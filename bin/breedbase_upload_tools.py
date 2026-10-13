"""Claude tools for converting a breeder's file into a Breedbase upload file.

Framework-independent like mcp_connection.py: nothing here imports Chainlit. The chat app keeps one UploadSession per
chat, calls register_file() for each attachment, and sends tool calls named in UPLOAD_TOOL_NAMES to run_upload_tool().

Claude only ever sees file and conversion IDs, column names, a sample of rows and the issues found; the full table
stays on the server, so a 100,000-row file costs the same as a 20-row one.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))  # the breedbase_upload package lives in src/

from breedbase_upload.engine import Conversion, convert, export, suggest_mapping
from breedbase_upload.reader import ReaderError, Table, read_table
from breedbase_upload.rules import Issue
from breedbase_upload.template import Template, TemplateError, available_templates, load_template

SAMPLE_ROWS = 20  # rows shown to Claude by inspect_file
PREVIEW_ROWS = 10  # rows shown to Claude after convert_file
MAX_ISSUES = 30  # issues shown to Claude per conversion (errors first)
MAX_ISSUE_ROWS = 10  # row numbers listed per issue
EXPORT_FORMATS = ('xlsx', 'csv')


@dataclass
class ConversionRecord:
  file_id: str
  upload_type: str
  result: Conversion


@dataclass
class UploadSession:
  work_dir: Path  # where exports go: the chat's own folder, cleaned up with the chat
  files: dict[str, Path] = field(default_factory=dict)  # file_id -> uploaded file on disk
  names: dict[str, str] = field(default_factory=dict)  # file_id -> file name as uploaded
  tables: dict[tuple, Table] = field(default_factory=dict)  # (file_id, sheet, upload_type) -> table, read once
  conversions: dict[str, ConversionRecord] = field(default_factory=dict)  # conversion_id -> convert_file result
  # Counters, not len(): IDs stay unique when a resumed chat restores only some files or none of its conversions
  next_file: int = 1
  next_conversion: int = 1


def register_file(session: UploadSession, path: Path, name: str) -> str:  # this stores the file's real path
  """Store an uploaded file and return the file_id Claude uses for it. Called by the chat app, not by Claude."""
  file_id = f'f{session.next_file}'
  session.next_file += 1
  session.files[file_id] = Path(path)
  session.names[file_id] = name
  return file_id


def restore_file(session: UploadSession, file_id: str, path: Path, name: str) -> None:
  """Put a file from a resumed chat back under its old ID, and keep new IDs after it."""
  session.files[file_id] = Path(path)
  session.names[file_id] = name
  session.next_file = max(session.next_file, int(file_id.removeprefix('f')) + 1)


# --- the tools: each takes (session, args) and returns a JSON-friendly dict ------------------------------------------


def _list_upload_types(session: UploadSession, args: dict) -> dict:
  types = []
  for name in available_templates():
    t = load_template(name)
    types.append(
      {
        'upload_type': name,
        'title': t.title,
        'required_columns': list(t.required_columns),
        **({'one_trial_per_file': t.group_by is None} if _is_trial(t) else {}),
      }
    )
  return {'upload_types': types}


def _inspect_file(session: UploadSession, args: dict) -> dict:
  file_id, sheet = args['file_id'], args.get('sheet')
  table = _load(session, file_id, sheet, args.get('upload_type'))
  return {
    'file_id': file_id,
    'file_name': session.names[file_id],
    'format': table.format,
    'sheets': table.sheets,
    'sheet': table.sheet,
    'header_row': table.header_row,
    'columns': list(table.df.columns),
    'row_count': len(table.df),
    'notes': table.notes,
    'sample_rows': _preview(table.df, SAMPLE_ROWS),
  }


def _get_template(session: UploadSession, args: dict) -> dict:
  t = load_template(args['upload_type'])
  return {
    'upload_type': t.name,
    'title': t.title,
    **({'one_trial_per_file': t.group_by is None} if _is_trial(t) else {}),
    **({'stock_types': {key: st.column for key, st in t.stock_types.items()}} if _is_trial(t) else {}),
    'columns': [
      {
        'name': c.name,
        'required': c.required,
        **({'header_required': True} if c.header_required else {}),  # header always written; values may be blank
        'type': c.type,
        **({'several_values_separated_by': c.separator} if c.multi_value else {}),
        **({'unique': True} if c.unique else {}),
        **({'allowed': list(c.allowed)} if c.allowed else {}),
        **({'unit': c.unit} if c.unit else {}),
        **({'description': c.description} if c.description else {}),
      }
      for c in (t.columns[name] for name in t.column_order)
    ],
    **(
      {
        'required_when': [
          {'column': r.column, 'when': r.when_column, 'is_one_of': list(r.values)} for r in t.required_when
        ]
      }
      if t.required_when
      else {}
    ),
  }


def _propose_mapping(session: UploadSession, args: dict) -> dict:
  t = load_template(args['upload_type'])
  table = _load(session, args['file_id'], args.get('sheet'), t.name)
  matches = suggest_mapping(list(table.df.columns), t)
  mapped = {m.target for m in matches if m.target in t.columns}
  return {
    'mapping': [
      {'column': pos, 'header': m.header, 'target': m.target, 'how': m.how, 'score': m.score}
      for pos, m in enumerate(matches)
    ],
    'missing_required': [c for c in t.required_columns if c not in mapped],
    # header-only columns (values may be blank) are added by convert_file when the file lacks them
    'added_as_blank_columns': [c for c in t.header_columns if c not in mapped and t.columns[c].header_required],
    **({'missing_stock_column': not any(c in mapped for c in t.stock_columns)} if _is_trial(t) else {}),
    'needs_confirmation': [m.header for m in matches if m.how == 'fuzzy'],
    'dropped': [m.header for m in matches if m.how == 'drop'],
  }


def _convert_file(session: UploadSession, args: dict) -> dict:
  t = load_template(args['upload_type'])
  file_id = args['file_id']
  table = _load(session, file_id, args.get('sheet'), t.name)

  targets = [m.target for m in suggest_mapping(list(table.df.columns), t)]
  for override in args.get('mapping_overrides') or []:
    pos = override['column']
    if not 0 <= pos < len(targets):
      raise ValueError(f'mapping_overrides column {pos} is out of range; the file has {len(targets)} columns (0-based)')
    targets[pos] = override.get('target')

  constants = {k: str(v) for k, v in (args.get('constants') or {}).items()}
  result = convert(table.df, t, targets, constants, args.get('date_order'))

  conversion_id = f'c{session.next_conversion}'
  session.next_conversion += 1
  session.conversions[conversion_id] = ConversionRecord(file_id, t.name, result)
  errors = sum(i.severity == 'error' for i in result.issues)
  return {
    'conversion_id': conversion_id,
    'ok': result.ok,
    'error_count': errors,
    'warning_count': len(result.issues) - errors,
    'row_count': len(result.df),
    'columns': list(result.df.columns),
    'issues': _issues_payload(result.issues),
    'changes': result.changes,
    'preview_rows': _preview(result.df, PREVIEW_ROWS),
    **({'dialog_stock_type': _dialog_stock_type(t, result.df)} if _is_trial(t) and t.group_by is None else {}),
  }


def _export_file(session: UploadSession, args: dict) -> dict:
  conversion_id = args['conversion_id']
  record = session.conversions.get(conversion_id)
  if record is None:
    raise ValueError(f'Unknown conversion_id {conversion_id!r}; conversions: {list(session.conversions)}')
  if not record.result.ok:
    return {'error': 'This conversion still has errors, so no file was exported. Fix them and convert again.'}
  fmt = args.get('format') or 'xlsx'
  session.work_dir.mkdir(parents=True, exist_ok=True)
  path = export(record.result.df, session.work_dir / f'{record.upload_type}_upload_{conversion_id}.{fmt}')
  return {
    'path': str(path),
    'file_name': path.name,
    'row_count': len(record.result.df),
    'warning_count': len(record.result.issues),
  }


_HANDLERS = {
  'list_upload_types': _list_upload_types,
  'inspect_file': _inspect_file,
  'get_template': _get_template,
  'propose_mapping': _propose_mapping,
  'convert_file': _convert_file,
  'export_file': _export_file,
}

# --- tool definitions for Claude -----------------------------------------------------------------------------------

_FILE_ID = {'type': 'string', 'description': "ID of an attached file, e.g. 'f1'"}
_SHEET = {'type': 'string', 'description': 'Sheet to read; only needed when the file has several sheets'}
_UPLOAD_TYPE = {
  'type': 'string',
  'description': "Upload type from list_upload_types: e.g. 'multi_trial', 'single_trial', 'accessions', 'crosses', "
  "'pedigree' or 'vector'",
}

UPLOAD_TOOLS: list[dict] = [
  {
    'name': 'list_upload_types',
    'description': (
      'List the Breedbase upload formats the converter supports (trials, accessions, crosses, pedigrees, vectors) '
      'with their required columns. Call this first when the user wants to convert a file, to choose the upload type '
      'with them.'
    ),
    'input_schema': {'type': 'object', 'properties': {}},
  },
  {
    'name': 'inspect_file',
    'description': (
      'Read an attached file and return its sheets, header row, column names, row count and a sample of rows. Call '
      'it before anything else for a new file. If the file has several sheets, ask the user which one to use.'
    ),
    'input_schema': {
      'type': 'object',
      'properties': {'file_id': _FILE_ID, 'sheet': _SHEET, 'upload_type': _UPLOAD_TYPE},
      'required': ['file_id'],
    },
    'eager_input_streaming': True,
  },
  {
    'name': 'get_template',
    'description': (
      'Return the Breedbase rules for one upload type: every column, whether it is required, its type and allowed '
      'values. Use it to explain what is missing and to ask the user precise questions.'
    ),
    'input_schema': {'type': 'object', 'properties': {'upload_type': _UPLOAD_TYPE}, 'required': ['upload_type']},
    'eager_input_streaming': True,
  },
  {
    'name': 'propose_mapping',
    'description': (
      "Suggest which Breedbase column each of the file's columns is, by position. Show the mapping to the user and "
      'ask them to confirm it, especially entries in needs_confirmation (fuzzy matches) and dropped columns that '
      'might matter. Also lists required columns no header matched.'
    ),
    'input_schema': {
      'type': 'object',
      'properties': {'file_id': _FILE_ID, 'upload_type': _UPLOAD_TYPE, 'sheet': _SHEET},
      'required': ['file_id', 'upload_type'],
    },
    'eager_input_streaming': True,
  },
  {
    'name': 'convert_file',
    'description': (
      'Convert the file to the Breedbase format and check it against every Breedbase rule. Starts from '
      'propose_mapping; pass mapping_overrides only for columns the user changed. Returns ok, issues (errors block '
      'the export, warnings do not), every change made, and preview rows. Call it again after each answer from the '
      'user. Never invent values: only pass constants the user gave. If an issue says dates could be day-first or '
      'month-first, ask the user and call again with date_order.'
    ),
    'input_schema': {
      'type': 'object',
      'properties': {
        'file_id': _FILE_ID,
        'upload_type': _UPLOAD_TYPE,
        'sheet': _SHEET,
        'mapping_overrides': {
          'type': 'array',
          'description': 'Changes to the proposed mapping: column position (0-based) and Breedbase column, or null '
          'to leave the column out',
          'items': {
            'type': 'object',
            'properties': {'column': {'type': 'integer'}, 'target': {'type': ['string', 'null']}},
            'required': ['column', 'target'],
          },
        },
        'constants': {
          'type': 'object',
          'description': "Values the user gave for every row, e.g. {'breeding_program': 'IITA'}. Fill blanks only.",
          'additionalProperties': {'type': 'string'},
        },
        'date_order': {'type': 'string', 'enum': ['DMY', 'MDY'], 'description': 'How to read dates like 03/05/2025'},
      },
      'required': ['file_id', 'upload_type'],
    },
    'eager_input_streaming': True,
  },
  {
    'name': 'export_file',
    'description': (
      'Write the converted file for the user to download and upload to Breedbase. Only works when the conversion '
      'has no errors. Tell the user about any remaining warnings before exporting.'
    ),
    'input_schema': {
      'type': 'object',
      'properties': {
        'conversion_id': {'type': 'string', 'description': "ID returned by convert_file, e.g. 'c2'"},
        'format': {'type': 'string', 'enum': list(EXPORT_FORMATS), 'description': 'Default xlsx'},
      },
      'required': ['conversion_id'],
    },
    'eager_input_streaming': True,
  },
]
UPLOAD_TOOL_NAMES = {t['name'] for t in UPLOAD_TOOLS}


def run_upload_tool(session: UploadSession, name: str, args: dict) -> dict:
  """Run one tool call. Errors the user or Claude can fix come back as {'error': ...} instead of raising."""
  try:
    _validate(name, args)
    return _HANDLERS[name](session, args)
  except (ReaderError, TemplateError, ValueError) as e:
    return {'error': str(e)}


_LABELS = {
  'list_upload_types': 'List upload types',
  'inspect_file': 'Inspect file',
  'get_template': 'Read Breedbase template',
  'propose_mapping': 'Propose column mapping',
  'convert_file': 'Convert and check',
  'export_file': 'Export file',
}


def describe_upload_call(name: str, args) -> str:
  """Short label for a tool call, used as the step name in the chat UI."""
  label = _LABELS.get(name, name)
  if not isinstance(args, dict):
    return label
  detail = ' · '.join(str(args[k]) for k in ('file_id', 'upload_type', 'conversion_id') if args.get(k))
  return f'{label} ({detail})' if detail else label


# --- helpers -------------------------------------------------------------------------------------------------------

_REQUIRED = {
  'list_upload_types': (),
  'inspect_file': ('file_id',),
  'get_template': ('upload_type',),
  'propose_mapping': ('file_id', 'upload_type'),
  'convert_file': ('file_id', 'upload_type'),
  'export_file': ('conversion_id',),
}


def _validate(name: str, args) -> None:
  """Eager input streaming skips server-side validation, so check the shape ourselves."""
  if name not in _HANDLERS:
    raise ValueError(f'Unknown tool {name!r}')
  if not isinstance(args, dict):
    raise ValueError('Tool input must be a JSON object')  # noqa: TRY004 -- ValueError so run_upload_tool reports it to Claude
  for key in _REQUIRED[name]:
    if key not in args:
      raise ValueError(f'Missing required field {key!r}')
  for key in ('file_id', 'upload_type', 'sheet', 'conversion_id'):
    if args.get(key) is not None and not isinstance(args[key], str):
      raise ValueError(f'{key} must be a string')
  if args.get('constants') is not None and not isinstance(args['constants'], dict):
    raise ValueError('constants must be an object')
  overrides = args.get('mapping_overrides')
  if overrides is not None:
    if not isinstance(overrides, list):
      raise ValueError('mapping_overrides must be a list')
    for o in overrides:
      if not isinstance(o, dict) or not isinstance(o.get('column'), int) or isinstance(o.get('column'), bool):
        raise ValueError('each mapping_overrides entry needs an integer column')  # noqa: TRY004 -- ValueError so run_upload_tool reports it to Claude
      if o.get('target') is not None and not isinstance(o['target'], str):
        raise ValueError('mapping_overrides target must be a string or null')
  if args.get('date_order') not in (None, 'DMY', 'MDY'):
    raise ValueError("date_order must be 'DMY' or 'MDY'")
  if args.get('format') not in (None, *EXPORT_FORMATS):
    raise ValueError(f'format must be one of {list(EXPORT_FORMATS)}')


def _load(session: UploadSession, file_id: str, sheet: str | None, upload_type: str | None = None) -> Table:
  """Read a file once per (file, sheet, upload type); the upload type's names help find the header row."""
  if file_id not in session.files:
    raise ValueError(f'Unknown file_id {file_id!r}; attached files: {list(session.files) or "none"}')
  key = (file_id, sheet, upload_type)
  if key not in session.tables:
    if upload_type:
      known = load_template(upload_type).known_headers
    else:  # type not chosen yet: any template's names
      known = frozenset().union(*(load_template(n).known_headers for n in available_templates()))
    session.tables[key] = read_table(session.files[file_id], sheet, known)
  return session.tables[key]


def _issues_payload(issues: list[Issue], limit: int = MAX_ISSUES) -> list[dict]:
  """Errors first, at most `limit` issues and MAX_ISSUE_ROWS row numbers each, so Claude's context stays small."""
  ordered = sorted(issues, key=lambda i: i.severity != 'error')
  payload = [
    {
      'severity': i.severity,
      'rule': i.rule,
      'message': i.message,
      **({'column': i.column} if i.column else {}),
      **({'trial': i.trial} if i.trial else {}),
      **({'rows': [int(r) for r in i.rows[:MAX_ISSUE_ROWS]], 'row_count': len(i.rows)} if i.rows else {}),
    }
    for i in ordered[:limit]
  ]
  if len(ordered) > limit:
    payload.append({'more': len(ordered) - limit, 'message': f'{len(ordered) - limit} more issue(s) not shown'})
  return payload


def _preview(df: pd.DataFrame, rows: int) -> list[dict]:
  """First rows with the user's row numbers. Repeated headers get ' (2)', ' (3)' so no value is lost."""
  seen: dict[str, int] = {}
  names = []
  for c in df.columns:
    seen[c] = seen.get(c, 0) + 1
    names.append(c if seen[c] == 1 else f'{c} ({seen[c]})')
  head = df.head(rows)
  return [{'row': int(i), **dict(zip(names, values))} for i, values in zip(head.index, head.to_numpy().tolist())]


def _is_trial(t: Template) -> bool:
  """Trial templates (single_trial, multi_trial) have stock types; other upload types do not."""
  return bool(t.stock_types)


def _dialog_stock_type(t: Template, df: pd.DataFrame) -> str:
  """The 'Stock type' to choose in Breedbase's single-trial upload dialog for this file."""
  for st in t.stock_types.values():
    if st.column in df.columns:
      return st.dialog_label
  return t.stock_types[t.default_stock_type].dialog_label
