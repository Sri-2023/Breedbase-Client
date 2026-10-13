"""Rebuild a resumed chat from what Chainlit stored for it (the thread), for @cl.on_chat_resume.

Chainlit shows the old messages, steps and files by itself. What it cannot give back is our own state: Claude's
conversation history (the API needs exact content blocks, which are not stored) and the attached files under the IDs
Claude knows them by. This module rebuilds both from the thread. Nothing here imports Chainlit, so it can be tested on
its own; the thread is the ThreadDict Chainlit passes to on_chat_resume.

Old tool calls are not replayed: their BrAPI results and conversions no longer exist, so Claude gets the visible
conversation as text plus RESUMED_NOTE, and calls the tools again when needed.
"""

import datetime as dt
from collections.abc import Mapping

HISTORY_DAYS = 5  # a chat is deleted this many days after its last message

RESUMED_NOTE = (
  'This conversation was resumed from the chat history. Tool results and conversion IDs from before (such as c1) '
  'no longer exist; call the tools again when you need data or a conversion. Attached files keep their IDs (f1, ...) '
  'unless the user is told a file is no longer available.'
)

_ROLES = {'user_message': 'user', 'assistant_message': 'assistant'}


def parse_time(text: str | None) -> dt.datetime | None:
  """A stored ISO timestamp as an aware UTC datetime; None when missing or unreadable."""
  if not text:
    return None
  try:
    value = dt.datetime.fromisoformat(text)  # Python 3.11+ also reads a trailing Z
  except ValueError:
    return None
  return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


def last_activity(thread: Mapping) -> dt.datetime | None:
  """When the chat was last used: its newest step, or when it was created."""
  times = [parse_time(s.get('createdAt')) for s in thread.get('steps') or []] + [parse_time(thread.get('createdAt'))]
  times = [t for t in times if t is not None]
  return max(times) if times else None


def is_expired(thread: Mapping, now: dt.datetime | None = None, days: int = HISTORY_DAYS) -> bool:
  """True if the chat had no activity for `days` days. Unknown times never count as expired."""
  last = last_activity(thread)
  now = now or dt.datetime.now(dt.UTC)
  return last is not None and now - last > dt.timedelta(days=days)


def _file_ids_by_element(upload_files: Mapping[str, Mapping]) -> dict[str, str]:
  return {info['element_id']: file_id for file_id, info in upload_files.items() if info.get('element_id')}


def claude_history(thread: Mapping, upload_files: Mapping[str, Mapping]) -> list[dict]:
  """The visible conversation as Claude messages: text only, alternating user/assistant, starting with the user.

  upload_files: file_id -> {'name', 'element_id'}, saved by the chat app when files were attached. The stored user
  messages only hold what the user typed, so the '[Attached file f1: name]' lines Claude saw are added back.
  """
  by_element = _file_ids_by_element(upload_files)
  attached: dict[str, list[str]] = {}  # step id -> attachment lines
  for element in thread.get('elements') or []:
    file_id = by_element.get(element.get('id'))
    if file_id and element.get('forId'):
      attached.setdefault(element['forId'], []).append(f'[Attached file {file_id}: {upload_files[file_id]["name"]}]')

  turns: list[dict] = []
  steps = sorted(thread.get('steps') or [], key=lambda s: s.get('createdAt') or '')
  for step in steps:
    role = _ROLES.get(step.get('type'))
    if role is None or step.get('isError'):
      continue  # tool steps, errors and other step types are not part of the conversation
    text = '\n\n'.join(p for p in [(step.get('output') or '').strip(), *attached.get(step.get('id'), [])] if p)
    if not text:
      continue
    if turns and turns[-1]['role'] == role:  # e.g. a reply sent in parts around tool calls
      turns[-1]['content'] += '\n\n' + text
    else:
      turns.append({'role': role, 'content': text})

  while turns and turns[0]['role'] != 'user':
    turns.pop(0)  # the welcome message comes before the first question
  if turns and turns[-1]['role'] == 'user':
    turns.pop()  # an unanswered question (the chat ended mid-turn): the user can ask it again
  return turns


def files_to_restore(thread: Mapping, upload_files: Mapping[str, Mapping]) -> list[tuple[str, str, Mapping | None]]:
  """(file_id, name, stored element or None) for every file attached in this chat, in ID order."""
  elements = {e.get('id'): e for e in thread.get('elements') or []}
  ordered = sorted(upload_files.items(), key=lambda item: int(item[0].removeprefix('f') or 0))
  return [(file_id, info['name'], elements.get(info.get('element_id'))) for file_id, info in ordered]
