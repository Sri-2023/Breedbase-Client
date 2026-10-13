"""BrAPI-chatbot (Chainlit): chat about Breedbase breeding data through BrAPI v2 and Claude, and convert breeders'
spreadsheets into Breedbase upload files.

Run from the repo root (Chainlit reads .chainlit/, public/ and chainlit.md from the working directory):
  uv run --extra chainlit chainlit run bin/chat-chainlit.py
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import chainlit as cl
from auth import GuestUsage, TokenStore, authenticate, guest_limit
from breedbase_upload_tools import (
  UPLOAD_TOOL_NAMES,
  UPLOAD_TOOLS,
  UploadSession,
  describe_upload_call,
  register_file,
  restore_file,
  run_upload_tool,
)
from chainlit.config import config as chainlit_config
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict
from databases import DB_SERVERS, site_url
from mcp_connection import McpConnection, McpUnavailable, ToolResult, result_address, server_for
from resume import RESUMED_NOTE, claude_history, files_to_restore, is_expired

MODEL = 'claude-opus-4-6'  # we are using for testing purposes. This model adapative thinking
MAX_TOOL_ROUNDS = 10  # limit for calling the tools by claude
MAX_TABLE_ROWS = 1000  # rows rendered in the chat; the CSV download always has everything
LOGO = '/public/favicon.png'
UPLOAD_INSTRUCTIONS = (
  Path(__file__).parent.parent / 'src' / 'breedbase_upload' / 'INSTRUCTIONS.md'
).read_text()  # this will load, the file irrespective of where the app is launched from


SYSTEM_TEMPLATE = """You are BrAPI-chatbot, a plant-breeding data assistant connected to the Breedbase database \
"{name}" ({base_url}) through the BrAPI v2 API. Answer the user's questions by calling the tools, then reply with a \
concise, well-formatted answer grounded in the data you retrieved.

Guidance:
- Start broad (programs, trials, studies) and narrow with DbId filters; DbIds are more reliable than names.
- Observations for a study: brapi_get service=observations params={{"studyDbId": "<id>"}}. Trait definitions: \
service=variables. Images: service=images. get_search_parameters lists the filters a search accepts.
- brapi_get and brapi_search save the full result on the server; every fetched table is already shown to the user with \
a CSV download, so don't reproduce large tables; summarise, highlight key numbers, and use small markdown tables only \
when helpful.
- To compute aggregates (means per germplasm, counts, etc.), read the rows you need with load_result (limit, offset, \
columns) instead of estimating from summaries. If a result is truncated, say how many rows exist in total.
- Some Breedbase features (e.g. product profiles, breeding-program settings) are not exposed through BrAPI. \
If the data isn't available, say so plainly and suggest where in the Breedbase web UI to find it.
- If a tool returns an error, tell the user what the error says; do not guess a cause it does not state."""

GUEST_NOTE = (
  'The user is a guest, so the tools only see public data. If they ask about private trials or data, explain that '
  'they need to log in with their own {name} account to see it.'
)

# Users' Breedbase tokens, by login identifier, in server memory only (never in the browser cookie or chat history).
TOKENS = TokenStore()
GUEST_USAGE = GuestUsage()  # questions asked per guest login (server memory)
GUEST_LIMIT_MESSAGE = "You've reached the guest limit. Log in to continue chatting."
# Only shown once chats are actually stored (a data layer is configured), so the app never promises what it can't do
SAVED_WHEN_LOGGED_IN = 'Your conversation will be saved if you log in.'

# Each login is for one database (please refer: bin/auth.py). Claude may answer general breeding questions, but questions about
# another database are sent there: the tools can only reach this database, and guessing would give wrong answers.
SCOPE_TEMPLATE = (
  'This chat is connected to {name} only; the user chose it on the start page. General plant-breeding questions '
  '(for example what an RCBD design is) are fine to answer. Do not answer questions about the data, trials, germplasm, '
  'people or programs of another breeding database ({others}), not even from general knowledge. Instead, say that '
  'this chat is for {name}, and that they should log out and choose that database on the start page to ask there.'
)

# The login page (public/login.js) shows these databases to choose from. Written at startup from DB_SERVERS, so the
# start page always matches the databases this app serves; the file is served without login and is not committed.
DATABASE_LIST = Path(__file__).parent.parent / 'public' / 'databases.json'
DATABASE_LIST.write_text(
  json.dumps([{'id': name, 'name': name, 'site': site_url(url)} for name, url in DB_SERVERS.items()], indent=2) + '\n'
)

# The login page script (public/login.js) is served without cache headers, so browsers keep an old copy after it
# changes and users see an outdated login page. Adding the file's modification time to its address makes browsers
# fetch it again whenever it changes (on the next app start). Chainlit builds each page from this setting per request.
LOGIN_SCRIPT = Path(__file__).parent.parent / 'public' / 'login.js'
chainlit_config.ui.custom_js = f'/public/login.js?v={int(LOGIN_SCRIPT.stat().st_mtime)}'

WELCOME = (
  'Welcome to the Breedbase chatbot, your interactive breeding chatbot for exploring [{name}]({url})! '
  'Ask me anything about breeding information, including phenomic, image, and genotyping data. '
  'You can also attach a spreadsheet of your field data, and I will help you convert it into a '
  'Breedbase-acceptable upload file.'
)


@dataclass
class ChatState:
  """Everything a chat needs that is not plain JSON, kept under one user_session key.

  When a chat ends, Chainlit saves the whole user_session with the stored chat, and anything that is not plain JSON
  is saved as null. Keeping these objects together means they are saved as a single null, so they can never push the
  saved data over Chainlit's 1 MB limit (above which it drops all of it, including the chosen database). Resumed
  chats rebuild this state in on_chat_resume instead.
  """

  uploads: UploadSession
  messages: list = field(default_factory=list)  # Claude conversation history, including tool calls
  mcp: McpConnection | None = None  # this conversation's session with the database's MCP server
  resumed: bool = False  # tells Claude that earlier tool results are gone (RESUMED_NOTE)


def _state() -> ChatState:
  return cl.user_session.get('state')


def _new_state() -> ChatState:
  # exports go in this chat's own Chainlit folder, which Chainlit deletes together with the uploads
  return ChatState(uploads=UploadSession(work_dir=cl.context.session.files_dir / 'exports'))


def _user_db() -> str:
  """The database this login is for (chosen on the start page); the first database if unknown."""
  user = cl.user_session.get('user')
  db = ((user.metadata or {}) if user else {}).get('db')
  return db if db in DB_SERVERS else next(iter(DB_SERVERS))


@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User | None = None):
  """Only the database this login is for. Switching databases means logging out and choosing another on the start
  page, so each database keeps its own login, history and guest limit."""
  db = (current_user.metadata or {}).get('db') if current_user else None
  names = [db] if db in DB_SERVERS else list(DB_SERVERS)
  return [
    cl.ChatProfile(
      name=name,
      markdown_description=f'Chat with [{name}]({site_url(DB_SERVERS[name])})',  # shown when hovering over the profile
      icon=LOGO,
      default=(i == 0),
    )
    for i, name in enumerate(names)
  ]


@cl.on_chat_start  # runs whenever user begins a chat
async def start():
  db = _user_db()  # chosen on the start page before logging in

  cl.user_session.set('db', db)
  cl.user_session.set('state', _new_state())
  # plain-JSON bookkeeping, saved with the stored chat so on_chat_resume can restore files under the same IDs
  cl.user_session.set('upload_files', {})  # file_id -> {'name', 'element_id'}

  # build and send the welcome msg
  db_web = site_url(DB_SERVERS[db])
  welcome_txt = WELCOME.format(name=db, url=db_web)
  if guest := _guest_id():
    left = GUEST_USAGE.remaining(guest, guest_limit())
    welcome_txt += (
      f'\n\nYou are using guest access: **{left} question(s) left**. Log in with your {db} '
      'account to keep chatting and to save your full conversation history so you never lose your progress.'
    )
  await cl.Message(content=welcome_txt).send()
  await _connect(db)

  # Warning an API key is missing. FYI, store the API key in the .env file and make sure to do .gitignore
  if not os.getenv('ANTHROPIC_API_KEY'):
    await cl.ErrorMessage(content='ANTHROPIC_API_KEY is not set. Add it  to the .env file and restart the app.').send()


async def _table_elements(tables: list) -> list:
  """Chat elements (a preview table + CSV download) for each (label, result_id, table) fetched from the MCP server."""
  elements = []

  for table_label, result_id, full_table in tables:
    if full_table.empty:
      continue
    preview_rows = full_table.head(MAX_TABLE_ROWS)  ## only show the first N rows in chat

    # shown table in the chat -
    preview_title = f'{table_label} ({len(full_table):,} rows)'
    elements.append(cl.Dataframe(name=preview_title, data=preview_rows, display='inline'))

    # full table as a downloadable csv
    csv_bytes = full_table.to_csv(index=False).encode()
    elements.append(cl.File(name=f'{result_id}.csv', content=csv_bytes, mime='text/csv', display='inline'))

  return elements


async def _respond(client: anthropic.AsyncAnthropic, db: str) -> None:
  """Run the tool loop for the latest user turn, streaming Claude's text and showing each BrAPI call as a step."""
  state = _state()
  messages, uploads = state.messages, state.uploads
  system_prompt = _system_prompt(db, state.resumed, guest=_guest_id() is not None)
  tables: list = []  # (label, result_id, table) fetched from the MCP server, shown under the reply
  downloads: list[cl.File] = []
  run_upload_tool_async = cl.make_async(run_upload_tool)
  reply_msg = None

  for _ in range(MAX_TOOL_ROUNDS):
    reply_msg = cl.Message(content='')
    async with client.beta.messages.stream(
      model=MODEL,
      max_tokens=64000,
      system=system_prompt,
      tools=state.mcp.tools_for_claude() + UPLOAD_TOOLS,  # the database's MCP tools and the file converter
      messages=messages,
      thinking={'type': 'adaptive'},  # this option is available on Claude Opus 4.6 and later models
      cache_control={'type': 'ephemeral'},
      betas=['server-side-fallback-2026-07-01'],
      fallbacks='default',
    ) as stream:
      async for delta in stream.text_stream:
        await reply_msg.stream_token(delta)
      response = await stream.get_final_message()

    if response.stop_reason == 'refusal':
      await reply_msg.stream_token('\n\n_Claude declined to answer this request._')
      break
    if response.stop_reason == 'max_tokens':
      await reply_msg.stream_token('\n\n_Response was cut off (hit the output limit)._')
      break

    # Keep full content blocks (including thinking) in history so later turns stay valid.
    messages.append({'role': 'assistant', 'content': response.content})
    if response.stop_reason == 'pause_turn':
      continue
    if response.stop_reason != 'tool_use':
      break

    if reply_msg.content:
      await reply_msg.send()
    tool_results = []
    for block in response.content:
      if block.type != 'tool_use':
        continue
      if block.name in UPLOAD_TOOL_NAMES:
        tool_results.append(await _run_upload_step(block, uploads, run_upload_tool_async, downloads))
        continue
      tool_results.append(await _run_mcp_step(block, state.mcp, tables))
    # all results for one assistant turn go back in a single user message.
    messages.append({'role': 'user', 'content': tool_results})
    reply_msg = None
  else:
    reply_msg = cl.Message(content='_Stopped after too many tool calls; try a more specific question._')

  if reply_msg is None:
    reply_msg = cl.Message(content='')
  reply_msg.elements = await _table_elements(tables) + downloads
  await reply_msg.send()


def _system_prompt(db: str, resumed: bool = False, guest: bool = False) -> str:
  others = ', '.join(name for name in DB_SERVERS if name != db) or 'none'
  parts = [
    SYSTEM_TEMPLATE.format(name=db, base_url=DB_SERVERS[db]),
    SCOPE_TEMPLATE.format(name=db, others=others),
    UPLOAD_INSTRUCTIONS,
  ]
  if guest:
    parts.append(GUEST_NOTE.format(name=db))
  if resumed:
    parts.append(RESUMED_NOTE)
  return '\n\n'.join(parts)


def _describe_mcp_call(name: str, args) -> str:
  """Short label for an MCP tool call, used as the step name in the UI."""
  if not isinstance(args, dict):
    return name
  target = '/'.join(str(args[k]) for k in ('service', 'db_id', 'sub') if args.get(k))
  if name == 'brapi_get':
    return f'Fetch {target}'
  if name == 'brapi_search':
    return f'Search {target}'
  return name.replace('_', ' ').capitalize()


async def _run_mcp_step(block, mcp: McpConnection, tables: list) -> dict:
  """Run one tool on the database's MCP server as a visible step. A brapi_get/brapi_search result is fetched from the
  server and added to `tables`, so the user sees it with a CSV download."""
  label = _describe_mcp_call(block.name, block.input)
  async with cl.Step(name=label, type='tool') as step:
    step.input = block.input
    try:
      result = await mcp.call(block.name, block.input)
    except McpUnavailable as e:  # reported to Claude, which tells the user
      result = ToolResult(text=f'Error: {e}', is_error=True, data=None)
    step.is_error = result.is_error
    step.output = result.text[:2000]
    if not result.is_error and block.name in ('brapi_get', 'brapi_search') and (address := result_address(result.data)):
      table = await mcp.fetch_table(address)
      if table is not None:
        tables.append((label, address.rsplit('/', 1)[-1], table))
  tool_result = {'type': 'tool_result', 'tool_use_id': block.id, 'content': result.text}
  if result.is_error:
    tool_result['is_error'] = True
  return tool_result


async def _run_upload_step(block, uploads: UploadSession, run_async, downloads: list) -> dict:
  """Run one upload-converter tool call as a visible step; an exported file is added to `downloads`."""

  step_name = describe_upload_call(block.name, block.input)

  async with cl.Step(name=step_name, type='tool') as step:
    step.input = block.input

    # run the upload tool in background thread
    payload = await run_async(uploads, block.name, block.input)

    # check whether the tool reported an error in its result
    is_error = 'error' in payload
    step.is_error = is_error
    step.output = json.dumps(payload, default=str)[:2000]

    # if the tool exported a file, offer it as download
    if 'path' in payload:
      download_file = cl.File(name=payload['file_name'], path=payload['path'], display='inline')
      downloads.append(download_file)

  # pack the result
  result = {
    'type': 'tool_result',
    'tool_use_id': block.id,
    'content': json.dumps(payload, default=str),
  }
  if is_error:
    result['is_error'] = True
  return result


def _with_attachments(
  message: cl.Message,
) -> str:  # this is necessary for claude to know, whether the file has been attached or not in the chat window
  """The user's text plus one line per attached file. Files are registered with the upload converter;
  Claude only sees their IDs and names, never their paths or contents."""
  uploads = _state().uploads
  upload_files = cl.user_session.get('upload_files')

  # register each attached file and describe it in one line
  file_lines = []
  for element in message.elements or []:
    if not getattr(element, 'path', None):
      continue  # skips elements
    file_id = register_file(uploads, Path(element.path), element.name)
    upload_files[file_id] = {'name': element.name, 'element_id': element.id}  # for on_chat_resume
    file_lines.append(f'[Attached file {file_id}: {element.name}]')

  ## sometimes, user mmight attached file but they might not type anything: ask Claude to check what they need
  text = message.content.strip()
  if not text and file_lines:
    text = (
      'I have attached a file for Breedbase. Please look at it and, if it is not clear, '
      'ask me what kind of upload I need (for example a single trial, multiple trials, or accessions).'
    )

  if not file_lines:
    return text
  return '\n\n'.join([text] + file_lines)


@cl.on_message  # runs every time the user sends a message
async def on_message(message: cl.Message):
  if not await _guest_may_ask():
    return
  db = cl.user_session.get('db')
  if not await _login_still_valid(db) or not await _connect(db):
    return
  messages = _state().messages
  history_len = len(messages)
  messages.append({'role': 'user', 'content': _with_attachments(message)})

  try:
    await _respond(anthropic.AsyncAnthropic(), db)
    if guest := _guest_id():
      GUEST_USAGE.record(guest)  # only answered questions count
    return
  except anthropic.AuthenticationError:
    error = 'Anthropic API key is missing or invalid. Set ANTHROPIC_API_KEY in .env and restart.'
  except anthropic.RateLimitError:
    error = 'Rate limited by the Anthropic API. Wait a moment and try again.'
  except anthropic.APIConnectionError:
    error = 'Could not reach the Anthropic API. Check your network.'
  except anthropic.APIStatusError as e:
    error = f'Anthropic API error ({e.status_code}): {e.message}'
  except TypeError as e:
    # The SDK raises TypeError when it finds no credentials at all
    missing = 'authentication method' in str(e)
    error = 'ANTHROPIC_API_KEY is not set. Add it to .env and restart.' if missing else f'Something went wrong: {e}'
  except Exception as e:  # noqa: BLE001 - keep the chat usable and the history valid on any failure
    error = f'Something went wrong: {e}'

  # The turn failed: roll back so no unanswered tool calls remain in history.
  del messages[history_len:]
  await cl.ErrorMessage(content=error).send()


# --- login and guest access ----------------------------------------------------------------------------------------


@cl.password_auth_callback
async def password_auth_callback(username: str, password: str) -> cl.User | None:
  """'<database>:<username>' from the start page, checked by that database's own Breedbase login (Option A); and
  '<database>:guest' gets a new limited guest login for that database (please refer: bin/auth.py).

  The user's Breedbase token goes into TOKENS (server memory), not into cl.User: Chainlit puts cl.User in the browser
  cookie and in stored chats, where a token must never be."""
  # The Breedbase login is a network request: run it in a thread so the app keeps serving other users meanwhile
  answer = await asyncio.to_thread(authenticate, username, password, DB_SERVERS)
  if answer is None:
    return None
  account, token = answer
  if token is not None:
    TOKENS.put(account.identifier, token)
  return cl.User(
    identifier=account.identifier,
    display_name=account.display_name,
    metadata={'role': account.role, 'db': account.database},
  )


def _guest_id() -> str | None:
  """The guest login's ID, or None for a user with an account."""
  user = cl.user_session.get('user')
  is_guest = user is not None and (user.metadata or {}).get('role') == 'guest'
  return user.identifier if is_guest else None


async def _guest_may_ask() -> bool:
  """False (after telling the guest to log in) once a guest has used all their questions."""
  guest = _guest_id()
  if guest is None or GUEST_USAGE.remaining(guest, guest_limit()) > 0:
    return True
  text = GUEST_LIMIT_MESSAGE + (f' {SAVED_WHEN_LOGGED_IN}' if get_data_layer() else '')
  db = _user_db()
  await cl.Message(content=text, actions=[cl.Action(name='login', payload={'db': db}, label='Log in')]).send()
  # public/login.js hides this database's guest button from now on
  await cl.send_window_message({'type': 'brapi-guest-limit', 'db': db})
  return False


@cl.action_callback('login')
async def on_login_clicked(action: cl.Action):
  # public/login.js logs the guest out and opens the login page of the same database
  await cl.send_window_message({'type': 'brapi-login', 'db': _user_db()})


async def _ask_to_log_in_again(db: str) -> None:
  await cl.Message(
    content=f'Your {db} login has expired (Breedbase logins last about 2 hours). Please log in again to continue.',
    actions=[cl.Action(name='login', payload={'db': db}, label='Log in')],
  ).send()


async def _login_still_valid(db: str) -> bool:
  """For logged-in users: True while their Breedbase token is valid. Otherwise asks them to log in again and returns
  False. Breedbase would not refuse an expired token (it answers with public data), so the expiry is checked here.
  The token is also gone after a server restart, which asks for a new login the same way. Guests have no token."""
  user = cl.user_session.get('user')
  if user is None or _guest_id() is not None:
    return True
  token = TOKENS.get(user.identifier)
  state = _state()
  if token is None:
    if state is not None and state.mcp is not None:
      await state.mcp.close()
      state.mcp = None
    await _ask_to_log_in_again(db)
    return False
  return True


async def _connect(db: str) -> bool:
  """This conversation's MCP session with the database's server, opened if needed: with the user's own Breedbase
  token, or none for guests (public data). Tells the user and returns False if the server cannot be reached."""
  state = _state()
  if state.mcp is not None:
    return True
  user = cl.user_session.get('user')
  token = None
  if user is not None and _guest_id() is None:
    stored = TOKENS.get(user.identifier)
    if stored is None:
      await _ask_to_log_in_again(db)
      return False
    token = stored.token
  try:
    connection = McpConnection(server_for(db), token)
    await connection.open()
  except McpUnavailable as e:
    await cl.ErrorMessage(content=f'{e} Please try again later.').send()
    return False
  state.mcp = connection
  return True


# --- chat history ----------------------------------------------------------------------------------------------


@cl.on_chat_end
async def on_chat_end() -> None:
  """Runs just before Chainlit saves user_session with the stored chat: record the next conversation number, so a
  resumed chat never reuses an ID (c1, c2, ...) that Claude saw before. Live state is left alone, because the same
  browser may reconnect to this session."""
  if state := _state():
    cl.user_session.set('next_conversion', state.uploads.next_conversion)
    if state.mcp is not None:  # end this conversation's MCP session; it is reopened if the browser reconnects
      await state.mcp.close()
      state.mcp = None


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
  """Continue a chat from the history sidebar. Chainlit has already shown its messages and files and restored the
  plain-JSON parts of user_session (chat profile, upload_files, next_conversion); this rebuilds the rest."""
  cl.user_session.set('db', _user_db())
  state = _new_state()
  cl.user_session.set('state', state)
  upload_files = cl.user_session.get('upload_files') or {}
  cl.user_session.set('upload_files', upload_files)

  if is_expired(thread):
    await cl.Message(
      content='This chat has expired, so its files and history are no longer available. Start a new chat to continue.'
    ).send()
    return

  state.resumed = True
  state.messages = claude_history(thread, upload_files)
  db = cl.user_session.get('db')
  if await _login_still_valid(db):
    await _connect(db)
  state.uploads.next_conversion = int(cl.user_session.get('next_conversion') or 1)
  # New files continue after every ID this chat used, including files that can no longer be restored
  state.uploads.next_file = 1 + max((int(f.removeprefix('f')) for f in upload_files), default=0)

  missing = []
  for file_id, name, element in files_to_restore(thread, upload_files):
    path = _stored_file(element)
    if path is None:
      missing.append(f'{name} ({file_id})')
    else:
      restore_file(state.uploads, file_id, path, name)
  if missing:
    await cl.Message(
      content='These files from this chat are no longer available: '
      + ', '.join(missing)
      + '. Attach them again to continue working with them.'
    ).send()


def _stored_file(element: dict | None) -> Path | None:
  """The file on disk for a stored attachment, or None. Filled in by the file-storage step (bin/history/storage.py);
  until then attachments are not stored, so resumed chats report them as no longer available."""
  path = (element or {}).get('path')
  return Path(path) if path and Path(path).is_file() else None
