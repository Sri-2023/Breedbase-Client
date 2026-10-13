"""
The chat app's connection to a database's MCP server.

Every database has its own MCP server (please refer: src/main.py over HTTP; one Docker container per database). The chat app is an
MCP client of the server for the database the user logged in to, set in .env:

  MCP_URL_CASSAVABASE=http://127.0.0.1:8102/mcp
  MCP_TOKEN_CASSAVABASE=<the server's MCP_SERVER_TOKEN>     # empty if that server does not require one

Each conversation opens its own MCP session, so the server keeps its results separate from other users'. 
Every request carries the server token (Authorization: Bearer) and, for logged-in users, their own Breedbase token
(X-Breedbase-Token), so Breedbase applies that user's permissions and also it uses the user-defined roles that were set to that user. 
Guests send no Breedbase token: public data only.

The X-Breedbase-Token authenticates users based on their active database login. 
Users must first log in to the target database using the link provided on the sign-in page. 
Once authenticated, a chat session remains valid for two hours. 
After this period, the user must refresh the token to continue


Tables: brapi_get and brapi_search store their full result on the server and return its address
(brapi://results/<session>/<result>). The chat app fetches the CSV from the server's /download route itself, so the
user's browser never talks to the MCP server.

Nothing here imports Chainlit, so it can be tested on its own.
"""

import contextlib
import io
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
import pandas as pd
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

TOKEN_HEADER = 'X-Breedbase-Token'
# Tools chat users do not get: download_images writes files on the server; the other two hand out links to the MCP
# server itself, which users' browsers cannot (and should not) reach; the chat app shows tables and downloads instead.
EXCLUDED_TOOLS = frozenset({'download_images', 'get_download_instructions', 'quick_download_link'})
RESULT_ADDRESS = re.compile(r'brapi://results/([^/\s\'"]+)/([^/\s\'"]+)')
DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_TABLE_BYTES = 50_000_000  # a stored result larger than this is not fetched into the chat


class McpUnavailable(RuntimeError):
  """The database's MCP server is not configured or cannot be reached."""


@dataclass(frozen=True)
class McpServer:
  url: str  # e.g. http://127.0.0.1:8102/mcp
  token: str | None  # the server's MCP_SERVER_TOKEN

  @property
  def base_url(self) -> str:
    return self.url.rstrip('/').removesuffix('/mcp')


def server_for(database: str, environ: Mapping[str, str] = os.environ) -> McpServer:
  """The MCP server of one database, from MCP_URL_<DB> and MCP_TOKEN_<DB>."""
  key = database.upper()
  url = (environ.get(f'MCP_URL_{key}') or '').strip()
  if not url:
    raise McpUnavailable(f'No MCP server is configured for {database} (MCP_URL_{key} in .env).')
  return McpServer(url=url, token=(environ.get(f'MCP_TOKEN_{key}') or '').strip() or None)


@dataclass
class ToolResult:
  text: str  # what Claude receives
  is_error: bool
  data: dict | None  # the structured result, when the tool returned one


class McpConnection:
  """
  One conversation's MCP session with one database's server. 
  Opened on first use; reopened once if the server restarted or the session expired 
  (the server's earlier results are then gone, and Claude is told so).
  """

  def __init__(self, server: McpServer, user_token: str | None):
    self.server = server
    self._user_token = user_token  # the user's Breedbase token; and none for guests
    self._client: Client | None = None
    self.tools: list = []

  def _headers(self) -> dict[str, str]:
    headers = {}
    if self.server.token:
      headers['Authorization'] = f'Bearer {self.server.token}'
    if self._user_token:
      headers[TOKEN_HEADER] = self._user_token
    return headers

  def set_user_token(self, token: str | None) -> None:
    """After the user logged in again: new token, new session."""
    self._user_token = token
    self._client = None  # the next call opens a session with the new headers

  async def open(self) -> None:
    if self._client is not None:
      return
    client = Client(StreamableHttpTransport(self.server.url, headers=self._headers()))
    try:
      await client.__aenter__()
      self.tools = [t for t in await client.list_tools() if t.name not in EXCLUDED_TOOLS]
    except Exception as e:
      raise McpUnavailable(f'The MCP server at {self.server.url} is not reachable ({type(e).__name__}).') from e
    self._client = client

  async def close(self) -> None:
    client, self._client = self._client, None
    if client is not None:
      # Closing a connection that is already gone (server restarted) is not worth reporting
      with contextlib.suppress(Exception):
        await client.__aexit__(None, None, None)

  def tools_for_claude(self) -> list[dict]:
    """The server's tools in the shape the Claude API expects."""
    return [{'name': t.name, 'description': t.description or '', 'input_schema': t.inputSchema} for t in self.tools]

  @property
  def tool_names(self) -> set[str]:
    return {t.name for t in self.tools}

  async def call(self, name: str, args: dict) -> ToolResult:
    """Run one tool on the server. A lost session (server restart, expiry) is reopened once and the call retried."""
    await self.open()  # first, so the server's tool list is known before checking the name
    if name in EXCLUDED_TOOLS or name not in self.tool_names:
      return ToolResult(text=f'Error: tool {name!r} is not available here.', is_error=True, data=None)
    for attempt in (1, 2):
      await self.open()
      try:
        result = await self._client.call_tool(name, args, raise_on_error=False)
        break
      except McpUnavailable:
        raise
      except Exception as e:
        await self.close()
        if attempt == 2:
          raise McpUnavailable(f'The MCP server stopped answering ({type(e).__name__}).') from e
    text = '\n'.join(getattr(block, 'text', '') for block in result.content if getattr(block, 'text', None))
    data = result.structured_content if isinstance(result.structured_content, dict) else None
    if data is not None and set(data) == {'result'} and isinstance(data['result'], dict):
      data = data['result']  # FastMCP wraps non-object return values as {'result': ...}
    is_error = bool(result.is_error) or bool(data and data.get('error'))
    return ToolResult(text=text or json.dumps(data or {}), is_error=is_error, data=data)

  async def fetch_table(self, address: str) -> pd.DataFrame | None:
    """The full stored result for a brapi://results/<session>/<result> address, as a table of strings; None if it
    cannot be fetched. Uses the server's /download route with the server token, from the chat app's side."""
    match = RESULT_ADDRESS.search(address or '')
    if not match:
      return None
    session_id, result_id = match.groups()
    headers = {'Authorization': f'Bearer {self.server.token}'} if self.server.token else {}
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS) as http:
      response = await http.get(f'{self.server.base_url}/download/{session_id}/{result_id}', headers=headers)
    if response.status_code != 200 or len(response.content) > MAX_TABLE_BYTES:
      return None
    return pd.read_csv(io.BytesIO(response.content), dtype=str, keep_default_na=False)


def result_address(data: dict | None) -> str | None:
  """The brapi://results/... address in a brapi_get/brapi_search result, if any."""
  if not data:
    return None
  match = RESULT_ADDRESS.search(json.dumps(data))
  return match.group(0) if match else None
