"""Logins for the chat app (Option A): each user logs in with their own Breedbase account for the chosen database.

The start page asks which database first and sends '<database>:<username>'. The password goes once to that
database's own Breedbase login (POST <base_url>/token, the same login the website and BrAPI clients use) and is then
forgotten. Breedbase answers with a token that is valid for a limited time; it is kept in server memory only
(TokenStore), never in the browser cookie or the chat history, and sent with every MCP call so Breedbase applies that
user's own permissions.

Guests sign in with the reserved username 'guest' (the "Continue as guest" button on the login page does this). They
have no token, so they only see public data, and they can ask GUEST_MESSAGE_LIMIT questions per database (default 3;
0 turns guest access off). The count is kept in server memory per guest login, so it is a soft limit.

Breedbase does not reject an expired token (it quietly answers with public data), so expiry is tracked here: a token
is treated as expired TOKEN_MARGIN_SECONDS before Breedbase's own expiry time, and the user is asked to log in again.

Nothing here imports Chainlit, so it can be tested on its own.
"""

import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import requests

LOGIN_SEPARATOR = ':'  # the login page sends '<database>:<username>'
GUEST_USERNAME = 'guest'
GUEST_PASSWORD = 'guest'  # not a secret: the guest button sends it; the limit is what protects the service
DEFAULT_GUEST_LIMIT = 3
LOGIN_TIMEOUT_SECONDS = 30
TOKEN_MARGIN_SECONDS = 300  # ask the user to log in again 5 minutes before the token expires

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Account:
  identifier: str  # unique per database: '<database>:<username>', or '<database>:guest-<random>' per guest login
  display_name: str
  role: str  # 'user' or 'guest'
  database: str  # the one database this login is for

  @property
  def is_guest(self) -> bool:
    return self.role == 'guest'


@dataclass(frozen=True)
class BreedbaseToken:
  """A user's Breedbase access token. Kept in server memory only; never logged, stored or sent to the browser."""

  token: str
  expires_at: float  # Unix time, from Breedbase's answer

  def valid(self, now: float | None = None) -> bool:
    return (now or time.time()) < self.expires_at - TOKEN_MARGIN_SECONDS

  def __repr__(self) -> str:  # so a token never ends up in a log by accident
    return f'BreedbaseToken(expires_at={self.expires_at:.0f})'


# --- Breedbase login -----------------------------------------------------------------------------------------------


def breedbase_login(base_url: str, username: str, password: str, post: Callable = requests.post):
  """Log in to one Breedbase site. Returns (BreedbaseToken, display name), or None if the site refuses the login.

  The request is Breedbase's own login (SGN's password grant, as in src/client/auth/sgn_auth.py). Network problems
  are logged without the password and also return None, so the user sees the same "could not log in" message.
  """
  try:
    response = post(
      f'{base_url.rstrip("/")}/token',
      data={'grant_type': 'password', 'username': username, 'password': password},
      timeout=LOGIN_TIMEOUT_SECONDS,
    )
  except requests.RequestException as e:
    log.warning('Breedbase login to %s failed: %s', base_url, type(e).__name__)
    return None
  if response.status_code != 200:
    return None
  try:
    data = response.json()
  except ValueError:
    return None
  if not data.get('access_token'):
    return None  # Breedbase answers 200 with an error message (no token) for a wrong password
  expires_in = float(data.get('expires_in') or 7200)  # Breedbase's default is 2 hours
  token = BreedbaseToken(token=data['access_token'], expires_at=time.time() + expires_in)
  return token, str(data.get('userDisplayName') or username)


# --- accounts ------------------------------------------------------------------------------------------------------


def split_login(login: str) -> tuple[str | None, str]:
  """'yambase:srikanth' -> ('yambase', 'srikanth'). Without a database part the database is None."""
  database, separator, name = (login or '').strip().partition(LOGIN_SEPARATOR)
  return (database.strip().lower(), name.strip()) if separator else (None, database)


def guest_limit(environ: Mapping[str, str] = os.environ) -> int:
  """GUEST_MESSAGE_LIMIT as a non-negative number; the default when missing or invalid."""
  try:
    return max(0, int(environ.get('GUEST_MESSAGE_LIMIT', DEFAULT_GUEST_LIMIT)))
  except ValueError:
    return DEFAULT_GUEST_LIMIT


def authenticate(
  login: str,
  password: str,
  databases: Mapping[str, str],
  environ: Mapping[str, str] = os.environ,
  login_to_breedbase: Callable | None = None,
) -> tuple[Account, BreedbaseToken | None] | None:
  """The account for '<database>:<username>' and this password, checked by that database's Breedbase login, with its
  token; a new guest account (no token) for '<database>:guest'; or None.

  databases: database name -> BrAPI base URL (DB_SERVERS).
  login_to_breedbase: replaces breedbase_login in tests (looked up when called, so tests can patch it).
  """
  database, name = split_login(login)
  if database not in databases or not name:
    return None  # no database chosen, or one this app does not serve
  if name.lower() == GUEST_USERNAME:
    if password == GUEST_PASSWORD and guest_limit(environ) > 0:
      guest_id = f'{database}{LOGIN_SEPARATOR}guest-{secrets.token_hex(6)}'
      return Account(identifier=guest_id, display_name='Guest', role='guest', database=database), None
    return None
  if not password:
    return None
  answer = (login_to_breedbase or breedbase_login)(databases[database], name, password)
  if answer is None:
    return None
  token, display_name = answer
  account = Account(
    identifier=f'{database}{LOGIN_SEPARATOR}{name.lower()}', display_name=display_name, role='user', database=database
  )
  return account, token


# --- tokens and guest limit (server memory) ------------------------------------------------------------------------


class TokenStore:
  """Users' Breedbase tokens by login identifier, in server memory only. A server restart forgets them, and the
  users are asked to log in again."""

  def __init__(self) -> None:
    self._tokens: dict[str, BreedbaseToken] = {}
    self._lock = threading.Lock()

  def put(self, identifier: str, token: BreedbaseToken) -> None:
    with self._lock:
      self._tokens[identifier] = token

  def get(self, identifier: str) -> BreedbaseToken | None:
    """The token if it is still valid; an expired one is dropped."""
    with self._lock:
      token = self._tokens.get(identifier)
      if token is not None and not token.valid():
        del self._tokens[identifier]
        return None
      return token

  def drop(self, identifier: str) -> None:
    with self._lock:
      self._tokens.pop(identifier, None)


class GuestUsage:
  """Questions asked per guest login, kept in server memory (reset when the server restarts)."""

  def __init__(self) -> None:
    self._asked: dict[str, int] = {}

  def asked(self, identifier: str) -> int:
    return self._asked.get(identifier, 0)

  def remaining(self, identifier: str, limit: int) -> int:
    return max(0, limit - self.asked(identifier))

  def record(self, identifier: str) -> None:
    self._asked[identifier] = self.asked(identifier) + 1
