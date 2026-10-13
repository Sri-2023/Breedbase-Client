"""
The Breedbase databases the chat app offers, and the app's settings from .env.
Each database here appears on the start page, has its own logins (each user's own Breedbase account) and
its own MCP server (MCP_URL_<DB> and MCP_TOKEN_<DB> in .env; see mcp_connection.py).
"""

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / '.env')  # ANTHROPIC_API_KEY, CHAINLIT_AUTH_SECRET, MCP_URL_<DB>, MCP_TOKEN_<DB>, ...

# we can add more databases, as you get along
DB_SERVERS = {
  'yambase': 'https://yambase.org/brapi/v2',
  'sweetpotatobase': 'https://sweetpotatobase.org/brapi/v2',
  'cassavabase': 'https://cassavabase.org/brapi/v2',
  'sugarcanebase': 'https://sugarcane.sgn.cornell.edu/brapi/v2',
  'cassbase': 'https://cassbase.org/brapi/v2',
  'musabase': 'https://musabase.org/brapi/v2',
}


def site_url(base_url: str) -> str:
  """The website of a database, from its BrAPI address."""
  return base_url.split('/brapi/')[0]
