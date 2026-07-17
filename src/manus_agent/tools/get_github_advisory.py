"""Tool to fetch data from the GitHub Advisory Database."""

import os
from typing import Any

import requests
from strands.tools import tool

from manus_agent.config import Config
from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Reusable library function (no Strands dependency)
# ---------------------------------------------------------------------------


def fetch_github_advisory(cve_id: str, *, github_token: str | None = None) -> dict[str, Any]:
    """Fetch advisory data from the GitHub Advisory Database for a CVE.

    Args:
        cve_id: CVE identifier (e.g. "CVE-2023-1234").
        github_token: Optional GitHub personal access token. If *None*, the
            function attempts to read from ``GITHUB_TOKEN`` env var or the
            manus-agent config file.

    Returns:
        A dictionary with keys:
        - On success: full advisory payload from GitHub (first match).
        - Not found: ``{"found": False, "message": "..."}``
        - Error: ``{"error": "..."}``
    """
    if not cve_id or not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        return {"error": "Invalid CVE ID format. It must be a string starting with 'CVE-'."}

    # Resolve token
    if github_token is None:
        try:
            config = Config.from_file()
            github_token = os.environ.get("GITHUB_TOKEN") or (config.github.api_token if config.github else None)
        except Exception:
            github_token = os.environ.get("GITHUB_TOKEN")

    url = f"https://api.github.com/advisories?cve_id={cve_id}"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data:
            return {"found": False, "message": f"No advisory found on GitHub for {cve_id}."}

        # The API returns a list; return the first (most relevant) advisory
        # plus a convenience flag.
        advisory = data[0]
        advisory["found"] = True
        return advisory

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return {"found": False, "message": f"No advisory found on GitHub for {cve_id}."}
        return {"error": f"HTTP error occurred while querying GitHub Advisory API: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"An error occurred while querying the GitHub Advisory API: {req_err}"}
    except (KeyError, IndexError, ValueError):
        return {"error": "Received an unexpected response format from the GitHub Advisory API."}


# ---------------------------------------------------------------------------
# Strands tool wrapper (agent-facing)
# ---------------------------------------------------------------------------


@tool
def get_github_advisory(cve_id: str) -> dict[str, Any]:
    """
    Fetches vulnerability advisory information from the GitHub Advisory Database for a given CVE ID.

    This tool queries the public GitHub REST API to find advisories associated with a specific CVE identifier.

    Args:
        cve_id: The CVE identifier (e.g., "CVE-2023-1234").

    Returns:
        A dictionary containing the advisory data from GitHub if found, otherwise a message indicating it was not found or an error.
    """
    result = fetch_github_advisory(cve_id)
    log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
    return result
