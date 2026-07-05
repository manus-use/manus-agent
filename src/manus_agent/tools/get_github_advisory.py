"""Tool to fetch data from the GitHub Advisory Database."""

import os
from typing import Any

import requests
from strands.tools import tool

from manus_agent.config import Config
from manus_agent.tools.tool_output_logger import log_tool_output_size


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
    if not cve_id or not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result = {"error": "Invalid CVE ID format. It must be a string starting with 'CVE-'."}
        log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
        return result

    # Use the official GitHub REST API endpoint for getting advisories by CVE ID.
    url = f"https://api.github.com/advisories?cve_id={cve_id}"

    try:
        config = Config.from_file()
        github_token = os.environ.get("GITHUB_TOKEN") or (config.github.api_token if config.github else None)
    except Exception:
        github_token = os.environ.get("GITHUB_TOKEN")

    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if not data:
            result = {"message": f"No advisory found on GitHub for {cve_id}."}
            log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
            return result

        # The API returns a list of advisories; we will return the first and most relevant one.
        result = data[0]
        log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
        return result

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            result = {"message": f"No advisory found on GitHub for {cve_id}."}
            log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
            return result
        result = {"error": f"HTTP error occurred while querying GitHub Advisory API: {http_err}"}
        log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
        return result
    except requests.exceptions.RequestException as req_err:
        result = {"error": f"An error occurred while querying the GitHub Advisory API: {req_err}"}
        log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
        return result
    except (KeyError, IndexError, ValueError):
        result = {"error": "Received an unexpected response format from the GitHub Advisory API."}
        log_tool_output_size("get_github_advisory", {"content": [{"json": result}]})
        return result
