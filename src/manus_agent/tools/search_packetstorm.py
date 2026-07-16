#!/usr/bin/env python3
"""
Custom tool for searching Packet Storm Security for exploits.
This module follows the Strands SDK's module-based tool specification.
"""

from __future__ import annotations

from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "search_packetstorm",
    "description": (
        "Searches the Packet Storm Security database for public exploits related to a given CVE ID or general keyword. "
        "This tool is useful for finding exploits that may not be available on other platforms."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The CVE ID (e.g., 'CVE-2024-3094') or general keyword to search for exploits.",
                }
            },
            "required": ["query"],
        }
    },
}


# ---------------------------------------------------------------------------
# Reusable library function (called by both the Strands tool and the CLI)
# ---------------------------------------------------------------------------


def fetch_packetstorm(query: str, *, max_results: int = 5) -> dict[str, Any]:
    """Search Packet Storm Security for *query* and return structured results.

    Returns a dict with keys:
      - ``source``: always ``"packetstorm"``
      - ``query``: the original query
      - ``exploits``: list of dicts with ``title``, ``link``
      - ``error``: optional error string (only on failure)
    """
    base_url = "https://packetstormsecurity.com/search/files/"
    url = f"{base_url}?q={requests.utils.quote(query)}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        html_content = response.text
        results: list[dict[str, Any]] = []

        exploit_entries = html_content.split('<dl class="file">')

        for entry in exploit_entries[1:]:
            title_start = entry.find('<dt><a href="')
            if title_start == -1:
                continue

            link_start = entry.find('">', title_start) + 2
            link_end = entry.find("</a></dt>", link_start)

            title = entry[link_start:link_end].strip()
            link = "https://packetstormsecurity.com" + entry[title_start + len('<dt><a href="') : link_start - 2]

            results.append({"title": title, "link": link})

            if len(results) >= max_results:
                break

        return {"source": "packetstorm", "query": query, "exploits": results}

    except requests.exceptions.RequestException as exc:
        return {"source": "packetstorm", "query": query, "exploits": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"source": "packetstorm", "query": query, "exploits": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def search_packetstorm(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    query = tool_input.get("query")

    if not isinstance(query, str) or not query.strip():
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid query. Must be a non-empty string."}],
        }
        log_tool_output_size("search_packetstorm", result)
        return result

    data = fetch_packetstorm(query)

    if data.get("error"):
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Request to Packet Storm failed: {data['error']}"}],
        }
        log_tool_output_size("search_packetstorm", result)
        return result

    exploits = data["exploits"]
    if not exploits:
        summary = f"No exploits found on Packet Storm for '{query}'."
    else:
        summary = f"Found {len(exploits)} potential exploits on Packet Storm for '{query}'."

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": {"summary": summary, "exploits": exploits}}],
    }
    log_tool_output_size("search_packetstorm", result)
    return result
