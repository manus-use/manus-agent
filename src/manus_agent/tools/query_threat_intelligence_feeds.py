#!/usr/bin/env python3
"""
Custom tool for querying open-source threat intelligence feeds.
This module follows the Strands SDK's module-based tool specification.
"""

from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {
    "name": "query_threat_intelligence_feeds",
    "description": (
        "Queries a curated list of open-source threat intelligence feeds for information related to a "
        "given CVE ID. This helps identify threat actor activity, campaigns, and broader context of "
        "exploitation beyond just PoC availability."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to search for in threat intelligence feeds (e.g., 'CVE-2024-3094').",
                }
            },
            "required": ["cve_id"],
        }
    },
}

# Default curated list of public threat intelligence feeds.
DEFAULT_THREAT_FEEDS: list[dict[str, str]] = [
    {
        "name": "CISA Cybersecurity Advisories",
        "url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "type": "rss",
    },
]


def fetch_threat_intelligence(
    cve_id: str,
    *,
    feeds: list[dict[str, str]] | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Query threat intelligence feeds for a CVE.

    Returns a dict with keys:
      - summary (str): human-readable summary
      - intelligence (list[dict]): list of feed match dicts
      - errors (list[dict]): list of feed errors (feed_name, error)
    """
    if feeds is None:
        feeds = DEFAULT_THREAT_FEEDS

    found_intelligence: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for feed in feeds:
        try:
            response = requests.get(feed["url"], timeout=timeout)
            response.raise_for_status()
            content = response.text

            # Basic search for CVE ID in the content
            if cve_id.upper() in content.upper():
                idx = content.upper().find(cve_id.upper())
                snippet_start = max(0, idx - 50)
                snippet_end = min(len(content), idx + 100)
                found_intelligence.append(
                    {
                        "feed_name": feed["name"],
                        "feed_url": feed["url"],
                        "cve_found": cve_id,
                        "snippet": content[snippet_start:snippet_end] + "...",
                    }
                )

        except requests.exceptions.RequestException as e:
            errors.append({"feed_name": feed["name"], "error": str(e)})
        except Exception as e:
            errors.append({"feed_name": feed["name"], "error": str(e)})

    if not found_intelligence:
        summary = f"No direct threat intelligence found for {cve_id} in curated feeds."
    else:
        summary = f"Found relevant threat intelligence for {cve_id} in {len(found_intelligence)} feed(s)."

    return {
        "summary": summary,
        "intelligence": found_intelligence,
        "errors": errors,
    }


def query_threat_intelligence_feeds(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id")

    if not isinstance(cve_id, str) or not cve_id.strip():
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID. Must be a non-empty string."}],
        }
        log_tool_output_size("query_threat_intelligence_feeds", result)
        return result

    payload = fetch_threat_intelligence(cve_id)

    if not payload["intelligence"]:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "json": {
                        "summary": payload["summary"],
                        "intelligence": [],
                    }
                }
            ],
        }
        log_tool_output_size("query_threat_intelligence_feeds", result)
        return result

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": {"summary": payload["summary"], "intelligence": payload["intelligence"]}}],
    }
    log_tool_output_size("query_threat_intelligence_feeds", result)
    return result
