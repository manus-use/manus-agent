#!/usr/bin/env python3
"""
Custom tool for querying open-source threat intelligence feeds.
This module follows the Strands SDK's module-based tool specification.
"""

from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

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


def query_threat_intelligence_feeds(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id")

    if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
        log_tool_output_size("query_threat_intelligence_feeds", err)
        return err

    # Curated list of public threat intelligence feeds (example URLs)
    # In a real-world scenario, this list would be more extensive and potentially configurable.
    # Parsing logic would also need to be more robust for different feed formats (RSS, JSON, HTML).
    threat_feeds = [
        {
            "name": "CISA Cybersecurity Advisories",
            "url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",  # Updated RSS feed
            "type": "rss",
        },
        # Removed US-CERT Alerts as it was causing 404 errors and may be deprecated.
    ]

    found_intelligence: list[dict[str, Any]] = []

    for feed in threat_feeds:
        try:
            response = requests.get(feed["url"], timeout=10)
            response.raise_for_status()
            content = response.text

            # Basic search for CVE ID in the content
            if cve_id.upper() in content.upper():
                # In a real tool, you'd parse the RSS/JSON/HTML more intelligently
                # to extract relevant snippets, titles, and links.
                found_intelligence.append(
                    {
                        "feed_name": feed["name"],
                        "feed_url": feed["url"],
                        "cve_found": cve_id,
                        "snippet": content[
                            content.upper().find(cve_id.upper()) - 50 : content.upper().find(cve_id.upper()) + 100
                        ]
                        + "...",  # Basic snippet
                    }
                )

        except requests.exceptions.RequestException as e:
            # Log the error but continue with other feeds
            print(f"Error fetching {feed['name']} ({feed['url']}): {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {feed['name']}: {e}")

    if not found_intelligence:
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "json": {
                        "summary": f"No direct threat intelligence found for {cve_id} in curated feeds.",
                        "intelligence": [],
                    }
                }
            ],
        }
        log_tool_output_size("query_threat_intelligence_feeds", result)
        return result

    summary = f"Found relevant threat intelligence for {cve_id} in {len(found_intelligence)} feeds."
    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": {"summary": summary, "intelligence": found_intelligence}}],
    }
    log_tool_output_size("query_threat_intelligence_feeds", result)
    return result
