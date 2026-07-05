import os
from typing import Any

import requests
from strands import Agent
from strands.types.tools import ToolResult, ToolUse

from manus_agent.config import Config

# MCP client is initialised lazily on first use so that importing this module
# does not immediately attempt a network connection to localhost:3001.
_sse_mcp_client = None
_mcp_tools = None


def _get_mcp_tools():
    """Return MCP tools, initialising the MCP client on first call."""
    global _sse_mcp_client, _mcp_tools  # noqa: PLW0603
    if _mcp_tools is not None:
        return _mcp_tools
    try:
        from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415
        from strands.tools.mcp.mcp_client import MCPClient  # noqa: PLC0415

        sse_url = os.environ.get("MCP_SSE_URL", "http://localhost:3001/mcp")
        _sse_mcp_client = MCPClient(lambda: streamablehttp_client(sse_url))
        _sse_mcp_client.start()
        _mcp_tools = _sse_mcp_client.list_tools_sync()
    except Exception:  # noqa: BLE001
        _mcp_tools = []
    return _mcp_tools


TOOL_SPEC = {
    "name": "submit_cves",
    "description": "Submits a discovered CVE, along with its associated intelligence data, to a downstream system or webhook.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_list": {
                    "type": "array",
                    "description": "A list of discovered vulnerabilities to submit.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cve_id": {
                                "type": "string",
                                "description": "The unique identifier for the vulnerability (e.g., 'CVE-2025-12345').",
                            },
                            "cvss_score": {
                                "type": "string",
                                "description": "The CVSS score and vector, formatted as 'Severity(Score),Vector' (e.g., 'Critical(9.8),CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H').",
                            },
                            "cisa_kev": {
                                "type": "boolean",
                                "description": "Indicates if the vulnerability is in the CISA Known Exploited Vulnerabilities (KEV) catalog.",
                            },
                            "exploited": {
                                "type": "boolean",
                                "description": "Indicates if the vulnerability is exploited in the wild, based on OTX or other threat intelligence.",
                            },
                            "epss_score": {
                                "type": "string",
                                "description": "The Exploit Prediction Scoring System (EPSS) score (e.g., 0.94577).",
                            },
                            "epss_percentile": {"type": "string", "description": "The EPSS percentile (e.g., 0.999)."},
                            "affected_products": {
                                "type": "string",
                                "description": "A single, comma-separated string listing the products including vendors, packages, or components confirmed to be affected by the CVE.",
                            },
                            "affected_versions": {
                                "type": "string",
                                "description": "A single string containing a comma-separated list of affected product versions.",
                            },
                            "cwe": {
                                "type": "string",
                                "description": "The Common Weakness Enumeration (CWE) identifier (e.g., 'CWE-79').",
                            },
                            "public_disclosure_date": {
                                "type": "string",
                                "description": "The date when the vulnerability was publicly disclosed (YYYY-MM-DD).",
                            },
                            "description": {
                                "type": "string",
                                "description": "The description of the CVE",
                            },
                            "priority": {
                                "type": "string",
                                "description": """Indicates the popularity and usage level of the affected software, including any third-party libraries or dependencies. Accepted values: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].
                                - 'CRITICAL': CVE affects core operating systems, foundational internet infrastructure, or extremely common libraries (e.g., Windows, OpenSSL, Apache).
                                - 'HIGH': CVE affects widely used applications, libraries, or components (e.g., Chrome, Log4j, jQuery).
                                - 'MEDIUM': CVE affects moderately popular software or libraries, but not universally adopted.
                                - 'LOW': CVE affects niche, specialized, or rarely used software or components.
                                Always consider both the primary software and any included third-party dependencies when assigning this value.""",
                            },
                        },
                        "required": [
                            "cve_id",
                            "priority",
                            "cvss_score",
                            "epss_score",
                            "epss_percentile",
                            "affected_products",
                            "cisa_kev",
                            "exploited",
                            "cwe",
                            "cpe",
                        ],
                    },
                }
            },
            "required": ["cve_list"],
        }
    },
}


def analyze_affected_assets(cves):
    for cve in cves:
        agent = Agent(tools=_get_mcp_tools())
        result = agent(f"please submit a match asset task for {cve}.")
        print(result)
    # return result


def submit_cves(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    # cve_list = tool["input"]
    cve_list = tool["input"].get("cve_list", [])
    print("============================")
    if not cve_list:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "The 'cve_list' parameter cannot be empty."}],
        }

    print(f"Submitting {len(cve_list)} CVEs to webhook...")

    """ First webhook (webhook.site) - with timeout
    try:
        url1 = "https://webhook.site/693c24d6-518f-48b7-8af5-e71563fabd5e"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url1, headers=headers, json=cve, timeout=10)
        print(f"Webhook.site response: {response.status_code}")
    except Exception as e:
        print(f"Warning: webhook.site failed: {e}")
    """
    # Main webhook (Lark)
    # NOTE: URL resolution and validation are inside the try block so that a
    # missing-URL configuration returns {"status": "error"} instead of raising
    # an uncaught ValueError to the caller.
    try:
        config = Config.from_file()
        url = getattr(getattr(config, "webhooks", None), "cve_submit_url", None)
        if not url:
            url = os.environ.get("CVE_SUBMIT_URL")
        if not url:
            raise ValueError(
                "CVE submission webhook URL not set in config or environment. Please set [webhooks] cve_submit_url in your config.toml or the CVE_SUBMIT_URL environment variable."
            )
        headers = {"Content-Type": "application/json"}

        critical_cves = list(
            map(
                lambda y: y.get("cve_id", "Unknown"),
                filter(lambda x: "CRITICAL" in x.get("priority", "") or "HIGH" in x.get("priority", ""), cve_list),
            )
        )
        print(f"{len(critical_cves)} critival vulnerabilities")
        for cve in cve_list:
            # if cve_analyzed.get(cve.get('cve_id', 'Unknown'), None):
            #   cve = cve | cve_analyzed.get(cve.get('cve_id', 'Unknown'))
            #    print(f"Critical: f{cve}")
            response = requests.post(url, headers=headers, json=cve)
            response.raise_for_status()
        submitted_ids = [cve.get("cve_id", "Unknown") for cve in cve_list]
        success_message = f"Successfully submitted {len(cve_list)} CVEs: {', '.join(submitted_ids)}"
        print(success_message)

        analyze_affected_assets(critical_cves)

        return {"toolUseId": tool_use_id, "status": "success", "content": [{"text": success_message}]}
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred while submitting CVEs: {http_err}"
        print(error_message)
        return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": error_message}]}
    except Exception as err:
        error_message = f"An unexpected error occurred: {err}"
        print(error_message)
        return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": error_message}]}
