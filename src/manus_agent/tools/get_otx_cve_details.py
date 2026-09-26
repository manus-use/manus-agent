#!/usr/bin/env python3
"""
Custom tool for fetching threat intelligence data from AlienVault OTX.
This module follows the Strands SDK's module-based tool specification.
"""

import json
import os
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.config import Config
from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

TOOL_SPEC = {
    "name": "get_otx_cve_details",
    "description": (
        "Fetches threat intelligence data for a given CVE ID from AlienVault OTX (Open Threat Exchange). "
        "This tool provides context on how a vulnerability is being used in the wild, including associated pulses, "
        "Indicators of Compromise (IoCs), and related threat actors."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2024-3094').",
                }
            },
            "required": ["cve_id"],
        }
    },
}


def get_otx_cve_details(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Searches AlienVault OTX for information about a given CVE ID.
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id")

    if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
        log_tool_output_size("get_otx_cve_details", err)
        return err

    try:
        config = Config.from_file()
        api_key = os.environ.get("OTX_API_KEY") or (config.otx.api_key if config.otx else None)
    except Exception:
        api_key = os.environ.get("OTX_API_KEY")

    if not api_key:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {
                    "text": "AlienVault OTX API key not found. Please set it in your config.toml or as an OTX_API_KEY environment variable."
                }
            ],
        }
        log_tool_output_size("get_otx_cve_details", result)
        return result

    headers = {"X-OTX-API-KEY": api_key}
    url = f"https://otx.alienvault.com/api/v1/indicators/cve/{cve_id.upper()}"

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not data or not data.get("pulse_info", {}).get("pulses"):
            result = {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {
                        "text": f"No specific threat intelligence pulses found for {cve_id} in AlienVault OTX. This may indicate it is not currently part of a major tracked campaign."
                    }
                ],
            }
            log_tool_output_size("get_otx_cve_details", result)
            return result

        result = {"toolUseId": tool_use_id, "status": "success", "content": [{"json": data}]}
        log_tool_output_size("get_otx_cve_details", result)
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            result = {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"No information found for {cve_id} in AlienVault OTX."}],
            }
            log_tool_output_size("get_otx_cve_details", result)
            return result
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Request to AlienVault OTX API failed with HTTP error: {e}"}],
        }
        log_tool_output_size("get_otx_cve_details", result)
        return result
    except requests.exceptions.RequestException as e:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Request to AlienVault OTX API failed: {e}"}],
        }
        log_tool_output_size("get_otx_cve_details", result)
        return result
    except json.JSONDecodeError:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Failed to parse JSON response from AlienVault OTX API."}],
        }
        log_tool_output_size("get_otx_cve_details", result)
        return result
    except Exception as e:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"An unexpected error occurred: {e}"}],
        }
        log_tool_output_size("get_otx_cve_details", result)
        return result
