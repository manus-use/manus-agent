#!/usr/bin/env python3
"""
Custom tool for fetching detailed NVD data for CVEs.
This module follows the Strands SDK's module-based tool specification.

Rate-limit resilience
---------------------
NVD's public API is throttled at **5 requests per 30 seconds** for
unauthenticated callers and **50 requests per 30 seconds** for API-key
holders.  A single transient 429 or network hiccup was previously fatal;
this version retries up to ``NVD_MAX_RETRIES`` times (default 3) with
exponential back-off (2 s, 4 s, 8 s).

Optional API key
----------------
Set ``NVD_API_KEY`` in the environment (or in your ``.env`` file) to
inject the key as an ``apiKey`` header on every request.  The key is
never logged.
"""

import json
import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

TOOL_SPEC = {  # Minor change to force re-evaluation
    "name": "get_nvd_data",
    "description": (
        "Fetches detailed, authoritative vulnerability data for a given CVE ID directly from the "
        "official National Vulnerability Database (NVD) API. This should be the primary and first tool used "
        "to gather information about a CVE. The output will also indicate if the CVE is in the CISA Known "
        "Exploited Vulnerabilities (KEV) Catalog."
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

# ---------------------------------------------------------------------------
# Retry / back-off constants (overridable via env for tests)
# ---------------------------------------------------------------------------
_NVD_MAX_RETRIES = int(os.environ.get("NVD_MAX_RETRIES", "3"))
_NVD_RETRY_BASE_DELAY = float(os.environ.get("NVD_RETRY_BASE_DELAY", "2"))  # seconds
# HTTP status codes worth retrying (rate-limit + transient server errors)
_NVD_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _build_nvd_headers() -> dict[str, str]:
    """Return request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _nvd_get_with_retry(url: str, *, timeout: int = 15) -> requests.Response:
    """GET *url* with exponential back-off retry on 429 / transient errors.

    Raises the underlying :class:`requests.exceptions.RequestException` (or
    :class:`requests.exceptions.HTTPError`) after all retries are exhausted.

    Back-off schedule (default, NVD_MAX_RETRIES=3):
      attempt 1 -> immediate
      attempt 2 -> sleep 2 s
      attempt 3 -> sleep 4 s
      attempt 4 -> sleep 8 s
    """
    headers = _build_nvd_headers()
    last_exc: Exception | None = None

    for attempt in range(_NVD_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _NVD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code in _NVD_RETRYABLE_STATUS:
                # Build a descriptive exception for this retryable status
                last_exc = requests.exceptions.HTTPError(f"HTTP {response.status_code}", response=response)
                if attempt < _NVD_MAX_RETRIES:
                    continue  # retry
                raise last_exc  # all retries exhausted
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as exc:
            # Non-retryable client errors (4xx other than 429) fail immediately
            if exc.response is not None and exc.response.status_code not in _NVD_RETRYABLE_STATUS:
                raise
            last_exc = exc
            if attempt < _NVD_MAX_RETRIES:
                continue
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _NVD_MAX_RETRIES:
                continue
            # Final attempt failed -- propagate
            raise
    # All retries exhausted via a retryable status code path
    raise last_exc  # type: ignore[misc]


def get_nvd_data(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id")

    if not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("get_nvd_data", result)
        return result

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    url = f"{base_url}?cveId={cve_id.upper()}"

    try:
        response = _nvd_get_with_retry(url)
        data = response.json()

        if not data.get("vulnerabilities"):
            result = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [
                    {"text": f"No vulnerability data found for {cve_id}. It may be an invalid or rejected CVE."}
                ],
            }
            log_tool_output_size("get_nvd_data", result)
            return result

        vulnerability_data = data["vulnerabilities"][0]

        # Extract CISA KEV information if available
        cisa_kev_info = {"is_in_kev": False}
        if "cisaExploitAdd" in vulnerability_data.get("cve", {}).get("vulnStatus", ""):
            cisa_kev_info["is_in_kev"] = True
            cisa_kev_info["date_added"] = vulnerability_data["cve"]["cisaExploitAdd"]
            cisa_kev_info["required_action"] = vulnerability_data["cve"]["cisaRequiredAction"]
            cisa_kev_info["due_date"] = vulnerability_data["cve"]["cisaActionDue"]
            # Add other relevant CISA fields if they exist and are needed

        # Add CISA KEV info to the main vulnerability data
        vulnerability_data["cisa_kev_info"] = cisa_kev_info

        result = {"toolUseId": tool_use_id, "status": "success", "content": [{"json": vulnerability_data}]}
        log_tool_output_size("get_nvd_data", result)
        return result

    except requests.exceptions.RequestException as e:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Request to NVD API failed: {e}"}],
        }
        log_tool_output_size("get_nvd_data", result)
        return result
    except json.JSONDecodeError:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Failed to parse JSON response from NVD API."}],
        }
        log_tool_output_size("get_nvd_data", result)
        return result
    except Exception as e:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"An unexpected error occurred: {e}"}],
        }
        log_tool_output_size("get_nvd_data", result)
        return result
