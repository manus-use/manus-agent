"""
Tool for fetching EPSS (Exploit Prediction Scoring System) score history for a CVE.

Uses the FIRST.org EPSS API time-series endpoint to retrieve historical scores,
then analyses the trend to flag significant jumps that indicate new exploitation activity.
"""

from __future__ import annotations

from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

# Threshold: a jump of this magnitude in a single week is flagged as a significant spike
_SPIKE_THRESHOLD = 0.10

TOOL_SPEC = {
    "name": "get_epss_trend",
    "description": (
        "Fetches the EPSS (Exploit Prediction Scoring System) score history for a given CVE "
        "using the FIRST.org time-series API. Returns a chronological list of daily scores, "
        "the current score, the maximum observed jump (spike) between consecutive days, "
        "and a flag indicating whether a significant spike was detected (suggesting new "
        "exploitation activity). Use this after get_nvd_data to understand whether exploitation "
        "risk is rising or stable."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2024-3094').",
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "How many days of history to return. Defaults to 30. Maximum is 365 (the FIRST.org API limit)."
                    ),
                    "default": 30,
                },
            },
            "required": ["cve_id"],
        }
    },
}


def _fetch_epss_time_series(cve_id: str, days: int) -> dict[str, Any]:
    """Call the FIRST.org EPSS API and return parsed JSON."""
    url = "https://api.first.org/data/v1/epss"
    params: dict[str, Any] = {
        "cve": cve_id.upper(),
        "scope": "time-series",
    }
    if days and days > 0:
        # The API returns up to `limit` data points; each point is one day
        params["limit"] = min(days, 365)

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _analyse_series(series: list[dict[str, str]]) -> dict[str, Any]:
    """
    Given a list of {"epss": "0.xxx", "percentile": "0.xxx", "date": "YYYY-MM-DD"} dicts
    (already sorted newest-first by the API), compute trend metrics.
    """
    # Normalise to floats and sort oldest-first for analysis
    points = sorted(
        [
            {
                "date": pt["date"],
                "epss": float(pt["epss"]),
                "percentile": float(pt["percentile"]),
            }
            for pt in series
        ],
        key=lambda x: x["date"],
    )

    if not points:
        return {"points": [], "spike_detected": False, "max_jump": 0.0, "trend": "unknown"}

    scores = [p["epss"] for p in points]

    # Largest single-day jump (absolute)
    jumps = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    max_jump = max(jumps) if jumps else 0.0
    max_jump_idx = jumps.index(max_jump) if jumps else -1
    max_jump_date = points[max_jump_idx + 1]["date"] if max_jump_idx >= 0 else None

    # Largest 7-day jump (sliding window)
    max_7d_jump = 0.0
    max_7d_end_date = None
    for i in range(len(scores) - 1):
        for j in range(i + 1, min(i + 8, len(scores))):
            delta = scores[j] - scores[i]
            if delta > max_7d_jump:
                max_7d_jump = delta
                max_7d_end_date = points[j]["date"]

    spike_detected = max_7d_jump >= _SPIKE_THRESHOLD

    # Simple trend: compare first half vs second half averages
    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid if mid else scores[0]
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid) if (len(scores) - mid) else scores[-1]

    if second_half_avg > first_half_avg + 0.02:
        trend = "rising"
    elif second_half_avg < first_half_avg - 0.02:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "points": points,
        "current_epss": scores[-1],
        "current_percentile": points[-1]["percentile"],
        "oldest_epss": scores[0],
        "oldest_date": points[0]["date"],
        "latest_date": points[-1]["date"],
        "max_single_day_jump": round(max_jump, 6),
        "max_single_day_jump_date": max_jump_date,
        "max_7d_jump": round(max_7d_jump, 6),
        "max_7d_jump_end_date": max_7d_end_date,
        "spike_detected": spike_detected,
        "spike_threshold_used": _SPIKE_THRESHOLD,
        "trend": trend,
    }


def get_epss_trend(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Fetch EPSS time-series for a CVE and analyse trend / spikes."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")
    days = int(tool_input.get("days", 30))

    if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
        log_tool_output_size("get_epss_trend", err)
        return err

    try:
        raw = _fetch_epss_time_series(cve_id, days)
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"FIRST.org EPSS API request failed: {exc}"}],
        }
        log_tool_output_size("get_epss_trend", result)
        return result

    data_entries = raw.get("data", [])
    if not data_entries:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"No EPSS data found for {cve_id.upper()}. The CVE may be too recent or unknown."}],
        }
        log_tool_output_size("get_epss_trend", result)
        return result

    entry = data_entries[0]
    time_series = entry.get("time-series", [])

    # The top-level entry is the current/most-recent score; include it in the series
    current_point = {
        "date": entry.get("date", ""),
        "epss": entry.get("epss", "0"),
        "percentile": entry.get("percentile", "0"),
    }
    if current_point["date"] and not any(p["date"] == current_point["date"] for p in time_series):
        time_series = [current_point] + list(time_series)

    analysis = _analyse_series(time_series)

    # Build a human-readable summary
    spike_flag = "⚠️  SPIKE DETECTED" if analysis["spike_detected"] else "✅  No significant spike"
    summary_lines = [
        f"EPSS trend for {cve_id.upper()} ({len(analysis['points'])} days)",
        f"  Current score : {analysis.get('current_epss', 'N/A'):.4f}  "
        f"(percentile {float(analysis.get('current_percentile', 0)):.1%})",
        f"  Trend         : {analysis['trend']}",
        f"  Max 7-day jump: {analysis['max_7d_jump']:.4f}"
        + (f" (peaked {analysis['max_7d_jump_end_date']})" if analysis["max_7d_jump_end_date"] else ""),
        f"  {spike_flag}",
    ]
    summary = "\n".join(summary_lines)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": summary},
            {
                "json": {
                    "cve_id": cve_id.upper(),
                    "analysis": analysis,
                }
            },
        ],
    }
    log_tool_output_size("get_epss_trend", result)
    return result
