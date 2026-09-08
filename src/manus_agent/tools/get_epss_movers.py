"""Tool for detecting CVEs with the largest EPSS score increases.

Queries the FIRST.org EPSS API for two dates (``recent`` and ``baseline``)
and computes the delta for every CVE that appears in both snapshots.  Returns
the top movers — CVEs whose exploitation-probability score jumped the most —
which represent **emerging threats** that security teams should investigate.

This is a *population-level* scan: it does not require a pre-selected list of
CVEs.  Instead it fetches the highest-EPSS CVEs on each date and identifies
which ones climbed the fastest.

Usage patterns:
- ``manus-agent epss-movers`` — top 20 biggest 7-day EPSS jumps
- ``manus-agent epss-movers --days 30 --top 50 --output json``

Graceful degradation: the FIRST.org API may return fewer results than
requested; the tool adapts and reports what it receives.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_epss_movers", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSS_API_URL = "https://api.first.org/data/v1/epss"
_REQUEST_TIMEOUT = int(os.environ.get("EPSS_MOVERS_TIMEOUT", "30"))

# Maximum CVEs to request per date snapshot.  The FIRST.org API allows up to
# 100 results per page; we fetch the top-scoring CVEs on each date.
_MAX_PER_PAGE = 100

# Default look-back window (days).
_DEFAULT_DAYS = 7

# Default number of top movers to return.
_DEFAULT_TOP = 20

# Minimum absolute EPSS delta to include in results (filters out noise).
_MIN_DELTA = float(os.environ.get("EPSS_MOVERS_MIN_DELTA", "0.001"))

# ---------------------------------------------------------------------------
# Retry / back-off (mirrors conventions in get_nvd_data / get_vulncheck_data)
# ---------------------------------------------------------------------------

_MAX_RETRIES = int(os.environ.get("EPSS_MOVERS_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("EPSS_MOVERS_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _epss_get_with_retry(
    params: dict[str, Any],
    *,
    timeout: int = _REQUEST_TIMEOUT,
) -> requests.Response:
    """GET the EPSS API with exponential back-off on transient errors."""
    import time as _time

    last_exc: requests.exceptions.RequestException | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(_EPSS_API_URL, params=params, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES:
                if attempt < _MAX_RETRIES:
                    _time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            http_resp = getattr(exc, "response", None)
            if http_resp is not None and http_resp.status_code not in _RETRYABLE_STATUSES:
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES:
                _time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    if last_exc is not None:
        raise last_exc
    raise requests.exceptions.RequestException(
        "EPSS request failed after all retries"
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _fetch_epss_snapshot(
    target_date: date,
    *,
    limit: int = _MAX_PER_PAGE,
    offset: int = 0,
) -> list[dict[str, str]]:
    """Fetch up to *limit* CVEs ordered by descending EPSS for *target_date*.

    Returns a list of ``{"cve": "CVE-...", "epss": "0.xxx", "percentile": "0.xxx", "date": "YYYY-MM-DD"}``
    dicts as returned by the FIRST.org API.
    """
    params: dict[str, Any] = {
        "date": target_date.isoformat(),
        "order": "!epss",  # descending by EPSS score
        "limit": min(limit, _MAX_PER_PAGE),
        "offset": offset,
    }
    resp = _epss_get_with_retry(params)
    payload = resp.json()
    return payload.get("data", [])


def _fetch_epss_for_cves(
    cve_ids: list[str],
    target_date: date,
) -> dict[str, float]:
    """Fetch EPSS scores for specific CVEs on a given date.

    Returns ``{cve_id: epss_score}`` mapping.  CVEs not found are omitted.
    Batches requests in groups of 30 CVEs (FIRST.org API supports comma-
    separated CVE lists).
    """
    result: dict[str, float] = {}
    batch_size = 30
    for i in range(0, len(cve_ids), batch_size):
        batch = cve_ids[i : i + batch_size]
        params: dict[str, Any] = {
            "cve": ",".join(batch),
            "date": target_date.isoformat(),
        }
        resp = _epss_get_with_retry(params)
        payload = resp.json()
        for entry in payload.get("data", []):
            cve = entry.get("cve", "").upper()
            try:
                score = float(entry.get("epss", "0"))
            except (ValueError, TypeError):
                score = 0.0
            if cve:
                result[cve] = score
    return result


def compute_movers(
    recent_date: date,
    baseline_date: date,
    *,
    top: int = _DEFAULT_TOP,
    min_delta: float = _MIN_DELTA,
    fetch_limit: int = _MAX_PER_PAGE,
) -> dict[str, Any]:
    """Compute the top EPSS movers between *baseline_date* and *recent_date*.

    Strategy:
    1. Fetch the top *fetch_limit* highest-EPSS CVEs on *recent_date*.
    2. Look up their EPSS scores on *baseline_date* via batch query.
    3. Compute delta (recent − baseline) for each CVE.
    4. Return the top *top* movers sorted by delta descending.

    This approach is efficient (2–3 API calls) and catches CVEs that
    have risen into the top tier.
    """
    if recent_date <= baseline_date:
        raise ValueError(
            f"recent_date ({recent_date}) must be after baseline_date ({baseline_date})"
        )

    # Step 1: top CVEs on the recent date
    recent_snapshot = _fetch_epss_snapshot(recent_date, limit=fetch_limit)

    if not recent_snapshot:
        return {
            "recent_date": recent_date.isoformat(),
            "baseline_date": baseline_date.isoformat(),
            "movers": [],
            "total_compared": 0,
            "error": None,
        }

    # Build recent scores map
    recent_scores: dict[str, dict[str, float]] = {}
    for entry in recent_snapshot:
        cve = entry.get("cve", "").upper()
        try:
            epss = float(entry.get("epss", "0"))
            pctl = float(entry.get("percentile", "0"))
        except (ValueError, TypeError):
            epss, pctl = 0.0, 0.0
        if cve:
            recent_scores[cve] = {"epss": epss, "percentile": pctl}

    # Step 2: look up baseline scores for the same CVEs
    cve_list = list(recent_scores.keys())
    baseline_scores = _fetch_epss_for_cves(cve_list, baseline_date)

    # Step 3: compute deltas
    movers: list[dict[str, Any]] = []
    for cve, recent_data in recent_scores.items():
        recent_epss = recent_data["epss"]
        baseline_epss = baseline_scores.get(cve, 0.0)
        delta = recent_epss - baseline_epss

        if delta >= min_delta:
            # Compute percentage change relative to baseline
            if baseline_epss > 0:
                pct_change = (delta / baseline_epss) * 100
            else:
                pct_change = float("inf") if delta > 0 else 0.0

            movers.append(
                {
                    "cve": cve,
                    "recent_epss": round(recent_epss, 6),
                    "baseline_epss": round(baseline_epss, 6),
                    "delta": round(delta, 6),
                    "pct_change": round(pct_change, 2) if pct_change != float("inf") else None,
                    "percentile": round(recent_data["percentile"], 6),
                    "is_new": cve not in baseline_scores,
                }
            )

    # Step 4: sort by delta descending, then by recent_epss descending
    movers.sort(key=lambda m: (-m["delta"], -m["recent_epss"]))
    top_movers = movers[:top]

    return {
        "recent_date": recent_date.isoformat(),
        "baseline_date": baseline_date.isoformat(),
        "movers": top_movers,
        "total_compared": len(recent_scores),
        "total_with_increase": len(movers),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_epss_movers",
    "description": (
        "Detects CVEs with the largest EPSS score increases over a configurable "
        "time window. Queries the FIRST.org EPSS API for two dates and computes "
        "deltas, returning the top movers — CVEs whose exploitation-probability "
        "score jumped the most — representing emerging threats. Use this for "
        "proactive threat monitoring: 'which CVEs became significantly more "
        "likely to be exploited this week?'"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": (
                        "Look-back window in days. Compares today's EPSS scores "
                        "against scores from this many days ago. "
                        "Default: 7. Range: 1–365."
                    ),
                    "default": 7,
                },
                "top": {
                    "type": "integer",
                    "description": (
                        "Number of top movers to return. Default: 20. Max: 100."
                    ),
                    "default": 20,
                },
            },
            "required": [],
        }
    },
}


def get_epss_movers(
    tool: ToolUse,
    *,
    now: date | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Strands tool entry point for EPSS movers detection.

    Parameters
    ----------
    tool : ToolUse
        Strands tool-use payload.
    now : date | None
        Override for the current date (used in tests for determinism).
    **kwargs
        Ignored; absorbed for forward-compatibility.

    Returns
    -------
    ToolResult
        ``status="success"`` with human-readable summary and JSON data,
        or ``status="error"`` on failure.
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {})

    days = int(tool_input.get("days", _DEFAULT_DAYS))
    top = int(tool_input.get("top", _DEFAULT_TOP))

    # Validate inputs
    if days < 1 or days > 365:
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {"text": "Invalid 'days' value. Must be between 1 and 365."}
            ],
        }
        log_tool_output_size("get_epss_movers", result)
        return result

    if top < 1 or top > 100:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {"text": "Invalid 'top' value. Must be between 1 and 100."}
            ],
        }
        log_tool_output_size("get_epss_movers", result)
        return result

    today = now or date.today()
    # EPSS data has a ~1 day lag; use yesterday as "recent"
    recent_date = today - timedelta(days=1)
    baseline_date = recent_date - timedelta(days=days)

    try:
        data = compute_movers(
            recent_date=recent_date,
            baseline_date=baseline_date,
            top=top,
        )
    except requests.exceptions.RequestException as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {
                    "text": (
                        f"FIRST.org EPSS API request failed: {exc}"
                    )
                }
            ],
        }
        log_tool_output_size("get_epss_movers", result)
        return result
    except ValueError as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": str(exc)}],
        }
        log_tool_output_size("get_epss_movers", result)
        return result

    movers = data["movers"]

    # Build human-readable summary
    if not movers:
        summary = (
            f"No significant EPSS movers found between "
            f"{data['baseline_date']} and {data['recent_date']} "
            f"(compared {data['total_compared']} CVEs)."
        )
    else:
        lines = [
            f"🔥 Top {len(movers)} EPSS movers "
            f"({data['baseline_date']} → {data['recent_date']}, "
            f"{days}-day window)",
            f"   Compared {data['total_compared']} CVEs, "
            f"{data.get('total_with_increase', 0)} showed increases",
            "",
        ]
        for i, m in enumerate(movers, 1):
            pct_str = (
                f"+{m['pct_change']:.0f}%"
                if m["pct_change"] is not None
                else "NEW"
            )
            new_tag = " 🆕" if m["is_new"] else ""
            lines.append(
                f"  {i:>2}. {m['cve']:<20} "
                f"{m['baseline_epss']:.4f} → {m['recent_epss']:.4f}  "
                f"Δ +{m['delta']:.4f} ({pct_str}){new_tag}"
            )
        summary = "\n".join(lines)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": summary},
            {"json": data},
        ],
    }
    log_tool_output_size("get_epss_movers", result)
    return result
