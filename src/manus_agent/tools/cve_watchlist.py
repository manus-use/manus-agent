#!/usr/bin/env python3
"""CVE Watchlist — persistent tracking with bulk EPSS + KEV status updates.

Provides a local watchlist (~/.manus-agent/watchlist.json) where users can
add/remove CVEs they're monitoring and retrieve bulk status updates showing
current EPSS scores, CISA KEV membership, and status changes since the last
check.

Design principles:
- Zero API keys required (EPSS is public, CISA KEV is public JSON feed)
- Persistent local storage with atomic writes
- Batch EPSS lookups (the FIRST.org API supports comma-separated CVE lists)
- Tracks previous EPSS scores to detect movement (spikes/drops)
- Human-friendly CLI output via Rich tables
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

__all__ = ["watchlist_manage"]

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

_DEFAULT_WATCHLIST_DIR = Path.home() / ".manus-agent"
_DEFAULT_WATCHLIST_FILE = _DEFAULT_WATCHLIST_DIR / "watchlist.json"

# Allow override via env var (useful for tests)
_WATCHLIST_PATH = Path(os.environ.get("MANUS_WATCHLIST_PATH", str(_DEFAULT_WATCHLIST_FILE)))

# CVE ID pattern
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

# EPSS batch size limit (FIRST.org API accepts up to ~100 CVEs per request)
_EPSS_BATCH_SIZE = 100

# CISA KEV feed URL
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# EPSS API base URL
_EPSS_API_URL = "https://api.first.org/data/v1/epss"

# HTTP timeout for all requests
_HTTP_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Watchlist data model
# ---------------------------------------------------------------------------


def _load_watchlist() -> dict[str, Any]:
    """Load the watchlist from disk. Returns empty structure if not found."""
    if not _WATCHLIST_PATH.exists():
        return {"cves": {}, "last_checked": None}
    try:
        data = json.loads(_WATCHLIST_PATH.read_text(encoding="utf-8"))
        if "cves" not in data:
            data["cves"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"cves": {}, "last_checked": None}


def _save_watchlist(data: dict[str, Any]) -> None:
    """Atomically save the watchlist to disk."""
    _WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _WATCHLIST_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(_WATCHLIST_PATH)


def _validate_cve_id(cve_id: str) -> str | None:
    """Validate and normalise a CVE ID. Returns normalised ID or None."""
    cve_id = cve_id.strip().upper()
    if _CVE_RE.match(cve_id):
        return cve_id
    return None


# ---------------------------------------------------------------------------
# External data fetching
# ---------------------------------------------------------------------------


def _fetch_epss_batch(cve_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch EPSS scores for a batch of CVE IDs.

    Returns a dict mapping CVE ID -> {epss, percentile, date}.
    """
    results: dict[str, dict[str, Any]] = {}
    for i in range(0, len(cve_ids), _EPSS_BATCH_SIZE):
        batch = cve_ids[i : i + _EPSS_BATCH_SIZE]
        try:
            resp = requests.get(
                _EPSS_API_URL,
                params={"cve": ",".join(batch)},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                results[item["cve"].upper()] = {
                    "epss": float(item.get("epss", 0)),
                    "percentile": float(item.get("percentile", 0)),
                    "date": item.get("date", ""),
                }
        except (requests.RequestException, ValueError, KeyError):
            # On failure, leave those CVEs without EPSS data
            continue
    return results


def _fetch_kev_set() -> set[str]:
    """Fetch the current CISA KEV catalog and return the set of CVE IDs."""
    try:
        resp = requests.get(_CISA_KEV_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return {v["cveID"].upper() for v in data.get("vulnerabilities", [])}
    except (requests.RequestException, ValueError, KeyError):
        return set()


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def watchlist_add(cve_ids: list[str], note: str | None = None) -> dict[str, Any]:
    """Add one or more CVEs to the watchlist.

    Args:
        cve_ids: List of CVE identifiers to add.
        note: Optional note/tag for these CVEs.

    Returns:
        Summary dict with added/invalid/duplicate lists.
    """
    wl = _load_watchlist()
    added = []
    invalid = []
    duplicate = []

    for raw_id in cve_ids:
        normalised = _validate_cve_id(raw_id)
        if normalised is None:
            invalid.append(raw_id)
            continue
        if normalised in wl["cves"]:
            duplicate.append(normalised)
            continue
        wl["cves"][normalised] = {
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": note or "",
            "last_epss": None,
            "last_percentile": None,
            "in_kev": None,
        }
        added.append(normalised)

    if added:
        _save_watchlist(wl)

    return {"added": added, "invalid": invalid, "duplicate": duplicate, "total": len(wl["cves"])}


def watchlist_remove(cve_ids: list[str]) -> dict[str, Any]:
    """Remove one or more CVEs from the watchlist.

    Args:
        cve_ids: List of CVE identifiers to remove.

    Returns:
        Summary dict with removed/not_found lists.
    """
    wl = _load_watchlist()
    removed = []
    not_found = []

    for raw_id in cve_ids:
        normalised = _validate_cve_id(raw_id)
        if normalised is None:
            not_found.append(raw_id)
            continue
        if normalised in wl["cves"]:
            del wl["cves"][normalised]
            removed.append(normalised)
        else:
            not_found.append(raw_id)

    if removed:
        _save_watchlist(wl)

    return {"removed": removed, "not_found": not_found, "total": len(wl["cves"])}


def watchlist_list() -> dict[str, Any]:
    """List all CVEs currently on the watchlist.

    Returns:
        Dict with cves list and metadata.
    """
    wl = _load_watchlist()
    entries = []
    for cve_id, info in sorted(wl["cves"].items()):
        entries.append(
            {
                "cve_id": cve_id,
                "added_at": info.get("added_at", ""),
                "note": info.get("note", ""),
                "last_epss": info.get("last_epss"),
                "last_percentile": info.get("last_percentile"),
                "in_kev": info.get("in_kev"),
            }
        )
    return {
        "entries": entries,
        "total": len(entries),
        "last_checked": wl.get("last_checked"),
    }


def watchlist_status() -> dict[str, Any]:
    """Fetch live EPSS + KEV status for all watched CVEs.

    Compares current EPSS scores against last-stored values and flags
    significant changes (spike >= 0.05, drop >= 0.05).

    Returns:
        Dict with per-CVE status, changes detected, and summary stats.
    """
    wl = _load_watchlist()
    cve_ids = list(wl["cves"].keys())

    if not cve_ids:
        return {"entries": [], "total": 0, "changes": [], "summary": {"kev_count": 0, "avg_epss": 0.0}}

    # Fetch live data
    epss_data = _fetch_epss_batch(cve_ids)
    kev_set = _fetch_kev_set()

    entries = []
    changes = []
    total_epss = 0.0
    epss_count = 0
    kev_count = 0

    for cve_id in sorted(cve_ids):
        info = wl["cves"][cve_id]
        prev_epss = info.get("last_epss")
        prev_kev = info.get("in_kev")

        # Current values
        current_epss_data = epss_data.get(cve_id, {})
        current_epss = current_epss_data.get("epss")
        current_percentile = current_epss_data.get("percentile")
        current_kev = cve_id in kev_set

        # Detect changes
        epss_delta = None
        if current_epss is not None and prev_epss is not None:
            epss_delta = current_epss - prev_epss

        if epss_delta is not None and abs(epss_delta) >= 0.05:
            direction = "spike" if epss_delta > 0 else "drop"
            changes.append(
                {
                    "cve_id": cve_id,
                    "type": f"epss_{direction}",
                    "previous": prev_epss,
                    "current": current_epss,
                    "delta": round(epss_delta, 4),
                }
            )

        if prev_kev is not None and current_kev and not prev_kev:
            changes.append(
                {
                    "cve_id": cve_id,
                    "type": "kev_added",
                    "previous": False,
                    "current": True,
                }
            )

        # Update stored state
        if current_epss is not None:
            info["last_epss"] = current_epss
            info["last_percentile"] = current_percentile
            total_epss += current_epss
            epss_count += 1
        info["in_kev"] = current_kev

        if current_kev:
            kev_count += 1

        entries.append(
            {
                "cve_id": cve_id,
                "epss": current_epss,
                "percentile": current_percentile,
                "epss_delta": round(epss_delta, 4) if epss_delta is not None else None,
                "in_kev": current_kev,
                "note": info.get("note", ""),
                "added_at": info.get("added_at", ""),
            }
        )

    # Persist updated state
    wl["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_watchlist(wl)

    avg_epss = round(total_epss / epss_count, 4) if epss_count > 0 else 0.0

    return {
        "entries": entries,
        "total": len(entries),
        "changes": changes,
        "summary": {
            "kev_count": kev_count,
            "avg_epss": avg_epss,
            "checked_at": wl["last_checked"],
        },
    }


def watchlist_clear() -> dict[str, Any]:
    """Clear the entire watchlist.

    Returns:
        Summary with count of removed entries.
    """
    wl = _load_watchlist()
    count = len(wl["cves"])
    wl["cves"] = {}
    wl["last_checked"] = None
    _save_watchlist(wl)
    return {"cleared": count}


# ---------------------------------------------------------------------------
# Unified dispatch (for CLI integration)
# ---------------------------------------------------------------------------


def watchlist_manage(
    action: str,
    cve_ids: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Unified watchlist management function.

    Args:
        action: One of 'add', 'remove', 'list', 'status', 'clear'.
        cve_ids: CVE IDs for add/remove actions.
        note: Optional note for add action.

    Returns:
        Action-specific result dict.
    """
    if action == "add":
        if not cve_ids:
            return {"error": "No CVE IDs provided for 'add' action."}
        return watchlist_add(cve_ids, note=note)
    elif action == "remove":
        if not cve_ids:
            return {"error": "No CVE IDs provided for 'remove' action."}
        return watchlist_remove(cve_ids)
    elif action == "list":
        return watchlist_list()
    elif action == "status":
        return watchlist_status()
    elif action == "clear":
        return watchlist_clear()
    else:
        return {"error": f"Unknown action: {action!r}. Valid: add, remove, list, status, clear."}
