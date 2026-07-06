#!/usr/bin/env python3
"""
Tool for fetching OSV.dev (Open Source Vulnerabilities) data for a CVE.

OSV.dev aggregates vulnerability records from many ecosystem databases
(GHSA, PyPA/PyPI, npm, Go, RustSec, Maven, etc.) into a single normalised
schema. Unlike NVD's CPE-centric model, OSV expresses affected software as
concrete *package + version-range* tuples keyed by ecosystem, which makes it
far more actionable for dependency-level triage.

This tool answers three questions NVD alone cannot:

1. Which ecosystems and packages does this CVE actually affect?
2. What are the exact vulnerable version ranges and first-fixed versions,
   per package (from OSV ``affected[].ranges`` events)?
3. What are all the aliases (GHSA/CVE/other) that refer to the same flaw?

Strategy: fetch the CVE's own OSV record from ``/v1/vulns/{cve}``; the CVE
record frequently lacks package-level ``affected`` data (which lives in the
GHSA advisory instead). When that happens we follow each ``GHSA-`` alias and
merge its per-package version ranges, deduplicating merged records by OSV id.

Public API, no key required:
- GET https://api.osv.dev/v1/vulns/{osv_id}
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off configuration (mirrors get_vulncheck_data conventions)
# ---------------------------------------------------------------------------
# Maximum number of HTTP attempts per request (1 = no retry).
# Override with OSV_MAX_RETRIES env var (useful in tests: set to "1").
_OSV_MAX_RETRIES: int = int(os.environ.get("OSV_MAX_RETRIES", "3"))

# Base delay between retries in seconds (doubles each attempt: 1s, 2s…).
# Override with OSV_RETRY_BASE_DELAY env var (set to "0" in tests).
_OSV_RETRY_BASE_DELAY: float = float(os.environ.get("OSV_RETRY_BASE_DELAY", "1.0"))

# HTTP status codes that are retryable (rate-limit or transient server error).
_OSV_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
_OSV_TIMEOUT = 15

# Cap on how many GHSA alias records we will follow for one CVE, to bound
# fan-out and output size for pathological cases.
_OSV_MAX_ALIAS_FOLLOWS = 8

TOOL_SPEC = {
    "name": "get_osv_data",
    "description": (
        "Fetches OSV.dev (Open Source Vulnerabilities) records for a given CVE ID. "
        "OSV normalises advisories from GHSA, PyPA/PyPI, npm, Go, RustSec, Maven and other "
        "ecosystem databases into concrete package + version-range tuples, which is far more "
        "actionable for dependency triage than NVD's CPE model. Returns, per affected package: "
        "the ecosystem, package name, vulnerable version ranges, and first-fixed versions "
        "(derived from OSV range events); plus all aliases (GHSA/CVE), severity (CVSS vectors), "
        "and references. Use this to answer 'which packages and versions does this CVE affect, "
        "and what version fixes it?'. Public API, no key required."
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
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------
def _osv_get_with_retry(osv_id: str) -> requests.Response:
    """GET a single OSV record with exponential back-off on transient errors.

    Retries on 429/5xx and on connection/timeout errors. Non-retryable 4xx
    (e.g. 404 for an unknown id) are returned to the caller as-is so they can
    be handled without raising.
    """
    url = _OSV_VULN_URL.format(osv_id=osv_id)
    last_exc: Exception | None = None
    for attempt in range(_OSV_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=_OSV_TIMEOUT, headers={"Accept": "application/json"})
            if resp.status_code in _OSV_RETRYABLE_STATUSES and attempt < _OSV_MAX_RETRIES - 1:
                time.sleep(_OSV_RETRY_BASE_DELAY * (2**attempt))
                continue
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < _OSV_MAX_RETRIES - 1:
                time.sleep(_OSV_RETRY_BASE_DELAY * (2**attempt))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OSV request failed without a specific exception")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_affected(affected: list[Any]) -> list[dict[str, Any]]:
    """Flatten OSV ``affected[]`` entries into package + version-range summaries."""
    packages: list[dict[str, Any]] = []
    for entry in affected or []:
        if not isinstance(entry, dict):
            continue
        pkg = entry.get("package") or {}
        ecosystem = pkg.get("ecosystem") if isinstance(pkg, dict) else None
        name = pkg.get("name") if isinstance(pkg, dict) else None
        # Skip fully package-less entries (nothing actionable to report).
        if ecosystem is None and name is None:
            continue

        introduced: list[str] = []
        fixed: list[str] = []
        last_affected: list[str] = []
        for rng in entry.get("ranges", []) or []:
            if not isinstance(rng, dict):
                continue
            for ev in rng.get("events", []) or []:
                if not isinstance(ev, dict):
                    continue
                if "introduced" in ev:
                    introduced.append(str(ev["introduced"]))
                if "fixed" in ev:
                    fixed.append(str(ev["fixed"]))
                if "last_affected" in ev:
                    last_affected.append(str(ev["last_affected"]))

        versions = [str(v) for v in (entry.get("versions", []) or []) if v is not None]

        packages.append(
            {
                "ecosystem": ecosystem,
                "package": name,
                "introduced": introduced,
                "fixed": fixed,
                "last_affected": last_affected,
                "affected_version_count": len(versions),
                "affected_versions_sample": versions[:10],
            }
        )
    return packages


def _parse_severity(record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract CVSS severity entries from an OSV record."""
    out: list[dict[str, str]] = []
    for sev in record.get("severity", []) or []:
        if isinstance(sev, dict):
            out.append({"type": str(sev.get("type", "")), "score": str(sev.get("score", ""))})
    return out


def _summarise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, structured summary from a single OSV vuln record."""
    packages = _parse_affected(record.get("affected", []))
    ecosystems = sorted({p["ecosystem"] for p in packages if p.get("ecosystem")})
    references = [r.get("url") for r in (record.get("references", []) or []) if isinstance(r, dict) and r.get("url")]
    return {
        "osv_id": record.get("id"),
        "summary": record.get("summary") or record.get("details"),
        "aliases": record.get("aliases", []) or [],
        "modified": record.get("modified"),
        "published": record.get("published"),
        "withdrawn": record.get("withdrawn"),
        "severity": _parse_severity(record),
        "affected_ecosystems": ecosystems,
        "affected_packages": packages,
        "references": references[:20],
    }


# ---------------------------------------------------------------------------
# Public fetch entry point (used by both the tool and the CLI)
# ---------------------------------------------------------------------------
def fetch_osv_data(cve_id: str) -> dict[str, Any]:
    """Query OSV.dev for a CVE and return a structured, normalised summary.

    Fetches the CVE's own OSV record and, when that record carries no
    package-level ``affected`` data, follows its ``GHSA-`` aliases to recover
    per-package version ranges. Merged records are deduplicated by OSV id.

    Returns a dict with keys: ``found`` (bool), ``cve_id``, ``records`` (list
    of per-record summaries), aggregated ``aliases`` and
    ``affected_ecosystems``, and a human-readable ``message``. Network/API
    errors are captured in ``error`` with ``found=False``.
    """
    cve_id = (cve_id or "").strip()
    if not cve_id:
        return {
            "found": False,
            "cve_id": cve_id,
            "records": [],
            "aliases": [],
            "affected_ecosystems": [],
            "error": "Invalid CVE ID. Must be a non-empty string.",
            "message": "Invalid CVE ID. Must be a non-empty string.",
        }

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_record(rec: dict[str, Any]) -> dict[str, Any]:
        summary = _summarise_record(rec)
        rid = summary.get("osv_id")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            records.append(summary)
        return summary

    try:
        resp = _osv_get_with_retry(cve_id)
    except Exception as exc:  # noqa: BLE001 - report any transport failure
        return {
            "found": False,
            "cve_id": cve_id,
            "records": [],
            "aliases": [],
            "affected_ecosystems": [],
            "error": f"OSV request failed: {exc}",
            "message": f"OSV.dev lookup failed for {cve_id}: {exc}",
        }

    if resp.status_code == 404:
        return {
            "found": False,
            "cve_id": cve_id,
            "records": [],
            "aliases": [],
            "affected_ecosystems": [],
            "message": f"No OSV.dev record found for {cve_id}.",
        }

    try:
        resp.raise_for_status()
        primary = resp.json()
    except (requests.exceptions.HTTPError, ValueError) as exc:
        return {
            "found": False,
            "cve_id": cve_id,
            "records": [],
            "aliases": [],
            "affected_ecosystems": [],
            "error": f"OSV response error: {exc}",
            "message": f"OSV.dev returned an unusable response for {cve_id}: {exc}",
        }

    primary_summary = _add_record(primary)

    # If the CVE record has no package-level data, follow GHSA aliases to
    # recover per-package version ranges (they live in the GHSA advisory).
    if not primary_summary["affected_packages"]:
        ghsa_aliases = [a for a in primary_summary["aliases"] if isinstance(a, str) and a.startswith("GHSA-")]
        for alias in ghsa_aliases[:_OSV_MAX_ALIAS_FOLLOWS]:
            try:
                a_resp = _osv_get_with_retry(alias)
                if a_resp.status_code == 200:
                    _add_record(a_resp.json())
            except Exception:  # noqa: BLE001 - alias enrichment is best-effort
                continue

    # Aggregate across all merged records.
    all_aliases = sorted(
        {a for rec in records for a in rec["aliases"] if isinstance(a, str)}
        | ({cve_id} if any(cve_id in rec["aliases"] or rec["osv_id"] == cve_id for rec in records) else set())
    )
    all_ecosystems = sorted({e for rec in records for e in rec["affected_ecosystems"]})
    total_packages = sum(len(rec["affected_packages"]) for rec in records)

    if all_ecosystems:
        eco = ", ".join(all_ecosystems)
        message = (
            f"OSV.dev: {cve_id} affects {total_packages} package(s) across {len(all_ecosystems)} ecosystem(s): {eco}."
        )
    else:
        message = (
            f"OSV.dev has a record for {cve_id} but no package-level version ranges (no ecosystem-specific advisory)."
        )

    return {
        "found": True,
        "cve_id": cve_id,
        "records": records,
        "aliases": all_aliases,
        "affected_ecosystems": all_ecosystems,
        "affected_package_count": total_packages,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def get_osv_data(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}
    cve_id = tool_input.get("cve_id")

    if not isinstance(cve_id, str) or not cve_id.strip():
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID. Must be a non-empty string."}],
        }
        log_tool_output_size("get_osv_data", result)
        return result

    payload = fetch_osv_data(cve_id)
    status = "success" if payload.get("found") else "error"
    if "error" in payload and not payload.get("found"):
        status = "error"

    import json

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("get_osv_data", result)
    return result
