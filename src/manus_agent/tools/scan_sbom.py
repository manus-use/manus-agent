#!/usr/bin/env python3
"""
Tool for scanning CycloneDX and SPDX SBOMs against OSV.dev for known
vulnerabilities, enriched with EPSS scores and CISA KEV membership.

Strategy
--------
1. **Parse** — Detect format (CycloneDX JSON, SPDX JSON) from top-level
   keys, then extract a flat list of ``(ecosystem, name, version)`` tuples
   from the component/package arrays.

2. **Query OSV.dev** — Use the ``/v1/querybatch`` endpoint (POST, no key
   required) to query all components in a single round-trip (batches of up
   to 1 000 queries per call, chunked if the SBOM is larger).

3. **Enrich** — For each vulnerability returned by OSV, fetch current EPSS
   score from the FIRST.org API (batch endpoint supports up to 100 CVEs)
   and check CISA KEV catalog membership.

4. **Rank** — Sort findings by: KEV membership (desc) → EPSS score (desc)
   → CVSS severity (desc) → CVE ID (asc), exactly as documented in the
   README.

Public APIs used (no keys required):
- OSV.dev ``/v1/querybatch`` — batch vulnerability lookup by package
- FIRST.org EPSS ``/api/data/v1/epss`` — batch EPSS scores by CVE
- CISA KEV JSON feed — known-exploited-vulnerabilities catalog
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("SBOM_SCAN_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("SBOM_SCAN_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_HTTP_TIMEOUT: int = 30

# OSV.dev batch query endpoint — max 1 000 queries per call.
_OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
_OSV_BATCH_CHUNK = 1000

# FIRST.org EPSS bulk endpoint (supports comma-separated CVEs, up to ~100).
_EPSS_URL = "https://api.first.org/data/v1/epss"
_EPSS_BATCH_SIZE = 100

# CISA KEV catalog feed.
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# ---------------------------------------------------------------------------
# Ecosystem mapping: SBOM package-URL type → OSV ecosystem name
# ---------------------------------------------------------------------------
_PURL_TYPE_TO_ECOSYSTEM: dict[str, str] = {
    "pypi": "PyPI",
    "npm": "npm",
    "maven": "Maven",
    "golang": "Go",
    "cargo": "crates.io",
    "nuget": "NuGet",
    "gem": "RubyGems",
    "composer": "Packagist",
    "pub": "Pub",
    "hex": "Hex",
    "hackage": "Hackage",
    "swift": "SwiftURL",
    "cocoapods": "CocoaPods",
    "cran": "CRAN",
    "apk": "Alpine",
    "deb": "Debian",
    "rpm": "AlmaLinux",  # conservative default
    "github": "GitHub Actions",
}

# CycloneDX component type hints for ecosystem fallback.
_CDX_TYPE_HINTS: dict[str, str] = {
    "library": "PyPI",  # last-resort default
}

# SPDX external-ref category/type for purl.
_SPDX_PURL_CATEGORY = "PACKAGE-MANAGER"
_SPDX_PURL_TYPE = "purl"

# Severity label mapping.
_SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}

# ---------------------------------------------------------------------------
# Strands TOOL_SPEC
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "scan_sbom",
    "description": (
        "Scans a CycloneDX or SPDX SBOM file for known vulnerabilities by "
        "querying OSV.dev in batch, enriching each finding with EPSS score "
        "and CISA KEV status, and ranking results by KEV membership then "
        "EPSS score. Accepts a local file path to a JSON SBOM."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sbom_path": {
                    "type": "string",
                    "description": "Path to a CycloneDX or SPDX JSON SBOM file.",
                },
            },
            "required": ["sbom_path"],
        }
    },
}


# ===================================================================
# HTTP helpers
# ===================================================================


def _request_with_retry(
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """HTTP request with exponential back-off on transient failures."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(
                method,
                url,
                json=json_body,
                params=params,
                timeout=timeout,
            )
            if resp.status_code not in _RETRYABLE_STATUSES:
                return resp
        except requests.RequestException:
            if attempt == _MAX_RETRIES - 1:
                raise
        delay = _RETRY_BASE_DELAY * (2**attempt)
        time.sleep(delay)
    # Unreachable in normal flow, but satisfy type checker.
    return resp  # type: ignore[possibly-undefined]


# ===================================================================
# SBOM parsing
# ===================================================================


def _parse_purl(purl: str) -> tuple[str, str, str]:
    """Extract (ecosystem, name, version) from a package URL.

    Minimal parser — covers the ``pkg:<type>/<namespace>/<name>@<version>``
    and ``pkg:<type>/<name>@<version>`` forms emitted by common SBOM tools.

    Returns ``("", "", "")`` if the purl cannot be parsed.
    """
    if not purl.startswith("pkg:"):
        return ("", "", "")
    rest = purl[4:]  # drop "pkg:"
    # Strip qualifiers and subpath: ?...#...
    for sep in ("?", "#"):
        idx = rest.find(sep)
        if idx != -1:
            rest = rest[:idx]
    # Split type from path: <type>/<path>
    slash_idx = rest.find("/")
    if slash_idx == -1:
        return ("", "", "")
    purl_type = rest[:slash_idx].lower()
    path = rest[slash_idx + 1 :]
    # Decode percent-encoded characters before further parsing.
    path = path.replace("%40", "@").replace("%2F", "/").replace("%2f", "/")
    # Split version: ...@<version> — but for scoped packages like
    # @scope/name@version, we need the *last* unambiguous '@'.
    at_idx = path.rfind("@")
    if at_idx == -1:
        return ("", "", "")  # no version → useless for vuln lookup
    version = path[at_idx + 1 :]
    name_part = path[:at_idx]
    ecosystem = _PURL_TYPE_TO_ECOSYSTEM.get(purl_type, "")
    # For maven, name is "group:artifact" in OSV parlance.
    if purl_type == "maven":
        name_part = name_part.replace("/", ":")
    # For Go modules, keep the full import path.
    # For most others, use the last component as the package name.
    if purl_type not in ("golang", "maven"):
        # npm scoped packages: @scope/name → keep as-is
        if "/" in name_part and not name_part.startswith("@"):
            name_part = name_part.rsplit("/", 1)[-1]
    return (ecosystem, name_part, version)


def _parse_cyclonedx(doc: dict[str, Any]) -> list[dict[str, str]]:
    """Extract components from a CycloneDX JSON SBOM."""
    components: list[dict[str, str]] = []
    for comp in doc.get("components", []):
        name = comp.get("name", "")
        version = comp.get("version", "")
        ecosystem = ""
        # Prefer purl if present.
        purl = comp.get("purl", "")
        if purl:
            eco, pname, pver = _parse_purl(purl)
            if eco and pname:
                ecosystem = eco
                name = pname
                if pver:
                    version = pver
        if not version:
            continue  # can't query without a version
        if not ecosystem:
            # Fallback: try to guess from component type or group.
            ecosystem = _CDX_TYPE_HINTS.get(comp.get("type", ""), "")
        if name and version:
            components.append({"ecosystem": ecosystem, "name": name, "version": version, "purl": purl})
    return components


def _parse_spdx(doc: dict[str, Any]) -> list[dict[str, str]]:
    """Extract packages from an SPDX 2.x JSON SBOM."""
    components: list[dict[str, str]] = []
    for pkg in doc.get("packages", []):
        name = pkg.get("name", "")
        version = pkg.get("versionInfo", "")
        ecosystem = ""
        purl = ""
        # Look for purl in externalRefs.
        for ref in pkg.get("externalRefs", []):
            cat = ref.get("referenceCategory", "")
            rtype = ref.get("referenceType", "")
            if cat == _SPDX_PURL_CATEGORY and rtype == _SPDX_PURL_TYPE:
                purl = ref.get("referenceLocator", "")
                break
        if purl:
            eco, pname, pver = _parse_purl(purl)
            if eco and pname:
                ecosystem = eco
                name = pname
                if pver:
                    version = pver
        if not version:
            continue
        if name and version:
            components.append({"ecosystem": ecosystem, "name": name, "version": version, "purl": purl})
    return components


def parse_sbom(path: str) -> tuple[str, list[dict[str, str]]]:
    """Parse an SBOM file and return (format_name, components).

    Raises ``ValueError`` for unrecognised or unparseable files.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"SBOM file not found: {path}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot parse SBOM as JSON: {exc}") from exc

    if not isinstance(doc, dict):
        raise ValueError("SBOM root must be a JSON object")

    # Detect format.
    if "bomFormat" in doc:
        # CycloneDX
        return ("CycloneDX", _parse_cyclonedx(doc))
    if "spdxVersion" in doc:
        # SPDX
        return ("SPDX", _parse_spdx(doc))

    raise ValueError("Unrecognised SBOM format — expected CycloneDX (bomFormat) or SPDX (spdxVersion)")


# ===================================================================
# OSV.dev batch query
# ===================================================================


def _build_osv_queries(
    components: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Build OSV querybatch query objects from parsed components."""
    queries: list[dict[str, Any]] = []
    for comp in components:
        q: dict[str, Any] = {
            "package": {"name": comp["name"]},
            "version": comp["version"],
        }
        if comp.get("ecosystem"):
            q["package"]["ecosystem"] = comp["ecosystem"]
        queries.append(q)
    return queries


def query_osv_batch(
    components: list[dict[str, str]],
) -> dict[int, list[dict[str, Any]]]:
    """Query OSV.dev /v1/querybatch for all components.

    Returns a dict mapping component index → list of OSV vulnerability
    records.  Components with no vulnerabilities are omitted.
    """
    queries = _build_osv_queries(components)
    results: dict[int, list[dict[str, Any]]] = {}
    # Process in chunks of _OSV_BATCH_CHUNK.
    for chunk_start in range(0, len(queries), _OSV_BATCH_CHUNK):
        chunk = queries[chunk_start : chunk_start + _OSV_BATCH_CHUNK]
        body = {"queries": chunk}
        try:
            resp = _request_with_retry("POST", _OSV_QUERYBATCH_URL, json_body=body)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue
        for i, result in enumerate(data.get("results", [])):
            vulns = result.get("vulns", [])
            if vulns:
                results[chunk_start + i] = vulns
    return results


# ===================================================================
# EPSS batch lookup
# ===================================================================


def fetch_epss_batch(cve_ids: list[str]) -> dict[str, float]:
    """Fetch EPSS scores for a batch of CVE IDs.

    Returns a dict mapping CVE-ID → EPSS probability (0.0–1.0).
    Missing CVEs are omitted from the result.
    """
    scores: dict[str, float] = {}
    for chunk_start in range(0, len(cve_ids), _EPSS_BATCH_SIZE):
        chunk = cve_ids[chunk_start : chunk_start + _EPSS_BATCH_SIZE]
        params = {"cve": ",".join(chunk)}
        try:
            resp = _request_with_retry("GET", _EPSS_URL, params=params)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue
        for entry in data.get("data", []):
            cve = entry.get("cve", "")
            epss = entry.get("epss")
            if cve and epss is not None:
                try:
                    scores[cve] = float(epss)
                except (TypeError, ValueError):
                    pass
    return scores


# ===================================================================
# CISA KEV lookup
# ===================================================================


def fetch_kev_set() -> set[str]:
    """Fetch the CISA KEV catalog and return a set of CVE IDs."""
    try:
        resp = _request_with_retry("GET", _KEV_URL)
        if resp.status_code != 200:
            return set()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return set()
    return {v.get("cveID", "").upper() for v in data.get("vulnerabilities", []) if v.get("cveID")}


# ===================================================================
# Finding assembly and ranking
# ===================================================================


def _extract_severity(vuln: dict[str, Any]) -> tuple[str, float]:
    """Extract the best severity label and numeric score from an OSV vuln."""
    best_label = "UNKNOWN"
    best_score = 0.0
    for sev in vuln.get("severity", []):
        score_str = sev.get("score", "")
        if not score_str:
            continue
        # OSV severity scores are CVSS vectors; extract base score.
        # Try to parse the numeric score from the vector.
        # Some OSV records use {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/..."}.
        if "CVSS:" in score_str:
            # Extract base score: the last metric group value after /
            parts = score_str.split("/")
            # Try to find a numeric token (sometimes appended).
            for part in reversed(parts):
                try:
                    val = float(part)
                    if val > best_score:
                        best_score = val
                        best_label = _score_to_label(val)
                    break
                except ValueError:
                    continue
    # Fall back to database_specific severity if available.
    db_spec = vuln.get("database_specific", {})
    if best_label == "UNKNOWN":
        sev_str = db_spec.get("severity", "")
        if isinstance(sev_str, str) and sev_str.upper() in _SEVERITY_ORDER:
            best_label = sev_str.upper()
            # Assign synthetic scores for sorting when no CVSS.
            synthetic = {"CRITICAL": 9.5, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.5}
            best_score = synthetic.get(best_label, 0.0)
    # Also check ecosystem-specific severity in affected[].
    for aff in vuln.get("affected", []):
        eco_sev = aff.get("database_specific", {}).get("severity", "")
        if isinstance(eco_sev, str) and eco_sev.upper() in _SEVERITY_ORDER:
            label = eco_sev.upper()
            if _SEVERITY_ORDER.get(label, 0) > _SEVERITY_ORDER.get(best_label, 0):
                best_label = label
                synthetic = {"CRITICAL": 9.5, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.5}
                best_score = synthetic.get(best_label, 0.0)
    return (best_label, best_score)


def _score_to_label(score: float) -> str:
    """Map a CVSS base score to a severity label."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "UNKNOWN"


def _extract_cve_ids(vuln: dict[str, Any]) -> list[str]:
    """Extract CVE IDs from an OSV vulnerability record."""
    cves: list[str] = []
    # Check aliases.
    for alias in vuln.get("aliases", []):
        if alias.upper().startswith("CVE-"):
            cves.append(alias.upper())
    # Check the id itself.
    vid = vuln.get("id", "")
    if vid.upper().startswith("CVE-") and vid.upper() not in cves:
        cves.append(vid.upper())
    return cves


def assemble_findings(
    components: list[dict[str, str]],
    osv_results: dict[int, list[dict[str, Any]]],
    epss_scores: dict[str, float],
    kev_set: set[str],
) -> list[dict[str, Any]]:
    """Assemble, deduplicate, and rank scan findings.

    Returns a sorted list of finding dicts, each containing:
    - vuln_id: OSV vulnerability ID
    - cve_ids: list of associated CVE IDs
    - component: affected component info
    - severity / severity_score
    - epss: highest EPSS score across associated CVEs
    - in_kev: whether any associated CVE is in CISA KEV
    - summary: short description
    """
    # Deduplicate: (vuln_id, component_name) → finding.
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for comp_idx, vulns in osv_results.items():
        comp = components[comp_idx]
        for vuln in vulns:
            vuln_id = vuln.get("id", "unknown")
            dedup_key = (vuln_id, comp["name"])
            if dedup_key in seen:
                continue
            cve_ids = _extract_cve_ids(vuln)
            severity_label, severity_score = _extract_severity(vuln)
            # EPSS: take the highest across associated CVEs.
            epss = max((epss_scores.get(c, 0.0) for c in cve_ids), default=0.0)
            # KEV: any associated CVE in the catalog?
            in_kev = any(c in kev_set for c in cve_ids)
            finding: dict[str, Any] = {
                "vuln_id": vuln_id,
                "cve_ids": cve_ids,
                "component": {
                    "name": comp["name"],
                    "version": comp["version"],
                    "ecosystem": comp.get("ecosystem", ""),
                },
                "severity": severity_label,
                "severity_score": severity_score,
                "epss": epss,
                "in_kev": in_kev,
                "summary": vuln.get("summary", vuln.get("details", ""))[:200],
            }
            seen[dedup_key] = finding
    # Rank: KEV desc → EPSS desc → severity_score desc → vuln_id asc.
    findings = sorted(
        seen.values(),
        key=lambda f: (
            -(1 if f["in_kev"] else 0),
            -f["epss"],
            -f["severity_score"],
            f["vuln_id"],
        ),
    )
    return findings


# ===================================================================
# Summary statistics
# ===================================================================


def compute_stats(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics for the scan results."""
    total = len(findings)
    kev_count = sum(1 for f in findings if f["in_kev"])
    severity_counts: dict[str, int] = {}
    for f in findings:
        sev = f["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    # Unique components affected.
    affected_components = len({(f["component"]["name"], f["component"]["version"]) for f in findings})
    return {
        "total_vulnerabilities": total,
        "kev_count": kev_count,
        "critical_count": severity_counts.get("CRITICAL", 0),
        "high_count": severity_counts.get("HIGH", 0),
        "medium_count": severity_counts.get("MEDIUM", 0),
        "low_count": severity_counts.get("LOW", 0),
        "unknown_count": severity_counts.get("UNKNOWN", 0),
        "affected_components": affected_components,
        "severity_breakdown": severity_counts,
    }


# ===================================================================
# Main scan entry point
# ===================================================================


def scan_sbom(sbom_path: str) -> dict[str, Any]:
    """Scan an SBOM file and return structured results.

    Returns a dict with keys:
    - sbom_format: "CycloneDX" or "SPDX"
    - total_components: number of parseable components
    - findings: ranked list of vulnerability findings
    - stats: aggregate statistics
    - message: human-readable summary
    """
    # 1. Parse SBOM.
    sbom_format, components = parse_sbom(sbom_path)
    if not components:
        return {
            "sbom_format": sbom_format,
            "total_components": 0,
            "findings": [],
            "stats": compute_stats([]),
            "message": f"Parsed {sbom_format} SBOM — no components with versions found.",
        }

    # 2. Query OSV.dev in batch.
    osv_results = query_osv_batch(components)

    # 3. Collect all unique CVE IDs for EPSS + KEV enrichment.
    all_cve_ids: set[str] = set()
    for vulns in osv_results.values():
        for vuln in vulns:
            all_cve_ids.update(_extract_cve_ids(vuln))

    # 4. Enrich: EPSS scores + KEV membership.
    epss_scores: dict[str, float] = {}
    kev_set: set[str] = set()
    if all_cve_ids:
        epss_scores = fetch_epss_batch(sorted(all_cve_ids))
        kev_set = fetch_kev_set()

    # 5. Assemble and rank findings.
    findings = assemble_findings(components, osv_results, epss_scores, kev_set)
    stats = compute_stats(findings)

    # 6. Build human-readable summary.
    msg_parts = [
        f"Scanned {sbom_format} SBOM: {len(components)} components, "
        f"{stats['total_vulnerabilities']} vulnerabilities found."
    ]
    if stats["kev_count"]:
        msg_parts.append(f"⚠️  {stats['kev_count']} in CISA KEV (actively exploited).")
    if stats["critical_count"]:
        msg_parts.append(f"🔴 {stats['critical_count']} CRITICAL")
    if stats["high_count"]:
        msg_parts.append(f"🟠 {stats['high_count']} HIGH")
    if stats["medium_count"]:
        msg_parts.append(f"🟡 {stats['medium_count']} MEDIUM")
    if stats["low_count"]:
        msg_parts.append(f"🟢 {stats['low_count']} LOW")

    return {
        "sbom_format": sbom_format,
        "total_components": len(components),
        "findings": findings,
        "stats": stats,
        "message": " ".join(msg_parts),
    }


# ===================================================================
# Strands tool handler
# ===================================================================


def handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool handler for scan_sbom."""
    tool_input = tool["input"]
    sbom_path = tool_input.get("sbom_path", "")
    if not sbom_path:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "sbom_path is required"}],
        }
    try:
        result = scan_sbom(sbom_path)
    except ValueError as exc:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": str(exc)}],
        }
    except Exception as exc:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": f"scan_sbom failed: {exc}"}],
        }

    text_output = json.dumps(result, indent=2)
    log_tool_output_size("scan_sbom", text_output)
    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": text_output}],
    }
