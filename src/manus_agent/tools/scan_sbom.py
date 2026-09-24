"""Tool for scanning CycloneDX / SPDX SBOMs against OSV.dev for known vulnerabilities.

Parses a CycloneDX 1.x (JSON) or SPDX 2.x (JSON) SBOM, extracts every
component's ecosystem + name + version, queries OSV.dev in batch for known
vulnerabilities, then enriches each finding with EPSS score and CISA KEV
membership.  Results are ranked by KEV status first, then EPSS score
descending.

Public APIs used (no keys required):
- POST https://api.osv.dev/v1/querybatch  — batch vulnerability lookup
- GET  https://api.first.org/data/v1/epss  — EPSS scores (supports CSV bulk)
- GET  https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["scan_sbom", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
_MAX_RETRIES: int = int(os.environ.get("SBOM_SCAN_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("SBOM_SCAN_RETRY_BASE_DELAY", "1.0"))
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
_EPSS_API_URL = "https://api.first.org/data/v1/epss"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

_OSV_TIMEOUT = 30
_EPSS_TIMEOUT = 20
_KEV_TIMEOUT = 15

# OSV batch API accepts up to 1000 queries per request.
_OSV_BATCH_SIZE = 1000

# CVE pattern
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Ecosystem mapping: PURL type → OSV ecosystem string
# ---------------------------------------------------------------------------
_PURL_TYPE_TO_OSV: dict[str, str] = {
    "pypi": "PyPI",
    "npm": "npm",
    "maven": "Maven",
    "golang": "Go",
    "cargo": "crates.io",
    "nuget": "NuGet",
    "gem": "RubyGems",
    "composer": "Packagist",
    "hex": "Hex",
    "pub": "Pub",
    "swift": "SwiftURL",
    "cocoapods": "CocoaPods",
    "hackage": "Hackage",
    "cran": "CRAN",
}

# SPDX externalRef type patterns for package managers
_SPDX_PKG_MANAGER_RE = re.compile(r"pkg:(\w+)/", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Strands tool spec
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "scan_sbom",
    "description": (
        "Scans a CycloneDX or SPDX SBOM (JSON) for known vulnerabilities. "
        "Extracts components, queries OSV.dev in batch, enriches findings with "
        "EPSS scores and CISA KEV membership, and ranks results by exploited-in-wild "
        "status then EPSS score. Returns vulnerability counts by severity and "
        "per-component finding details."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sbom_path": {
                    "type": "string",
                    "description": ("Path to a CycloneDX (JSON) or SPDX (JSON) SBOM file."),
                }
            },
            "required": ["sbom_path"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> requests.Response:
    """GET *url* with exponential back-off on retryable failures."""
    retries = max_retries if max_retries is not None else _MAX_RETRIES
    delay = base_delay if base_delay is not None else _RETRY_BASE_DELAY

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES and attempt < retries - 1:
                time.sleep(delay * (2**attempt))
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay * (2**attempt))
    raise last_exc or RuntimeError("HTTP request failed after retries")


def _post_with_retry(
    url: str,
    *,
    json_body: Any = None,
    timeout: int = 30,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> requests.Response:
    """POST *url* with exponential back-off on retryable failures."""
    retries = max_retries if max_retries is not None else _MAX_RETRIES
    delay = base_delay if base_delay is not None else _RETRY_BASE_DELAY

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=json_body, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUSES and attempt < retries - 1:
                time.sleep(delay * (2**attempt))
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay * (2**attempt))
    raise last_exc or RuntimeError("HTTP request failed after retries")


# ---------------------------------------------------------------------------
# SBOM parsing
# ---------------------------------------------------------------------------
def _parse_purl(purl: str) -> dict[str, str]:
    """Extract ecosystem, name, and version from a Package URL (purl).

    Handles standard purl format: ``pkg:<type>/<namespace>/<name>@<version>``
    and the namespace-less form ``pkg:<type>/<name>@<version>``.
    """
    result: dict[str, str] = {"ecosystem": "", "name": "", "version": ""}
    if not purl or not purl.startswith("pkg:"):
        return result

    # Strip qualifiers/subpath: everything after ? or #
    core = purl.split("?")[0].split("#")[0]
    # pkg:<type>/<rest>
    without_prefix = core[4:]  # remove "pkg:"
    slash_idx = without_prefix.find("/")
    if slash_idx < 0:
        return result

    purl_type = without_prefix[:slash_idx].lower()
    remainder = without_prefix[slash_idx + 1 :]

    # URL-decode common percent-encoded characters (e.g. %40 → @)
    from urllib.parse import unquote

    remainder = unquote(remainder)

    # Split version
    version = ""
    if "@" in remainder:
        name_part, version = remainder.rsplit("@", 1)
    else:
        name_part = remainder

    # Decode %2F → / for namespaced packages (e.g. Maven group/artifact)
    # (already handled by unquote above, but kept for clarity)

    # For Maven, keep group/artifact; for others, take last segment if namespaced
    if purl_type == "maven":
        name = name_part.replace("/", ":")
    elif "/" in name_part:
        # npm scoped packages: @scope/name → keep as-is
        if purl_type == "npm":
            name = name_part
        else:
            name = name_part.rsplit("/", 1)[-1]
    else:
        name = name_part

    ecosystem = _PURL_TYPE_TO_OSV.get(purl_type, purl_type)

    result["ecosystem"] = ecosystem
    result["name"] = name
    result["version"] = version
    return result


def _parse_cyclonedx(data: dict) -> list[dict[str, str]]:
    """Extract components from a CycloneDX 1.x JSON SBOM.

    Returns a list of ``{"ecosystem": ..., "name": ..., "version": ...}``
    dicts.
    """
    components: list[dict[str, str]] = []
    raw_components = data.get("components", [])

    for comp in raw_components:
        if not isinstance(comp, dict):
            continue

        name = comp.get("name", "")
        version = comp.get("version", "")
        purl = comp.get("purl", "")
        comp_type = comp.get("type", "library")

        # Skip non-library components (framework, application, etc. may be
        # included but are not meaningful for vuln scanning)
        if comp_type not in ("library", "framework"):
            continue

        if purl:
            parsed = _parse_purl(purl)
            if parsed["name"]:
                # Prefer purl-derived fields but fall back to component-level
                components.append(
                    {
                        "ecosystem": parsed["ecosystem"],
                        "name": parsed["name"],
                        "version": parsed["version"] or version,
                    }
                )
                continue

        # No purl — best-effort from name + version
        if name:
            components.append({"ecosystem": "", "name": name, "version": version})

    return components


def _parse_spdx(data: dict) -> list[dict[str, str]]:
    """Extract packages from an SPDX 2.x JSON SBOM.

    Returns a list of ``{"ecosystem": ..., "name": ..., "version": ...}``
    dicts.
    """
    components: list[dict[str, str]] = []
    packages = data.get("packages", [])

    for pkg in packages:
        if not isinstance(pkg, dict):
            continue

        name = pkg.get("name", "")
        version = pkg.get("versionInfo", "")

        # Skip the root document package (SPDX convention)
        spdx_id = pkg.get("SPDXID", "")
        if spdx_id == "SPDXRef-DOCUMENT":
            continue

        # Try externalRefs for purl
        ext_refs = pkg.get("externalRefs", [])
        purl_found = False
        for ref in ext_refs:
            if not isinstance(ref, dict):
                continue
            ref_type = ref.get("referenceType", "")
            if ref_type == "purl" or ref_type == "pkg":
                locator = ref.get("referenceLocator", "")
                if locator.startswith("pkg:"):
                    parsed = _parse_purl(locator)
                    if parsed["name"]:
                        components.append(
                            {
                                "ecosystem": parsed["ecosystem"],
                                "name": parsed["name"],
                                "version": parsed["version"] or version,
                            }
                        )
                        purl_found = True
                        break

        if not purl_found and name:
            components.append({"ecosystem": "", "name": name, "version": version})

    return components


def detect_sbom_format(data: dict) -> str:
    """Detect whether *data* is CycloneDX or SPDX.

    Returns ``"cyclonedx"``, ``"spdx"``, or ``"unknown"``.
    """
    if "bomFormat" in data:
        return "cyclonedx"
    if data.get("bomFormat", "").lower() == "cyclonedx":
        return "cyclonedx"
    if "spdxVersion" in data:
        return "spdx"
    if "components" in data and isinstance(data.get("components"), list):
        return "cyclonedx"
    if "packages" in data and isinstance(data.get("packages"), list):
        return "spdx"
    return "unknown"


def parse_sbom(data: dict) -> tuple[str, list[dict[str, str]]]:
    """Parse an SBOM dict and return ``(format, components)``.

    *components* is a list of ``{"ecosystem": ..., "name": ..., "version": ...}``
    dicts.
    """
    fmt = detect_sbom_format(data)
    if fmt == "cyclonedx":
        return fmt, _parse_cyclonedx(data)
    if fmt == "spdx":
        return fmt, _parse_spdx(data)
    return "unknown", []


# ---------------------------------------------------------------------------
# OSV batch query
# ---------------------------------------------------------------------------
def _build_osv_queries(
    components: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Build OSV querybatch queries for a list of components.

    Each query uses the ``package`` field with ``name`` + ``version`` + optional
    ``ecosystem``.  Components without a version are queried without a version
    constraint (matches all known vulnerabilities for the package).
    """
    queries: list[dict[str, Any]] = []
    for comp in components:
        q: dict[str, Any] = {
            "package": {"name": comp["name"]},
        }
        if comp.get("ecosystem"):
            q["package"]["ecosystem"] = comp["ecosystem"]
        if comp.get("version"):
            q["version"] = comp["version"]
        queries.append(q)
    return queries


def _query_osv_batch(
    queries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Send queries to OSV.dev querybatch in chunks and return per-query vulns.

    Returns a list aligned with *queries* — each element is a list of
    vulnerability summary dicts (may be empty).
    """
    all_results: list[list[dict[str, Any]]] = []

    for start in range(0, len(queries), _OSV_BATCH_SIZE):
        chunk = queries[start : start + _OSV_BATCH_SIZE]
        try:
            resp = _post_with_retry(
                _OSV_QUERYBATCH_URL,
                json_body={"queries": chunk},
                timeout=_OSV_TIMEOUT,
            )
            if resp.status_code != 200:
                # Append empty results for this chunk
                all_results.extend([[] for _ in chunk])
                continue

            body = resp.json()
            results = body.get("results", [])
            for r in results:
                vulns = r.get("vulns", [])
                all_results.append(vulns)

            # Pad if API returned fewer results than queries
            while len(all_results) < start + len(chunk):
                all_results.append([])

        except Exception:  # noqa: BLE001
            all_results.extend([[] for _ in chunk])

    return all_results


# ---------------------------------------------------------------------------
# EPSS enrichment
# ---------------------------------------------------------------------------
def _fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Fetch current EPSS scores for a list of CVE IDs.

    Returns a dict mapping CVE-ID → EPSS score (0.0–1.0).
    The FIRST.org API supports bulk queries via comma-separated CVE list.
    """
    if not cve_ids:
        return {}

    scores: dict[str, float] = {}

    # API supports up to ~100 CVEs per request via comma-separated list
    chunk_size = 100
    for start in range(0, len(cve_ids), chunk_size):
        chunk = cve_ids[start : start + chunk_size]
        try:
            resp = _get_with_retry(
                _EPSS_API_URL,
                params={"cve": ",".join(chunk)},
                timeout=_EPSS_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", [])
            for entry in data:
                cve = entry.get("cve", "")
                epss = entry.get("epss")
                if cve and epss is not None:
                    try:
                        scores[cve.upper()] = float(epss)
                    except (ValueError, TypeError):
                        pass
        except Exception:  # noqa: BLE001
            continue

    return scores


# ---------------------------------------------------------------------------
# CISA KEV enrichment
# ---------------------------------------------------------------------------
def _fetch_kev_set() -> set[str]:
    """Fetch the CISA KEV catalog and return a set of CVE IDs."""
    try:
        resp = _get_with_retry(_CISA_KEV_URL, timeout=_KEV_TIMEOUT)
        if resp.status_code != 200:
            return set()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        return {v["cveID"].upper() for v in vulns if isinstance(v, dict) and "cveID" in v}
    except Exception:  # noqa: BLE001
        return set()


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------
def _extract_cve_ids(vulns: list[dict[str, Any]]) -> list[str]:
    """Extract CVE IDs from OSV vulnerability records."""
    cves: list[str] = []
    for v in vulns:
        osv_id = v.get("id", "")
        if _CVE_RE.match(osv_id):
            cves.append(osv_id.upper())
        for alias in v.get("aliases", []):
            if isinstance(alias, str) and _CVE_RE.match(alias):
                cves.append(alias.upper())
    return list(dict.fromkeys(cves))  # dedupe preserving order


def _severity_label(cvss_score: float | None) -> str:
    """Map a CVSS 3.x base score to a severity label."""
    if cvss_score is None:
        return "UNKNOWN"
    if cvss_score >= 9.0:
        return "CRITICAL"
    if cvss_score >= 7.0:
        return "HIGH"
    if cvss_score >= 4.0:
        return "MEDIUM"
    if cvss_score > 0.0:
        return "LOW"
    return "NONE"


def _extract_cvss(vuln: dict[str, Any]) -> float | None:
    """Extract the highest CVSS base score from an OSV vulnerability record."""
    severity_list = vuln.get("severity", [])
    if not isinstance(severity_list, list):
        return None

    best: float | None = None
    for sev in severity_list:
        if not isinstance(sev, dict):
            continue
        score_str = sev.get("score")
        if score_str is not None:
            try:
                score = float(score_str)
                if best is None or score > best:
                    best = score
            except (ValueError, TypeError):
                pass
    return best


def scan_sbom_file(sbom_path: str) -> dict[str, Any]:
    """Scan an SBOM file and return structured vulnerability findings.

    Parameters
    ----------
    sbom_path:
        Path to a CycloneDX or SPDX JSON SBOM file.

    Returns
    -------
    dict with keys:
        sbom_format, component_count, vulnerable_component_count,
        total_vulnerability_count, critical_count, high_count,
        medium_count, low_count, unknown_count, kev_count,
        findings (list), message.
    """
    path = Path(sbom_path).expanduser().resolve()

    if not path.exists():
        return {
            "error": f"File not found: {sbom_path}",
            "findings": [],
            "message": f"SBOM file not found: {sbom_path}",
        }

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "error": f"Failed to parse SBOM: {exc}",
            "findings": [],
            "message": f"Could not parse {sbom_path} as JSON: {exc}",
        }

    sbom_format, components = parse_sbom(data)
    if sbom_format == "unknown":
        return {
            "error": "Unrecognised SBOM format (expected CycloneDX or SPDX JSON)",
            "findings": [],
            "message": ("Could not detect SBOM format. Ensure the file is CycloneDX 1.x JSON or SPDX 2.x JSON."),
        }

    if not components:
        return {
            "sbom_format": sbom_format,
            "component_count": 0,
            "vulnerable_component_count": 0,
            "total_vulnerability_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unknown_count": 0,
            "kev_count": 0,
            "findings": [],
            "message": "SBOM parsed successfully but contains no scannable components.",
        }

    # 1. Query OSV in batch
    queries = _build_osv_queries(components)
    osv_results = _query_osv_batch(queries)

    # 2. Collect all unique CVE IDs for enrichment
    all_cves: set[str] = set()
    for vulns in osv_results:
        for v in vulns:
            for cve in _extract_cve_ids([v]):
                all_cves.add(cve)

    # 3. Enrich in parallel: EPSS + KEV
    epss_scores: dict[str, float] = {}
    kev_set: set[str] = set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        epss_future = pool.submit(_fetch_epss_scores, sorted(all_cves))
        kev_future = pool.submit(_fetch_kev_set)

        try:
            epss_scores = epss_future.result(timeout=30)
        except Exception:  # noqa: BLE001
            epss_scores = {}

        try:
            kev_set = kev_future.result(timeout=20)
        except Exception:  # noqa: BLE001
            kev_set = set()

    # 4. Build findings
    findings: list[dict[str, Any]] = []
    severity_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "UNKNOWN": 0,
    }
    kev_count = 0

    for _idx, (comp, vulns) in enumerate(zip(components, osv_results, strict=False)):
        if not vulns:
            continue

        for vuln in vulns:
            osv_id = vuln.get("id", "")
            cve_ids = _extract_cve_ids([vuln])
            cvss = _extract_cvss(vuln)
            severity = _severity_label(cvss)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # EPSS: use highest score across all CVE aliases
            max_epss: float | None = None
            for cve in cve_ids:
                score = epss_scores.get(cve.upper())
                if score is not None:
                    if max_epss is None or score > max_epss:
                        max_epss = score

            # KEV membership
            in_kev = any(cve.upper() in kev_set for cve in cve_ids)
            if in_kev:
                kev_count += 1

            summary = vuln.get("summary", "")

            finding: dict[str, Any] = {
                "component": {
                    "name": comp["name"],
                    "version": comp.get("version", ""),
                    "ecosystem": comp.get("ecosystem", ""),
                },
                "vulnerability": {
                    "id": osv_id,
                    "cve_ids": cve_ids,
                    "summary": summary,
                    "cvss_score": cvss,
                    "severity": severity,
                    "epss_score": max_epss,
                    "in_kev": in_kev,
                },
            }
            findings.append(finding)

    # 5. Sort: KEV first, then by EPSS descending, then by CVSS descending
    def _sort_key(f: dict) -> tuple:
        v = f["vulnerability"]
        return (
            0 if v.get("in_kev") else 1,
            -(v.get("epss_score") or 0.0),
            -(v.get("cvss_score") or 0.0),
        )

    findings.sort(key=_sort_key)

    vuln_components = len({idx for idx, vulns in enumerate(osv_results) if vulns})
    total_vulns = len(findings)

    if total_vulns == 0:
        msg = f"Scanned {len(components)} components from {sbom_format} SBOM — no known vulnerabilities found."
    else:
        msg = (
            f"Scanned {len(components)} components from {sbom_format} SBOM — "
            f"found {total_vulns} vulnerabilities across "
            f"{vuln_components} component(s)."
        )
        if kev_count:
            msg += f" {kev_count} exploited-in-wild (CISA KEV)."

    return {
        "sbom_format": sbom_format,
        "component_count": len(components),
        "vulnerable_component_count": vuln_components,
        "total_vulnerability_count": total_vulns,
        "critical_count": severity_counts.get("CRITICAL", 0),
        "high_count": severity_counts.get("HIGH", 0),
        "medium_count": severity_counts.get("MEDIUM", 0),
        "low_count": severity_counts.get("LOW", 0),
        "unknown_count": severity_counts.get("UNKNOWN", 0),
        "kev_count": kev_count,
        "findings": findings,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------
def scan_sbom(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK entry point for the scan_sbom tool."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}
    sbom_path = tool_input.get("sbom_path")

    if not isinstance(sbom_path, str) or not sbom_path.strip():
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid sbom_path. Must be a non-empty string."}],
        }
        log_tool_output_size("scan_sbom", result)
        return result

    payload = scan_sbom_file(sbom_path.strip())
    status = "success" if "error" not in payload else "error"

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("scan_sbom", result)
    return result
