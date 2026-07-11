"""
Tool: get_dependency_blast_radius

Given a vulnerable package specification (``package@version`` or a CVE ID),
estimates the downstream exposure — how many packages and repositories depend
on the affected package, and how widely it is downloaded.

Data sources (all free, no API key required):
  1. NVD CVE 2.0 API  — maps CVE → affected packages + version ranges
  2. OSV.dev API       — cross-ecosystem vulnerability data (PyPI, npm, Maven,
                         Go, RubyGems, crates.io, …)
  3. GitHub Advisory DB — GHSA packages + ecosystems
  4. npm registry API  — dependents count + weekly download stats (npm only)
  5. PyPI JSON API     — package metadata (PyPI only; download counts from
                         pypistats.org with graceful degradation on 429)
  6. Maven Central     — artifact metadata + version count (Maven only)
  7. Bioconductor      — R/Bioconductor package metadata; dependsOnMe +
                         importsMe counts for blast-radius estimation

The tool deliberately avoids paid/rate-limited APIs so it works without
configuration.  When a source is unavailable the result degrades gracefully.

CLI: ``manus-agent blast-radius requests@2.28.0``
     ``manus-agent blast-radius CVE-2021-44228``
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from strands import tool

__all__ = ["get_dependency_blast_radius"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_PKG_RE = re.compile(r"^(?P<name>[^@]+?)(?:@(?P<version>.+))?$")

_NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_OSV_QUERY_URL = "https://api.osv.dev/v1/query"
_OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{}"
_GHSA_URL = "https://api.github.com/advisories"
_NPM_SEARCH_URL = "https://registry.npmjs.org/-/v1/search"
_NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/point/last-week/{}"
_PYPI_JSON_URL = "https://pypi.org/pypi/{}/json"
_PYPISTATS_URL = "https://pypistats.org/api/packages/{}/recent"
_MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
_BIOC_PACKAGES_URL = "https://bioconductor.org/packages/json/3.19/bioc/packages.json"

_TIMEOUT = 20

# OSV ecosystem → display name mapping
_ECOSYSTEM_LABEL: dict[str, str] = {
    "PyPI": "PyPI (Python)",
    "npm": "npm (JavaScript/Node.js)",
    "Maven": "Maven (Java)",
    "Go": "Go modules",
    "crates.io": "crates.io (Rust)",
    "RubyGems": "RubyGems (Ruby)",
    "NuGet": "NuGet (.NET)",
    "Packagist": "Packagist (PHP)",
    "Hex": "Hex (Elixir/Erlang)",
    "Pub": "Pub (Dart/Flutter)",
    "Bioconductor": "Bioconductor (R/Bioinformatics)",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> Any:
    """GET with timeout; returns parsed JSON or raises."""
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: dict) -> Any:
    """POST JSON; returns parsed JSON or raises."""
    r = requests.post(url, json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _parse_input(spec: str) -> dict[str, str | None]:
    """Return {'kind': 'cve'|'package', 'cve_id': ..., 'name': ..., 'version': ..., 'ecosystem': ...}."""
    spec = spec.strip()
    if _CVE_RE.match(spec):
        return {"kind": "cve", "cve_id": spec.upper(), "name": None, "version": None, "ecosystem": None}

    # Accept ecosystem-qualified specs: pypi:requests@2.28.0, npm:axios@1.6.0, maven:log4j-core@2.14.1
    ecosystem = None
    if ":" in spec:
        parts = spec.split(":", 1)
        if not parts[0].startswith("http"):
            ecosystem = parts[0]
            spec = parts[1]

    m = _PKG_RE.match(spec)
    if not m:
        raise ValueError(f"Cannot parse package spec: {spec!r}")
    return {
        "kind": "package",
        "cve_id": None,
        "name": m.group("name").strip(),
        "version": (m.group("version") or "").strip() or None,
        "ecosystem": ecosystem,
    }


# ---------------------------------------------------------------------------
# Source 1: NVD — CVE → affected packages
# ---------------------------------------------------------------------------


def _fetch_nvd_affected(cve_id: str) -> list[dict[str, str]]:
    """Return list of {ecosystem, name, version_range} from NVD CPE configurations."""
    try:
        data = _get(_NVD_URL, params={"cveId": cve_id})
    except Exception as exc:
        logger.debug("NVD fetch failed for %s: %s", cve_id, exc)
        return []

    result: list[dict[str, str]] = []
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return result

    cve = vulns[0].get("cve", {})
    configs = cve.get("configurations", [])

    for config in configs:
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if not cpe_match.get("vulnerable", False):
                    continue
                cpe_uri = cpe_match.get("criteria", "")
                parts = cpe_uri.split(":")
                # CPE 2.3: cpe:2.3:a:vendor:product:version:...
                if len(parts) >= 6:
                    product = parts[4] if len(parts) > 4 else "unknown"
                    version = parts[5] if len(parts) > 5 else "*"
                    ver_start_inc = cpe_match.get("versionStartIncluding", "")
                    ver_end_exc = cpe_match.get("versionEndExcluding", "")
                    ver_end_inc = cpe_match.get("versionEndIncluding", "")

                    if ver_start_inc or ver_end_exc or ver_end_inc:
                        ver_range = ""
                        if ver_start_inc:
                            ver_range += f">={ver_start_inc}"
                        if ver_end_exc:
                            ver_range += f"<{ver_end_exc}" if not ver_range else f", <{ver_end_exc}"
                        elif ver_end_inc:
                            ver_range += f"<={ver_end_inc}" if not ver_range else f", <={ver_end_inc}"
                    else:
                        ver_range = version if version != "*" else "all versions"

                    result.append(
                        {
                            "name": product,
                            "version_range": ver_range,
                            "ecosystem": "unknown",
                            "source": "nvd",
                        }
                    )

    # Deduplicate by name
    seen: set[str] = set()
    deduped = []
    for r in result:
        if r["name"] not in seen:
            seen.add(r["name"])
            deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# Source 2: OSV.dev — CVE → affected packages with ecosystem tags
# ---------------------------------------------------------------------------


def _fetch_osv_affected(cve_id: str) -> list[dict[str, str]]:
    """Query OSV for the CVE and return affected package records."""
    try:
        data = _post(_OSV_QUERY_URL, {"id": cve_id})
    except Exception as exc:
        logger.debug("OSV query failed: %s", exc)
        return []

    result: list[dict[str, str]] = []
    for vuln in data.get("vulns", []):
        vuln_id = vuln.get("id", "")
        # Fetch the full record to get affected packages
        try:
            full = _get(_OSV_VULN_URL.format(vuln_id))
        except Exception:
            continue

        for affected in full.get("affected", []):
            pkg = affected.get("package", {})
            name = pkg.get("name", "")
            ecosystem = pkg.get("ecosystem", "")
            if not name:
                continue

            # Extract version range summary
            ranges = affected.get("ranges", [])
            ver_range = _summarise_osv_ranges(ranges, affected.get("versions", []))

            result.append(
                {
                    "name": name,
                    "ecosystem": ecosystem,
                    "version_range": ver_range,
                    "source": "osv",
                }
            )

    return result


def _summarise_osv_ranges(ranges: list[dict], versions: list[str]) -> str:
    """Produce a short human-readable version range from OSV range data."""
    summaries: list[str] = []
    for rng in ranges:
        if rng.get("type") == "SEMVER" or rng.get("type") == "ECOSYSTEM":
            events = rng.get("events", [])
            introduced = None
            fixed = None
            for ev in events:
                if "introduced" in ev and ev["introduced"] != "0":
                    introduced = ev["introduced"]
                if "fixed" in ev:
                    fixed = ev["fixed"]
            if introduced and fixed:
                summaries.append(f">={introduced}, <{fixed}")
            elif introduced:
                summaries.append(f">={introduced}")
            elif fixed:
                summaries.append(f"<{fixed}")

    if summaries:
        return "; ".join(summaries)
    if versions:
        sample = versions[:5]
        suffix = f" (+{len(versions) - 5} more)" if len(versions) > 5 else ""
        return ", ".join(sample) + suffix
    return "unspecified"


# ---------------------------------------------------------------------------
# Source 3: GitHub Advisory API
# ---------------------------------------------------------------------------


def _fetch_ghsa_affected(cve_id: str) -> list[dict[str, str]]:
    """Return affected packages from GitHub Security Advisory DB."""
    try:
        data = _get(_GHSA_URL, params={"cve_id": cve_id, "per_page": 10})
    except Exception as exc:
        logger.debug("GHSA fetch failed: %s", exc)
        return []

    result: list[dict[str, str]] = []
    for advisory in data if isinstance(data, list) else []:
        for vuln in advisory.get("vulnerabilities", []):
            pkg = vuln.get("package", {})
            name = pkg.get("name", "")
            ecosystem = pkg.get("ecosystem", "")
            if not name:
                continue
            vulnerable_range = vuln.get("vulnerable_version_range", "")
            patched = vuln.get("first_patched_version", {})
            patched_ver = patched.get("identifier", "") if isinstance(patched, dict) else ""
            ver_range = vulnerable_range or (f"<{patched_ver}" if patched_ver else "unspecified")
            result.append(
                {
                    "name": name,
                    "ecosystem": ecosystem,
                    "version_range": ver_range,
                    "source": "ghsa",
                }
            )

    return result


# ---------------------------------------------------------------------------
# Package-level enrichment: download/dependent stats
# ---------------------------------------------------------------------------


def _enrich_npm(name: str) -> dict[str, Any]:
    """Fetch npm dependent count + weekly downloads."""
    result: dict[str, Any] = {"ecosystem": "npm", "package_name": name}
    try:
        # Search returns dependents count
        search_data = _get(_NPM_SEARCH_URL, params={"text": name, "size": 5})
        for obj in search_data.get("objects", []):
            if obj.get("package", {}).get("name", "").lower() == name.lower():
                result["dependent_packages_count"] = int(obj.get("dependents", 0))
                dls = obj.get("downloads", {})
                result["weekly_downloads"] = dls.get("weekly", 0)
                result["monthly_downloads"] = dls.get("monthly", 0)
                break
        # Fallback: direct downloads API
        if "weekly_downloads" not in result:
            dl_data = _get(_NPM_DOWNLOADS_URL.format(name))
            result["weekly_downloads"] = dl_data.get("downloads", 0)
    except Exception as exc:
        logger.debug("npm enrich failed for %s: %s", name, exc)
    return result


def _enrich_pypi(name: str) -> dict[str, Any]:
    """Fetch PyPI package metadata."""
    result: dict[str, Any] = {"ecosystem": "PyPI", "package_name": name}
    try:
        pypi_data = _get(_PYPI_JSON_URL.format(name))
        info = pypi_data.get("info", {})
        result["description"] = info.get("summary", "")[:120]
        result["latest_version"] = info.get("version", "")
        result["home_page"] = info.get("project_url", info.get("home_page", ""))
        # Total release count as a proxy for maturity
        result["release_count"] = len(pypi_data.get("releases", {}))
        # Try pypistats for download counts (rate-limited; degrade gracefully)
        try:
            stats = _get(_PYPISTATS_URL.format(name))
            data = stats.get("data", {})
            result["weekly_downloads"] = data.get("last_week", 0)
            result["monthly_downloads"] = data.get("last_month", 0)
        except Exception:
            result["weekly_downloads"] = None
            result["monthly_downloads"] = None
    except Exception as exc:
        logger.debug("PyPI enrich failed for %s: %s", name, exc)
    return result


def _enrich_maven(name: str) -> dict[str, Any]:
    """Fetch Maven Central artifact metadata."""
    result: dict[str, Any] = {"ecosystem": "Maven", "package_name": name}
    try:
        # Accept both 'groupId:artifactId' and plain 'artifactId' forms
        if ":" in name:
            g, a = name.split(":", 1)
            query = f"g:{g} AND a:{a}"
        else:
            query = f"a:{name}"
        data = _get(_MAVEN_SEARCH_URL, params={"q": query, "rows": 5, "wt": "json"})
        docs = data.get("response", {}).get("docs", [])
        if docs:
            doc = docs[0]
            result["latest_version"] = doc.get("latestVersion", "")
            result["version_count"] = doc.get("versionCount", 0)
            result["group_id"] = doc.get("g", "")
            result["artifact_id"] = doc.get("a", "")
            result["full_id"] = doc.get("id", name)
        result["total_artifacts_found"] = data.get("response", {}).get("numFound", 0)
    except Exception as exc:
        logger.debug("Maven enrich failed for %s: %s", name, exc)
    return result


def _enrich_bioconductor(name: str) -> dict[str, Any]:
    """
    Fetch Bioconductor package metadata from the release JSON manifest.

    Uses ``https://bioconductor.org/packages/json/3.19/bioc/packages.json``
    which is a single-call, unauthenticated endpoint that returns all ~2300
    packages with version, reverse-dependency lists, download rank, and
    biocViews category tags.

    Blast-radius signal: ``dependent_packages_count`` = len(dependsOnMe) +
    len(importsMe) — i.e., the packages that would *break* if this one changed.
    ``suggestsMe`` is deliberately excluded: soft suggestions rarely cause
    build failures and would inflate the score.
    """
    result: dict[str, Any] = {"ecosystem": "Bioconductor", "package_name": name}
    try:
        data = _get(_BIOC_PACKAGES_URL)
    except Exception as exc:
        logger.debug("Bioconductor manifest fetch failed: %s", exc)
        return result

    pkg = data.get(name) or data.get(name.lower())
    if pkg is None:
        # Case-insensitive fallback
        name_lower = name.lower()
        for k, v in data.items():
            if k.lower() == name_lower:
                pkg = v
                break
    if pkg is None:
        logger.debug("Package %r not found in Bioconductor manifest", name)
        return result

    depends_on_me: list[str] = pkg.get("dependsOnMe") or []
    imports_me: list[str] = pkg.get("importsMe") or []
    # Hard reverse-dep count (Depends + Imports); excludes soft Suggests
    hard_rev_deps = len(depends_on_me) + len(imports_me)

    result["latest_version"] = pkg.get("Version", "")
    result["title"] = pkg.get("Title", "")
    result["description"] = (pkg.get("Description") or "")[:120]
    result["bioc_views"] = ", ".join((pkg.get("biocViews") or [])[:5])
    result["download_rank"] = pkg.get("Rank")  # lower = more popular
    result["dependency_count"] = pkg.get("dependencyCount")  # how many deps it has
    result["dependent_packages_count"] = hard_rev_deps  # drives _blast_score
    result["depends_on_me"] = len(depends_on_me)
    result["imports_me"] = len(imports_me)
    result["suggests_me"] = len(pkg.get("suggestsMe") or [])
    result["git_url"] = pkg.get("git_url", "")
    result["bioc_page"] = f"https://bioconductor.org/packages/release/bioc/html/{name}.html"
    return result


def _enrich_package(name: str, ecosystem: str) -> dict[str, Any]:
    """Dispatch to the right enrichment function based on ecosystem."""
    eco_lower = (ecosystem or "").lower()
    if eco_lower in ("npm", "javascript", "node"):
        return _enrich_npm(name)
    elif eco_lower in ("pypi", "python", "pip"):
        return _enrich_pypi(name)
    elif eco_lower in ("maven", "java", "gradle"):
        return _enrich_maven(name)
    elif eco_lower in ("bioconductor", "bioc", "r-bioc", "rbioconductor"):
        return _enrich_bioconductor(name)
    # Unknown ecosystem — return minimal record
    return {"ecosystem": ecosystem, "package_name": name}


# ---------------------------------------------------------------------------
# Blast radius scoring
# ---------------------------------------------------------------------------


def _blast_score(stats: dict[str, Any]) -> str:
    """
    Produce a qualitative blast-radius label based on download/dependent counts.

    Returns one of: critical / high / medium / low / unknown
    """
    weekly = stats.get("weekly_downloads") or 0
    dependents = stats.get("dependent_packages_count") or 0

    if weekly >= 5_000_000 or dependents >= 50_000:
        return "CRITICAL"
    elif weekly >= 500_000 or dependents >= 5_000:
        return "HIGH"
    elif weekly >= 50_000 or dependents >= 500:
        return "MEDIUM"
    elif weekly > 0 or dependents > 0:
        return "LOW"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


@tool
def get_dependency_blast_radius(  # noqa: C901
    package_or_cve: str,
    max_packages: int = 10,
) -> str:
    """
    Estimate the downstream blast radius of a vulnerable package or CVE.

    Given a package specification (e.g. ``requests@2.28.0``) or a CVE ID
    (e.g. ``CVE-2021-44228``), this tool:

    1. Identifies the affected package(s) and version range(s) from NVD,
       OSV.dev, and the GitHub Advisory Database.
    2. For each affected package, fetches downstream exposure metrics:
       - **npm**: dependent package count + weekly/monthly download stats
       - **PyPI**: package metadata + download stats (when pypistats is available)
       - **Maven**: artifact metadata from Maven Central
    3. Computes a qualitative blast-radius label: CRITICAL / HIGH / MEDIUM / LOW.

    Use after ``get_nvd_data`` to understand *how many projects are exposed*
    to a vulnerability — the answer is often orders of magnitude larger than
    the single vulnerable package.

    Args:
        package_or_cve: Package spec (``name@version``, ``ecosystem:name@version``)
                        or CVE ID (``CVE-YYYY-NNNN``).
        max_packages:   Maximum number of packages to enrich with stats (default 10).

    Returns:
        A structured text report with per-package exposure stats and a
        summary blast-radius label.
    """
    try:
        parsed = _parse_input(package_or_cve)
    except ValueError as e:
        return f"Error: {e}"

    sections: list[str] = []
    all_packages: list[dict[str, Any]] = []

    if parsed["kind"] == "cve":
        cve_id = parsed["cve_id"]
        sections.append(f"Dependency Blast Radius — {cve_id}\n{'=' * 50}")

        # Gather affected packages from multiple sources
        nvd_pkgs = _fetch_nvd_affected(cve_id)
        osv_pkgs = _fetch_osv_affected(cve_id)
        ghsa_pkgs = _fetch_ghsa_affected(cve_id)

        # Merge: OSV and GHSA have ecosystem tags; NVD often does not
        # Prefer OSV/GHSA records since they include ecosystem
        seen_names: dict[str, dict] = {}
        for pkg in osv_pkgs + ghsa_pkgs + nvd_pkgs:
            key = (pkg["name"].lower(), (pkg.get("ecosystem") or "").lower())
            if key not in seen_names:
                seen_names[key] = pkg

        merged = list(seen_names.values())
        if not merged:
            return (
                f"No affected package records found for {cve_id} in NVD, OSV, or GHSA.\n"
                "The CVE may be too recent, use a product/version CPE rather than a named package,\n"
                "or may not yet be indexed."
            )

        sections.append(f"Affected packages found: {len(merged)}")
        all_packages = merged[:max_packages]

    else:
        # Direct package query
        name = parsed["name"]
        version = parsed["version"]
        ecosystem = parsed["ecosystem"] or ""
        label = f"{ecosystem}:{name}" if ecosystem else name
        ver_label = f"@{version}" if version else " (all versions)"
        sections.append(f"Dependency Blast Radius — {label}{ver_label}\n{'=' * 50}")
        all_packages = [
            {
                "name": name,
                "ecosystem": ecosystem,
                "version_range": version or "all",
                "source": "direct",
            }
        ]

    # Enrich each package with stats
    enriched_results: list[dict[str, Any]] = []
    for pkg in all_packages:
        eco = pkg.get("ecosystem", "")
        name = pkg["name"]
        stats = _enrich_package(name, eco)
        stats["version_range"] = pkg.get("version_range", "")
        stats["source"] = pkg.get("source", "")
        blast = _blast_score(stats)
        stats["blast_radius"] = blast
        enriched_results.append(stats)

    # Sort by blast severity
    _severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    enriched_results.sort(key=lambda r: _severity_order.get(r.get("blast_radius", "UNKNOWN"), 4))

    # Build output
    for i, r in enumerate(enriched_results):
        eco = r.get("ecosystem") or "Unknown"
        eco_label = _ECOSYSTEM_LABEL.get(eco, eco)
        pkg_name = r.get("package_name") or r.get("name", "unknown")
        blast = r.get("blast_radius", "UNKNOWN")
        ver_range = r.get("version_range", "")
        source = r.get("source", "")

        lines: list[str] = [
            f"\n[{i + 1}] {pkg_name}  ({eco_label})",
            f"    Blast radius:     {blast}",
        ]
        if ver_range:
            lines.append(f"    Vulnerable range: {ver_range}")

        # npm-specific stats
        if "dependent_packages_count" in r:
            lines.append(f"    npm dependents:   {r['dependent_packages_count']:,}")
        if r.get("weekly_downloads") is not None:
            lines.append(f"    Weekly downloads: {r['weekly_downloads']:,}")
        if r.get("monthly_downloads") is not None:
            lines.append(f"    Monthly downloads:{r['monthly_downloads']:,}")

        # PyPI-specific stats
        if eco.lower() in ("pypi", "python"):
            if r.get("latest_version"):
                lines.append(f"    Latest version:   {r['latest_version']}")
            if r.get("release_count"):
                lines.append(f"    Total releases:   {r['release_count']}")
            if r.get("description"):
                lines.append(f"    Description:      {r['description'][:80]}")

        # Maven-specific stats
        if eco.lower() in ("maven", "java"):
            if r.get("full_id"):
                lines.append(f"    Maven artifact:   {r['full_id']}")
            if r.get("latest_version"):
                lines.append(f"    Latest version:   {r['latest_version']}")
            if r.get("version_count"):
                lines.append(f"    Version count:    {r['version_count']}")

        # Bioconductor-specific stats
        if eco.lower() in ("bioconductor", "bioc"):
            if r.get("latest_version"):
                lines.append(f"    Latest version:   {r['latest_version']}")
            if r.get("title"):
                lines.append(f"    Title:            {r['title'][:80]}")
            if r.get("dependent_packages_count") is not None:
                lines.append(
                    f"    Hard rev-deps:    {r['dependent_packages_count']:,}"
                    f"  (dependsOnMe={r.get('depends_on_me', 0)},"
                    f" importsMe={r.get('imports_me', 0)})"
                )
            if r.get("suggests_me"):
                lines.append(f"    Soft rev-deps:    {r['suggests_me']:,}  (suggestsMe, not scored)")
            if r.get("download_rank"):
                lines.append(f"    Bioc DL rank:     #{r['download_rank']}  (lower → more downloaded)")
            if r.get("bioc_views"):
                lines.append(f"    biocViews:        {r['bioc_views']}")
            if r.get("description"):
                lines.append(f"    Description:      {r['description'][:80]}")
            if r.get("git_url"):
                lines.append(f"    Git source:       {r['git_url']}")
            if r.get("bioc_page"):
                lines.append(f"    Bioconductor:     {r['bioc_page']}")

        if source:
            lines.append(f"    Data sources:     {source}")

        sections.extend(lines)

    # Summary line
    if enriched_results:
        top = enriched_results[0]
        max_blast = top.get("blast_radius", "UNKNOWN")
        max_pkg = top.get("package_name", "")
        sections.append(f"\nSummary: highest blast radius is {max_blast} ({max_pkg})")
        total_weekly = sum(
            r.get("weekly_downloads") or 0 for r in enriched_results if r.get("weekly_downloads") is not None
        )
        total_dependents = sum(r.get("dependent_packages_count") or 0 for r in enriched_results)
        if total_weekly:
            sections.append(f"         Total weekly downloads across all packages: {total_weekly:,}")
        if total_dependents:
            sections.append(f"         Total npm dependent packages: {total_dependents:,}")

    return "\n".join(sections)
