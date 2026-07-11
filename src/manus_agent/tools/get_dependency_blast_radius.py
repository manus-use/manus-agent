"""
Tool: get_dependency_blast_radius

Given a vulnerable package specification (``package@version`` or a CVE ID),
estimates the downstream exposure — how many packages and repositories depend
on the affected package, and how widely it is downloaded.

Data sources (all free, no API key required):
  1. NVD CVE 2.0 API  — maps CVE → affected packages + version ranges
  2. OSV.dev API       — cross-ecosystem vulnerability data (PyPI, npm, Maven,
                         Go, RubyGems, crates.io, SwiftURL, …)
  3. GitHub Advisory DB — GHSA packages + ecosystems
  4. npm registry API  — dependents count + weekly download stats (npm only)
  5. PyPI JSON API     — package metadata (PyPI only; download counts from
                         pypistats.org with graceful degradation on 429)
  6. Maven Central     — artifact metadata + version count (Maven only)
  7. GitHub REST API   — repo metadata + release history (Swift packages only)

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
_GITHUB_REPO_URL = "https://api.github.com/repos/{}"
_GITHUB_RELEASES_URL = "https://api.github.com/repos/{}/releases"
_GITHUB_TAGS_URL = "https://api.github.com/repos/{}/tags"

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
    "SwiftURL": "Swift Package Index (Swift/Apple)",
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


def _parse_swift_timestamp(ts: str) -> str:
    """Normalise a GitHub ISO-8601 timestamp to ``YYYY-MM-DD``.

    GitHub returns timestamps as ``2014-07-31T05:56:19Z``.
    """
    if not ts:
        return ""
    # Strip trailing Z or +HH:MM offset before slicing
    ts = ts.rstrip("Z")
    if "+" in ts:
        ts = ts.split("+")[0]
    return ts[:10]  # YYYY-MM-DD


def _enrich_swift(name: str) -> dict[str, Any]:
    """Fetch Swift package metadata from the GitHub API.

    *name* must be an ``owner/repo`` slug (e.g. ``Alamofire/Alamofire``).
    Swift packages are always hosted on GitHub (or another VCS), and their
    identity in the OSV ``SwiftURL`` ecosystem is ``github.com/{owner}/{repo}``.

    Metrics returned:
      - ``github_stars``        — repository star count (main engagement proxy)
      - ``forks``               — fork count
      - ``latest_version``      — latest release tag (or latest tag if no releases)
      - ``total_versions``      — total published release/tag count
      - ``first_release_date``  — date of the oldest release/tag (ISO YYYY-MM-DD)
      - ``age_years``           — approximate package age
      - ``description``         — repo description (120-char truncated)
      - ``language``            — primary language reported by GitHub
      - ``github_page``         — HTML URL

    ``weekly_downloads`` is set to ``github_stars * 100`` so that
    :func:`_blast_score` produces meaningful CRITICAL/HIGH/MEDIUM/LOW labels
    without requiring a download-stats API.  This proxy is conservative:
    a package with 40 k stars is depended upon by vastly more than 4 M
    weekly users, but the scaling keeps the label roughly calibrated with
    the other ecosystems.
    """
    result: dict[str, Any] = {"ecosystem": "SwiftURL", "package_name": name}

    # ---------- Step 1: repository metadata ----------
    try:
        repo_data = _get(_GITHUB_REPO_URL.format(name))
        result["github_stars"] = repo_data.get("stargazers_count", 0)
        result["forks"] = repo_data.get("forks_count", 0)
        result["language"] = repo_data.get("language") or ""
        raw_desc = repo_data.get("description") or ""
        result["description"] = raw_desc.replace("\n", " ")[:120]
        result["github_page"] = repo_data.get("html_url", "")
        created_at = repo_data.get("created_at", "")
        # Use repo creation as a conservative lower bound for age
        result["_repo_created_at"] = _parse_swift_timestamp(created_at)
    except Exception as exc:
        logger.debug("Swift GitHub repo fetch failed for %s: %s", name, exc)
        return result

    # Blast-score proxy: stars × 100 → weekly_downloads equivalent
    stars = result.get("github_stars", 0) or 0
    result["weekly_downloads"] = stars * 100

    # ---------- Step 2: release history ----------
    try:
        releases = _get(_GITHUB_RELEASES_URL.format(name), params={"per_page": 100})
        if isinstance(releases, list) and releases:
            result["total_versions"] = len(releases)
            # GitHub returns newest-first
            result["latest_version"] = releases[0].get("tag_name", "")
            oldest_ts = ""
            for rel in reversed(releases):
                ts = rel.get("published_at", "")
                if ts:
                    oldest_ts = ts
                    break
            if oldest_ts:
                result["first_release_date"] = _parse_swift_timestamp(oldest_ts)
    except Exception as exc:
        logger.debug("Swift releases fetch failed for %s: %s", name, exc)

    # ---------- Step 3: tag fallback (no formal releases) ----------
    if "total_versions" not in result:
        try:
            tags = _get(_GITHUB_TAGS_URL.format(name), params={"per_page": 100})
            if isinstance(tags, list) and tags:
                result["total_versions"] = len(tags)
                result["latest_version"] = tags[0].get("name", "")
        except Exception as exc:
            logger.debug("Swift tags fetch failed for %s: %s", name, exc)

    # ---------- Step 4: age_years from first_release_date or repo creation ----------
    base_date = result.get("first_release_date") or result.get("_repo_created_at", "")
    if base_date:
        try:
            import datetime  # noqa: PLC0415

            year = int(base_date[:4])
            month = int(base_date[5:7])
            day = int(base_date[8:10])
            origin = datetime.date(year, month, day)
            today = datetime.date.today()
            result["age_years"] = round((today - origin).days / 365.25, 1)
        except Exception:
            pass

    # Remove internal key
    result.pop("_repo_created_at", None)
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
    elif eco_lower in ("swift", "swifturl", "swiftpackage", "ios", "macos", "spm"):
        return _enrich_swift(name)
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

    Supported ecosystems and their enrichment data sources:
      - **npm**:      dependent package count + weekly/monthly download stats
      - **PyPI**:     package metadata + download stats (when pypistats is available)
      - **Maven**:    artifact metadata from Maven Central
      - **Swift**:    GitHub repo stars, release history, age (via GitHub REST API)

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

        # Swift-specific stats
        if eco.lower() in ("swifturl", "swift", "swiftpackage", "ios", "macos", "spm"):
            if r.get("github_stars") is not None:
                lines.append(f"    GitHub stars:     {r['github_stars']:,}")
            if r.get("forks") is not None:
                lines.append(f"    Forks:            {r['forks']:,}")
            if r.get("latest_version"):
                lines.append(f"    Latest version:   {r['latest_version']}")
            if r.get("total_versions"):
                lines.append(f"    Total versions:   {r['total_versions']}")
            if r.get("first_release_date"):
                age = r.get("age_years", "")
                age_str = f" ({age} yrs)" if age else ""
                lines.append(f"    First released:   {r['first_release_date']}{age_str}")
            if r.get("description"):
                lines.append(f"    Description:      {r['description'][:80]}")
            if r.get("language"):
                lines.append(f"    Language:         {r['language']}")
            if r.get("github_page"):
                lines.append(f"    GitHub page:      {r['github_page']}")

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
