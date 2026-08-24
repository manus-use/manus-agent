"""Tool: dep_audit

Audit a project's dependencies for known vulnerabilities by parsing local
manifest/lockfiles and querying OSV.dev in batch.

Supported manifest formats:
  - requirements.txt (Python/pip)
  - package.json + package-lock.json (npm)
  - go.mod (Go)
  - Cargo.toml + Cargo.lock (Rust)
  - Gemfile.lock (Ruby)
  - pom.xml (Maven/Java)
  - composer.json (PHP)

Unlike sbom-scan (which requires a pre-generated CycloneDX/SPDX file), this
tool works directly from raw project manifests — no SBOM generation step needed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from strands import tool

__all__ = ["dep_audit", "audit_dependencies"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
_EPSS_BATCH_URL = "https://api.first.org/data/v1/epss"
_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known-exploited-vulnerabilities.json"

_USER_AGENT = "manus-agent/dep-audit (github.com/manus-use/manus-agent)"
_MAX_BATCH_SIZE = 1000  # OSV batch limit
_REQUEST_TIMEOUT = 20

# Ecosystem mappings for OSV.dev
_ECOSYSTEM_MAP = {
    "requirements.txt": "PyPI",
    "setup.py": "PyPI",
    "pyproject.toml": "PyPI",
    "package.json": "npm",
    "package-lock.json": "npm",
    "go.mod": "Go",
    "Cargo.toml": "crates.io",
    "Cargo.lock": "crates.io",
    "Gemfile.lock": "RubyGems",
    "pom.xml": "Maven",
    "composer.json": "Packagist",
    "composer.lock": "Packagist",
}

# ---------------------------------------------------------------------------
# Parsers — extract (name, version) tuples from manifest files
# ---------------------------------------------------------------------------

# PEP 508 requirement line: name[extras] == version
_PIP_RE = re.compile(
    r"^([A-Za-z0-9][-A-Za-z0-9_.]*)"
    r"(?:\[[^\]]*\])?"
    r"\s*==\s*"
    r"([^\s;#,]+)",
)

_GO_MOD_RE = re.compile(
    r"^\s+([^\s]+)\s+v([^\s/]+)",
)

_CARGO_DEP_RE = re.compile(
    r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]+)"',
)

_CARGO_LOCK_RE = re.compile(
    r'name\s*=\s*"([^"]+)"\s*\n\s*version\s*=\s*"([^"]+)"',
    re.MULTILINE,
)

_GEMFILE_LOCK_RE = re.compile(
    r"^\s{4}([A-Za-z0-9_.-]+)\s+\(([^\s)]+)\)",
    re.MULTILINE,
)

_MAVEN_DEP_RE = re.compile(
    r"<groupId>([^<]+)</groupId>\s*"
    r"<artifactId>([^<]+)</artifactId>\s*"
    r"<version>([^<$]+)</version>",
    re.DOTALL,
)


def _parse_requirements_txt(content: str) -> list[dict[str, str]]:
    """Parse requirements.txt for pinned packages."""
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _PIP_RE.match(line)
        if m:
            deps.append({"name": m.group(1).lower(), "version": m.group(2)})
    return deps


def _parse_package_json(content: str) -> list[dict[str, str]]:
    """Parse package.json for dependencies with semver-exact versions."""
    deps = []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return deps

    for section in ("dependencies", "devDependencies"):
        for name, version in data.get(section, {}).items():
            clean = re.sub(r"^[~^>=<!\s]+", "", version)
            if clean and re.match(r"\d", clean):
                deps.append({"name": name, "version": clean})
    return deps


def _parse_package_lock_json(content: str) -> list[dict[str, str]]:
    """Parse package-lock.json (v2/v3) for resolved packages."""
    deps = []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return deps

    # v2/v3 format
    packages = data.get("packages", {})
    for path, info in packages.items():
        if not path:
            continue
        name = info.get("name") or path.rsplit("node_modules/", 1)[-1]
        version = info.get("version", "")
        if name and version:
            deps.append({"name": name, "version": version})

    # v1 fallback
    if not deps:
        for name, info in data.get("dependencies", {}).items():
            version = info.get("version", "")
            if version:
                deps.append({"name": name, "version": version})

    return deps


def _parse_go_mod(content: str) -> list[dict[str, str]]:
    """Parse go.mod require block."""
    deps = []
    in_require = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require = True
            continue
        if in_require and stripped == ")":
            in_require = False
            continue
        if in_require:
            m = _GO_MOD_RE.match(line)
            if m:
                deps.append({"name": m.group(1), "version": m.group(2)})
        elif stripped.startswith("require "):
            m = re.match(r"^require\s+([^\s]+)\s+v([^\s]+)", stripped)
            if m:
                deps.append({"name": m.group(1), "version": m.group(2)})
    return deps


def _parse_cargo_lock(content: str) -> list[dict[str, str]]:
    """Parse Cargo.lock for package entries."""
    deps = []
    for m in _CARGO_LOCK_RE.finditer(content):
        deps.append({"name": m.group(1), "version": m.group(2)})
    return deps


def _parse_cargo_toml(content: str) -> list[dict[str, str]]:
    """Parse Cargo.toml [dependencies] section (simple format only)."""
    deps = []
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if re.match(r"^\[dependencies\]", stripped, re.IGNORECASE):
            in_deps = True
            continue
        if stripped.startswith("[") and in_deps:
            in_deps = False
            continue
        if in_deps:
            m = _CARGO_DEP_RE.match(stripped)
            if m:
                deps.append({"name": m.group(1), "version": m.group(2)})
    return deps


def _parse_gemfile_lock(content: str) -> list[dict[str, str]]:
    """Parse Gemfile.lock GEM specs section."""
    deps = []
    for m in _GEMFILE_LOCK_RE.finditer(content):
        deps.append({"name": m.group(1), "version": m.group(2)})
    return deps


def _parse_pom_xml(content: str) -> list[dict[str, str]]:
    """Parse pom.xml <dependency> blocks with literal versions."""
    deps = []
    for m in _MAVEN_DEP_RE.finditer(content):
        group_id = m.group(1).strip()
        artifact_id = m.group(2).strip()
        version = m.group(3).strip()
        if "${" in version:
            continue
        deps.append({
            "name": f"{group_id}:{artifact_id}",
            "version": version,
        })
    return deps


def _parse_composer_json(content: str) -> list[dict[str, str]]:
    """Parse composer.json require section."""
    deps = []
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return deps

    for section in ("require", "require-dev"):
        for name, version in data.get(section, {}).items():
            if name == "php" or name.startswith("ext-"):
                continue
            clean = re.sub(r"^[~^>=<!\s|]+", "", version)
            if clean and re.match(r"\d", clean):
                deps.append({"name": name, "version": clean})
    return deps


# Dispatcher
_PARSERS = {
    "requirements.txt": _parse_requirements_txt,
    "package.json": _parse_package_json,
    "package-lock.json": _parse_package_lock_json,
    "go.mod": _parse_go_mod,
    "Cargo.toml": _parse_cargo_toml,
    "Cargo.lock": _parse_cargo_lock,
    "Gemfile.lock": _parse_gemfile_lock,
    "pom.xml": _parse_pom_xml,
    "composer.json": _parse_composer_json,
}


def _detect_and_parse(directory: str) -> list[dict[str, Any]]:
    """Auto-detect manifest files in directory and parse them.

    Returns list of dicts: {name, version, ecosystem, source_file}
    """
    results: list[dict[str, Any]] = []
    dir_path = Path(directory)

    if not dir_path.is_dir():
        return results

    for filename, parser in _PARSERS.items():
        filepath = dir_path / filename
        if filepath.exists():
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                deps = parser(content)
                ecosystem = _ECOSYSTEM_MAP.get(filename, "")
                for dep in deps:
                    results.append({
                        "name": dep["name"],
                        "version": dep["version"],
                        "ecosystem": ecosystem,
                        "source_file": filename,
                    })
            except (OSError, UnicodeDecodeError):
                continue

    return results


# ---------------------------------------------------------------------------
# OSV.dev query
# ---------------------------------------------------------------------------


def _osv_batch_query(
    packages: list[dict[str, Any]],
    timeout: int = _REQUEST_TIMEOUT,
) -> list[list[dict[str, Any]]]:
    """Query OSV.dev batch endpoint for vulnerability data.

    Returns a list (same length as packages) where each element is a list
    of vulnerability dicts for that package.
    """
    if not packages:
        return []

    queries = []
    for pkg in packages:
        q: dict[str, Any] = {
            "package": {
                "name": pkg["name"],
                "ecosystem": pkg["ecosystem"],
            },
            "version": pkg["version"],
        }
        queries.append(q)

    all_results: list[list[dict[str, Any]]] = []

    for i in range(0, len(queries), _MAX_BATCH_SIZE):
        batch = queries[i: i + _MAX_BATCH_SIZE]
        try:
            resp = requests.post(
                _OSV_BATCH_URL,
                json={"queries": batch},
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            for r in results:
                vulns = r.get("vulns", [])
                all_results.append(vulns)
        except (requests.RequestException, ValueError, KeyError):
            all_results.extend([[] for _ in batch])

    return all_results


# ---------------------------------------------------------------------------
# EPSS enrichment
# ---------------------------------------------------------------------------


def _fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Fetch EPSS scores for a list of CVE IDs. Returns {cve_id: score}."""
    if not cve_ids:
        return {}

    scores: dict[str, float] = {}

    for i in range(0, len(cve_ids), 100):
        batch = cve_ids[i: i + 100]
        try:
            resp = requests.get(
                _EPSS_BATCH_URL,
                params={"cve": ",".join(batch)},
                headers={"User-Agent": _USER_AGENT},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                cve = item.get("cve", "")
                epss = item.get("epss")
                if cve and epss is not None:
                    try:
                        scores[cve] = float(epss)
                    except (ValueError, TypeError):
                        pass
        except (requests.RequestException, ValueError, KeyError):
            continue

    return scores


# ---------------------------------------------------------------------------
# KEV enrichment
# ---------------------------------------------------------------------------


def _fetch_kev_set() -> set[str]:
    """Fetch CISA KEV catalog and return set of CVE IDs."""
    try:
        resp = requests.get(
            _KEV_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            v.get("cveID", "").upper()
            for v in data.get("vulnerabilities", [])
            if v.get("cveID")
        }
    except (requests.RequestException, ValueError, KeyError):
        return set()


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------


def audit_dependencies(
    directory: str,
    *,
    include_dev: bool = True,
    skip_epss: bool = False,
    skip_kev: bool = False,
) -> dict[str, Any]:
    """Audit dependencies in a directory for known vulnerabilities.

    Args:
        directory: Path to the project directory containing manifest files.
        include_dev: Whether to include devDependencies (default: True).
        skip_epss: Skip EPSS score enrichment (faster).
        skip_kev: Skip CISA KEV lookup (faster).

    Returns:
        Dict with audit results including findings, summary, and metadata.
    """
    start_time = time.time()

    # 1. Parse manifests
    packages = _detect_and_parse(directory)
    if not packages:
        return {
            "status": "no_manifests",
            "message": f"No supported manifest files found in {directory}",
            "supported_files": list(_PARSERS.keys()),
            "packages_scanned": 0,
            "vulnerabilities_found": 0,
            "findings": [],
        }

    # 2. Query OSV.dev
    osv_results = _osv_batch_query(packages)

    # 3. Collect all unique CVE IDs for enrichment
    all_cve_ids: set[str] = set()
    findings: list[dict[str, Any]] = []

    for pkg, vulns in zip(packages, osv_results):
        if not vulns:
            continue
        for vuln in vulns:
            vuln_id = vuln.get("id", "")
            aliases = vuln.get("aliases", [])
            cve_ids = [a for a in aliases if a.startswith("CVE-")]
            if vuln_id.startswith("CVE-"):
                cve_ids.append(vuln_id)
            cve_ids = list(set(cve_ids))
            all_cve_ids.update(cve_ids)

            # Extract severity
            severity = ""
            for s in vuln.get("severity", []):
                if s.get("type") == "CVSS_V3":
                    severity = s.get("score", "")
                    break

            # Extract affected version ranges
            affected_ranges = []
            for affected in vuln.get("affected", []):
                pkg_info = affected.get("package", {})
                if (
                    pkg_info.get("name", "").lower() == pkg["name"].lower()
                    or pkg_info.get("name", "") == pkg["name"]
                ):
                    for r in affected.get("ranges", []):
                        events = r.get("events", [])
                        introduced = ""
                        fixed = ""
                        for ev in events:
                            if "introduced" in ev:
                                introduced = ev["introduced"]
                            if "fixed" in ev:
                                fixed = ev["fixed"]
                        if introduced or fixed:
                            affected_ranges.append({
                                "introduced": introduced,
                                "fixed": fixed,
                            })

            finding = {
                "package": pkg["name"],
                "version": pkg["version"],
                "ecosystem": pkg["ecosystem"],
                "source_file": pkg["source_file"],
                "vuln_id": vuln_id,
                "aliases": cve_ids,
                "summary": vuln.get("summary", "")[:200],
                "severity_vector": severity,
                "affected_ranges": affected_ranges,
                "fixed_version": (
                    affected_ranges[0]["fixed"]
                    if affected_ranges and affected_ranges[0].get("fixed")
                    else None
                ),
            }
            findings.append(finding)

    # 4. EPSS enrichment
    epss_scores: dict[str, float] = {}
    if not skip_epss and all_cve_ids:
        epss_scores = _fetch_epss_scores(list(all_cve_ids))

    # 5. KEV enrichment
    kev_set: set[str] = set()
    if not skip_kev and all_cve_ids:
        kev_set = _fetch_kev_set()

    # 6. Enrich findings with EPSS + KEV
    for finding in findings:
        best_epss = 0.0
        in_kev = False
        for cve in finding["aliases"]:
            cve_upper = cve.upper()
            if cve_upper in epss_scores:
                best_epss = max(best_epss, epss_scores[cve_upper])
            if cve_upper in kev_set:
                in_kev = True
        finding["epss_score"] = best_epss
        finding["in_kev"] = in_kev

    # 7. Sort findings by severity: KEV first, then EPSS desc
    findings.sort(key=lambda f: (not f["in_kev"], -f["epss_score"]))

    # 8. Build summary
    unique_vulns = len({f["vuln_id"] for f in findings})
    unique_packages_affected = len({f["package"] for f in findings})
    kev_count = sum(1 for f in findings if f["in_kev"])
    high_epss_count = sum(1 for f in findings if f["epss_score"] >= 0.5)

    elapsed = round(time.time() - start_time, 2)

    return {
        "status": "completed",
        "directory": directory,
        "packages_scanned": len(packages),
        "vulnerabilities_found": unique_vulns,
        "packages_affected": unique_packages_affected,
        "kev_count": kev_count,
        "high_epss_count": high_epss_count,
        "elapsed_seconds": elapsed,
        "manifests_parsed": list({p["source_file"] for p in packages}),
        "findings": findings,
        "summary": {
            "total_packages": len(packages),
            "total_findings": len(findings),
            "unique_vulnerabilities": unique_vulns,
            "packages_with_vulns": unique_packages_affected,
            "kev_findings": kev_count,
            "high_epss_findings": high_epss_count,
        },
    }


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "dep_audit",
    "description": (
        "Audit a project directory's dependencies for known vulnerabilities. "
        "Parses manifest files (requirements.txt, package.json, go.mod, "
        "Cargo.lock, Gemfile.lock, pom.xml, composer.json) and queries "
        "OSV.dev for known CVEs. Enriches results with EPSS scores and "
        "CISA KEV status. Returns findings sorted by severity."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": (
                        "Path to the project directory containing manifest files "
                        "(e.g. requirements.txt, package.json, go.mod)."
                    ),
                },
                "skip_epss": {
                    "type": "boolean",
                    "description": "Skip EPSS score enrichment for faster results.",
                    "default": False,
                },
                "skip_kev": {
                    "type": "boolean",
                    "description": "Skip CISA KEV lookup for faster results.",
                    "default": False,
                },
            },
            "required": ["directory"],
        }
    },
}


@tool
def dep_audit(
    directory: str,
    skip_epss: bool = False,
    skip_kev: bool = False,
) -> str:
    """Audit project dependencies for known vulnerabilities.

    Parses manifest files in the given directory and queries OSV.dev for
    known CVEs affecting the declared dependencies.

    Args:
        directory: Path to project directory with manifest files.
        skip_epss: Skip EPSS enrichment (default: False).
        skip_kev: Skip CISA KEV lookup (default: False).

    Returns:
        JSON string with audit results.
    """
    result = audit_dependencies(
        directory,
        skip_epss=skip_epss,
        skip_kev=skip_kev,
    )
    return json.dumps(result, indent=2)
