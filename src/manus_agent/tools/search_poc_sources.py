"""
Tool: search_poc_sources

Multi-source PoC aggregator for a CVE ID.  Queries five public sources in
parallel and returns a unified, deduplicated, and ranked result set:

  1. trickest/cve  — GitHub tree API (always checked)
  2. VulnCheck KEV — exploited-in-wild signal (skipped gracefully when no key)
  3. Exploit-DB    — CSV index (cached 24 h at /tmp/exploitdb_cache.csv)
  4. GitHub search — repos/code mentioning the CVE
  5. NVD refs      — filtered for github.com / exploit-db.com / packetstorm

The tool is deliberately source-agnostic: all results share a common schema
so consumers can sort, filter, and display them uniformly.

CLI: ``manus-agent poc-search CVE-XXXX-YYYY``
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from strands import tool

from manus_agent.utils.cve_utils import extract_cve_year, is_valid_cve_id, validate_cve_id

__all__ = ["search_poc_sources"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-(\d{4})-\d+$", re.IGNORECASE)
_EXPLOITDB_CACHE = "/tmp/exploitdb_cache.csv"
_EXPLOITDB_CSV_URL = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
_EXPLOITDB_CACHE_TTL = 86_400  # 24 hours in seconds
_GITHUB_API = "https://api.github.com"
_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_VULNCHECK_KEV = "https://api.vulncheck.com/v3/index/vulncheck-kev"
_POC_URL_PATTERNS = re.compile(r"(github\.com|exploit-db\.com|packetstormsecurity\.com)", re.IGNORECASE)
_REQUEST_TIMEOUT = 20  # seconds

# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------


def _get(url: str, headers: dict[str, str] | None = None) -> Any:
    """Minimal HTTP GET returning parsed JSON; raises on failure."""
    import json

    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# URL normalisation (for deduplication)
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    url = url.strip().lower()
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


# ---------------------------------------------------------------------------
# Source 1: trickest/cve
# ---------------------------------------------------------------------------


def _fetch_trickest(cve_id: str) -> list[dict]:
    """Return PoC entries from the trickest/cve GitHub tree."""
    results: list[dict] = []
    year = extract_cve_year(cve_id)
    if year is None:
        return results
    year = str(year)

    tree_url = "https://api.github.com/repos/trickest/cve/git/trees/main?recursive=0"
    try:
        tree_data = _get(tree_url)
    except Exception as exc:
        logger.debug("trickest tree fetch failed: %s", exc)
        return results

    # Find the year folder SHA
    year_sha = None
    for item in tree_data.get("tree", []):
        if item.get("path") == year and item.get("type") == "tree":
            year_sha = item["sha"]
            break

    if not year_sha:
        return results

    # Fetch the year subtree (non-recursive — just list CVE folders)
    try:
        year_tree = _get(f"https://api.github.com/repos/trickest/cve/git/trees/{year_sha}")
    except Exception as exc:
        logger.debug("trickest year tree fetch failed: %s", exc)
        return results

    cve_upper = cve_id.upper()
    for item in year_tree.get("tree", []):
        if item.get("path", "").upper() == cve_upper:
            url = f"https://github.com/trickest/cve/tree/main/{year}/{cve_upper}"
            results.append(
                {
                    "source": "trickest",
                    "url": url,
                    "title": f"{cve_upper} — trickest/cve index",
                    "published": None,
                    "author": "trickest",
                    "tags": ["index"],
                    "exploited_in_wild": False,
                }
            )
            break

    return results


# ---------------------------------------------------------------------------
# Source 2: VulnCheck KEV
# ---------------------------------------------------------------------------


def _fetch_vulncheck_kev(cve_id: str) -> list[dict]:
    """Query VulnCheck KEV for exploited-in-wild status."""
    api_key = os.environ.get("VULNCHECK_API_KEY", "")
    if not api_key:
        logger.debug("VULNCHECK_API_KEY not set — skipping VulnCheck KEV")
        return []

    url = f"{_VULNCHECK_KEV}?cve={urllib.parse.quote(cve_id)}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        data = _get(url, headers=headers)
    except Exception as exc:
        logger.debug("VulnCheck KEV fetch failed: %s", exc)
        return []

    results: list[dict] = []
    for entry in data.get("data", []):
        entry_cve = entry.get("cveID", "") or entry.get("cve_id", "")
        if entry_cve.upper() != cve_id.upper():
            continue
        date_added = entry.get("dateAdded") or entry.get("date_added")
        sources = entry.get("sources") or []
        tags = ["kev"]
        if entry.get("ransomwareUse") or entry.get("ransomware_use"):
            tags.append("ransomware")
        results.append(
            {
                "source": "vulncheck_kev",
                "url": f"https://vulncheck.com/browse/kev/{cve_id.lower()}",
                "title": f"{cve_id} — VulnCheck KEV (exploited in wild)",
                "published": date_added,
                "author": ", ".join(sources) if sources else None,
                "tags": tags,
                "exploited_in_wild": True,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Source 3: Exploit-DB CSV
# ---------------------------------------------------------------------------


def _ensure_exploitdb_cache() -> str | None:
    """Return path to a fresh Exploit-DB CSV, downloading if needed."""
    cache_path = _EXPLOITDB_CACHE
    try:
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < _EXPLOITDB_CACHE_TTL:
            return cache_path
    except FileNotFoundError:
        pass

    try:
        req = urllib.request.Request(_EXPLOITDB_CSV_URL, headers={"User-Agent": "manus-agent"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = resp.read()
        with open(cache_path, "wb") as fh:
            fh.write(data)
        return cache_path
    except Exception as exc:
        logger.debug("Exploit-DB CSV download failed: %s", exc)
        return None


def _fetch_exploitdb(cve_id: str) -> list[dict]:
    """Filter the Exploit-DB CSV for rows matching the CVE ID."""
    cache_path = _ensure_exploitdb_cache()
    if not cache_path:
        return []

    results: list[dict] = []
    cve_upper = cve_id.upper()
    try:
        with open(cache_path, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                codes = row.get("codes", "") or ""
                if cve_upper not in codes.upper():
                    continue
                edb_id = (row.get("id") or "").strip()
                title = (row.get("description") or row.get("title") or "").strip()
                date_str = (row.get("date_published") or row.get("date") or "").strip()
                url = f"https://www.exploit-db.com/exploits/{edb_id}" if edb_id else "https://www.exploit-db.com"
                etype = (row.get("type") or "").strip().lower()
                platform = (row.get("platform") or "").strip().lower()
                tags = [t for t in [etype, platform] if t]
                results.append(
                    {
                        "source": "exploitdb",
                        "url": url,
                        "title": title or f"Exploit-DB #{edb_id}",
                        "published": date_str or None,
                        "author": (row.get("author") or "").strip() or None,
                        "tags": tags,
                        "exploited_in_wild": False,
                    }
                )
    except Exception as exc:
        logger.debug("Exploit-DB CSV parse failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Source 4: GitHub repo search
# ---------------------------------------------------------------------------


def _fetch_github(cve_id: str) -> list[dict]:
    """Search GitHub repositories mentioning the CVE ID."""
    query = urllib.parse.quote(cve_id)
    url = f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&per_page=5"
    headers = {"Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    try:
        data = _get(url, headers=headers)
    except Exception as exc:
        logger.debug("GitHub repo search failed: %s", exc)
        return []

    results: list[dict] = []
    for item in data.get("items", []):
        pushed = item.get("pushed_at")
        published = pushed[:10] if pushed else None
        results.append(
            {
                "source": "github",
                "url": item.get("html_url", ""),
                "title": item.get("full_name", item.get("name", "")),
                "published": published,
                "author": (item.get("owner") or {}).get("login"),
                "tags": ["github-repo"],
                "exploited_in_wild": False,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Source 5: NVD references
# ---------------------------------------------------------------------------


def _fetch_nvd(cve_id: str) -> list[dict]:
    """Extract PoC-relevant URLs from NVD references."""
    url = f"{_NVD_API}?cveId={urllib.parse.quote(cve_id)}"
    try:
        data = _get(url)
    except Exception as exc:
        logger.debug("NVD fetch failed: %s", exc)
        return []

    results: list[dict] = []
    for vuln in data.get("vulnerabilities", []):
        cve_data = vuln.get("cve", {})
        pub = cve_data.get("published", "")
        published = pub[:10] if pub else None
        for ref in cve_data.get("references", []):
            ref_url = ref.get("url", "")
            if not _POC_URL_PATTERNS.search(ref_url):
                continue
            tags = [t.lower() for t in (ref.get("tags") or [])]
            results.append(
                {
                    "source": "nvd",
                    "url": ref_url,
                    "title": ref_url,
                    "published": published,
                    "author": None,
                    "tags": tags,
                    "exploited_in_wild": False,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_ALL_SOURCES = ["trickest", "vulncheck_kev", "exploitdb", "github", "nvd"]

# Map source name -> module-level attribute name; looked up dynamically so
# unit tests can patch individual _fetch_* functions on the module object.
_SOURCE_FN_NAMES = {
    "trickest": "_fetch_trickest",
    "vulncheck_kev": "_fetch_vulncheck_kev",
    "exploitdb": "_fetch_exploitdb",
    "github": "_fetch_github",
    "nvd": "_fetch_nvd",
}


def _get_source_fn(source: str):
    """Resolve a source fetcher by name from this module (supports test patching)."""
    import manus_agent.tools.search_poc_sources as _self

    attr = _SOURCE_FN_NAMES.get(source)
    if attr is None:
        return None
    return getattr(_self, attr, None)


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse ISO date string; return None on failure."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except ValueError:
            continue
    return None


def _is_recent(date_str: str | None, days: int = 30) -> bool:
    dt = _parse_date(date_str)
    if dt is None:
        return False
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - dt).days <= days


def aggregate_poc_results(
    cve_id: str,
    sources: list[str] | None = None,
) -> dict:
    """Fetch PoC data from requested sources in parallel and aggregate."""
    if sources is None:
        sources = _ALL_SOURCES

    raw_results: list[dict] = []
    sources_checked: list[str] = []
    sources_failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
        futures = {pool.submit(_get_source_fn(s), cve_id): s for s in sources if _get_source_fn(s) is not None}
        for future in as_completed(futures):
            src = futures[future]
            sources_checked.append(src)
            try:
                raw_results.extend(future.result())
            except Exception as exc:
                logger.debug("Source %s raised: %s", src, exc)
                sources_failed.append(src)

    # Deduplicate by normalised URL
    seen: dict[str, dict] = {}
    for item in raw_results:
        key = _normalize_url(item.get("url", ""))
        if not key:
            continue
        if key not in seen:
            seen[key] = item
        elif item.get("exploited_in_wild"):
            seen[key]["exploited_in_wild"] = True

    deduped = list(seen.values())

    def _sort_key(r: dict):
        eaw = 0 if r.get("exploited_in_wild") else 1
        dt = _parse_date(r.get("published"))
        date_ts = -dt.timestamp() if dt else float("inf")
        return (eaw, date_ts)

    deduped.sort(key=_sort_key)

    any_kev = any(r.get("exploited_in_wild") for r in deduped)
    any_recent = any(_is_recent(r.get("published")) for r in deduped)

    return {
        "cve_id": cve_id.upper(),
        "total_found": len(deduped),
        "exploited_in_wild": any_kev,
        "recent_activity": any_recent,
        "sources_checked": sorted(sources_checked),
        "sources_failed": sorted(sources_failed),
        "results": deduped,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


@tool
def search_poc_sources(cve_id: str, sources: str = "") -> dict:
    """Search multiple public sources for PoC exploits related to a CVE.

    Queries trickest/cve, VulnCheck KEV (exploited-in-wild), Exploit-DB,
    GitHub repositories, and NVD references in parallel.  Results are
    deduplicated by URL and sorted with exploited-in-wild entries first,
    then by publication date descending.

    Args:
        cve_id: CVE identifier (e.g. CVE-2024-3094).
        sources: Optional comma-separated list of sources to query.
            Valid values: trickest, vulncheck_kev, exploitdb, github, nvd.
            Leave empty to query all sources.

    Returns:
        Dictionary with keys: cve_id, total_found, exploited_in_wild,
        recent_activity, sources_checked, sources_failed, results.
        Each result has: source, url, title, published, author, tags,
        exploited_in_wild.
    """
    cve_id = cve_id.strip()
    if (err_msg := validate_cve_id(cve_id)) is not None:
        return {
            "error": err_msg,
            "cve_id": cve_id,
            "total_found": 0,
            "exploited_in_wild": False,
            "recent_activity": False,
            "sources_checked": [],
            "sources_failed": [],
            "results": [],
        }

    source_list: list[str] | None = None
    if sources.strip():
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    return aggregate_poc_results(cve_id, source_list)
