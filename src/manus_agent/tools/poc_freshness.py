"""
Tool: poc_freshness

Measures how recently proof-of-concept (PoC) activity occurred for a given
CVE, producing a 0-100 freshness score. A high score means attacker interest
is ongoing or very recent.

Signals examined (all public, no authentication required by default):

  1. GitHub PoC repos - last commit date, push recency, star recency
  2. trickest/cve index - presence and number of known PoC links
  3. Exploit-DB - publication date of matching entries (via CSV index)
  4. NVD references - count of exploit-tagged references

The score is an exponential decay function: activity today scores 100, activity
one year ago scores near 0.  Multiple recent signals amplify the score.

CLI: ``manus-agent poc-freshness CVE-XXXX-YYYY``
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from strands import tool

__all__ = ["poc_freshness"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-(\d{4})-\d+$", re.IGNORECASE)
_GITHUB_API = "https://api.github.com"
_TRICKEST_RAW = "https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md"
_EXPLOITDB_CACHE = "/tmp/exploitdb_cache.csv"
_EXPLOITDB_CSV_URL = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
_EXPLOITDB_CACHE_TTL = 86_400  # 24h
_NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_REQUEST_TIMEOUT = 15
_HALF_LIFE_DAYS = 60  # activity half-life for decay scoring
_URL_RE = re.compile(r"https?://\S+")

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """Minimal HTTP GET returning parsed JSON."""
    import json as _json

    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return _json.loads(resp.read().decode())


def _http_get_text(url: str, headers: dict[str, str] | None = None) -> str:
    """HTTP GET returning text."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return resp.read().decode()


# ---------------------------------------------------------------------------
# Decay scoring
# ---------------------------------------------------------------------------


def _days_ago(date_str: str) -> float:
    """Parse an ISO-ish date string and return days from now."""
    date_str = date_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        # Try date-only
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = (now - dt).total_seconds() / 86400
    return max(delta, 0)


def _decay_score(days: float) -> float:
    """Exponential decay: 100 for today, ~50 at half-life, asymptotic to 0."""
    return 100.0 * math.exp(-0.693 * days / _HALF_LIFE_DAYS)


# ---------------------------------------------------------------------------
# Signal 1: GitHub PoC repos
# ---------------------------------------------------------------------------


def _search_github_pocs(cve_id: str) -> list[dict]:
    """Search GitHub for repos matching the CVE ID; return freshness signals."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    query = urllib.parse.quote(f"{cve_id} poc OR exploit OR proof-of-concept")
    url = f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&per_page=10"

    try:
        data = _http_get_json(url, headers)
    except Exception as exc:
        logger.debug("GitHub search failed: %s", exc)
        return []

    results = []
    for repo in data.get("items", [])[:10]:
        pushed_at = repo.get("pushed_at", "")
        created_at = repo.get("created_at", "")
        stars = repo.get("stargazers_count", 0)
        results.append(
            {
                "name": repo.get("full_name", ""),
                "url": repo.get("html_url", ""),
                "pushed_at": pushed_at,
                "created_at": created_at,
                "stars": stars,
                "days_since_push": _days_ago(pushed_at) if pushed_at else 9999,
                "days_since_create": _days_ago(created_at) if created_at else 9999,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Signal 2: trickest/cve
# ---------------------------------------------------------------------------


def _check_trickest(cve_id: str) -> dict:
    """Check trickest/cve for PoC links count."""
    match = _CVE_RE.match(cve_id)
    if not match:
        return {"found": False, "poc_count": 0}
    year = match.group(1)
    url = _TRICKEST_RAW.format(year=year, cve_id=cve_id.upper())
    try:
        md = _http_get_text(url)
    except Exception:
        return {"found": False, "poc_count": 0}

    urls = _URL_RE.findall(md)
    return {"found": True, "poc_count": len(set(urls))}


# ---------------------------------------------------------------------------
# Signal 3: Exploit-DB CSV
# ---------------------------------------------------------------------------


def _ensure_exploitdb_csv() -> str | None:
    """Download/cache Exploit-DB CSV; return path or None on failure."""
    if os.path.isfile(_EXPLOITDB_CACHE):
        age = time.time() - os.path.getmtime(_EXPLOITDB_CACHE)
        if age < _EXPLOITDB_CACHE_TTL:
            return _EXPLOITDB_CACHE
    try:
        req = urllib.request.Request(
            _EXPLOITDB_CSV_URL,
            headers={"User-Agent": "manus-agent/poc-freshness"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(_EXPLOITDB_CACHE, "wb") as f:
            f.write(data)
        return _EXPLOITDB_CACHE
    except Exception as exc:
        logger.debug("Exploit-DB CSV fetch failed: %s", exc)
        return None


def _search_exploitdb(cve_id: str) -> list[dict]:
    """Search cached Exploit-DB CSV for CVE matches."""
    csv_path = _ensure_exploitdb_csv()
    if not csv_path:
        return []

    results = []
    cve_upper = cve_id.upper()
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                codes = row.get("codes", "")
                if cve_upper in codes.upper():
                    date_published = row.get("date_published", "")
                    results.append(
                        {
                            "edb_id": row.get("id", ""),
                            "description": row.get("description", "")[:120],
                            "date_published": date_published,
                            "days_since_publish": (_days_ago(date_published) if date_published else 9999),
                        }
                    )
    except Exception as exc:
        logger.debug("Exploit-DB CSV parse failed: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Signal 4: NVD exploit references
# ---------------------------------------------------------------------------


def _count_nvd_exploit_refs(cve_id: str) -> int:
    """Count NVD references tagged as exploit/third-party-advisory."""
    url = f"{_NVD_API}?cveId={cve_id.upper()}"
    headers = {"User-Agent": "manus-agent/poc-freshness"}
    api_key = os.environ.get("NVD_API_KEY", "")
    if api_key:
        headers["apiKey"] = api_key
    try:
        data = _http_get_json(url, headers)
    except Exception as exc:
        logger.debug("NVD API failed: %s", exc)
        return 0

    count = 0
    for vuln in data.get("vulnerabilities", []):
        cve_obj = vuln.get("cve", {})
        for ref in cve_obj.get("references", []):
            tags = ref.get("tags", [])
            if any(t.lower() in ("exploit", "third party advisory") for t in tags):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def _compute_freshness(
    github_repos: list[dict],
    trickest: dict,
    exploitdb_entries: list[dict],
    nvd_exploit_refs: int,
) -> dict:
    """Compute composite freshness score and breakdown."""
    signals: list[dict] = []

    # GitHub: score from most recently pushed repo
    if github_repos:
        best = min(github_repos, key=lambda r: r["days_since_push"])
        gh_score = _decay_score(best["days_since_push"])
        signals.append(
            {
                "source": "github_poc_repos",
                "score": gh_score,
                "detail": (
                    f"{len(github_repos)} repos found; "
                    f"most recent push {best['days_since_push']:.0f}d ago "
                    f"({best['name']})"
                ),
            }
        )

        # Bonus for newly-created repos (last 30 days)
        new_repos = [r for r in github_repos if r["days_since_create"] < 30]
        if new_repos:
            newest = min(new_repos, key=lambda r: r["days_since_create"])
            bonus = _decay_score(newest["days_since_create"]) * 0.5
            signals.append(
                {
                    "source": "github_new_repo",
                    "score": bonus,
                    "detail": (
                        f"{len(new_repos)} repo(s) created in last 30d; "
                        f"newest {newest['days_since_create']:.0f}d ago "
                        f"({newest['name']})"
                    ),
                }
            )

        # Star signal: high stars = ongoing community interest
        max_stars = max(r["stars"] for r in github_repos)
        if max_stars >= 50:
            star_bonus = min(30.0, max_stars / 10.0)
            signals.append(
                {
                    "source": "github_stars",
                    "score": star_bonus,
                    "detail": f"Top repo has {max_stars} stars (ongoing interest)",
                }
            )

    # trickest: presence + density bonus
    if trickest["found"]:
        poc_count = trickest["poc_count"]
        trickest_score = min(40.0, 15.0 + poc_count * 3)
        signals.append(
            {
                "source": "trickest_cve_index",
                "score": trickest_score,
                "detail": f"{poc_count} PoC URLs indexed in trickest/cve",
            }
        )

    # Exploit-DB: most recent publication
    if exploitdb_entries:
        best_edb = min(exploitdb_entries, key=lambda e: e["days_since_publish"])
        edb_score = _decay_score(best_edb["days_since_publish"])
        signals.append(
            {
                "source": "exploit_db",
                "score": edb_score,
                "detail": (
                    f"{len(exploitdb_entries)} entries; "
                    f"newest {best_edb['days_since_publish']:.0f}d ago "
                    f"(EDB-{best_edb['edb_id']})"
                ),
            }
        )

    # NVD exploit references: presence signal
    if nvd_exploit_refs > 0:
        nvd_score = min(25.0, 10.0 + nvd_exploit_refs * 3)
        signals.append(
            {
                "source": "nvd_exploit_refs",
                "score": nvd_score,
                "detail": f"{nvd_exploit_refs} exploit-tagged NVD references",
            }
        )

    # Composite: weighted geometric-ish blend capped at 100
    if not signals:
        freshness_score = 0.0
    else:
        # Take the max signal as base, add diminishing returns from others
        sorted_scores = sorted((s["score"] for s in signals), reverse=True)
        composite = sorted_scores[0]
        for i, s in enumerate(sorted_scores[1:], start=1):
            composite += s * (0.5**i)
        freshness_score = min(100.0, composite)

    # Classification
    if freshness_score >= 80:
        classification = "CRITICAL - Active, very recent PoC activity"
    elif freshness_score >= 60:
        classification = "HIGH - Recent PoC activity (last few weeks)"
    elif freshness_score >= 40:
        classification = "MODERATE - PoC activity within last few months"
    elif freshness_score >= 20:
        classification = "LOW - Stale PoC activity (several months old)"
    else:
        classification = "MINIMAL - No recent PoC activity detected"

    return {
        "freshness_score": round(freshness_score, 1),
        "classification": classification,
        "signals": signals,
        "github_repos_found": len(github_repos),
        "exploitdb_entries_found": len(exploitdb_entries),
        "trickest_indexed": trickest["found"],
        "trickest_poc_count": trickest.get("poc_count", 0),
        "nvd_exploit_refs": nvd_exploit_refs,
    }


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


@tool
def poc_freshness(cve_id: str) -> str:
    """Measure how recently PoC (proof-of-concept) activity occurred for a CVE.

    Produces a 0-100 freshness score based on: last commit recency in known PoC
    repos, recently-starred/created repositories, new Exploit-DB entries, and
    trickest/cve index presence.  A high freshness score means attacker interest
    is ongoing.

    Args:
        cve_id: CVE identifier, e.g. "CVE-2024-3094".

    Returns:
        Formatted freshness report with score, classification, and signal breakdown.
    """

    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return f"Invalid CVE ID format: {cve_id!r}. Expected CVE-YYYY-NNNNN."

    # Gather signals (best-effort, never fatal)
    github_repos = _search_github_pocs(cve_id)
    trickest = _check_trickest(cve_id)
    exploitdb_entries = _search_exploitdb(cve_id)
    nvd_refs = _count_nvd_exploit_refs(cve_id)

    result = _compute_freshness(github_repos, trickest, exploitdb_entries, nvd_refs)
    result["cve_id"] = cve_id

    # Format as readable report
    lines = [
        f"## PoC Freshness Report: {cve_id}",
        "",
        f"**Freshness Score:** {result['freshness_score']}/100",
        f"**Classification:** {result['classification']}",
        "",
        "### Signal Breakdown",
    ]
    for sig in result["signals"]:
        lines.append(f"  - [{sig['source']}] score={sig['score']:.1f} -- {sig['detail']}")

    if not result["signals"]:
        lines.append("  (no PoC activity signals detected)")

    lines += [
        "",
        "### Summary",
        f"  GitHub PoC repos: {result['github_repos_found']}",
        f"  Exploit-DB entries: {result['exploitdb_entries_found']}",
        f"  trickest/cve indexed: {'Yes' if result['trickest_indexed'] else 'No'}"
        + (f" ({result['trickest_poc_count']} URLs)" if result["trickest_indexed"] else ""),
        f"  NVD exploit refs: {result['nvd_exploit_refs']}",
    ]

    return "\n".join(lines)
