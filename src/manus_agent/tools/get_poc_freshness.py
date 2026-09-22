"""
Tool: get_poc_freshness

Measures how recently Proof-of-Concept (PoC) activity occurred for a CVE.

Checks multiple freshness signals in parallel:

  1. **GitHub PoC repos** — searches for repos mentioning the CVE, examines
     pushed_at timestamps, star counts, and fork counts to gauge active interest.
  2. **Exploit-DB** — checks for recent entries via the GitLab CSV index.
  3. **EPSS trend** — fetches the current EPSS score and recent change via
     the FIRST.org API as a proxy for exploitation prediction momentum.

Produces a composite **freshness_score** (0–100) where:
  - 0–25  = stale / no recent activity
  - 26–50 = moderate / some signals
  - 51–75 = fresh / active PoC development
  - 76–100 = hot / very recent, high-interest activity

CLI: ``manus-agent poc-freshness CVE-XXXX-YYYY``
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from strands import tool

__all__ = ["get_poc_freshness"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)
_REQUEST_TIMEOUT = 20  # seconds
_USER_AGENT = "manus-agent/poc-freshness (github.com/manus-use/manus-agent)"
_GITHUB_API = "https://api.github.com"
_EPSS_API = "https://api.first.org/data/v1/epss"
_EXPLOITDB_CACHE = "/tmp/exploitdb_cache.csv"
_EXPLOITDB_CSV_URL = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
_EXPLOITDB_CACHE_TTL = 86_400  # 24 hours

# Freshness decay half-life in days — activity older than this contributes
# exponentially less to the freshness score.
_HALF_LIFE_DAYS = 30

# Weight allocation for the composite score (must sum to 1.0)
_W_GITHUB = 0.50
_W_EXPLOITDB = 0.25
_W_EPSS = 0.25


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """Minimal HTTP GET returning parsed JSON."""
    hdrs = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get_text(url: str) -> str:
    """Minimal HTTP GET returning UTF-8 text."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime (naive, for consistent comparisons)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_iso(date_str: str | None) -> datetime | None:
    """Parse an ISO-ish date string; return None on failure."""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except ValueError:
            continue
    return None


def _days_ago(dt: datetime | None) -> float | None:
    """Return how many days ago *dt* was, or None."""
    if dt is None:
        return None
    delta = _now_utc() - dt
    return max(0.0, delta.total_seconds() / 86_400)


def _decay(days: float | None) -> float:
    """Exponential decay: 1.0 for today, 0.5 at half-life, → 0."""
    if days is None or days < 0:
        return 0.0
    return math.exp(-0.693 * days / _HALF_LIFE_DAYS)  # ln(2) ≈ 0.693


# ---------------------------------------------------------------------------
# Signal 1: GitHub PoC repos
# ---------------------------------------------------------------------------


def _fetch_github_signal(cve_id: str) -> dict[str, Any]:
    """Search GitHub for repos mentioning the CVE; return freshness metrics."""
    result: dict[str, Any] = {
        "repos_found": 0,
        "most_recent_push": None,
        "most_recent_push_days_ago": None,
        "total_stars": 0,
        "total_forks": 0,
        "top_repos": [],
        "score": 0.0,
        "error": None,
    }

    query = urllib.parse.quote(cve_id)
    url = f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&per_page=10"
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    try:
        data = _http_get_json(url, headers=headers)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    items = data.get("items", [])
    result["repos_found"] = data.get("total_count", len(items))

    if not items:
        return result

    most_recent_dt: datetime | None = None
    total_stars = 0
    total_forks = 0
    top_repos: list[dict[str, Any]] = []

    for repo in items[:10]:
        pushed_at = _parse_iso(repo.get("pushed_at"))
        stars = repo.get("stargazers_count", 0) or 0
        forks = repo.get("forks_count", 0) or 0
        total_stars += stars
        total_forks += forks

        if pushed_at and (most_recent_dt is None or pushed_at > most_recent_dt):
            most_recent_dt = pushed_at

        top_repos.append(
            {
                "full_name": repo.get("full_name", ""),
                "url": repo.get("html_url", ""),
                "pushed_at": repo.get("pushed_at"),
                "stars": stars,
                "forks": forks,
                "description": (repo.get("description") or "")[:120],
            }
        )

    result["most_recent_push"] = most_recent_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if most_recent_dt else None
    days = _days_ago(most_recent_dt)
    result["most_recent_push_days_ago"] = round(days, 1) if days is not None else None
    result["total_stars"] = total_stars
    result["total_forks"] = total_forks
    result["top_repos"] = top_repos

    # Score: recency decay × repo-count boost × star boost
    recency = _decay(days)
    repo_boost = min(1.0, result["repos_found"] / 5)  # saturates at 5 repos
    star_boost = min(1.0, math.log1p(total_stars) / 5)  # saturates ~147 stars
    result["score"] = round(recency * 0.5 + repo_boost * 0.3 + star_boost * 0.2, 4)

    return result


# ---------------------------------------------------------------------------
# Signal 2: Exploit-DB entries
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
        req = urllib.request.Request(_EXPLOITDB_CSV_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310
            data = resp.read()
        with open(cache_path, "wb") as fh:
            fh.write(data)
        return cache_path
    except Exception as exc:
        logger.debug("Exploit-DB CSV download failed: %s", exc)
        return None


def _fetch_exploitdb_signal(cve_id: str) -> dict[str, Any]:
    """Check Exploit-DB CSV for entries matching the CVE."""
    result: dict[str, Any] = {
        "entries_found": 0,
        "most_recent_date": None,
        "most_recent_days_ago": None,
        "entries": [],
        "score": 0.0,
        "error": None,
    }

    cache_path = _ensure_exploitdb_cache()
    if not cache_path:
        result["error"] = "Could not download or access Exploit-DB CSV"
        return result

    cve_upper = cve_id.upper()
    entries: list[dict[str, Any]] = []
    most_recent_dt: datetime | None = None

    try:
        with open(cache_path, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                codes = (row.get("codes") or "").upper()
                if cve_upper not in codes:
                    continue
                edb_id = (row.get("id") or "").strip()
                date_str = (row.get("date_published") or row.get("date") or "").strip()
                dt = _parse_iso(date_str)
                if dt and (most_recent_dt is None or dt > most_recent_dt):
                    most_recent_dt = dt

                entries.append(
                    {
                        "edb_id": edb_id,
                        "title": (row.get("description") or row.get("title") or "").strip()[:120],
                        "date": date_str or None,
                        "url": (f"https://www.exploit-db.com/exploits/{edb_id}" if edb_id else None),
                    }
                )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["entries_found"] = len(entries)
    result["entries"] = entries[:5]  # top 5

    if most_recent_dt:
        result["most_recent_date"] = most_recent_dt.strftime("%Y-%m-%d")
        days = _days_ago(most_recent_dt)
        result["most_recent_days_ago"] = round(days, 1) if days is not None else None
    else:
        days = None

    # Score: recency decay × entry count boost
    recency = _decay(days if days is not None else 9999)
    count_boost = min(1.0, len(entries) / 3)  # saturates at 3 entries
    result["score"] = round(recency * 0.6 + count_boost * 0.4, 4)

    return result


# ---------------------------------------------------------------------------
# Signal 3: EPSS trend
# ---------------------------------------------------------------------------


def _fetch_epss_signal(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score and recent change from FIRST.org API."""
    result: dict[str, Any] = {
        "epss_score": None,
        "epss_percentile": None,
        "score": 0.0,
        "error": None,
    }

    url = f"{_EPSS_API}?cve={urllib.parse.quote(cve_id)}"
    try:
        data = _http_get_json(url)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    entries = data.get("data", [])
    if not entries:
        return result

    entry = entries[0]
    epss = float(entry.get("epss", 0))
    percentile = float(entry.get("percentile", 0))

    result["epss_score"] = round(epss, 6)
    result["epss_percentile"] = round(percentile, 4)

    # Score: EPSS itself is a probability of exploitation — use it directly
    # as the freshness signal (high EPSS = high active exploitation likelihood)
    result["score"] = round(min(1.0, epss * 5), 4)  # amplify: 0.2 EPSS → 1.0

    return result


# ---------------------------------------------------------------------------
# Composite freshness calculation
# ---------------------------------------------------------------------------


def _compute_freshness(
    github: dict[str, Any],
    exploitdb: dict[str, Any],
    epss: dict[str, Any],
) -> dict[str, Any]:
    """Combine signals into a composite freshness score (0–100)."""
    raw = github["score"] * _W_GITHUB + exploitdb["score"] * _W_EXPLOITDB + epss["score"] * _W_EPSS
    freshness_score = round(min(100.0, raw * 100), 1)

    if freshness_score >= 76:
        label = "hot"
    elif freshness_score >= 51:
        label = "fresh"
    elif freshness_score >= 26:
        label = "moderate"
    else:
        label = "stale"

    return {
        "freshness_score": freshness_score,
        "freshness_label": label,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_poc_freshness(cve_id: str) -> dict[str, Any]:
    """Measure PoC freshness for a CVE across multiple signals.

    Args:
        cve_id: CVE identifier (e.g. ``CVE-2024-3094``).

    Returns:
        Dictionary with composite freshness score and per-signal detail::

            {
                "cve_id": "CVE-2024-3094",
                "freshness_score": 72.5,
                "freshness_label": "fresh",
                "signals": {
                    "github": { ... },
                    "exploitdb": { ... },
                    "epss": { ... },
                },
            }
    """
    cve_id = cve_id.strip().upper()
    if not _CVE_RE.match(cve_id):
        return {
            "cve_id": cve_id,
            "error": f"Invalid CVE identifier: {cve_id!r}",
            "freshness_score": 0,
            "freshness_label": "unknown",
            "signals": {},
        }

    # Fetch all signals in parallel
    signal_fns = {
        "github": (_fetch_github_signal, cve_id),
        "exploitdb": (_fetch_exploitdb_signal, cve_id),
        "epss": (_fetch_epss_signal, cve_id),
    }

    signals: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn, arg): name for name, (fn, arg) in signal_fns.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                signals[name] = future.result()
            except Exception as exc:
                logger.debug("Signal %s raised: %s", name, exc)
                signals[name] = {"score": 0.0, "error": str(exc)}

    # Ensure all keys exist with defaults
    for key in ("github", "exploitdb", "epss"):
        if key not in signals:
            signals[key] = {"score": 0.0, "error": "signal not collected"}

    composite = _compute_freshness(signals["github"], signals["exploitdb"], signals["epss"])

    return {
        "cve_id": cve_id,
        **composite,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


@tool
def get_poc_freshness(cve_id: str) -> dict:
    """Measure how recently PoC activity occurred for a CVE.

    Checks GitHub PoC repositories (push recency, stars, forks),
    Exploit-DB entries (publication dates), and EPSS score (exploitation
    prediction) to produce a composite freshness score (0–100).

    A high freshness score means attacker interest is ongoing.

    Score bands:
      - 0–25:  stale — no recent activity
      - 26–50: moderate — some signals
      - 51–75: fresh — active PoC development
      - 76–100: hot — very recent, high-interest activity

    Args:
        cve_id: CVE identifier (e.g. CVE-2024-3094).

    Returns:
        Dictionary with freshness_score (0–100), freshness_label,
        and per-signal detail (github, exploitdb, epss).
    """
    return check_poc_freshness(cve_id)
