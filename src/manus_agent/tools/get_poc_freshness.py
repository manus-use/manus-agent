"""Tool: get_poc_freshness

Measures how recently PoC (Proof-of-Concept) activity occurred for a CVE.

Checks three signals of ongoing attacker interest:

1. **GitHub PoC repos** -- last commit date, star count, recent stars, fork
   count for repositories mentioning the CVE.
2. **Exploit-DB** -- publication date of matching entries.
3. **trickest/cve** -- existence and PoC link count in the curated index.

Each signal contributes to a composite **freshness score** (0-100):
- 90-100: Active development in the last 7 days
- 70-89:  Recent activity in the last 30 days
- 40-69:  Moderate activity in the last 90 days
- 10-39:  Stale -- last activity 90+ days ago
-  0-9:   No PoC activity found

A high freshness score means attacker interest is **ongoing** and the CVE
deserves urgent attention regardless of its CVSS/EPSS scores.

CLI: ``manus-agent poc-freshness CVE-XXXX-YYYY``
"""

from __future__ import annotations

import csv
import json as _json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

__all__ = ["get_poc_freshness"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-(\d{4})-\d+$", re.IGNORECASE)
_GITHUB_API = "https://api.github.com"
_TRICKEST_RAW = "https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md"
_EXPLOITDB_CSV_URL = (
    "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
)
_EXPLOITDB_CACHE = "/tmp/exploitdb_cache.csv"
_EXPLOITDB_CACHE_TTL = 86_400  # 24 hours

_REQUEST_TIMEOUT = 20  # seconds
_USER_AGENT = "manus-agent/poc-freshness (github.com/manus-use/manus-agent)"

# Freshness decay curve: days -> score contribution
# Exponential decay: score = base * exp(-lambda * days)
_DECAY_LAMBDA = 0.03  # half-life ~ 23 days


# ---------------------------------------------------------------------------
# HTTP helper with retry
# ---------------------------------------------------------------------------


def _http_get(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = _REQUEST_TIMEOUT,
    max_retries: int = 2,
    retry_codes: tuple[int, ...] = (429, 503),
) -> bytes:
    """HTTP GET with simple retry/back-off.  Returns raw bytes."""
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in retry_codes and attempt < max_retries:
                wait = 2 ** (attempt + 1)
                logger.debug("HTTP %d from %s -- retrying in %ds", exc.code, url, wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

    raise last_exc or RuntimeError("_http_get: unexpected fall-through")  # pragma: no cover


def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Any:
    """HTTP GET returning parsed JSON."""
    raw = _http_get(url, headers=headers, timeout=timeout)
    return _json.loads(raw.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Source 1: GitHub search -- PoC repos
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    hdrs: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs


def _search_github_repos(cve_id: str) -> list[dict[str, Any]]:
    """Search GitHub for repositories mentioning the CVE.

    Returns a list of repo dicts with keys: full_name, html_url, description,
    stargazers_count, forks_count, pushed_at, created_at, updated_at.
    """
    query = urllib.request.quote(f"{cve_id} in:name,description,readme")
    url = f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&per_page=10"
    try:
        data = _http_get_json(url, headers=_github_headers())
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        logger.warning("GitHub repo search failed for %s: %s", cve_id, exc)
        return []
    except Exception:  # noqa: BLE001
        logger.warning("GitHub repo search failed for %s (unexpected)", cve_id, exc_info=True)
        return []

    items = data.get("items", [])
    repos: list[dict[str, Any]] = []
    for item in items:
        repos.append(
            {
                "full_name": item.get("full_name", ""),
                "html_url": item.get("html_url", ""),
                "description": (item.get("description") or "")[:200],
                "stargazers_count": item.get("stargazers_count", 0),
                "forks_count": item.get("forks_count", 0),
                "pushed_at": item.get("pushed_at", ""),
                "created_at": item.get("created_at", ""),
                "updated_at": item.get("updated_at", ""),
            }
        )
    return repos


# ---------------------------------------------------------------------------
# Source 2: Exploit-DB -- CSV index
# ---------------------------------------------------------------------------


def _load_exploitdb_csv() -> list[dict[str, str]]:
    """Load the Exploit-DB CSV index (with 24h local cache)."""
    # Check cache freshness
    try:
        stat = os.stat(_EXPLOITDB_CACHE)
        if time.time() - stat.st_mtime < _EXPLOITDB_CACHE_TTL:
            with open(_EXPLOITDB_CACHE, encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                return list(reader)
    except FileNotFoundError:
        pass

    # Fetch fresh copy
    try:
        raw = _http_get(_EXPLOITDB_CSV_URL, timeout=30)
        with open(_EXPLOITDB_CACHE, "wb") as fh:
            fh.write(raw)
        with open(_EXPLOITDB_CACHE, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            return list(reader)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to fetch Exploit-DB CSV index", exc_info=True)
        return []


def _search_exploitdb(cve_id: str) -> list[dict[str, Any]]:
    """Search the Exploit-DB CSV for entries matching the CVE."""
    rows = _load_exploitdb_csv()
    cve_upper = cve_id.upper()
    matches: list[dict[str, Any]] = []
    for row in rows:
        codes = row.get("codes", "")
        if cve_upper in codes.upper():
            matches.append(
                {
                    "id": row.get("id", ""),
                    "description": (row.get("description", ""))[:200],
                    "date_published": row.get("date_published", ""),
                    "platform": row.get("platform", ""),
                    "type": row.get("type", ""),
                    "url": f"https://www.exploit-db.com/exploits/{row.get('id', '')}",
                }
            )
    return matches


# ---------------------------------------------------------------------------
# Source 3: trickest/cve
# ---------------------------------------------------------------------------


def _check_trickest(cve_id: str) -> dict[str, Any]:
    """Check the trickest/cve index for the CVE.

    Returns a dict with keys: found, poc_count, poc_urls.
    """
    match = _CVE_RE.match(cve_id)
    if not match:
        return {"found": False, "poc_count": 0, "poc_urls": []}

    year = match.group(1)
    url = _TRICKEST_RAW.format(year=year, cve_id=cve_id.upper())

    try:
        raw = _http_get(url, timeout=10)
        text = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"found": False, "poc_count": 0, "poc_urls": []}
        logger.warning("trickest/cve fetch failed for %s: HTTP %d", cve_id, exc.code)
        return {"found": False, "poc_count": 0, "poc_urls": []}
    except Exception:  # noqa: BLE001
        logger.warning("trickest/cve fetch failed for %s", cve_id, exc_info=True)
        return {"found": False, "poc_count": 0, "poc_urls": []}

    # Extract URLs from the markdown
    url_re = re.compile(r"https?://\S+")
    urls = [u.rstrip(").,]") for u in url_re.findall(text)]
    # Filter to likely PoC URLs (GitHub repos, exploit-db, etc.)
    poc_urls = [
        u
        for u in urls
        if any(
            domain in u.lower()
            for domain in ("github.com", "exploit-db.com", "packetstormsecurity.com")
        )
    ]

    return {
        "found": True,
        "poc_count": len(poc_urls),
        "poc_urls": poc_urls[:20],  # cap at 20 for output sanity
    }


# ---------------------------------------------------------------------------
# Freshness scoring
# ---------------------------------------------------------------------------


def _parse_iso_date(date_str: str) -> datetime | None:
    """Parse an ISO-8601 date/datetime string; returns None on failure."""
    if not date_str:
        return None
    try:
        # Handle GitHub-style "2024-01-15T12:00:00Z"
        clean = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        # Ensure timezone-aware (date-only strings parse as naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass
    # Try date-only format "2024-01-15"
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _days_ago(dt: datetime, now: datetime | None = None) -> float:
    """Return how many days ago *dt* was relative to *now* (UTC)."""
    now = now or datetime.now(tz=timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 86_400, 0)


def _decay_score(days: float, base: float = 100.0) -> float:
    """Exponential decay: older activity -> lower score."""
    return base * math.exp(-_DECAY_LAMBDA * days)


def _compute_freshness(
    github_repos: list[dict[str, Any]],
    exploitdb_entries: list[dict[str, Any]],
    trickest: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute a composite freshness score (0-100) from all signals.

    Returns a dict with:
        - freshness_score: int (0-100)
        - label: str (Active/Recent/Moderate/Stale/None)
        - signals: dict with per-source detail
        - most_recent_activity: ISO date string or None
        - days_since_last_activity: float or None
    """
    now = now or datetime.now(tz=timezone.utc)
    activity_dates: list[datetime] = []
    signals: dict[str, Any] = {}

    # --- GitHub signal ---
    github_score = 0.0
    github_detail: dict[str, Any] = {
        "repos_found": len(github_repos),
        "total_stars": 0,
        "total_forks": 0,
        "most_recent_push": None,
        "score_contribution": 0.0,
    }

    if github_repos:
        total_stars = sum(r.get("stargazers_count", 0) for r in github_repos)
        total_forks = sum(r.get("forks_count", 0) for r in github_repos)
        github_detail["total_stars"] = total_stars
        github_detail["total_forks"] = total_forks

        # Find most recent push across all repos
        push_dates: list[datetime] = []
        for repo in github_repos:
            dt = _parse_iso_date(repo.get("pushed_at", ""))
            if dt:
                push_dates.append(dt)
                activity_dates.append(dt)

        if push_dates:
            most_recent = max(push_dates)
            github_detail["most_recent_push"] = most_recent.isoformat()
            days = _days_ago(most_recent, now)

            # Recency score (up to 50 points)
            recency = _decay_score(days, base=50.0)

            # Popularity bonus (up to 10 points for stars/forks)
            pop_bonus = min(10.0, math.log2(max(total_stars + total_forks, 1)) * 2)

            # Multiple repos bonus (up to 5 points)
            multi_bonus = min(5.0, len(github_repos) * 1.0)

            github_score = recency + pop_bonus + multi_bonus

        github_detail["score_contribution"] = round(github_score, 1)

    signals["github"] = github_detail

    # --- Exploit-DB signal ---
    edb_score = 0.0
    edb_detail: dict[str, Any] = {
        "entries_found": len(exploitdb_entries),
        "most_recent_date": None,
        "score_contribution": 0.0,
    }

    if exploitdb_entries:
        edb_dates: list[datetime] = []
        for entry in exploitdb_entries:
            dt = _parse_iso_date(entry.get("date_published", ""))
            if dt:
                edb_dates.append(dt)
                activity_dates.append(dt)

        if edb_dates:
            most_recent = max(edb_dates)
            edb_detail["most_recent_date"] = most_recent.isoformat()
            days = _days_ago(most_recent, now)

            # Exploit-DB recency (up to 25 points)
            edb_score = _decay_score(days, base=25.0)

            # Multiple entries bonus (up to 5 points)
            edb_score += min(5.0, len(exploitdb_entries) * 1.5)

        edb_detail["score_contribution"] = round(edb_score, 1)

    signals["exploit_db"] = edb_detail

    # --- trickest/cve signal ---
    trickest_score = 0.0
    trickest_detail: dict[str, Any] = {
        "indexed": trickest.get("found", False),
        "poc_count": trickest.get("poc_count", 0),
        "score_contribution": 0.0,
    }

    if trickest.get("found"):
        poc_count = trickest.get("poc_count", 0)
        # Presence bonus: 5 points for being indexed
        trickest_score = 5.0
        # PoC count bonus: up to 10 points
        if poc_count > 0:
            trickest_score += min(10.0, math.log2(max(poc_count, 1)) * 3)

        trickest_detail["score_contribution"] = round(trickest_score, 1)

    signals["trickest"] = trickest_detail

    # --- Composite score ---
    raw_score = github_score + edb_score + trickest_score
    # Normalise to 0-100 (max theoretical ~ 50+10+5 + 25+5 + 5+10 = 110)
    freshness_score = min(100, round(raw_score))

    # Find most recent activity across all sources
    most_recent_activity: str | None = None
    days_since: float | None = None
    if activity_dates:
        latest = max(activity_dates)
        most_recent_activity = latest.isoformat()
        days_since = round(_days_ago(latest, now), 1)

    # Label
    if freshness_score >= 90:
        label = "Active"
    elif freshness_score >= 70:
        label = "Recent"
    elif freshness_score >= 40:
        label = "Moderate"
    elif freshness_score >= 10:
        label = "Stale"
    else:
        label = "None"

    return {
        "freshness_score": freshness_score,
        "label": label,
        "signals": signals,
        "most_recent_activity": most_recent_activity,
        "days_since_last_activity": days_since,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_poc_freshness(
    cve_id: str,
    *,
    timeout: int = _REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Measure PoC freshness for a CVE -- how recently exploit activity occurred.

    Queries GitHub (repository search), Exploit-DB (CSV index), and
    trickest/cve (curated PoC index) to build a composite freshness score.

    Args:
        cve_id: CVE identifier, e.g. ``"CVE-2024-3094"``.
        timeout: HTTP timeout in seconds per request.

    Returns:
        A dict with keys:

        - ``"cve_id"`` -- normalised CVE identifier
        - ``"freshness_score"`` -- int 0-100
        - ``"label"`` -- ``"Active"`` / ``"Recent"`` / ``"Moderate"`` /
          ``"Stale"`` / ``"None"``
        - ``"most_recent_activity"`` -- ISO datetime or *None*
        - ``"days_since_last_activity"`` -- float or *None*
        - ``"signals"`` -- per-source breakdown (github, exploit_db, trickest)
        - ``"github_repos"`` -- list of matching GitHub repos with metadata
        - ``"exploitdb_entries"`` -- list of matching Exploit-DB entries
        - ``"trickest"`` -- trickest/cve lookup result
        - ``"error"`` -- only present on total failure
    """
    cve_id = cve_id.strip().upper()

    if not _CVE_RE.match(cve_id):
        return {
            "cve_id": cve_id,
            "error": f"Invalid CVE ID format: {cve_id!r}. Expected CVE-YYYY-NNNNN.",
        }

    # Fetch all three sources (sequential to avoid complexity; fast enough)
    github_repos = _search_github_repos(cve_id)
    exploitdb_entries = _search_exploitdb(cve_id)
    trickest = _check_trickest(cve_id)

    # Compute composite freshness
    freshness = _compute_freshness(github_repos, exploitdb_entries, trickest)

    return {
        "cve_id": cve_id,
        "freshness_score": freshness["freshness_score"],
        "label": freshness["label"],
        "most_recent_activity": freshness["most_recent_activity"],
        "days_since_last_activity": freshness["days_since_last_activity"],
        "signals": freshness["signals"],
        "github_repos": github_repos,
        "exploitdb_entries": exploitdb_entries,
        "trickest": trickest,
    }
