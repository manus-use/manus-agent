"""
Tool: get_poc_freshness

PoC freshness checker for a CVE ID.  Measures how recently proof-of-concept
exploit activity occurred across multiple public sources, producing a 0–100
freshness score that indicates whether attacker interest is ongoing.

Three signal dimensions are evaluated:

  1. **Commit recency** — last push/commit timestamps in known PoC GitHub
     repositories (via trickest/cve tree + GitHub repo API).
  2. **Star velocity** — recently-starred GitHub repositories mentioning the
     CVE (recent stars = rising attacker interest).
  3. **Exploit-DB recency** — publication dates of matching Exploit-DB entries
     (new EDB IDs = fresh weaponisation).

The freshness score is a weighted composite:

  - Commit recency:   40 %  (most actionable signal)
  - Star velocity:    30 %  (community interest proxy)
  - Exploit-DB:       30 %  (formal weaponisation signal)

Score tiers:

  - 80–100  ACTIVE    — PoC activity in the last 7 days
  - 60–79   RECENT    — PoC activity in the last 30 days
  - 40–59   AGING     — PoC activity in the last 90 days
  - 20–39   STALE     — PoC activity in the last 365 days
  -  0–19   DORMANT   — no recent PoC activity (>1 year or none found)

CLI: ``manus-agent poc-freshness CVE-XXXX-YYYY``
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_poc_freshness", "TOOL_SPEC"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_GITHUB_API = "https://api.github.com"
_REQUEST_TIMEOUT = 20  # seconds
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5  # seconds multiplier

# Exploit-DB CSV cache (shared with search_poc_sources)
_EXPLOITDB_CACHE = "/tmp/exploitdb_cache.csv"
_EXPLOITDB_CSV_URL = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
_EXPLOITDB_CACHE_TTL = 86_400  # 24 hours

# Freshness score weights
_WEIGHT_COMMIT_RECENCY = 0.40
_WEIGHT_STAR_VELOCITY = 0.30
_WEIGHT_EXPLOITDB = 0.30

# Tier thresholds (days since last activity → sub-score)
_TIER_BOUNDARIES = [
    (7, 100),  # ≤7 days  → 100
    (30, 75),  # ≤30 days → 75
    (90, 50),  # ≤90 days → 50
    (365, 25),  # ≤365 days → 25
]
_TIER_FLOOR = 5  # >365 days → 5

# Tier labels
_TIER_LABELS = [
    (80, "ACTIVE"),
    (60, "RECENT"),
    (40, "AGING"),
    (20, "STALE"),
    (0, "DORMANT"),
]


# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands agent integration)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_poc_freshness",
    "description": (
        "Measures how recently proof-of-concept exploit activity occurred for "
        "a given CVE. Checks commit recency in known PoC repositories, "
        "recently-starred GitHub repos, and new Exploit-DB entries. Returns a "
        "0–100 freshness score where higher means more recent attacker interest. "
        "Use after poc-search to understand whether exploitation activity is "
        "ongoing or dormant."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": ("The CVE identifier to check (e.g., 'CVE-2024-3094')."),
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers with retry + back-off
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including token if available."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "manus-agent/poc-freshness",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Any:
    """GET request with retry/back-off, returning parsed JSON."""
    import json as _json

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _http_get_text(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> str:
    """GET request returning raw text with retry/back-off."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper: parse ISO-8601 date strings
# ---------------------------------------------------------------------------


def _parse_iso_date(date_str: str | None) -> datetime | None:
    """Parse an ISO-8601 date/datetime string, returning None on failure."""
    if not date_str:
        return None
    try:
        # Handle GitHub's Z-suffix timestamps
        cleaned = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _days_since(dt: datetime | None, now: datetime | None = None) -> float | None:
    """Return days between *dt* and *now* (defaults to utcnow)."""
    if dt is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 86_400, 0)


def _days_to_subscore(days: float | None) -> float:
    """Convert days-since-activity to a 0–100 sub-score."""
    if days is None:
        return 0.0
    for max_days, score in _TIER_BOUNDARIES:
        if days <= max_days:
            return float(score)
    return float(_TIER_FLOOR)


def _freshness_tier(score: float) -> str:
    """Map a 0–100 freshness score to a tier label."""
    for threshold, label in _TIER_LABELS:
        if score >= threshold:
            return label
    return "DORMANT"


# ---------------------------------------------------------------------------
# Signal 1: Commit recency (trickest/cve + GitHub repo API)
# ---------------------------------------------------------------------------


def _fetch_trickest_repos(cve_id: str) -> list[str]:
    """Return GitHub repo URLs found in the trickest/cve tree for a CVE.

    The trickest/cve repo organises PoCs by year: ``<year>/<CVE-ID>.md``.
    Each markdown file contains lines with GitHub repository URLs.
    """
    match = _CVE_RE.match(cve_id)
    if not match:
        return []

    year = cve_id.split("-")[1]
    raw_url = f"https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id.upper()}.md"
    try:
        text = _http_get_text(raw_url, timeout=_REQUEST_TIMEOUT)
    except Exception:  # noqa: BLE001
        return []

    # Extract GitHub repo URLs from the markdown content
    repo_urls: list[str] = []
    for line in text.splitlines():
        urls = re.findall(r"https?://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", line)
        repo_urls.extend(urls)

    # Normalise and deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for url in repo_urls:
        norm = url.lower().rstrip("/")
        if norm.endswith(".git"):
            norm = norm[:-4]
        if norm not in seen:
            seen.add(norm)
            unique.append(url)
    return unique


def _get_repo_last_push(repo_url: str) -> dict[str, Any]:
    """Query GitHub API for a repo's last push and recent commit dates.

    Returns dict with keys: repo_url, pushed_at, last_commit_at,
    stargazers_count, error.
    """
    result: dict[str, Any] = {
        "repo_url": repo_url,
        "pushed_at": None,
        "last_commit_at": None,
        "stargazers_count": 0,
        "error": None,
    }

    # Extract owner/repo from URL
    match = re.match(
        r"https?://github\.com/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)",
        repo_url,
    )
    if not match:
        result["error"] = "Could not parse owner/repo from URL"
        return result

    owner, repo = match.group(1), match.group(2)
    headers = _github_headers()

    # Fetch repo metadata (pushed_at, stargazers_count)
    try:
        data = _http_get_json(
            f"{_GITHUB_API}/repos/{owner}/{repo}",
            headers=headers,
        )
        result["pushed_at"] = data.get("pushed_at")
        result["stargazers_count"] = data.get("stargazers_count", 0)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Repo API error: {exc}"
        return result

    # Fetch last commit date (more accurate than pushed_at)
    try:
        commits = _http_get_json(
            f"{_GITHUB_API}/repos/{owner}/{repo}/commits?per_page=1",
            headers=headers,
        )
        if commits and isinstance(commits, list) and len(commits) > 0:
            commit_date = commits[0].get("commit", {}).get("committer", {}).get("date")
            result["last_commit_at"] = commit_date
    except Exception:  # noqa: BLE001
        pass  # pushed_at is still available as fallback

    return result


def _assess_commit_recency(
    cve_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess commit recency across known PoC repos for a CVE.

    Returns dict with: repos_found, repos_checked, most_recent_commit,
    most_recent_push, days_since_last_activity, subscore, repos (details).
    """
    result: dict[str, Any] = {
        "repos_found": 0,
        "repos_checked": 0,
        "most_recent_commit": None,
        "most_recent_push": None,
        "days_since_last_activity": None,
        "subscore": 0.0,
        "repos": [],
    }

    repo_urls = _fetch_trickest_repos(cve_id)
    result["repos_found"] = len(repo_urls)

    if not repo_urls:
        return result

    # Cap at 10 repos to avoid API rate-limit exhaustion
    repo_urls = repo_urls[:10]

    most_recent: datetime | None = None
    repo_details: list[dict[str, Any]] = []

    for url in repo_urls:
        info = _get_repo_last_push(url)
        result["repos_checked"] += 1

        commit_dt = _parse_iso_date(info.get("last_commit_at"))
        push_dt = _parse_iso_date(info.get("pushed_at"))
        best_dt = commit_dt or push_dt

        repo_details.append(
            {
                "repo_url": info["repo_url"],
                "pushed_at": info.get("pushed_at"),
                "last_commit_at": info.get("last_commit_at"),
                "stargazers_count": info.get("stargazers_count", 0),
                "days_since_activity": (round(_days_since(best_dt, now), 1) if best_dt else None),
                "error": info.get("error"),
            }
        )

        if best_dt is not None:
            if most_recent is None or best_dt > most_recent:
                most_recent = best_dt

        # Brief delay to be polite to GitHub API
        time.sleep(0.25)

    result["repos"] = repo_details

    if most_recent:
        if most_recent.tzinfo is None:
            most_recent = most_recent.replace(tzinfo=timezone.utc)
        result["most_recent_commit"] = most_recent.isoformat()
        days = _days_since(most_recent, now)
        result["days_since_last_activity"] = round(days, 1) if days is not None else None
        result["subscore"] = _days_to_subscore(days)
    else:
        result["most_recent_push"] = None
        result["subscore"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Signal 2: Star velocity (GitHub search for recently-starred repos)
# ---------------------------------------------------------------------------


def _assess_star_velocity(
    cve_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess star velocity for GitHub repos mentioning the CVE.

    Searches GitHub for repositories matching the CVE ID, sorted by
    recently updated, and evaluates the collective star count and
    freshness of the results.

    Returns dict with: total_repos, recent_repos (updated in last 90 days),
    total_stars, most_recent_update, days_since_most_recent, subscore,
    top_repos (details).
    """
    result: dict[str, Any] = {
        "total_repos": 0,
        "recent_repos": 0,
        "total_stars": 0,
        "most_recent_update": None,
        "days_since_most_recent": None,
        "subscore": 0.0,
        "top_repos": [],
    }

    if now is None:
        now = datetime.now(timezone.utc)

    headers = _github_headers()
    query = urllib.parse.quote(cve_id.upper())

    try:
        data = _http_get_json(
            f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&order=desc&per_page=10",
            headers=headers,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("GitHub search failed for %s: %s", cve_id, exc)
        return result

    items = data.get("items", [])
    result["total_repos"] = data.get("total_count", len(items))

    most_recent_dt: datetime | None = None
    recent_count = 0
    total_stars = 0

    for repo in items[:10]:
        updated_dt = _parse_iso_date(repo.get("updated_at"))
        pushed_dt = _parse_iso_date(repo.get("pushed_at"))
        best_dt = pushed_dt or updated_dt
        stars = repo.get("stargazers_count", 0)
        total_stars += stars

        days = _days_since(best_dt, now)
        if days is not None and days <= 90:
            recent_count += 1

        if best_dt is not None:
            if most_recent_dt is None or best_dt > most_recent_dt:
                most_recent_dt = best_dt

        result["top_repos"].append(
            {
                "full_name": repo.get("full_name", ""),
                "url": repo.get("html_url", ""),
                "stars": stars,
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "days_since_activity": (round(_days_since(best_dt, now), 1) if best_dt else None),
            }
        )

    result["recent_repos"] = recent_count
    result["total_stars"] = total_stars

    if most_recent_dt:
        result["most_recent_update"] = most_recent_dt.isoformat()
        days = _days_since(most_recent_dt, now)
        result["days_since_most_recent"] = round(days, 1) if days is not None else None

        # Boost subscore if there are multiple recent repos with stars
        base_subscore = _days_to_subscore(days)
        if recent_count >= 3 and total_stars >= 10:
            base_subscore = min(100.0, base_subscore * 1.2)
        result["subscore"] = round(base_subscore, 1)
    else:
        result["subscore"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Signal 3: Exploit-DB recency
# ---------------------------------------------------------------------------


def _load_exploitdb_csv() -> str:
    """Load the Exploit-DB CSV, using a 24-hour cache."""
    # Check cache freshness
    try:
        stat = os.stat(_EXPLOITDB_CACHE)
        if time.time() - stat.st_mtime < _EXPLOITDB_CACHE_TTL:
            with open(_EXPLOITDB_CACHE, encoding="utf-8", errors="replace") as fh:
                return fh.read()
    except FileNotFoundError:
        pass

    # Fetch fresh copy
    text = _http_get_text(_EXPLOITDB_CSV_URL, timeout=30)

    # Write cache
    try:
        with open(_EXPLOITDB_CACHE, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass  # cache write failure is non-fatal

    return text


def _assess_exploitdb_recency(
    cve_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check Exploit-DB for entries matching the CVE and assess recency.

    Returns dict with: total_entries, most_recent_date,
    days_since_most_recent, subscore, entries (details).
    """
    result: dict[str, Any] = {
        "total_entries": 0,
        "most_recent_date": None,
        "days_since_most_recent": None,
        "subscore": 0.0,
        "entries": [],
    }

    if now is None:
        now = datetime.now(timezone.utc)

    try:
        csv_text = _load_exploitdb_csv()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load Exploit-DB CSV: %s", exc)
        return result

    cve_upper = cve_id.upper()
    most_recent_dt: datetime | None = None
    entries: list[dict[str, Any]] = []

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # Exploit-DB CSV has a 'codes' column with CVE references
        codes = row.get("codes", "")
        if cve_upper not in codes.upper():
            continue

        edb_id = row.get("id", "")
        date_published = row.get("date_published", "")
        description = row.get("description", "")

        pub_dt = _parse_date_published(date_published)

        entries.append(
            {
                "edb_id": edb_id,
                "url": f"https://www.exploit-db.com/exploits/{edb_id}" if edb_id else "",
                "description": description[:200] if description else "",
                "date_published": date_published,
                "days_since_published": (round(_days_since(pub_dt, now), 1) if pub_dt else None),
            }
        )

        if pub_dt is not None:
            if most_recent_dt is None or pub_dt > most_recent_dt:
                most_recent_dt = pub_dt

    result["total_entries"] = len(entries)
    result["entries"] = entries[:10]  # Cap output

    if most_recent_dt:
        result["most_recent_date"] = most_recent_dt.isoformat()
        days = _days_since(most_recent_dt, now)
        result["days_since_most_recent"] = round(days, 1) if days is not None else None
        result["subscore"] = _days_to_subscore(days)
    else:
        result["subscore"] = 0.0

    return result


def _parse_date_published(date_str: str) -> datetime | None:
    """Parse Exploit-DB date_published column (YYYY-MM-DD format)."""
    if not date_str or not date_str.strip():
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_freshness(
    cve_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute the composite PoC freshness score for a CVE.

    Returns a rich result dict containing per-signal assessments, the
    composite freshness_score (0–100), tier label, and summary.
    """
    cve_id = cve_id.strip().upper()

    if not _CVE_RE.match(cve_id):
        return {
            "error": f"Invalid CVE identifier: {cve_id!r}",
            "cve_id": cve_id,
            "freshness_score": 0,
            "tier": "DORMANT",
            "signals": {},
            "summary": f"Invalid CVE identifier: {cve_id!r}",
        }

    # Collect signals
    commit_signal = _assess_commit_recency(cve_id, now=now)
    star_signal = _assess_star_velocity(cve_id, now=now)
    exploitdb_signal = _assess_exploitdb_recency(cve_id, now=now)

    # Compute weighted composite score
    composite = (
        commit_signal["subscore"] * _WEIGHT_COMMIT_RECENCY
        + star_signal["subscore"] * _WEIGHT_STAR_VELOCITY
        + exploitdb_signal["subscore"] * _WEIGHT_EXPLOITDB
    )
    freshness_score = round(min(100.0, max(0.0, composite)), 1)
    tier = _freshness_tier(freshness_score)

    # Determine most recent activity across all signals
    all_dates: list[datetime] = []
    for dt_str in [
        commit_signal.get("most_recent_commit"),
        star_signal.get("most_recent_update"),
        exploitdb_signal.get("most_recent_date"),
    ]:
        dt = _parse_iso_date(dt_str) if isinstance(dt_str, str) else None
        if dt is not None:
            all_dates.append(dt)

    most_recent_activity = max(all_dates).isoformat() if all_dates else None

    # Build summary
    summary_parts: list[str] = []
    if commit_signal["repos_found"] > 0:
        summary_parts.append(f"{commit_signal['repos_found']} PoC repo(s) found")
        if commit_signal["days_since_last_activity"] is not None:
            summary_parts.append(f"last commit {commit_signal['days_since_last_activity']:.0f}d ago")
    if star_signal["total_repos"] > 0:
        summary_parts.append(f"{star_signal['total_repos']} GitHub repo(s) mentioning CVE")
        if star_signal["recent_repos"] > 0:
            summary_parts.append(f"{star_signal['recent_repos']} updated in last 90d")
    if exploitdb_signal["total_entries"] > 0:
        summary_parts.append(f"{exploitdb_signal['total_entries']} Exploit-DB entry(ies)")

    summary = "; ".join(summary_parts) if summary_parts else "No PoC activity found"

    return {
        "cve_id": cve_id,
        "freshness_score": freshness_score,
        "tier": tier,
        "most_recent_activity": most_recent_activity,
        "summary": summary,
        "signals": {
            "commit_recency": {
                "weight": _WEIGHT_COMMIT_RECENCY,
                "subscore": commit_signal["subscore"],
                "repos_found": commit_signal["repos_found"],
                "repos_checked": commit_signal["repos_checked"],
                "most_recent_commit": commit_signal["most_recent_commit"],
                "days_since_last_activity": commit_signal["days_since_last_activity"],
                "repos": commit_signal["repos"],
            },
            "star_velocity": {
                "weight": _WEIGHT_STAR_VELOCITY,
                "subscore": star_signal["subscore"],
                "total_repos": star_signal["total_repos"],
                "recent_repos": star_signal["recent_repos"],
                "total_stars": star_signal["total_stars"],
                "most_recent_update": star_signal["most_recent_update"],
                "days_since_most_recent": star_signal["days_since_most_recent"],
                "top_repos": star_signal["top_repos"],
            },
            "exploitdb_recency": {
                "weight": _WEIGHT_EXPLOITDB,
                "subscore": exploitdb_signal["subscore"],
                "total_entries": exploitdb_signal["total_entries"],
                "most_recent_date": exploitdb_signal["most_recent_date"],
                "days_since_most_recent": exploitdb_signal["days_since_most_recent"],
                "entries": exploitdb_signal["entries"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def get_poc_freshness(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool handler for get_poc_freshness."""
    tool_input = tool.get("input", {}) if isinstance(tool, dict) else tool.input
    cve_id = tool_input.get("cve_id", "")

    result = compute_freshness(cve_id)
    log_tool_output_size("get_poc_freshness", result)

    return {
        "toolUseId": tool.get("toolUseId", "") if isinstance(tool, dict) else tool.toolUseId,
        "status": "error" if "error" in result else "success",
        "content": [{"text": str(result)}],
    }
