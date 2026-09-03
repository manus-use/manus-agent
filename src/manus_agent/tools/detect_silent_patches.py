"""
Tool for detecting silent security patches in a GitHub repository.

Scans a repository's commit history for security-relevant fixes that were
never assigned a CVE. Uses a two-stage heuristic scoring pipeline:

  Stage 1 — Commit message keywords: scores each commit message against
  14 security-relevant bug-class keyword lists plus general security
  fix indicators (e.g. "fix", "vuln", "security", "sanitize").

  Stage 2 — Diff keywords (optional, skipped in --fast mode): fetches the
  unified diff for high-scoring commits and rescores against the same
  keyword lists applied to the actual code changes.

Each candidate commit is labelled with one of 14 bug classes:
  sql_injection, command_injection, path_traversal, buffer_overflow,
  integer_overflow, use_after_free, null_dereference, auth_bypass,
  deserialization, xss, ssrf, input_validation, race_condition,
  cryptographic

Only public GitHub repositories are supported (unauthenticated or via
GITHUB_TOKEN).  Private repositories return a descriptive error.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Bug-class keyword mapping (same 14 classes as get_patch_diff)
# ---------------------------------------------------------------------------
_BUG_CLASSES: list[tuple[str, list[str]]] = [
    (
        "sql_injection",
        [
            "sql",
            "query",
            "execute",
            "cursor",
            "select ",
            "insert ",
            "update ",
            "delete ",
            "where ",
            "parameteriz",
            "prepared statement",
        ],
    ),
    (
        "command_injection",
        [
            "os.system",
            "subprocess",
            "shell=True",
            "exec(",
            "eval(",
            "popen",
            "execv",
            "system(",
            "passthru",
        ],
    ),
    (
        "path_traversal",
        [
            "../",
            "..\\",
            "os.path.join",
            "realpath",
            "abspath",
            "traverse",
            "chroot",
            "directory traversal",
        ],
    ),
    (
        "buffer_overflow",
        [
            "memcpy",
            "strcpy",
            "strcat",
            "sprintf",
            "gets(",
            "scanf",
            "overflow",
            "struct.pack",
            "ctypes",
            "bounds check",
        ],
    ),
    (
        "integer_overflow",
        [
            "integer overflow",
            "int overflow",
            "wrap around",
            "wraparound",
            "size_t",
            "ssize_t",
            "uint",
            "truncat",
        ],
    ),
    (
        "use_after_free",
        [
            "use after free",
            "use-after-free",
            "uaf",
            "free(",
            "kfree(",
            "dangling",
            "double free",
        ],
    ),
    (
        "null_dereference",
        [
            "null check",
            "nullptr",
            "null pointer",
            "is None",
            "if not ",
            "== null",
            "!= null",
            "nullpointer",
        ],
    ),
    (
        "auth_bypass",
        [
            "authentication",
            "authoriz",
            "permission",
            "privilege",
            "access control",
            "bypass",
            "check_permission",
            "is_authenticated",
            "acl",
        ],
    ),
    (
        "deserialization",
        [
            "deserializ",
            "pickle",
            "yaml.load",
            "json.loads",
            "unmarshal",
            "unserializ",
            "objectinputstream",
        ],
    ),
    (
        "xss",
        [
            "escape",
            "sanitize",
            "sanitise",
            "html.escape",
            "encode",
            "htmlentities",
            "innerhtml",
            "xss",
            "cross-site",
        ],
    ),
    (
        "ssrf",
        [
            "ssrf",
            "urlopen",
            "requests.get",
            "fetch(",
            "curl",
            "allowed_hosts",
            "internal",
            "localhost",
            "server-side request",
        ],
    ),
    (
        "input_validation",
        [
            "validate",
            "validation",
            "sanitize",
            "sanitise",
            "allowlist",
            "whitelist",
            "blacklist",
            "regex",
            "pattern",
            "filter",
        ],
    ),
    (
        "race_condition",
        [
            "race condition",
            "toctou",
            "time-of-check",
            "mutex",
            "lock(",
            "synchronized",
            "atomic",
            "concurren",
        ],
    ),
    (
        "cryptographic",
        [
            "encrypt",
            "decrypt",
            "cipher",
            "hash",
            "random",
            "entropy",
            "tls",
            "ssl",
            "hmac",
            "signature",
            "constant.time",
            "timing attack",
        ],
    ),
]

# Keywords in commit messages that indicate a security-relevant fix.
_SECURITY_MSG_KEYWORDS: list[str] = [
    "fix",
    "vuln",
    "security",
    "cve-",
    "exploit",
    "patch",
    "sanitiz",
    "sanitise",
    "overflow",
    "injection",
    "bypass",
    "traversal",
    "xss",
    "csrf",
    "ssrf",
    "rce",
    "dos",
    "denial of service",
    "privilege escalation",
    "buffer",
    "heap",
    "stack",
    "out of bounds",
    "out-of-bounds",
    "memory corruption",
    "use after free",
    "use-after-free",
    "null pointer",
    "null dereference",
    "race condition",
    "authentication",
    "authorization",
    "access control",
    "hardcoded",
    "hardened",
    "unsafe",
    "insecure",
    "malicious",
]

# CVE pattern — commits that already have a CVE are NOT silent patches.
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

# Message score threshold for stage-2 diff analysis.
_MSG_SCORE_THRESHOLD = 2

# Combined score threshold for reporting a commit as a candidate.
_CANDIDATE_THRESHOLD = 3

# Maximum diff size to fetch (bytes).
_MAX_DIFF_BYTES = 256_000

# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def _github_headers() -> dict[str, str]:
    """Build GitHub API request headers, including auth token if available."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _github_get_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 15,
    max_retries: int = 2,
) -> dict[str, Any] | list[Any] | None:
    """GET *url* and return parsed JSON, or None on any error.

    Retries on 429 / 5xx with exponential back-off.
    """
    headers = _github_headers()
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None
            if resp.status_code == 422:
                return None
            if resp.status_code in (429, 500, 502, 503):
                wait = 2 ** (attempt + 1)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, int(retry_after))
                    except ValueError:
                        pass
                if attempt < max_retries:
                    time.sleep(min(wait, 30))
                    continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < max_retries:
                time.sleep(2 ** (attempt + 1))
                continue
            return None
    return None


def _github_get_text(url: str, *, timeout: int = 20) -> str | None:
    """GET *url* and return response text, or None on any error."""
    headers = _github_headers()
    headers["Accept"] = "application/vnd.github.diff"
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        # Cap downloaded diff to _MAX_DIFF_BYTES.
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(8192):
            chunks.append(chunk)
            total += len(chunk)
            if total >= _MAX_DIFF_BYTES:
                break
        return b"".join(chunks).decode("utf-8", errors="replace")
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_text_against_bug_classes(text: str) -> tuple[int, list[str]]:
    """Score *text* against all bug-class keyword lists.

    Returns ``(total_hits, matched_classes)`` where *total_hits* is the
    number of distinct keyword hits across all classes, and
    *matched_classes* is a deduplicated list of bug-class names that
    matched.
    """
    text_lower = text.lower()
    total_hits = 0
    matched_classes: list[str] = []
    for cls_name, keywords in _BUG_CLASSES:
        for kw in keywords:
            if kw in text_lower:
                total_hits += 1
                if cls_name not in matched_classes:
                    matched_classes.append(cls_name)
                break  # one hit per class is enough
    return total_hits, matched_classes


def _score_message(message: str) -> int:
    """Score a commit message for security relevance.

    Returns an integer score (higher = more likely a security fix).
    """
    msg_lower = message.lower()
    score = 0

    # General security keywords.
    for kw in _SECURITY_MSG_KEYWORDS:
        if kw in msg_lower:
            score += 1

    # Bug-class keywords in message.
    bug_hits, _ = _score_text_against_bug_classes(message)
    score += bug_hits

    return score


def _score_diff(diff_text: str) -> tuple[int, list[str]]:
    """Score a unified diff for security-relevant changes.

    Only scores added/removed lines (lines starting with + or -).

    Returns ``(score, matched_classes)``.
    """
    # Extract only changed lines (skip diff headers).
    changed_lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed_lines.append(line[1:])  # strip the +/- prefix

    changed_text = "\n".join(changed_lines)
    hits, matched_classes = _score_text_against_bug_classes(changed_text)

    # Also count general security keywords in changed lines.
    changed_lower = changed_text.lower()
    general_hits = 0
    for kw in _SECURITY_MSG_KEYWORDS:
        if kw in changed_lower:
            general_hits += 1

    return hits + general_hits, matched_classes


def _classify_commit(
    message: str,
    diff_text: str | None,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    """Classify a single commit.

    Returns a dict with scoring details:
    - ``message_score``: int
    - ``diff_score``: int (0 if fast or no diff)
    - ``total_score``: int
    - ``classifications``: list of matched bug-class names
    - ``is_candidate``: bool
    """
    msg_score = _score_message(message)
    diff_score = 0
    diff_classes: list[str] = []

    if not fast and diff_text:
        diff_score, diff_classes = _score_diff(diff_text)

    # Merge classifications from message and diff.
    _, msg_classes = _score_text_against_bug_classes(message)
    all_classes = list(dict.fromkeys(msg_classes + diff_classes))  # deduplicated, ordered

    total = msg_score + diff_score
    threshold = _MSG_SCORE_THRESHOLD if fast else _CANDIDATE_THRESHOLD

    return {
        "message_score": msg_score,
        "diff_score": diff_score,
        "total_score": total,
        "classifications": all_classes,
        "is_candidate": total >= threshold,
    }


# ---------------------------------------------------------------------------
# Commit fetching
# ---------------------------------------------------------------------------


def _fetch_commits(
    owner: str,
    repo: str,
    *,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
) -> list[dict[str, Any]]:
    """Fetch commit list from GitHub REST API.

    Returns a list of commit dicts with ``sha``, ``message``, ``date``,
    ``author``, and ``url`` keys.  Paginates up to *max_commits*.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params: dict[str, str] = {"per_page": str(min(max_commits, 100))}
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    all_commits: list[dict[str, Any]] = []
    fetched = 0

    while fetched < max_commits:
        data = _github_get_json(url, params=params)
        if not data or not isinstance(data, list):
            break

        for item in data:
            if fetched >= max_commits:
                break
            commit_data = item.get("commit", {})
            message = commit_data.get("message", "")
            sha = item.get("sha", "")
            author_info = commit_data.get("author") or {}
            committer_info = commit_data.get("committer") or {}
            date = author_info.get("date") or committer_info.get("date") or ""
            author = author_info.get("name") or ""
            html_url = item.get("html_url") or ""

            all_commits.append(
                {
                    "sha": sha,
                    "message": message,
                    "date": date,
                    "author": author,
                    "url": html_url,
                }
            )
            fetched += 1

        if len(data) < int(params.get("per_page", "100")):
            break  # last page

        # Paginate: use the last sha as the "before" cursor via `sha` param.
        # GitHub's commit listing uses page-based pagination.
        page = int(params.get("page", "1"))
        params["page"] = str(page + 1)

    return all_commits


def _fetch_commit_diff(owner: str, repo: str, sha: str) -> str | None:
    """Fetch the unified diff for a single commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    return _github_get_text(url)


# ---------------------------------------------------------------------------
# Main scanning pipeline
# ---------------------------------------------------------------------------


def _parse_repo_spec(spec: str) -> tuple[str, str]:
    """Parse ``owner/repo`` from a spec string.

    Accepts:
    - ``owner/repo``
    - ``https://github.com/owner/repo``
    - ``github.com/owner/repo``

    Returns ``(owner, repo)`` or raises ``ValueError``.
    """
    spec = spec.strip().rstrip("/")

    # Strip common URL prefixes.
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if spec.lower().startswith(prefix):
            spec = spec[len(prefix) :]
            break

    parts = spec.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid repo spec: expected 'owner/repo', got '{spec}'")

    return parts[0], parts[1]


def scan_silent_patches(
    repo_spec: str,
    *,
    since: str | None = None,
    until: str | None = None,
    max_commits: int = 500,
    fast: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Scan a repository for silent security patches.

    Parameters
    ----------
    repo_spec:
        Repository in ``owner/repo`` or URL format.
    since:
        ISO-8601 date string for scan start (default: 90 days ago).
    until:
        ISO-8601 date string for scan end (default: now).
    max_commits:
        Hard cap on commits fetched (default: 500).
    fast:
        If True, skip diff analysis (message keywords only).
    now:
        Override "now" for deterministic testing.

    Returns
    -------
    dict with keys:
        - ``repo``: str
        - ``scan_window``: dict with ``since`` and ``until``
        - ``commits_scanned``: int
        - ``candidates``: list of candidate dicts
        - ``summary``: dict with counts and classification breakdown
    """
    _now = now or datetime.now(tz=timezone.utc)
    owner, repo = _parse_repo_spec(repo_spec)

    # Default window: 90 days.
    if not since:
        default_since = _now - timedelta(days=90)
        since = default_since.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Ensure ISO format.
        if "T" not in since:
            since = since + "T00:00:00Z"

    if not until:
        until = _now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        if "T" not in until:
            until = until + "T23:59:59Z"

    max_commits = max(1, min(max_commits, 5000))

    # Stage 0: fetch commits.
    commits = _fetch_commits(owner, repo, since=since, until=until, max_commits=max_commits)

    candidates: list[dict[str, Any]] = []
    skipped_cve = 0

    for commit in commits:
        message = commit["message"]

        # Skip commits that already reference a CVE.
        if _CVE_RE.search(message):
            skipped_cve += 1
            continue

        # Stage 1: message scoring.
        msg_score = _score_message(message)

        # Only fetch diff for commits that pass message threshold.
        diff_text: str | None = None
        if not fast and msg_score >= _MSG_SCORE_THRESHOLD:
            diff_text = _fetch_commit_diff(owner, repo, commit["sha"])

        # Stage 2: classify.
        classification = _classify_commit(message, diff_text, fast=fast)

        if classification["is_candidate"]:
            # Truncate message for output.
            first_line = message.split("\n", 1)[0][:200]
            candidates.append(
                {
                    "sha": commit["sha"][:12],
                    "full_sha": commit["sha"],
                    "date": commit["date"],
                    "author": commit["author"],
                    "message": first_line,
                    "url": commit["url"],
                    "message_score": classification["message_score"],
                    "diff_score": classification["diff_score"],
                    "total_score": classification["total_score"],
                    "classification": classification["classifications"],
                }
            )

    # Sort candidates by total_score descending, then date descending.
    candidates.sort(key=lambda c: (-c["total_score"], c["date"]), reverse=False)

    # Build classification summary.
    class_counts: dict[str, int] = {}
    for c in candidates:
        for cls in c["classification"]:
            class_counts[cls] = class_counts.get(cls, 0) + 1

    return {
        "repo": f"{owner}/{repo}",
        "scan_window": {"since": since, "until": until},
        "commits_scanned": len(commits),
        "commits_skipped_cve": skipped_cve,
        "candidates_found": len(candidates),
        "fast_mode": fast,
        "candidates": candidates,
        "summary": {
            "total_candidates": len(candidates),
            "classification_breakdown": class_counts,
            "top_classifications": sorted(class_counts, key=class_counts.get, reverse=True)[:5]  # type: ignore[arg-type]
            if class_counts
            else [],
        },
    }


# ---------------------------------------------------------------------------
# Strands TOOL_SPEC
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "detect_silent_patches",
    "description": (
        "Scans a GitHub repository's commit history for security-relevant "
        "fixes that were never assigned a CVE (silent patches). Uses a "
        "two-stage heuristic: commit message keyword scoring then diff "
        "keyword scoring. Each candidate is classified into one of 14 bug "
        "classes (e.g. auth_bypass, buffer_overflow, xss). "
        "Accepts owner/repo or full GitHub URL. Requires GITHUB_TOKEN for "
        "higher rate limits. Use --fast to skip diff analysis."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": ("Repository to scan in 'owner/repo' format or full GitHub URL."),
                },
                "since": {
                    "type": "string",
                    "description": ("Start date for commit scan in YYYY-MM-DD format (default: 90 days ago)."),
                },
                "until": {
                    "type": "string",
                    "description": ("End date for commit scan in YYYY-MM-DD format (default: today)."),
                },
                "max_commits": {
                    "type": "integer",
                    "description": "Hard limit on commits fetched (default: 500, max: 5000).",
                },
                "fast": {
                    "type": "boolean",
                    "description": ("If true, skip diff scoring (message keywords only). Faster but less accurate."),
                },
            },
            "required": ["repo"],
        }
    },
}


def detect_silent_patches(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for silent patch detection."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    repo_spec: str = tool_input.get("repo", "")
    if not repo_spec:
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Repository spec is required (e.g. 'owner/repo')."}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result

    try:
        _parse_repo_spec(repo_spec)
    except ValueError as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": str(exc)}],
        }
        log_tool_output_size("detect_silent_patches", result)
        return result

    since = tool_input.get("since")
    until_date = tool_input.get("until")
    max_commits = tool_input.get("max_commits", 500)
    fast = tool_input.get("fast", False)

    if not isinstance(max_commits, int):
        try:
            max_commits = int(max_commits)
        except (TypeError, ValueError):
            max_commits = 500

    payload = scan_silent_patches(
        repo_spec,
        since=since,
        until=until_date,
        max_commits=max_commits,
        fast=fast,
    )

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": payload}],
    }
    log_tool_output_size("detect_silent_patches", result)
    return result
