"""Probe vulnerability-intelligence API endpoints and report health.

Performs lightweight connectivity checks against every external API that the
manus-agent tool suite depends on.  Each probe issues a single, cheap HTTP
request (HEAD where possible, otherwise a minimal GET) and reports:

- reachability (HTTP status code)
- latency (milliseconds)
- rate-limit headroom (from ``X-RateLimit-Remaining`` / ``Retry-After``)
- API-key validity (for key-gated services like VulnCheck and NVD)

Designed to run from ``manus-agent doctor --apis`` and as a standalone Strands
tool (``check_api_health``).

All network I/O uses :mod:`concurrent.futures` so probes run in parallel and
the whole check completes in a few seconds even when one endpoint is slow.
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

# Each probe is a dict:
#   name        – human label
#   url         – endpoint to hit
#   method      – HTTP method (default GET)
#   headers_fn  – optional callable returning extra headers
#   params      – optional query params
#   key_env     – env-var name that provides an API key (if any)
#   key_header  – header name for the key
#   note        – extra context shown in the report

_DUMMY_CVE = "CVE-2021-44228"  # well-known CVE used for lightweight GETs


def _nvd_headers() -> dict[str, str]:
    key = os.environ.get("NVD_API_KEY", "")
    return {"apiKey": key} if key else {}


def _vulncheck_headers() -> dict[str, str]:
    key = os.environ.get("VULNCHECK_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "") or os.environ.get("MANUS_GITHUB_TOKEN", "")
    h: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


_PROBES: list[dict[str, Any]] = [
    {
        "name": "NVD (NIST)",
        "url": f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={_DUMMY_CVE}&resultsPerPage=1",
        "headers_fn": _nvd_headers,
        "key_env": "NVD_API_KEY",
        "note": "National Vulnerability Database — CVSS, CWE, CPE data",
    },
    {
        "name": "EPSS (FIRST.org)",
        "url": f"https://api.first.org/data/v1/epss?cve={_DUMMY_CVE}",
        "note": "Exploit Prediction Scoring System — no key required",
    },
    {
        "name": "OSV.dev",
        "url": "https://api.osv.dev/v1/vulns/GHSA-jfh8-c2jp-5v3q",
        "note": "Open Source Vulnerability database — no key required",
    },
    {
        "name": "CISA KEV",
        "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "method": "HEAD",
        "note": "Known Exploited Vulnerabilities catalog — no key required",
    },
    {
        "name": "GitHub Advisories",
        "url": f"https://api.github.com/advisories?cve_id={_DUMMY_CVE}&per_page=1",
        "headers_fn": _github_headers,
        "key_env": "GITHUB_TOKEN",
        "note": "GitHub Security Advisories — optional GITHUB_TOKEN for higher rate limits",
    },
    {
        "name": "VulnCheck KEV",
        "url": "https://api.vulncheck.com/v3/index/vulncheck-kev?cve=CVE-2021-44228&limit=1",
        "headers_fn": _vulncheck_headers,
        "key_env": "VULNCHECK_API_KEY",
        "note": "VulnCheck exploited-in-wild intelligence — requires VULNCHECK_API_KEY",
    },
]


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def _run_single_probe(probe: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """Execute one probe and return a result dict."""
    name = probe["name"]
    url = probe["url"]
    method = probe.get("method", "GET").upper()
    headers: dict[str, str] = {}
    if probe.get("headers_fn"):
        headers = probe["headers_fn"]()
    params = probe.get("params")
    key_env = probe.get("key_env")

    result: dict[str, Any] = {
        "name": name,
        "url": url,
        "note": probe.get("note", ""),
        "key_env": key_env,
        "key_set": bool(os.environ.get(key_env, "")) if key_env else None,
    }

    start = time.monotonic()
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            allow_redirects=True,
        )
        elapsed_ms = round((time.monotonic() - start) * 1000)

        result["status_code"] = resp.status_code
        result["latency_ms"] = elapsed_ms
        result["ok"] = resp.status_code < 400

        # Rate-limit headers (various conventions)
        rl_remaining = resp.headers.get("X-RateLimit-Remaining")
        rl_limit = resp.headers.get("X-RateLimit-Limit")
        rl_reset = resp.headers.get("X-RateLimit-Reset")
        retry_after = resp.headers.get("Retry-After")

        rate_limit: dict[str, Any] = {}
        if rl_remaining is not None:
            rate_limit["remaining"] = rl_remaining
        if rl_limit is not None:
            rate_limit["limit"] = rl_limit
        if rl_reset is not None:
            rate_limit["reset"] = rl_reset
        if retry_after is not None:
            rate_limit["retry_after"] = retry_after
            result["ok"] = False  # rate-limited => not healthy
            result["error"] = f"Rate-limited (Retry-After: {retry_after}s)"

        if rate_limit:
            result["rate_limit"] = rate_limit

        # Classify specific error codes
        if resp.status_code == 401:
            result["error"] = "Authentication failed — check API key"
        elif resp.status_code == 403:
            if key_env and not os.environ.get(key_env, ""):
                result["error"] = f"Forbidden — set {key_env} environment variable"
            else:
                result["error"] = "Forbidden — API key may be invalid or expired"
        elif resp.status_code == 429:
            result["error"] = "Rate-limited — too many requests"
        elif resp.status_code >= 500:
            result["error"] = f"Server error (HTTP {resp.status_code})"

    except requests.exceptions.ConnectTimeout:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        result["status_code"] = None
        result["latency_ms"] = elapsed_ms
        result["ok"] = False
        result["error"] = "Connection timed out"
    except requests.exceptions.ConnectionError as exc:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        result["status_code"] = None
        result["latency_ms"] = elapsed_ms
        result["ok"] = False
        result["error"] = f"Connection error: {_short_error(exc)}"
    except requests.exceptions.Timeout:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        result["status_code"] = None
        result["latency_ms"] = elapsed_ms
        result["ok"] = False
        result["error"] = "Request timed out"
    except requests.exceptions.RequestException as exc:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        result["status_code"] = None
        result["latency_ms"] = elapsed_ms
        result["ok"] = False
        result["error"] = f"Request failed: {_short_error(exc)}"

    return result


def _short_error(exc: Exception) -> str:
    """Return a concise, single-line error description."""
    msg = str(exc)
    # Strip nested exception chains for readability
    if ":" in msg:
        msg = msg.split(":")[-1].strip()
    return msg[:120]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_api_health(
    *,
    timeout: float = 10.0,
    probes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all API health probes in parallel and return a summary.

    Parameters
    ----------
    timeout:
        Per-probe HTTP timeout in seconds.
    probes:
        Override the default probe list (mainly for testing).

    Returns
    -------
    dict with keys:
        ``results``  – list of per-probe result dicts
        ``summary``  – {healthy: int, degraded: int, down: int, total: int}
        ``all_healthy`` – bool
    """
    probe_list = probes if probes is not None else _PROBES
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, len(probe_list))) as pool:
        futures = {pool.submit(_run_single_probe, p, timeout): p["name"] for p in probe_list}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover – defensive
                results.append(
                    {
                        "name": futures[future],
                        "ok": False,
                        "error": f"Probe crashed: {exc}",
                    }
                )

    # Sort by original probe order
    name_order = {p["name"]: i for i, p in enumerate(probe_list)}
    results.sort(key=lambda r: name_order.get(r["name"], 999))

    healthy = sum(1 for r in results if r.get("ok"))
    down = sum(1 for r in results if not r.get("ok"))

    return {
        "results": results,
        "summary": {
            "healthy": healthy,
            "down": down,
            "total": len(results),
        },
        "all_healthy": healthy == len(results),
    }


# ---------------------------------------------------------------------------
# Rich console output (used by `manus-agent doctor --apis`)
# ---------------------------------------------------------------------------


def print_api_health_report(report: dict[str, Any]) -> None:
    """Pretty-print the health report to the terminal using Rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:  # pragma: no cover
        # Fallback to plain text
        _print_plain(report)
        return

    console = Console()

    table = Table(
        title="API Health Check",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Status", width=6)
    table.add_column("API", style="cyan", min_width=20)
    table.add_column("HTTP", width=5)
    table.add_column("Latency", width=10)
    table.add_column("Rate Limit", width=14)
    table.add_column("Key", width=10)
    table.add_column("Notes", style="dim")

    for r in report["results"]:
        ok = r.get("ok", False)
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"

        http_code = str(r.get("status_code", "-"))
        latency = f"{r.get('latency_ms', '?')} ms"

        # Rate limit info
        rl = r.get("rate_limit", {})
        if rl.get("remaining") is not None and rl.get("limit") is not None:
            rl_text = f"{rl['remaining']}/{rl['limit']}"
        elif rl.get("retry_after") is not None:
            rl_text = f"[red]retry {rl['retry_after']}s[/red]"
        else:
            rl_text = "-"

        # Key status
        key_env = r.get("key_env")
        if key_env is None:
            key_text = "[dim]n/a[/dim]"
        elif r.get("key_set"):
            key_text = "[green]set[/green]"
        else:
            key_text = "[yellow]unset[/yellow]"

        notes = r.get("error", r.get("note", ""))
        # Truncate long notes
        if len(notes) > 50:
            notes = notes[:47] + "..."

        table.add_row(status, r["name"], http_code, latency, rl_text, key_text, notes)

    console.print(table)

    summary = report["summary"]
    if report["all_healthy"]:
        console.print(
            Panel(
                f"[bold green]All {summary['total']} APIs reachable![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]{summary['down']}/{summary['total']} API(s) unreachable or degraded[/bold red]",
                border_style="red",
            )
        )


def _print_plain(report: dict[str, Any]) -> None:
    """Fallback plain-text output when Rich is not available."""
    print("\n=== API Health Check ===\n")
    for r in report["results"]:
        ok = "OK" if r.get("ok") else "FAIL"
        code = r.get("status_code", "-")
        latency = r.get("latency_ms", "?")
        error = r.get("error", "")
        print(f"  [{ok}] {r['name']}  HTTP {code}  {latency} ms  {error}")

    summary = report["summary"]
    print(f"\n  {summary['healthy']}/{summary['total']} healthy")


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

TOOL_SPEC: dict[str, Any] = {
    "name": "check_api_health",
    "description": (
        "Probe all external vulnerability-intelligence APIs (NVD, EPSS, "
        "OSV.dev, CISA KEV, GitHub Advisories, VulnCheck) and report "
        "connectivity, latency, rate-limit headroom, and API-key validity. "
        "Use this to diagnose why a tool is failing or before running a "
        "long analysis pipeline."
    ),
    "inputSchema": {
        "type": "object",
        "json": {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Per-probe HTTP timeout in seconds (default 10).",
                    "default": 10,
                },
                "output": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format (default text).",
                    "default": "text",
                },
            },
            "required": [],
        },
    },
}


def check_api_health_tool(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for ``check_api_health``."""
    tool_use_id = tool.get("toolUseId", "")
    tool_input = tool.get("input", {})

    timeout = tool_input.get("timeout", 10)
    output_fmt = tool_input.get("output", "text")

    try:
        report = check_api_health(timeout=timeout)

        if output_fmt == "json":
            text = _json.dumps(report, indent=2, default=str)
        else:
            lines = _format_text_report(report)
            text = "\n".join(lines)

        return {
            "toolUseId": tool_use_id,
            "status": "success" if report["all_healthy"] else "error",
            "content": [{"text": text}],
        }
    except Exception as exc:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Health check failed: {exc}"}],
        }


def _format_text_report(report: dict[str, Any]) -> list[str]:
    """Format the report as plain-text lines for the Strands tool output."""
    lines: list[str] = ["API Health Check", "=" * 40]

    for r in report["results"]:
        ok = "✓" if r.get("ok") else "✗"
        code = r.get("status_code", "-")
        latency = r.get("latency_ms", "?")
        error = r.get("error", "")

        line = f"  {ok} {r['name']:20s}  HTTP {code!s:>4s}  {latency!s:>5s} ms"

        key_env = r.get("key_env")
        if key_env is not None:
            key_status = "set" if r.get("key_set") else "UNSET"
            line += f"  key:{key_status}"

        if error:
            line += f"  — {error}"

        lines.append(line)

    summary = report["summary"]
    lines.append("")
    lines.append(f"  {summary['healthy']}/{summary['total']} APIs healthy")
    return lines
