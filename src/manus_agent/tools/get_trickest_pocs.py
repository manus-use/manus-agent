"""
Tool: get_trickest_pocs

Fetches known PoC links for a CVE from the trickest/cve repository
(https://github.com/trickest/cve), a continuously auto-updated index of
250k+ CVEs with PoC links sourced from CVE references, GitHub search, and
HackerOne reports.

This is a fast pre-flight PoC lookup — returns results in milliseconds with
no rate limits, complementing the slower ExploitDB/Packetstorm/GitHub-search tools.
"""

from __future__ import annotations

import re

from strands import tool

from manus_agent.utils.cve_utils import extract_cve_year, validate_cve_id

__all__ = ["get_trickest_pocs"]

_RAW_URL = "https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md"
_CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-\d+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


def _parse_pocs(markdown: str) -> dict:
    """Parse the trickest markdown file into structured PoC data."""
    result = {
        "description": "",
        "reference_pocs": [],
        "github_pocs": [],
        "all_pocs": [],
    }

    lines = markdown.splitlines()
    section = None
    subsection = None
    desc_lines = []
    in_description = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("### Description"):
            in_description = True
            section = "description"
            continue
        elif stripped.startswith("### POC"):
            in_description = False
            section = "poc"
            continue
        elif stripped.startswith("#### Reference") and section == "poc":
            subsection = "reference"
            continue
        elif stripped.startswith("#### Github") and section == "poc":
            subsection = "github"
            continue
        elif stripped.startswith("###"):
            in_description = False
            section = None
            subsection = None
            continue

        if section == "description" and in_description and stripped:
            desc_lines.append(stripped)

        if section == "poc":
            urls = _URL_RE.findall(stripped)
            for url in urls:
                url = url.rstrip(")")
                if subsection == "reference":
                    result["reference_pocs"].append(url)
                elif subsection == "github":
                    result["github_pocs"].append(url)
                result["all_pocs"].append(url)

    result["description"] = " ".join(desc_lines)
    # Deduplicate while preserving order
    seen: set = set()
    deduped = []
    for url in result["all_pocs"]:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    result["all_pocs"] = deduped

    return result


@tool
def get_trickest_pocs(cve_id: str) -> str:
    """Fetch known PoC links for a CVE from the trickest/cve repository.

    trickest/cve is a continuously auto-updated index of 250k+ CVEs with PoC
    links sourced from CVE references, GitHub search, and HackerOne reports.
    This is a fast, rate-limit-free alternative/complement to ExploitDB and
    Packetstorm searches.

    Args:
        cve_id: CVE identifier, e.g. "CVE-2025-6554" or "cve-2024-3094".

    Returns:
        A formatted string with the CVE description and all known PoC URLs,
        or a message indicating no entry was found.
    """
    import urllib.error
    import urllib.request

    cve_id = cve_id.strip().upper()

    if (err_msg := validate_cve_id(cve_id)) is not None:
        return err_msg

    year = str(extract_cve_year(cve_id))
    url = _RAW_URL.format(year=year, cve_id=cve_id)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "manus-use/trickest-cve-tool"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            markdown = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return f"No trickest/cve entry found for {cve_id}. The CVE may be too new or not yet indexed."
        return f"HTTP error fetching trickest/cve data for {cve_id}: {exc.code} {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching trickest/cve data for {cve_id}: {exc}"

    data = _parse_pocs(markdown)

    if not data["all_pocs"]:
        return (
            f"trickest/cve entry found for {cve_id}, but no PoC links listed.\n\n"
            f"Description: {data['description'] or 'N/A'}\n\n"
            f"Source: {url}"
        )

    lines = [
        f"## trickest/cve PoC Report: {cve_id}",
        f"Source: {url}",
        "",
    ]

    if data["description"]:
        lines += [f"**Description:** {data['description']}", ""]

    lines.append(f"**Total PoC links found:** {len(data['all_pocs'])}")

    if data["reference_pocs"]:
        lines += ["", f"### Reference PoCs ({len(data['reference_pocs'])})"]
        for poc in data["reference_pocs"]:
            lines.append(f"- {poc}")

    if data["github_pocs"]:
        lines += ["", f"### GitHub PoCs ({len(data['github_pocs'])})"]
        for poc in data["github_pocs"]:
            lines.append(f"- {poc}")

    return "\n".join(lines)
