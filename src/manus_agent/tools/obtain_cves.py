from typing import Any

import requests
from strands.types.tools import ToolUse

from manus_agent.tools.get_nvd_data import _nvd_get_with_retry

# --- Strands Tool Definition ---
TOOL_SPEC = {
    "name": "obtain_cves",
    "description": "Discover, filter, and return high-impact vulnerabilities for a given time slice.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in ISO 8601 format: %Y-%m-%dT%H:%M:%S.000Z.",
                },
                "end_date": {"type": "string", "description": "End date in ISO 8601 format: %Y-%m-%dT%H:%M:%S.000Z."},
            },
            "required": ["start_date", "end_date"],
        }
    },
}

# --- Helper Functions for Data Fetching and Processing ---


def _get_all_cves_from_nvd(start_date, end_date):
    # Fetches all CVEs from the NVD API for a given date range and severity
    start = start_date
    end = end_date
    print(f"start_date:{start} to {end}")
    cves = []
    filtered_cves = []
    start_index = 0
    while True:
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?pubStartDate={start}&pubEndDate={end}&cvssV3Severity=HIGH&cvssV3Severity=CRITICAL&cvssV4Severity=HIGH&cvssV4Severity=CRITICAL&resultsPerPage=100&startIndex={start_index}"
        response = _nvd_get_with_retry(url)
        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])
        cves.extend(vulnerabilities)
        # filtered_cves.extend(_filter_cves_by_epss(vulnerabilities))
        total_results = data.get("totalResults", 0)
        start_index += len(vulnerabilities)
        if start_index >= total_results:
            break
    print(f"{len(filtered_cves)} / {len(cves)}")
    return cves


def _get_all_cves_from_github(start_date, end_date):
    """Fetches all CVEs from the GitHub Advisories API, handling pagination and rate limiting."""
    cves = []
    start = start_date.split("T")[0]
    end = end_date.split("T")[0]
    url = f"https://api.github.com/advisories?severity=high&severity=critical&published={start}..{end}"
    headers = {"Accept": "application/vnd.github+json"}

    while url:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Will raise an HTTPError for 4xx/5xx status codes
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch data from GitHub API: {e}. Skipping this source.")
            return []  # Return an empty list to allow the workflow to continue

        advisories = response.json()

        for adv in advisories:
            if adv.get("cve_id"):
                cves.append(
                    {
                        "cve": {
                            "id": adv["cve_id"],
                            "descriptions": [{"lang": "en", "value": adv.get("summary", "No description.")}],
                            "summary": adv.get("summary"),
                            "published": adv.get("published_at"),
                            "cvss_score": adv.get("cvss_severities"),
                        }
                    }
                )

        # Handle pagination
        if "next" in response.links:
            url = response.links["next"]["url"]
        else:
            url = None

    return cves


def _filter_cves_by_epss(cves):
    # Filters a list of CVEs based on EPSS score and percentile
    if not cves:
        return []
    cve_ids = [cve["cve"]["id"] for cve in cves]
    epss_url = f"https://api.first.org/data/v1/epss?cve={','.join(cve_ids)}"
    response = requests.get(epss_url)
    response.raise_for_status()

    epss_data = {item["cve"]: item for item in response.json().get("data", [])}
    print(f"epss: {len(epss_data.keys())}")
    filtered_cves = []
    for cve in cves:
        cve_id = cve["cve"]["id"]
        if cve_id in epss_data:
            epss_info = epss_data[cve_id]
            if float(epss_info.get("epss", 0)) > 0.05 or float(epss_info.get("percentile", 0)) > 0.5:
                cve["epss_data"] = epss_info
                filtered_cves.append(cve)
    return filtered_cves


def _enrich_with_cisa_kev(cves):
    # Enriches CVEs with CISA KEV information
    response = requests.get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
    response.raise_for_status()
    kev_data = response.json().get("vulnerabilities", [])
    kev_cves = {item["cveID"]: True for item in kev_data}

    for cve in cves:
        cve["cisa_kev"] = kev_cves.get(cve["cve"]["id"], False)
    return cves


def _submit_in_batches(cves):
    """Formats and submits the final, enriched data in batches."""
    url = "https://webhook.site/693c24d6-518f-48b7-8af5-e71563fabd5e"
    headers = {"Content-Type": "application/json"}

    for i in range(0, len(cves), 100):
        chunk = cves[i : i + 100]
        formatted_chunk = []
        for cve_data in chunk:
            cve_item = cve_data.get("cve", {})

            # CVSS V3 Metrics
            cvss_metrics_v3 = next((m for m in cve_item.get("metrics", {}).get("cvssMetricV31", [])), {})
            cvss_data = cvss_metrics_v3.get("cvssData", {})
            cvss_score_str = f"{cvss_data.get('baseSeverity', '')} ({cvss_data.get('baseScore', '')})"

            # CWE
            cwe_value = "N/A"
            if cve_item.get("weaknesses"):
                cve_item.get("weaknesses", [{}])[0].get("description", [{}])[0].get("value", "N/A")
            description = "No description available."
            if cve_item.get("descriptions"):
                description = next((d["value"] for d in cve_item["descriptions"] if d["lang"] == "en"), description)

            # Affected Products and CPEs
            affected_products_list = []
            cpe_list = []
            if cve_item.get("configurations"):
                for conf in cve_item.get("configurations", []):
                    for node in conf.get("nodes", []):
                        for cpe_match in node.get("cpeMatch", []):
                            if cpe_match.get("vulnerable", False):
                                uri = cpe_match.get("criteria", "")
                                cpe_list.append(uri)
                                try:
                                    parts = uri.split(":")
                                    affected_products_list.append(f"{parts[3]}:{parts[4]}")
                                except IndexError:
                                    continue

            formatted_cve = {
                "cve_id": cve_item.get("id", "N/A"),
                "cvss_score": cvss_score_str,
                "epss_score": cve_data.get("epss_data", {}).get("epss"),
                "epss_percentile": cve_data.get("epss_data", {}).get("percentile"),
                "cisa_kev": cve_data.get("cisa_kev", False),
                "exploited": cve_data.get("cisa_kev", False),  # Using KEV as a proxy
                "cwe": cwe_value,
                "cpe": ", ".join(cpe_list),
                "affected_products": ", ".join(list(set(affected_products_list))),
                "public_disclosure_date": cve_item.get("published"),
                "description": description,
            }
            response = requests.post(url=url, json=formatted_cve, headers=headers)
            response.raise_for_status()
            formatted_chunk.append(formatted_cve)

        print(f"Submitting chunk of {len(chunk)} formatted CVEs.")


def obtain_cves(tool: ToolUse, **kwargs: Any) -> dict:
    tool_use_id = tool["toolUseId"]
    start_date = tool["input"].get("start_date")
    end_date = tool["input"].get("end_date")

    try:
        # 1. Discover from both NVD and GitHub
        nvd_cves = _get_all_cves_from_nvd(start_date, end_date)
        github_cves = _get_all_cves_from_github(start_date, end_date)

        # Merge and de-duplicate results
        all_cves_dict = {cve["cve"]["id"]: cve for cve in nvd_cves}
        for cve in github_cves:
            if cve["cve"]["id"] not in all_cves_dict:
                all_cves_dict[cve["cve"]["id"]] = cve

        final_cves = list(all_cves_dict.values())
        print(f"{len(final_cves)} CVEs found")
        if not final_cves:
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "No new high/critical CVEs found."}],
            }

        filtered_cves = []
        for i in range(0, len(final_cves), 100):
            chunk = final_cves[i : i + 100]
            filtered_cves.extend(_filter_cves_by_epss(chunk))
        print(f"{len(filtered_cves)}/{len(final_cves)} CVEs with high EPSS")
        # 2. Enrich and Submit
        # enriched_cves = _enrich_with_cisa_kev(final_cves)
        # _submit_in_batches(filtered_cves)
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {
                    "text": f"{start_date}-{end_date}: Total {len(final_cves)} CVEs found. {len(filtered_cves)} CVEs with High EPSS found. High EPSS rate: ({len(filtered_cves)}/{len(final_cves)})"
                },
                {
                    "json": {
                        "cves_with_high_epss": filtered_cves,
                        "total_with_high_epss": len(filtered_cves),
                        "total_found": len(final_cves),
                    }
                },
            ],
        }
    except Exception as e:
        return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": f"An error occurred: {e}"}]}
