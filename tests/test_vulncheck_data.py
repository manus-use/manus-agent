"""Tests for get_vulncheck_data, track_vendor_response, and VI agent wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(cve_id: str = "CVE-2024-3094", tool_use_id: str = "test-001") -> dict:
    return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}


def _make_kev_response(cve_id: str = "CVE-2024-3094") -> dict:
    """Minimal VulnCheck KEV API response payload with a hit."""
    return {
        "data": [
            {
                "cveID": cve_id,
                "dateAdded": "2024-03-29",
                "sources": ["CISA KEV", "FBI Flash", "MS-ISAC"],
                "ransomwareUse": False,
                "notes": "Supply-chain backdoor in XZ Utils.",
            }
        ],
        "_meta": {"timestamp": "2024-04-01T00:00:00Z"},
    }


def _make_kev_empty_response() -> dict:
    """VulnCheck KEV API response with no data (CVE not in KEV)."""
    return {"data": [], "_meta": {}}


def _make_nvd2_response(cve_id: str = "CVE-2024-3094") -> dict:
    """Minimal VulnCheck NVD2 API response."""
    return {
        "data": [
            {
                "id": cve_id,
                "published": "2024-03-29T00:00:00.000",
                "lastModified": "2024-04-01T00:00:00.000",
                "descriptions": [{"lang": "en", "value": "Supply-chain attack in xz-utils."}],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "baseScore": 10.0,
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                                "baseSeverity": "CRITICAL",
                            }
                        }
                    ]
                },
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:tukaani:xz_utils:5.6.0:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:tukaani:xz_utils:5.6.1:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ],
            }
        ]
    }


def _make_requests_response(payload: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response for the given payload."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _make_requests_error_response() -> MagicMock:
    """Mock that raises requests.exceptions.RequestException on raise_for_status."""
    import requests

    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.ConnectionError("network failure")
    return resp


# ===========================================================================
# get_vulncheck_data — no API key (graceful degradation)
# ===========================================================================


def test_no_api_key_returns_available_false(monkeypatch):
    monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())

    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["available"] is False
    assert payload["kev"]["in_kev"] is False
    assert payload["nvd2"]["cvss_v3_score"] is None
    assert "VULNCHECK_API_KEY" in payload["error"]


def test_no_api_key_includes_cve_id(monkeypatch):
    monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use("CVE-2021-44228"))
    payload = result["content"][0]["json"]
    assert payload["cve_id"] == "CVE-2021-44228"


def test_no_api_key_kev_sources_empty(monkeypatch):
    monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["kev"]["sources"] == []


def test_no_api_key_cpe_matches_empty(monkeypatch):
    monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["nvd2"]["cpe_matches"] == []


# ===========================================================================
# get_vulncheck_data — invalid CVE ID
# ===========================================================================


def test_invalid_cve_id_returns_error(monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data({"toolUseId": "x", "input": {"cve_id": "INVALID"}})
    assert result["status"] == "error"
    assert "Invalid CVE ID" in result["content"][0]["text"]


def test_missing_cve_id_returns_error(monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data({"toolUseId": "x", "input": {}})
    assert result["status"] == "error"


# ===========================================================================
# get_vulncheck_data — valid key + KEV hit
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_hit_in_kev_true(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["available"] is True
    assert payload["kev"]["in_kev"] is True


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_hit_sources_populated(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert "CISA KEV" in payload["kev"]["sources"]
    assert "FBI Flash" in payload["kev"]["sources"]


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_hit_date_added_parsed(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["kev"]["date_added"] == "2024-03-29"


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_hit_no_error_field(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["error"] is None


# ===========================================================================
# get_vulncheck_data — ransomware flag
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_ransomware_flag_parsed_true(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    kev_payload = {
        "data": [
            {
                "cveID": "CVE-2024-3094",
                "dateAdded": "2024-03-29",
                "sources": ["CISA KEV"],
                "ransomwareUse": True,
            }
        ]
    }
    mock_get.side_effect = [
        _make_requests_response(kev_payload),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["kev"]["ransomware_use"] is True


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_ransomware_flag_parsed_false(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["kev"]["ransomware_use"] is False


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_ransomware_known_campaign_use_field(mock_get, monkeypatch):
    """knownRansomwareCampaignUse field should set ransomware_use=True."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    kev_payload = {
        "data": [
            {
                "cveID": "CVE-2024-3094",
                "knownRansomwareCampaignUse": True,
                "sources": [],
            }
        ]
    }
    mock_get.side_effect = [
        _make_requests_response(kev_payload),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["kev"]["ransomware_use"] is True


# ===========================================================================
# get_vulncheck_data — no KEV match
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_no_kev_match_in_kev_false(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_empty_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use("CVE-2021-99999"))
    payload = result["content"][0]["json"]
    assert payload["kev"]["in_kev"] is False


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_no_kev_match_sources_empty(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_empty_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use("CVE-2021-99999"))
    payload = result["content"][0]["json"]
    assert payload["kev"]["sources"] == []


# ===========================================================================
# get_vulncheck_data — NVD2 enriched CPE extraction
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_nvd2_cpe_matches_extracted(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    cpe_matches = payload["nvd2"]["cpe_matches"]
    assert len(cpe_matches) == 2
    assert "cpe:2.3:a:tukaani:xz_utils:5.6.0:*:*:*:*:*:*:*" in cpe_matches


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_nvd2_cvss_score_extracted(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["nvd2"]["cvss_v3_score"] == 10.0
    assert payload["nvd2"]["cvss_v3_severity"] == "CRITICAL"


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_nvd2_description_extracted(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_response()),
        _make_requests_response(_make_nvd2_response()),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert "xz" in payload["nvd2"]["description"].lower()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_nvd2_empty_data_returns_none_fields(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_empty_response()),
        _make_requests_response({"data": []}),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use("CVE-2021-99999"))
    payload = result["content"][0]["json"]
    assert payload["nvd2"]["cvss_v3_score"] is None
    assert payload["nvd2"]["cpe_matches"] == []


# ===========================================================================
# get_vulncheck_data — network / API errors
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_network_failure_sets_error(mock_get, monkeypatch):
    import requests as req_lib

    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = req_lib.exceptions.ConnectionError("timeout")
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["error"] is not None
    assert "KEV" in payload["error"] or "request" in payload["error"].lower()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_failure_still_tries_nvd2(mock_get, monkeypatch):
    """Even if KEV fails (after all retries), NVD2 should still be attempted."""
    import requests as req_lib

    import manus_agent.tools.get_vulncheck_data as vc_mod

    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    # Disable retries for this test so the KEV error propagates immediately.
    orig_max = vc_mod._VC_MAX_RETRIES
    orig_delay = vc_mod._VC_RETRY_BASE_DELAY
    try:
        vc_mod._VC_MAX_RETRIES = 1
        vc_mod._VC_RETRY_BASE_DELAY = 0.0
        # First call (KEV) → error; second call (NVD2) → success
        mock_get.side_effect = [
            req_lib.exceptions.ConnectionError("timeout"),
            _make_requests_response(_make_nvd2_response()),
        ]
        from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

        result = get_vulncheck_data(_make_tool_use())
    finally:
        vc_mod._VC_MAX_RETRIES = orig_max
        vc_mod._VC_RETRY_BASE_DELAY = orig_delay
    payload = result["content"][0]["json"]
    # NVD2 should have populated CPE matches
    assert len(payload["nvd2"]["cpe_matches"]) > 0


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_both_endpoints_fail_error_combined(mock_get, monkeypatch):
    """Both failures should produce a combined error string."""
    import requests as req_lib

    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = req_lib.exceptions.ConnectionError("timeout")
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["available"] is True  # key was present; enrichment attempted
    assert payload["error"] is not None


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_http_401_error_propagated(mock_get, monkeypatch):
    import requests as req_lib

    monkeypatch.setenv("VULNCHECK_API_KEY", "invalid-key")
    http_err = req_lib.exceptions.HTTPError("401 Unauthorized")
    mock_get.side_effect = http_err
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["error"] is not None


# ===========================================================================
# get_vulncheck_data — cve_id normalisation
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_cve_id_uppercased(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    mock_get.side_effect = [
        _make_requests_response(_make_kev_empty_response()),
        _make_requests_response({"data": []}),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data({"toolUseId": "t", "input": {"cve_id": "cve-2024-3094"}})
    payload = result["content"][0]["json"]
    assert payload["cve_id"] == "CVE-2024-3094"


# ===========================================================================
# get_vulncheck_data — KEV sources from alternative field names
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_kev_reported_by_field_used_as_sources(mock_get, monkeypatch):
    """reportedBy (alternative field) should populate kev.sources."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    kev_payload = {
        "data": [
            {
                "cveID": "CVE-2024-3094",
                "reportedBy": ["NVD", "CERT-EU"],
            }
        ]
    }
    mock_get.side_effect = [
        _make_requests_response(kev_payload),
        _make_requests_response({"data": []}),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())
    payload = result["content"][0]["json"]
    assert "NVD" in payload["kev"]["sources"]


# ===========================================================================
# get_vulncheck_data — NVD2 cvssMetricV30 fallback
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_nvd2_cvss_v30_fallback(mock_get, monkeypatch):
    """Should parse cvssMetricV30 when cvssMetricV31 is absent."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    nvd2_payload = {
        "data": [
            {
                "id": "CVE-2021-44228",
                "descriptions": [],
                "metrics": {
                    "cvssMetricV30": [
                        {
                            "cvssData": {
                                "baseScore": 9.8,
                                "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                                "baseSeverity": "CRITICAL",
                            }
                        }
                    ]
                },
                "configurations": [],
            }
        ]
    }
    mock_get.side_effect = [
        _make_requests_response(_make_kev_empty_response()),
        _make_requests_response(nvd2_payload),
    ]
    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use("CVE-2021-44228"))
    payload = result["content"][0]["json"]
    assert payload["nvd2"]["cvss_v3_score"] == 9.8


# ===========================================================================
# VI agent tool list includes get_vulncheck_data
# ===========================================================================


def test_vi_agent_imports_get_vulncheck_data():
    """get_vulncheck_data module must be importable."""
    from manus_agent.tools.get_vulncheck_data import TOOL_SPEC, get_vulncheck_data

    assert callable(get_vulncheck_data)
    assert TOOL_SPEC["name"] == "get_vulncheck_data"


def test_vi_agent_system_prompt_mentions_vulncheck():
    """The VI agent system prompt should reference VulnCheck KEV."""
    from manus_agent.agents.vi_agent import SYSTEM_PROMPT

    assert "get_vulncheck_data" in SYSTEM_PROMPT
    assert "VulnCheck" in SYSTEM_PROMPT


def test_vi_agent_system_prompt_mentions_actively_exploited():
    """System prompt should instruct agent to flag active exploitation."""
    from manus_agent.agents.vi_agent import SYSTEM_PROMPT

    assert "ACTIVELY EXPLOITED" in SYSTEM_PROMPT


def test_vi_agent_system_prompt_mentions_ransomware():
    """System prompt should instruct agent to surface ransomware association."""
    from manus_agent.agents.vi_agent import SYSTEM_PROMPT

    assert "RANSOMWARE" in SYSTEM_PROMPT or "ransomware" in SYSTEM_PROMPT


def test_vi_agent_module_imports_without_strands():
    """vi_agent module should import cleanly without strands installed."""
    import manus_agent.agents.vi_agent as vi

    assert hasattr(vi, "VulnerabilityIntelligenceAgent")
    assert hasattr(vi, "SYSTEM_PROMPT")


# ===========================================================================
# track_vendor_response — basic structure
# ===========================================================================


def test_track_vendor_response_module_importable():
    from manus_agent.tools.track_vendor_response import TOOL_SPEC, track_vendor_response

    assert callable(track_vendor_response)
    assert TOOL_SPEC["name"] == "track_vendor_response"


def test_track_vendor_response_invalid_cve():
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response({"toolUseId": "x", "input": {"cve_id": "NOT-A-CVE"}})
    assert result["status"] == "error"


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_unknown_state_when_no_data(mock_vc, mock_cisa, mock_nvd):
    mock_nvd.return_value = []
    mock_cisa.return_value = {}
    mock_vc.return_value = {}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["vendor_response_state"] == "unknown"
    assert "cve_id" in payload


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_patch_tag_gives_patch_available(mock_vc, mock_cisa, mock_nvd):
    mock_nvd.return_value = [{"tags": ["Patch", "Vendor Advisory"], "url": "https://example.com/patch"}]
    mock_cisa.return_value = {}
    mock_vc.return_value = {}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["vendor_response_state"] == "patch_available"


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_vulncheck_kev_hit_elevates_state(mock_vc, mock_cisa, mock_nvd):
    """VulnCheck KEV hit should elevate unknown → investigating."""
    mock_nvd.return_value = []
    mock_cisa.return_value = {}
    mock_vc.return_value = {"cveID": "CVE-2024-3094", "sources": ["FBI Flash"]}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    # Should be elevated from unknown to at least investigating.
    assert payload["vendor_response_state"] != "unknown"
    assert payload["signals"]["vulncheck_kev_hit"] is True


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_vulncheck_kev_increases_confidence(mock_vc, mock_cisa, mock_nvd):
    """VulnCheck KEV hit should increase confidence above baseline."""
    mock_nvd.return_value = []
    mock_cisa.return_value = {}
    mock_vc.return_value = {"cveID": "CVE-2024-3094"}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["confidence"] > 0.2  # above the absolute minimum


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_cisa_kev_update_action(mock_vc, mock_cisa, mock_nvd):
    """CISA KEV with 'Apply update' action should yield patch_available."""
    mock_nvd.return_value = []
    mock_cisa.return_value = {
        "cveID": "CVE-2024-3094",
        "requiredAction": "Apply update per vendor instructions.",
        "shortDescription": "Actively exploited.",
    }
    mock_vc.return_value = {}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["vendor_response_state"] == "patch_available"
    assert payload["signals"]["cisa_kev_hit"] is True


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_evidence_list_populated(mock_vc, mock_cisa, mock_nvd):
    mock_nvd.return_value = [{"tags": ["Patch"], "url": "https://example.com"}]
    mock_cisa.return_value = {}
    mock_vc.return_value = {}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert isinstance(payload["evidence"], list)
    assert len(payload["evidence"]) > 0


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_state_always_valid(mock_vc, mock_cisa, mock_nvd):
    """Returned state must always be one of the 6 valid values."""
    mock_nvd.return_value = []
    mock_cisa.return_value = {}
    mock_vc.return_value = {}
    from manus_agent.tools.track_vendor_response import _VALID_STATES, track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    assert payload["vendor_response_state"] in _VALID_STATES


@patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
@patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
@patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
def test_track_vendor_response_ransomware_in_evidence(mock_vc, mock_cisa, mock_nvd):
    """Ransomware signal from VulnCheck should appear in evidence list."""
    mock_nvd.return_value = []
    mock_cisa.return_value = {}
    mock_vc.return_value = {"cveID": "CVE-2024-3094", "ransomwareUse": True}
    from manus_agent.tools.track_vendor_response import track_vendor_response

    result = track_vendor_response(_make_tool_use())
    payload = result["content"][0]["json"]
    evidence_text = " ".join(payload["evidence"]).lower()
    assert "ransomware" in evidence_text
