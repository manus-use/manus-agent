"""
Comprehensive tests for four foundation tools that power every VA pipeline run:

  - check_cisa_kev   (CISA KEV catalog lookup with caching)
  - get_nvd_data     (NVD API enrichment)
  - get_cwe_details  (CWE page scraper)
  - get_otx_cve_details (AlienVault OTX threat intel)

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across all tests
# ---------------------------------------------------------------------------


def _make_tool_use(
    tool_name: str = "check_cisa_kev",
    input_data: dict | None = None,
    tool_use_id: str = "tu-001",
) -> dict:
    return {"toolUseId": tool_use_id, "input": input_data or {}}


def _cve_tool_use(
    cve_id: str = "CVE-2024-3094",
    tool_use_id: str = "tu-001",
) -> dict:
    return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}


def _cwe_tool_use(
    cwe_id: str = "CWE-79",
    tool_use_id: str = "tu-001",
) -> dict:
    return {"toolUseId": tool_use_id, "input": {"cwe_id": cwe_id}}


def _result_status(result: dict) -> str:
    return result.get("status", "")


def _result_json(result: dict) -> dict:
    content = result.get("content", [])
    for block in content:
        if "json" in block:
            return block["json"]
    return {}


def _result_text(result: dict) -> str:
    content = result.get("content", [])
    for block in content:
        if "text" in block:
            return block["text"]
    return ""


# ---------------------------------------------------------------------------
# Fixtures — KEV payload
# ---------------------------------------------------------------------------

_FULL_KEV_CATALOG = {
    "title": "CISA Known Exploited Vulnerabilities Catalog",
    "catalogVersion": "2024.04.01",
    "dateReleased": "2024-04-01T00:00:00Z",
    "count": 2,
    "vulnerabilities": [
        {
            "cveID": "CVE-2024-3094",
            "vendorProject": "XZ Utils",
            "product": "XZ Utils",
            "vulnerabilityName": "XZ Utils Supply Chain Backdoor",
            "dateAdded": "2024-03-29",
            "shortDescription": "Supply-chain backdoor.",
            "requiredAction": "Apply updates.",
            "dueDate": "2024-04-12",
            "notes": "",
        },
        {
            "cveID": "CVE-2021-44228",
            "vendorProject": "Apache",
            "product": "Log4j2",
            "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "dateAdded": "2021-12-10",
            "shortDescription": "Log4Shell RCE.",
            "requiredAction": "Apply updates.",
            "dueDate": "2021-12-24",
            "notes": "",
        },
    ],
}

_EMPTY_KEV_CATALOG = {
    "title": "CISA Known Exploited Vulnerabilities Catalog",
    "catalogVersion": "2024.01.01",
    "count": 0,
    "vulnerabilities": [],
}


# ===========================================================================
# check_cisa_kev tests
# ===========================================================================


class TestCheckCisaKevImport:
    def test_module_importable(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        assert callable(check_cisa_kev)

    def test_tool_spec_present(self):
        from manus_agent.tools.check_cisa_kev import TOOL_SPEC

        assert TOOL_SPEC["name"] == "check_cisa_kev"
        assert "inputSchema" in TOOL_SPEC
        assert "cve_id" in TOOL_SPEC["inputSchema"]["json"]["properties"]

    def test_exported_from_tools_package(self):
        from manus_agent.tools import check_cisa_kev as mod

        assert mod is not None


class TestCheckCisaKevInputValidation:
    def test_empty_string_returns_error(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        result = check_cisa_kev(_cve_tool_use(""))
        assert _result_status(result) == "error"
        assert "Invalid" in _result_text(result)

    def test_whitespace_string_returns_error(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        result = check_cisa_kev(_cve_tool_use("   "))
        assert _result_status(result) == "error"

    def test_non_string_cve_id_returns_error(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        result = check_cisa_kev(_make_tool_use(input_data={"cve_id": 12345}))
        assert _result_status(result) == "error"

    def test_none_cve_id_returns_error(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        result = check_cisa_kev(_make_tool_use(input_data={"cve_id": None}))
        assert _result_status(result) == "error"

    def test_missing_cve_id_key_returns_error(self):
        from manus_agent.tools.check_cisa_kev import check_cisa_kev

        result = check_cisa_kev(_make_tool_use(input_data={}))
        assert _result_status(result) == "error"


class TestCheckCisaKevFound:
    def _mock_requests_get(self, catalog: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = catalog
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_cve_in_kev_returns_exploited_true(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"
        data = _result_json(result)
        assert data["exploited"] is True
        assert data["details"]["cveID"] == "CVE-2024-3094"

    def test_cve_in_kev_summary_mentions_critical(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        summary = _result_json(result).get("summary", "")
        assert "CRITICAL" in summary or "CISA KEV" in summary

    def test_cve_lookup_is_case_insensitive(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("cve-2024-3094"))

        assert _result_status(result) == "success"
        assert _result_json(result)["exploited"] is True

    def test_second_cve_in_catalog_found(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2021-44228"))

        assert _result_json(result)["exploited"] is True

    def test_cve_not_in_kev_returns_exploited_false(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-1999-0001"))

        assert _result_status(result) == "success"
        assert _result_json(result)["exploited"] is False

    def test_cve_not_found_summary_mentions_not_found(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-1999-0001"))

        summary = _result_json(result).get("summary", "")
        assert "not found" in summary.lower() or "was not found" in summary.lower()

    def test_tool_use_id_preserved_in_result(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094", tool_use_id="unique-id-42"))

        assert result["toolUseId"] == "unique-id-42"


class TestCheckCisaKevCaching:
    def _mock_requests_get(self, catalog: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = catalog
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_cache_miss_calls_requests_get(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        cache_file = tmp_path / "kev_cache.json"
        with patch.object(mod, "CACHE_FILE", cache_file):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)) as mock_get:
                mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))
                mock_get.assert_called_once()

    def test_cache_hit_does_not_call_requests_get(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        cache_file = tmp_path / "kev_cache.json"
        # Pre-populate cache with a fresh timestamp
        cache_file.write_text(json.dumps({"timestamp": time.time(), "data": _FULL_KEV_CATALOG}))
        with patch.object(mod, "CACHE_FILE", cache_file):
            with patch("requests.get") as mock_get:
                mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))
                mock_get.assert_not_called()

    def test_expired_cache_triggers_refresh(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        cache_file = tmp_path / "kev_cache.json"
        # Write a stale cache (timestamp = epoch 0)
        cache_file.write_text(json.dumps({"timestamp": 0, "data": _FULL_KEV_CATALOG}))
        with patch.object(mod, "CACHE_FILE", cache_file):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)) as mock_get:
                mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))
                mock_get.assert_called_once()

    def test_fresh_cache_is_written_after_fetch(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        cache_file = tmp_path / "kev_cache.json"
        with patch.object(mod, "CACHE_FILE", cache_file):
            with patch("requests.get", return_value=self._mock_requests_get(_FULL_KEV_CATALOG)):
                mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert "timestamp" in cached
        assert "data" in cached


class TestCheckCisaKevErrorHandling:
    def test_network_failure_returns_error(self, tmp_path):
        import requests as _requests

        from manus_agent.tools import check_cisa_kev as mod

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            # Use RequestException so _get_kev_data catches it and returns {}
            with patch("requests.get", side_effect=_requests.exceptions.RequestException("Network down")):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        # _get_kev_data returns {} → tool returns error for missing/empty data
        assert _result_status(result) == "error"

    def test_empty_catalog_no_vulnerabilities_key_returns_error(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        bad_catalog = {"title": "test", "count": 0}  # missing "vulnerabilities" key
        mock_resp = MagicMock()
        mock_resp.json.return_value = bad_catalog
        mock_resp.raise_for_status.return_value = None

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=mock_resp):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"

    def test_empty_vulnerabilities_list_cve_not_found(self, tmp_path):
        from manus_agent.tools import check_cisa_kev as mod

        mock_resp = MagicMock()
        mock_resp.json.return_value = _EMPTY_KEV_CATALOG
        mock_resp.raise_for_status.return_value = None

        with patch.object(mod, "CACHE_FILE", tmp_path / "kev_cache.json"):
            with patch("requests.get", return_value=mock_resp):
                result = mod.check_cisa_kev(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"
        assert _result_json(result)["exploited"] is False


# ===========================================================================
# get_nvd_data tests
# ===========================================================================


def _make_nvd_response(cve_id: str = "CVE-2024-3094") -> dict:
    """Minimal NVD API v2 response containing one vulnerability."""
    return {
        "resultsPerPage": 1,
        "startIndex": 0,
        "totalResults": 1,
        "format": "NVD_CVE",
        "version": "2.0",
        "timestamp": "2024-04-01T00:00:00.000",
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "sourceIdentifier": "cve@mitre.org",
                    "published": "2024-03-29T17:15:08.833",
                    "lastModified": "2024-04-01T11:15:08.983",
                    "vulnStatus": "Analyzed",
                    "descriptions": [{"lang": "en", "value": "Supply-chain backdoor in XZ Utils liblzma."}],
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "source": "nvd@nist.gov",
                                "type": "Primary",
                                "cvssData": {
                                    "version": "3.1",
                                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                                    "baseScore": 10.0,
                                    "baseSeverity": "CRITICAL",
                                },
                                "exploitabilityScore": 3.9,
                                "impactScore": 6.0,
                            }
                        ]
                    },
                    "weaknesses": [
                        {
                            "source": "nvd@nist.gov",
                            "type": "Primary",
                            "description": [{"lang": "en", "value": "CWE-506"}],
                        }
                    ],
                    "references": [{"url": "https://example.com/advisory", "source": "cve@mitre.org"}],
                }
            }
        ],
    }


class TestGetNvdDataImport:
    def test_module_importable(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        assert callable(get_nvd_data)

    def test_tool_spec_name(self):
        from manus_agent.tools.get_nvd_data import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_nvd_data"

    def test_tool_spec_has_cve_id_input(self):
        from manus_agent.tools.get_nvd_data import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cve_id" in props


class TestGetNvdDataInputValidation:
    def test_invalid_format_no_cve_prefix_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        result = get_nvd_data(_cve_tool_use("12345-6789"))
        assert _result_status(result) == "error"
        assert "Invalid" in _result_text(result)

    def test_non_string_cve_id_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        result = get_nvd_data(_make_tool_use(input_data={"cve_id": 99999}))
        assert _result_status(result) == "error"

    def test_empty_cve_id_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        result = get_nvd_data(_cve_tool_use(""))
        assert _result_status(result) == "error"

    def test_none_cve_id_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        result = get_nvd_data(_make_tool_use(input_data={"cve_id": None}))
        assert _result_status(result) == "error"


class TestGetNvdDataSuccess:
    def _mock_get(self, payload: dict):
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_successful_lookup_returns_success(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"

    def test_vulnerability_data_in_result(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        data = _result_json(result)
        assert "cve" in data
        assert data["cve"]["id"] == "CVE-2024-3094"

    def test_cve_id_uppercased_in_request(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())) as mock_get:
            get_nvd_data(_cve_tool_use("cve-2024-3094"))

        call_url = mock_get.call_args[0][0]
        assert "CVE-2024-3094" in call_url

    def test_cisa_kev_info_added_to_result(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        data = _result_json(result)
        assert "cisa_kev_info" in data

    def test_cisa_kev_not_in_kev_by_default(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        data = _result_json(result)
        # No cisaExploitAdd in vulnStatus → is_in_kev should be False
        assert data["cisa_kev_info"]["is_in_kev"] is False

    def test_tool_use_id_preserved(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", return_value=self._mock_get(_make_nvd_response())):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094", tool_use_id="nvd-test-99"))

        assert result["toolUseId"] == "nvd-test-99"

    def test_no_vulnerabilities_in_response_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        empty = {"totalResults": 0, "vulnerabilities": []}
        with patch("requests.get", return_value=self._mock_get(empty)):
            result = get_nvd_data(_cve_tool_use("CVE-9999-0000"))

        assert _result_status(result) == "error"
        assert (
            "9999-0000" in _result_text(result)
            or "invalid" in _result_text(result).lower()
            or "No vulnerability" in _result_text(result)
        )

    def test_empty_vulnerabilities_key_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        payload = {"totalResults": 0}  # no "vulnerabilities" key
        with patch("requests.get", return_value=self._mock_get(payload)):
            result = get_nvd_data(_cve_tool_use("CVE-9999-0000"))

        assert _result_status(result) == "error"


class TestGetNvdDataErrorHandling:
    def test_request_exception_returns_error(self):
        import requests

        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", side_effect=requests.exceptions.RequestException("timeout")):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"
        assert "failed" in _result_text(result).lower() or "Request" in _result_text(result)

    def test_json_decode_error_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad JSON", "", 0)

        with patch("requests.get", return_value=mock_resp):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"
        assert "JSON" in _result_text(result) or "parse" in _result_text(result).lower()

    def test_generic_exception_returns_error(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with patch("requests.get", side_effect=RuntimeError("unexpected")):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"

    def test_http_error_4xx_returns_error(self):
        import requests

        from manus_agent.tools.get_nvd_data import get_nvd_data

        mock_resp = MagicMock()
        http_err = requests.exceptions.HTTPError("403 Forbidden")
        mock_resp.raise_for_status.side_effect = http_err
        with patch("requests.get", return_value=mock_resp):
            result = get_nvd_data(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"


# ===========================================================================
# get_cwe_details tests
# ===========================================================================

_CWE_HTML_TEMPLATE = """\
<html><body>
<div id="Description">
<p>The software does not properly neutralize input before placing it into an output.</p>
</div>
<div id="Extended_Description">
<p>This allows attackers to inject malicious scripts.</p>
</div>
</body></html>
"""

_CWE_HTML_NO_EXTENDED = """\
<html><body>
<div id="Description">
<p>Buffer overflow description here.</p>
</div>
<div id="SomeOtherSection">Extra content.</div>
</body></html>
"""

_CWE_HTML_MISSING_DESCRIPTION = """\
<html><body>
<div id="OtherContent">Nothing here.</div>
</body></html>
"""


class TestGetCweDetailsImport:
    def test_module_importable(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        assert callable(get_cwe_details)

    def test_tool_spec_name(self):
        from manus_agent.tools.get_cwe_details import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_cwe_details"

    def test_tool_spec_has_cwe_id_input(self):
        from manus_agent.tools.get_cwe_details import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cwe_id" in props


class TestGetCweDetailsInputValidation:
    def test_cwe_without_prefix_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        result = get_cwe_details(_cwe_tool_use("79"))
        assert _result_status(result) == "error"
        assert "Invalid" in _result_text(result)

    def test_non_string_cwe_id_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        result = get_cwe_details(_make_tool_use(input_data={"cwe_id": 79}))
        assert _result_status(result) == "error"

    def test_none_cwe_id_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        result = get_cwe_details(_make_tool_use(input_data={"cwe_id": None}))
        assert _result_status(result) == "error"

    def test_cwe_with_non_numeric_suffix_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        result = get_cwe_details(_cwe_tool_use("CWE-ABC"))
        assert _result_status(result) == "error"
        assert "invalid" in _result_text(result).lower() or "Invalid" in _result_text(result)

    def test_empty_string_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        result = get_cwe_details(_cwe_tool_use(""))
        assert _result_status(result) == "error"


class TestGetCweDetailsSuccess:
    def _mock_get(self, html: str, status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status.return_value = None
        mock_resp.status_code = status_code
        return mock_resp

    def test_successful_parse_returns_success(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        assert _result_status(result) == "success"

    def test_returns_cwe_id_in_result(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        data = _result_json(result)
        assert data["cwe_id"] == "CWE-79"

    def test_returns_description_text(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        data = _result_json(result)
        assert "neutralize" in data["description"] or len(data["description"]) > 0

    def test_returns_url_in_result(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        data = _result_json(result)
        assert "cwe.mitre.org" in data["url"]
        assert "79" in data["url"]

    def test_cwe_id_is_case_insensitive(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("cwe-79"))

        assert _result_status(result) == "success"

    def test_url_uses_numeric_id_only(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)) as mock_get:
            get_cwe_details(_cwe_tool_use("CWE-120"))

        call_url = mock_get.call_args[0][0]
        assert "120.html" in call_url
        assert "CWE-" not in call_url

    def test_html_without_extended_description_still_parses(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_NO_EXTENDED)):
            result = get_cwe_details(_cwe_tool_use("CWE-120"))

        assert _result_status(result) == "success"
        data = _result_json(result)
        assert len(data["description"]) > 0

    def test_missing_description_div_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_MISSING_DESCRIPTION)):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        assert _result_status(result) == "error"

    def test_tool_use_id_preserved(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", return_value=self._mock_get(_CWE_HTML_TEMPLATE)):
            result = get_cwe_details(_cwe_tool_use("CWE-79", tool_use_id="cwe-test-id"))

        assert result["toolUseId"] == "cwe-test-id"


class TestGetCweDetailsErrorHandling:
    def test_request_exception_returns_error(self):
        import requests

        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", side_effect=requests.exceptions.RequestException("conn refused")):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        assert _result_status(result) == "error"
        assert "failed" in _result_text(result).lower() or "Request" in _result_text(result)

    def test_generic_exception_returns_error(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        with patch("requests.get", side_effect=RuntimeError("unexpected")):
            result = get_cwe_details(_cwe_tool_use("CWE-79"))

        assert _result_status(result) == "error"

    def test_http_404_returns_error(self):
        import requests

        from manus_agent.tools.get_cwe_details import get_cwe_details

        mock_resp = MagicMock()
        http_err = requests.exceptions.HTTPError("404 Not Found")
        mock_resp.raise_for_status.side_effect = http_err
        with patch("requests.get", return_value=mock_resp):
            result = get_cwe_details(_cwe_tool_use("CWE-99999"))

        assert _result_status(result) == "error"


# ===========================================================================
# get_otx_cve_details tests
# ===========================================================================


def _make_otx_response(cve_id: str = "CVE-2024-3094", pulse_count: int = 3) -> dict:
    """Minimal OTX CVE indicator API response."""
    pulses = [
        {
            "id": f"pulse-{i}",
            "name": f"Threat Campaign {i}",
            "created": "2024-04-01T00:00:00",
            "modified": "2024-04-01T00:00:00",
            "author": {"username": "researcher"},
            "tags": ["CVE", "backdoor"],
        }
        for i in range(pulse_count)
    ]
    return {
        "type": "CVE",
        "id": cve_id,
        "pulse_info": {
            "count": pulse_count,
            "pulses": pulses,
        },
        "sections": ["general", "pulse_info"],
    }


class TestGetOtxCveDetailsImport:
    def test_module_importable(self):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        assert callable(get_otx_cve_details)

    def test_tool_spec_name(self):
        from manus_agent.tools.get_otx_cve_details import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_otx_cve_details"

    def test_tool_spec_has_cve_id_input(self):
        from manus_agent.tools.get_otx_cve_details import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cve_id" in props


class TestGetOtxCveDetailsInputValidation:
    def test_non_cve_prefix_returns_error(self):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        result = get_otx_cve_details(_cve_tool_use("XSS-2024-3094"))
        assert _result_status(result) == "error"
        assert "Invalid" in _result_text(result)

    def test_non_string_returns_error(self):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        result = get_otx_cve_details(_make_tool_use(input_data={"cve_id": 12345}))
        assert _result_status(result) == "error"

    def test_none_cve_id_returns_error(self):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        result = get_otx_cve_details(_make_tool_use(input_data={"cve_id": None}))
        assert _result_status(result) == "error"

    def test_empty_string_returns_error(self):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        result = get_otx_cve_details(_cve_tool_use(""))
        assert _result_status(result) == "error"


class TestGetOtxCveDetailsMissingApiKey:
    def test_no_api_key_returns_error(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.delenv("OTX_API_KEY", raising=False)
        with patch("manus_agent.tools.get_otx_cve_details.Config") as mock_cfg:
            # Config object with no OTX config
            mock_instance = MagicMock()
            mock_instance.otx = None
            mock_cfg.from_file.return_value = mock_instance
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"
        assert "API key" in _result_text(result) or "api_key" in _result_text(result).lower()

    def test_config_load_exception_falls_back_to_env_key(self, monkeypatch):
        """When Config.from_file() raises, tool still tries OTX_API_KEY env var."""
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key-from-env")
        with patch("manus_agent.tools.get_otx_cve_details.Config") as mock_cfg:
            mock_cfg.from_file.side_effect = RuntimeError("config not found")
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_otx_response()
            mock_resp.raise_for_status.return_value = None
            with patch("requests.get", return_value=mock_resp):
                result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"


class TestGetOtxCveDetailsSuccess:
    def _setup(self, monkeypatch, payload: dict | None = None):
        monkeypatch.setenv("OTX_API_KEY", "test-api-key-123")
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload if payload is not None else _make_otx_response()
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_successful_response_with_pulses_returns_success(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        mock_resp = self._setup(monkeypatch)
        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"

    def test_pulse_data_present_in_result(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        mock_resp = self._setup(monkeypatch)
        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        data = _result_json(result)
        assert "pulse_info" in data

    def test_api_key_sent_in_header(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        mock_resp = self._setup(monkeypatch)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        _, kwargs = mock_get.call_args
        headers = kwargs.get("headers", {})
        assert "X-OTX-API-KEY" in headers
        assert headers["X-OTX-API-KEY"] == "test-api-key-123"

    def test_url_contains_uppercased_cve_id(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        mock_resp = self._setup(monkeypatch)
        with patch("requests.get", return_value=mock_resp) as mock_get:
            get_otx_cve_details(_cve_tool_use("cve-2024-3094"))

        call_url = mock_get.call_args[0][0]
        assert "CVE-2024-3094" in call_url

    def test_config_api_key_used_when_env_absent(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.delenv("OTX_API_KEY", raising=False)
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_otx_response()
        mock_resp.raise_for_status.return_value = None

        with patch("manus_agent.tools.get_otx_cve_details.Config") as mock_cfg:
            mock_otx = MagicMock()
            mock_otx.api_key = "config-api-key-xyz"
            mock_instance = MagicMock()
            mock_instance.otx = mock_otx
            mock_cfg.from_file.return_value = mock_instance
            with patch("requests.get", return_value=mock_resp) as mock_get:
                get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        _, kwargs = mock_get.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("X-OTX-API-KEY") == "config-api-key-xyz"

    def test_empty_pulses_returns_success_with_text(self, monkeypatch):
        """OTX returning 0 pulses is a successful query with no-pulse notice."""
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        no_pulse = _make_otx_response(pulse_count=0)
        mock_resp = self._setup(monkeypatch, payload=no_pulse)
        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"

    def test_tool_use_id_preserved(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        mock_resp = self._setup(monkeypatch)
        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094", tool_use_id="otx-id-77"))

        assert result["toolUseId"] == "otx-id-77"


class TestGetOtxCveDetailsErrorHandling:
    def test_request_exception_returns_error(self, monkeypatch):
        import requests

        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key")
        with patch("requests.get", side_effect=requests.exceptions.RequestException("network down")):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"
        assert "failed" in _result_text(result).lower() or "Request" in _result_text(result)

    def test_http_404_returns_success_with_no_info_message(self, monkeypatch):
        """OTX 404 means CVE not indexed, which is a valid (not error) outcome."""
        import requests

        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key")
        mock_resp = MagicMock()
        http_err = requests.exceptions.HTTPError("404 Not Found")
        http_err.response = MagicMock()
        http_err.response.status_code = 404
        mock_resp.raise_for_status.side_effect = http_err

        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "success"
        assert "No information" in _result_text(result) or "not found" in _result_text(result).lower()

    def test_http_5xx_returns_error(self, monkeypatch):
        import requests

        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key")
        mock_resp = MagicMock()
        http_err = requests.exceptions.HTTPError("503 Service Unavailable")
        http_err.response = MagicMock()
        http_err.response.status_code = 503
        mock_resp.raise_for_status.side_effect = http_err

        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"

    def test_json_decode_error_returns_error(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)

        with patch("requests.get", return_value=mock_resp):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"
        assert "JSON" in _result_text(result) or "parse" in _result_text(result).lower()

    def test_generic_exception_returns_error(self, monkeypatch):
        from manus_agent.tools.get_otx_cve_details import get_otx_cve_details

        monkeypatch.setenv("OTX_API_KEY", "test-key")
        with patch("requests.get", side_effect=RuntimeError("unexpected")):
            result = get_otx_cve_details(_cve_tool_use("CVE-2024-3094"))

        assert _result_status(result) == "error"


# ===========================================================================
# Cross-tool integration: TOOL_SPEC contract
# ===========================================================================


class TestToolSpecContract:
    """All four tools must satisfy the Strands TOOL_SPEC contract."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_nvd_data",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.get_otx_cve_details",
        ],
    )
    def test_tool_spec_has_required_keys(self, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        spec = mod.TOOL_SPEC
        assert "name" in spec
        assert "description" in spec
        assert "inputSchema" in spec

    @pytest.mark.parametrize(
        "module_path",
        [
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_nvd_data",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.get_otx_cve_details",
        ],
    )
    def test_tool_spec_name_matches_function(self, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        fn_name = mod.TOOL_SPEC["name"]
        assert hasattr(mod, fn_name), f"{fn_name} not found in {module_path}"
        assert callable(getattr(mod, fn_name))

    @pytest.mark.parametrize(
        "module_path",
        [
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_nvd_data",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.get_otx_cve_details",
        ],
    )
    def test_tool_spec_description_non_empty(self, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        assert len(mod.TOOL_SPEC["description"]) > 20

    @pytest.mark.parametrize(
        "module_path,expected_input_field",
        [
            ("manus_agent.tools.check_cisa_kev", "cve_id"),
            ("manus_agent.tools.get_nvd_data", "cve_id"),
            ("manus_agent.tools.get_cwe_details", "cwe_id"),
            ("manus_agent.tools.get_otx_cve_details", "cve_id"),
        ],
    )
    def test_tool_spec_required_fields(self, module_path, expected_input_field):
        import importlib

        mod = importlib.import_module(module_path)
        schema = mod.TOOL_SPEC["inputSchema"]["json"]
        required = schema.get("required", [])
        assert expected_input_field in required


# ===========================================================================
# VI agent wiring: all four tools must appear in the vi_agent system prompt
# ===========================================================================


class TestViAgentWiring:
    """The four core tools must be wired into the VI agent's system prompt and tool list."""

    def test_check_cisa_kev_in_vi_system_prompt(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "check_cisa_kev" in SYSTEM_PROMPT

    def test_get_nvd_data_in_vi_system_prompt(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "get_nvd_data" in SYSTEM_PROMPT

    def test_get_cwe_details_in_vi_system_prompt(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "get_cwe_details" in SYSTEM_PROMPT

    def test_get_otx_cve_details_in_vi_system_prompt(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "get_otx_cve_details" in SYSTEM_PROMPT

    def test_all_four_tools_loadable_via_vi_agent(self):
        """The VI agent registers tools by module path; all four core modules must load."""
        import importlib

        # Check that the tool module paths the VI agent uses all import cleanly
        for mod_path in (
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_nvd_data",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.get_otx_cve_details",
        ):
            mod = importlib.import_module(mod_path)
            assert hasattr(mod, "TOOL_SPEC")

    def test_vi_agent_module_path_list_contains_all_four(self):
        """vi_agent source must reference all four core tool module paths."""
        import inspect

        from manus_agent.agents import vi_agent

        source = inspect.getsource(vi_agent)
        for mod_path in (
            "manus_agent.tools.get_nvd_data",
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.get_otx_cve_details",
        ):
            assert mod_path in source, f"{mod_path!r} not found in vi_agent source"
