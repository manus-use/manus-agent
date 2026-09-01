#!/usr/bin/env python3
"""Comprehensive test suite for search_cpe tool and cpe-search CLI subcommand.

All HTTP calls are mocked — no real network requests.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest
import requests

# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Validate the Strands TOOL_SPEC contract."""

    def _get_spec(self):
        from manus_agent.tools.search_cpe import TOOL_SPEC

        return TOOL_SPEC

    def test_spec_has_name(self):
        spec = self._get_spec()
        assert spec["name"] == "search_cpe"

    def test_spec_has_description(self):
        spec = self._get_spec()
        assert "CPE" in spec["description"]
        assert len(spec["description"]) > 20

    def test_spec_has_input_schema(self):
        spec = self._get_spec()
        schema = spec["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "keyword" in schema["properties"]
        assert "keyword" in schema["required"]

    def test_spec_keyword_is_string(self):
        spec = self._get_spec()
        assert spec["inputSchema"]["json"]["properties"]["keyword"]["type"] == "string"

    def test_spec_optional_params(self):
        spec = self._get_spec()
        props = spec["inputSchema"]["json"]["properties"]
        for key in ("version", "cpe_type", "fetch_cves", "max_cpes", "max_cves_per_cpe"):
            assert key in props, f"Missing optional param: {key}"

    def test_spec_fetch_cves_is_boolean(self):
        spec = self._get_spec()
        assert spec["inputSchema"]["json"]["properties"]["fetch_cves"]["type"] == "boolean"

    def test_spec_max_cpes_is_integer(self):
        spec = self._get_spec()
        assert spec["inputSchema"]["json"]["properties"]["max_cpes"]["type"] == "integer"


# ---------------------------------------------------------------------------
# CPE URI parser tests
# ---------------------------------------------------------------------------


class TestParseCpeUri:
    """Test _parse_cpe_uri parsing logic."""

    def test_full_cpe23_uri(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(uri)
        assert result["cpe_uri"] == uri
        assert result["part"] == "a"
        assert result["vendor"] == "apache"
        assert result["product"] == "log4j"
        assert result["version"] == "2.14.1"

    def test_wildcard_version(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:a:openssl:openssl:*:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(uri)
        assert result["part"] == "a"
        assert result["vendor"] == "openssl"
        assert result["product"] == "openssl"
        assert "version" not in result  # wildcard excluded

    def test_os_type(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:o:linux:linux_kernel:5.15:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(uri)
        assert result["part"] == "o"
        assert result["vendor"] == "linux"
        assert result["product"] == "linux_kernel"
        assert result["version"] == "5.15"

    def test_hardware_type(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:h:cisco:asr_1000:*:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(uri)
        assert result["part"] == "h"
        assert result["vendor"] == "cisco"

    def test_short_uri(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:a:vendor:product"
        result = _parse_cpe_uri(uri)
        assert result["cpe_uri"] == uri
        assert result["part"] == "a"
        assert result["vendor"] == "vendor"
        assert result["product"] == "product"

    def test_dash_treated_as_not_applicable(self):
        from manus_agent.tools.search_cpe import _parse_cpe_uri

        uri = "cpe:2.3:a:vendor:product:-:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(uri)
        # '-' means NA, should be excluded like '*'
        assert "version" not in result


# ---------------------------------------------------------------------------
# _first_title helper tests
# ---------------------------------------------------------------------------


class TestFirstTitle:
    """Test _first_title extraction from NVD CPE titles."""

    def test_english_title_preferred(self):
        from manus_agent.tools.search_cpe import _first_title

        titles = [
            {"lang": "de", "title": "German Title"},
            {"lang": "en", "title": "English Title"},
        ]
        assert _first_title(titles) == "English Title"

    def test_en_us_accepted(self):
        from manus_agent.tools.search_cpe import _first_title

        titles = [{"lang": "en-US", "title": "US English"}]
        assert _first_title(titles) == "US English"

    def test_fallback_to_first(self):
        from manus_agent.tools.search_cpe import _first_title

        titles = [{"lang": "ja", "title": "Japanese Title"}]
        assert _first_title(titles) == "Japanese Title"

    def test_empty_titles(self):
        from manus_agent.tools.search_cpe import _first_title

        assert _first_title([]) == ""


# ---------------------------------------------------------------------------
# CVSS extraction tests
# ---------------------------------------------------------------------------


class TestExtractCvss:
    """Test _extract_cvss with various NVD metrics formats."""

    def test_cvss_v31(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV31": [
                {
                    "cvssData": {
                        "version": "3.1",
                        "baseScore": 9.8,
                        "baseSeverity": "CRITICAL",
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    },
                    "exploitabilityScore": 3.9,
                    "impactScore": 5.9,
                }
            ]
        }
        result = _extract_cvss(metrics)
        assert result["cvss_version"] == "3.1"
        assert result["cvss_score"] == 9.8
        assert result["cvss_severity"] == "CRITICAL"

    def test_cvss_v30_fallback(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV30": [
                {
                    "cvssData": {
                        "version": "3.0",
                        "baseScore": 7.5,
                        "baseSeverity": "HIGH",
                        "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
                    },
                    "exploitabilityScore": 3.9,
                    "impactScore": 3.6,
                }
            ]
        }
        result = _extract_cvss(metrics)
        assert result["cvss_version"] == "3.0"
        assert result["cvss_score"] == 7.5

    def test_cvss_v2_fallback(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV2": [
                {
                    "cvssData": {
                        "baseScore": 7.5,
                        "vectorString": "AV:N/AC:L/Au:N/C:P/I:P/A:P",
                    },
                    "exploitabilityScore": 10.0,
                    "impactScore": 6.4,
                }
            ]
        }
        result = _extract_cvss(metrics)
        assert result["cvss_version"] == "2.0"
        assert result["cvss_score"] == 7.5
        assert result["cvss_severity"] == "HIGH"

    def test_cvss_v2_medium(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV2": [
                {
                    "cvssData": {"baseScore": 5.0, "vectorString": ""},
                    "exploitabilityScore": 0.0,
                    "impactScore": 0.0,
                }
            ]
        }
        result = _extract_cvss(metrics)
        assert result["cvss_severity"] == "MEDIUM"

    def test_cvss_v2_low(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV2": [
                {
                    "cvssData": {"baseScore": 2.0, "vectorString": ""},
                    "exploitabilityScore": 0.0,
                    "impactScore": 0.0,
                }
            ]
        }
        result = _extract_cvss(metrics)
        assert result["cvss_severity"] == "LOW"

    def test_no_metrics(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        result = _extract_cvss({})
        assert result["cvss_score"] == 0.0
        assert result["cvss_severity"] == ""

    def test_v31_preferred_over_v2(self):
        from manus_agent.tools.search_cpe import _extract_cvss

        metrics = {
            "cvssMetricV31": [
                {
                    "cvssData": {
                        "version": "3.1",
                        "baseScore": 9.8,
                        "baseSeverity": "CRITICAL",
                        "vectorString": "",
                    },
                    "exploitabilityScore": 3.9,
                    "impactScore": 5.9,
                }
            ],
            "cvssMetricV2": [
                {
                    "cvssData": {"baseScore": 7.5, "vectorString": ""},
                    "exploitabilityScore": 0.0,
                    "impactScore": 0.0,
                }
            ],
        }
        result = _extract_cvss(metrics)
        assert result["cvss_version"] == "3.1"
        assert result["cvss_score"] == 9.8


# ---------------------------------------------------------------------------
# _first_en_description tests
# ---------------------------------------------------------------------------


class TestFirstEnDescription:
    """Test _first_en_description extraction."""

    def test_english_preferred(self):
        from manus_agent.tools.search_cpe import _first_en_description

        descs = [
            {"lang": "es", "value": "Descripción en español"},
            {"lang": "en", "value": "English description"},
        ]
        assert _first_en_description(descs) == "English description"

    def test_fallback(self):
        from manus_agent.tools.search_cpe import _first_en_description

        descs = [{"lang": "fr", "value": "Description française"}]
        assert _first_en_description(descs) == "Description française"

    def test_empty(self):
        from manus_agent.tools.search_cpe import _first_en_description

        assert _first_en_description([]) == ""


# ---------------------------------------------------------------------------
# NVD headers and rate limit helpers
# ---------------------------------------------------------------------------


class TestNvdHelpers:
    """Test _build_nvd_headers, _has_api_key, _inter_request_delay."""

    def test_headers_without_key(self):
        from manus_agent.tools.search_cpe import _build_nvd_headers

        with mock.patch.dict(os.environ, {}, clear=True):
            headers = _build_nvd_headers()
            assert "apiKey" not in headers

    def test_headers_with_key(self):
        from manus_agent.tools.search_cpe import _build_nvd_headers

        with mock.patch.dict(os.environ, {"NVD_API_KEY": "test-key-123"}):
            headers = _build_nvd_headers()
            assert headers["apiKey"] == "test-key-123"

    def test_has_api_key_false(self):
        from manus_agent.tools.search_cpe import _has_api_key

        with mock.patch.dict(os.environ, {}, clear=True):
            assert _has_api_key() is False

    def test_has_api_key_true(self):
        from manus_agent.tools.search_cpe import _has_api_key

        with mock.patch.dict(os.environ, {"NVD_API_KEY": "key"}):
            assert _has_api_key() is True

    def test_has_api_key_empty_string(self):
        from manus_agent.tools.search_cpe import _has_api_key

        with mock.patch.dict(os.environ, {"NVD_API_KEY": "  "}):
            assert _has_api_key() is False

    def test_inter_request_delay_with_key(self):
        from manus_agent.tools.search_cpe import _inter_request_delay

        with mock.patch.dict(os.environ, {"NVD_API_KEY": "key"}):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep") as m:
                _inter_request_delay()
                m.assert_called_once()
                assert m.call_args[0][0] <= 0.2

    def test_inter_request_delay_without_key(self):
        from manus_agent.tools.search_cpe import _inter_request_delay

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep") as m:
                _inter_request_delay()
                m.assert_called_once()
                assert m.call_args[0][0] >= 0.5


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestNvdGetWithRetry:
    """Test _nvd_get_with_retry retry/back-off logic."""

    def _make_response(self, status=200, json_data=None):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = status
        resp.json.return_value = json_data or {}
        if status >= 400:
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                f"HTTP {status}",
                response=resp,
            )
        else:
            resp.raise_for_status.return_value = None
        return resp

    def test_success_first_attempt(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        resp = self._make_response(200, {"ok": True})
        with mock.patch("manus_agent.tools.search_cpe.requests.get", return_value=resp):
            result = _nvd_get_with_retry("https://example.com")
            assert result.status_code == 200

    def test_retry_on_429(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        fail = self._make_response(429)
        success = self._make_response(200, {"ok": True})
        with mock.patch("manus_agent.tools.search_cpe.requests.get", side_effect=[fail, success]):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep"):
                result = _nvd_get_with_retry("https://example.com")
                assert result.status_code == 200

    def test_retry_on_503(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        fail = self._make_response(503)
        success = self._make_response(200)
        with mock.patch("manus_agent.tools.search_cpe.requests.get", side_effect=[fail, success]):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep"):
                result = _nvd_get_with_retry("https://example.com")
                assert result.status_code == 200

    def test_retry_on_connection_error(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        success = self._make_response(200)
        with mock.patch(
            "manus_agent.tools.search_cpe.requests.get",
            side_effect=[requests.exceptions.ConnectionError("fail"), success],
        ):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep"):
                result = _nvd_get_with_retry("https://example.com")
                assert result.status_code == 200

    def test_retry_on_timeout(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        success = self._make_response(200)
        with mock.patch(
            "manus_agent.tools.search_cpe.requests.get",
            side_effect=[requests.exceptions.Timeout("timeout"), success],
        ):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep"):
                result = _nvd_get_with_retry("https://example.com")
                assert result.status_code == 200

    def test_all_retries_exhausted(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        fail = self._make_response(429)
        with mock.patch("manus_agent.tools.search_cpe.requests.get", return_value=fail):
            with mock.patch("manus_agent.tools.search_cpe.time.sleep"):
                import manus_agent.tools.search_cpe as mod

                original = mod._MAX_RETRIES
                mod._MAX_RETRIES = 2
                try:
                    with pytest.raises((requests.exceptions.HTTPError, RuntimeError)):
                        _nvd_get_with_retry("https://example.com")
                finally:
                    mod._MAX_RETRIES = original

    def test_non_retryable_http_error_raises_immediately(self):
        from manus_agent.tools.search_cpe import _nvd_get_with_retry

        resp = self._make_response(404)
        with mock.patch("manus_agent.tools.search_cpe.requests.get", return_value=resp):
            with pytest.raises(requests.exceptions.HTTPError):
                _nvd_get_with_retry("https://example.com")


# ---------------------------------------------------------------------------
# Fixtures: mock NVD API responses
# ---------------------------------------------------------------------------

_MOCK_CPE_RESPONSE = {
    "totalResults": 3,
    "products": [
        {
            "cpe": {
                "cpeName": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                "titles": [{"lang": "en", "title": "Apache Log4j 2.14.1"}],
                "deprecated": False,
                "lastModified": "2023-01-01T00:00:00.000",
            }
        },
        {
            "cpe": {
                "cpeName": "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
                "titles": [{"lang": "en", "title": "Apache Log4j 2.15.0"}],
                "deprecated": False,
                "lastModified": "2023-02-01T00:00:00.000",
            }
        },
        {
            "cpe": {
                "cpeName": "cpe:2.3:o:linux:linux_kernel:5.15:*:*:*:*:*:*:*",
                "titles": [{"lang": "en", "title": "Linux Kernel 5.15"}],
                "deprecated": False,
                "lastModified": "2023-03-01T00:00:00.000",
            }
        },
    ],
}

_MOCK_CVE_RESPONSE = {
    "totalResults": 2,
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2021-44228",
                "descriptions": [{"lang": "en", "value": "Apache Log4j2 allows RCE via JNDI"}],
                "published": "2021-12-10T10:00:00.000",
                "lastModified": "2023-01-01T00:00:00.000",
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "version": "3.1",
                                "baseScore": 10.0,
                                "baseSeverity": "CRITICAL",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                            },
                            "exploitabilityScore": 3.9,
                            "impactScore": 6.0,
                        }
                    ],
                },
            }
        },
        {
            "cve": {
                "id": "CVE-2021-45046",
                "descriptions": [
                    {
                        "lang": "en",
                        "value": "Log4j2 incomplete fix for CVE-2021-44228",
                    }
                ],
                "published": "2021-12-14T10:00:00.000",
                "lastModified": "2023-01-01T00:00:00.000",
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "version": "3.1",
                                "baseScore": 9.0,
                                "baseSeverity": "CRITICAL",
                                "vectorString": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H",
                            },
                            "exploitabilityScore": 2.2,
                            "impactScore": 6.0,
                        }
                    ],
                },
            }
        },
    ],
}


# ---------------------------------------------------------------------------
# search_cpes tests
# ---------------------------------------------------------------------------


class TestSearchCpes:
    """Test the CPE search function."""

    def _make_resp(self, json_data, status=200):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = status
        resp.json.return_value = json_data
        resp.raise_for_status.return_value = None
        return resp

    def test_basic_search(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("apache log4j")
            assert result["total_results"] == 3
            assert result["keyword"] == "apache log4j"
            # Only 'a' type should be returned (filters out the linux kernel 'o')
            for cpe in result["cpes"]:
                parsed_parts = cpe["cpe_uri"].split(":")
                assert parsed_parts[2] == "a"

    def test_version_filter(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("apache log4j", version="2.14.1")
            assert len(result["cpes"]) == 1
            assert result["cpes"][0]["version"] == "2.14.1"

    def test_os_type_filter(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("linux kernel", cpe_type="o")
            assert len(result["cpes"]) == 1
            assert result["cpes"][0]["vendor"] == "linux"

    def test_empty_keyword(self):
        from manus_agent.tools.search_cpe import search_cpes

        result = search_cpes("")
        assert result.get("error") == "keyword is required"
        assert result["cpes"] == []

    def test_whitespace_keyword(self):
        from manus_agent.tools.search_cpe import search_cpes

        result = search_cpes("   ")
        assert result.get("error") == "keyword is required"

    def test_invalid_cpe_type_defaults_to_a(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("apache", cpe_type="z")
            assert result["cpe_type"] == "a"

    def test_max_results_clamped(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("apache", max_results=100)
            # Should clamp to 50
            assert result["returned"] <= 50

    def test_api_error_handled(self):
        from manus_agent.tools.search_cpe import search_cpes

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=requests.exceptions.ConnectionError("fail"),
        ):
            result = search_cpes("apache")
            assert "error" in result
            assert result["cpes"] == []

    def test_deprecated_cpe_included(self):
        from manus_agent.tools.search_cpe import search_cpes

        data = {
            "totalResults": 1,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:old:software:1.0:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": True,
                        "lastModified": "2020-01-01T00:00:00.000",
                    },
                }
            ],
        }
        resp = self._make_resp(data)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("old software")
            assert len(result["cpes"]) == 1
            assert result["cpes"][0]["deprecated"] is True

    def test_empty_cpe_name_skipped(self):
        from manus_agent.tools.search_cpe import search_cpes

        data = {
            "totalResults": 1,
            "products": [{"cpe": {"cpeName": "", "titles": []}}],
        }
        resp = self._make_resp(data)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("test")
            assert len(result["cpes"]) == 0


# ---------------------------------------------------------------------------
# fetch_cves_for_cpe tests
# ---------------------------------------------------------------------------


class TestFetchCvesForCpe:
    """Test the CVE lookup function."""

    def _make_resp(self, json_data, status=200):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = status
        resp.json.return_value = json_data
        resp.raise_for_status.return_value = None
        return resp

    def test_basic_cve_fetch(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        resp = self._make_resp(_MOCK_CVE_RESPONSE)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = fetch_cves_for_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
            assert result["total_results"] == 2
            assert len(result["cves"]) == 2
            # Sorted by CVSS descending
            assert result["cves"][0]["cvss_score"] >= result["cves"][1]["cvss_score"]

    def test_empty_cpe_uri(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        result = fetch_cves_for_cpe("")
        assert "error" in result
        assert result["cves"] == []

    def test_api_error(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=Exception("Network error"),
        ):
            result = fetch_cves_for_cpe("cpe:2.3:a:test:test:*:*:*:*:*:*:*:*")
            assert "error" in result

    def test_no_cves_found(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        data = {"totalResults": 0, "vulnerabilities": []}
        resp = self._make_resp(data)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = fetch_cves_for_cpe("cpe:2.3:a:niche:product:1.0:*:*:*:*:*:*:*")
            assert result["total_results"] == 0
            assert result["cves"] == []

    def test_cve_description_truncated(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        long_desc = "A" * 500
        data = {
            "totalResults": 1,
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "descriptions": [{"lang": "en", "value": long_desc}],
                        "published": "2024-01-01T00:00:00.000",
                        "lastModified": "2024-01-01T00:00:00.000",
                        "metrics": {},
                    },
                }
            ],
        }
        resp = self._make_resp(data)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = fetch_cves_for_cpe("cpe:2.3:a:test:test:*:*:*:*:*:*:*:*")
            assert len(result["cves"][0]["description"]) <= 300

    def test_empty_cve_id_skipped(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        data = {
            "totalResults": 1,
            "vulnerabilities": [{"cve": {"id": "", "descriptions": [], "metrics": {}}}],
        }
        resp = self._make_resp(data)
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = fetch_cves_for_cpe("cpe:2.3:a:test:test:*:*:*:*:*:*:*:*")
            assert len(result["cves"]) == 0


# ---------------------------------------------------------------------------
# search_cpe_and_cves combined tests
# ---------------------------------------------------------------------------


class TestSearchCpeAndCves:
    """Test the combined two-stage pipeline."""

    def _make_resp(self, json_data):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = json_data
        resp.raise_for_status.return_value = None
        return resp

    def test_full_pipeline(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        cpe_resp = self._make_resp(_MOCK_CPE_RESPONSE)
        cve_resp = self._make_resp(_MOCK_CVE_RESPONSE)

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=[cpe_resp, cve_resp, cve_resp],
        ):
            with mock.patch("manus_agent.tools.search_cpe._inter_request_delay"):
                result = search_cpe_and_cves("apache log4j")
                assert result["keyword"] == "apache log4j"
                summary = result["summary"]
                assert summary["cpes_found"] > 0
                assert summary["total_unique_cves"] > 0

    def test_cpe_only_mode(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        cpe_resp = self._make_resp(_MOCK_CPE_RESPONSE)
        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            return_value=cpe_resp,
        ):
            result = search_cpe_and_cves("apache log4j", fetch_cves=False)
            assert result["cve_results"] == []
            assert result["summary"]["total_unique_cves"] == 0
            assert result["summary"]["cpes_found"] > 0

    def test_no_cpes_found(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        empty_resp = self._make_resp({"totalResults": 0, "products": []})
        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            return_value=empty_resp,
        ):
            result = search_cpe_and_cves("nonexistent-product-xyz")
            assert result["summary"]["cpes_found"] == 0
            assert result["summary"]["total_unique_cves"] == 0

    def test_cpe_error_propagated(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=Exception("Boom"),
        ):
            result = search_cpe_and_cves("apache")
            assert "error" in result
            assert result["summary"]["cpes_found"] == 0

    def test_cve_deduplication(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        # Same CVE response for two different CPEs
        cpe_data = {
            "totalResults": 2,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                },
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                },
            ],
        }
        cpe_resp = self._make_resp(cpe_data)
        cve_resp = self._make_resp(_MOCK_CVE_RESPONSE)

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=[cpe_resp, cve_resp, cve_resp],
        ):
            with mock.patch("manus_agent.tools.search_cpe._inter_request_delay"):
                result = search_cpe_and_cves("apache log4j")
                # Both CPEs return same 2 CVEs, so unique count should be 2
                assert result["summary"]["total_unique_cves"] == 2

    def test_severity_counts(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        cpe_data = {
            "totalResults": 1,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:test:test:1.0:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                }
            ],
        }
        cve_data = {
            "totalResults": 4,
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "descriptions": [],
                        "published": "",
                        "lastModified": "",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "version": "3.1",
                                        "baseScore": 9.8,
                                        "baseSeverity": "CRITICAL",
                                        "vectorString": "",
                                    },
                                    "exploitabilityScore": 0,
                                    "impactScore": 0,
                                }
                            ]
                        },
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2024-0002",
                        "descriptions": [],
                        "published": "",
                        "lastModified": "",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "version": "3.1",
                                        "baseScore": 7.5,
                                        "baseSeverity": "HIGH",
                                        "vectorString": "",
                                    },
                                    "exploitabilityScore": 0,
                                    "impactScore": 0,
                                }
                            ]
                        },
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2024-0003",
                        "descriptions": [],
                        "published": "",
                        "lastModified": "",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "version": "3.1",
                                        "baseScore": 5.0,
                                        "baseSeverity": "MEDIUM",
                                        "vectorString": "",
                                    },
                                    "exploitabilityScore": 0,
                                    "impactScore": 0,
                                }
                            ]
                        },
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2024-0004",
                        "descriptions": [],
                        "published": "",
                        "lastModified": "",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "version": "3.1",
                                        "baseScore": 3.0,
                                        "baseSeverity": "LOW",
                                        "vectorString": "",
                                    },
                                    "exploitabilityScore": 0,
                                    "impactScore": 0,
                                }
                            ]
                        },
                    }
                },
            ],
        }
        cpe_resp = self._make_resp(cpe_data)
        cve_resp = self._make_resp(cve_data)
        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=[cpe_resp, cve_resp],
        ):
            with mock.patch("manus_agent.tools.search_cpe._inter_request_delay"):
                result = search_cpe_and_cves("test")
                summary = result["summary"]
                assert summary["critical_count"] == 1
                assert summary["high_count"] == 1
                assert summary["medium_count"] == 1
                assert summary["low_count"] == 1

    def test_unique_cves_sorted_descending(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        cpe_data = {
            "totalResults": 1,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:test:test:1.0:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                }
            ],
        }
        cve_resp = self._make_resp(_MOCK_CVE_RESPONSE)
        cpe_resp = self._make_resp(cpe_data)
        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=[cpe_resp, cve_resp],
        ):
            with mock.patch("manus_agent.tools.search_cpe._inter_request_delay"):
                result = search_cpe_and_cves("test")
                cves = result["unique_cves"]
                for i in range(len(cves) - 1):
                    assert cves[i]["cvss_score"] >= cves[i + 1]["cvss_score"]

    def test_max_cpes_parameter(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        many_products = {
            "totalResults": 5,
            "products": [
                {
                    "cpe": {
                        "cpeName": f"cpe:2.3:a:vendor:product{i}:1.0:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                }
                for i in range(5)
            ],
        }
        cpe_resp = self._make_resp(many_products)
        cve_resp = self._make_resp({"totalResults": 0, "vulnerabilities": []})

        with mock.patch(
            "manus_agent.tools.search_cpe._nvd_get_with_retry",
            side_effect=[cpe_resp] + [cve_resp] * 2,
        ):
            with mock.patch("manus_agent.tools.search_cpe._inter_request_delay"):
                result = search_cpe_and_cves("vendor", max_cpes=2)
                assert result["summary"]["cpes_found"] == 2


# ---------------------------------------------------------------------------
# Strands handler tests
# ---------------------------------------------------------------------------


class TestSearchCpeHandler:
    """Test the Strands-compatible handler."""

    def test_handler_success(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {"input": {"keyword": "apache log4j"}}
        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = {
                "keyword": "apache log4j",
                "summary": {"cpes_found": 2, "total_unique_cves": 3},
            }
            result = search_cpe_handler(tool)
            assert result["status"] == "success"
            content = result["content"][0]["text"]
            parsed = json.loads(content)
            assert parsed["keyword"] == "apache log4j"

    def test_handler_empty_keyword(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {"input": {"keyword": ""}}
        result = search_cpe_handler(tool)
        assert result["status"] == "error"

    def test_handler_missing_keyword(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {"input": {}}
        result = search_cpe_handler(tool)
        assert result["status"] == "error"

    def test_handler_passes_all_params(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {
            "input": {
                "keyword": "openssl",
                "version": "3.0.7",
                "cpe_type": "a",
                "fetch_cves": False,
                "max_cpes": 5,
                "max_cves_per_cpe": 10,
            }
        }
        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = {"keyword": "openssl", "summary": {}}
            search_cpe_handler(tool)
            m.assert_called_once_with(
                "openssl",
                version="3.0.7",
                cpe_type="a",
                fetch_cves=False,
                max_cpes=5,
                max_cves_per_cpe=10,
            )

    def test_handler_error_result(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {"input": {"keyword": "test"}}
        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = {"error": "NVD API failed", "summary": {}}
            result = search_cpe_handler(tool)
            assert result["status"] == "error"

    def test_handler_defaults(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {"input": {"keyword": "test"}}
        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = {"keyword": "test", "summary": {}}
            search_cpe_handler(tool)
            m.assert_called_once_with(
                "test",
                version="",
                cpe_type="a",
                fetch_cves=True,
                max_cpes=10,
                max_cves_per_cpe=20,
            )

    def test_handler_with_no_input_key(self):
        from manus_agent.tools.search_cpe import search_cpe_handler

        tool = {}
        result = search_cpe_handler(tool)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCpeSearchCli:
    """Test the manus-agent cpe-search CLI subcommand."""

    def test_cli_parser_builds(self):
        from manus_agent.cli import _build_cpe_search_parser

        parser = _build_cpe_search_parser()
        args = parser.parse_args(["apache log4j"])
        assert args.keyword == "apache log4j"
        assert args.version == ""
        assert args.cpe_type == "a"
        assert args.output == "text"
        assert args.no_cves is False
        assert args.max_cpes == 10
        assert args.max_cves == 20

    def test_cli_parser_all_flags(self):
        from manus_agent.cli import _build_cpe_search_parser

        parser = _build_cpe_search_parser()
        args = parser.parse_args(
            [
                "openssl",
                "--version",
                "3.0.7",
                "--type",
                "a",
                "--max-cpes",
                "5",
                "--max-cves",
                "50",
                "--no-cves",
                "--output",
                "json",
            ]
        )
        assert args.keyword == "openssl"
        assert args.version == "3.0.7"
        assert args.cpe_type == "a"
        assert args.max_cpes == 5
        assert args.max_cves == 50
        assert args.no_cves is True
        assert args.output == "json"

    def test_cli_text_output(self):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "apache log4j",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {
                "cpes": [
                    {
                        "cpe_uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                        "vendor": "apache",
                        "product": "log4j",
                        "version": "2.14.1",
                        "deprecated": False,
                        "title": "Apache Log4j 2.14.1",
                    },
                ],
                "total_results": 1,
            },
            "unique_cves": [
                {
                    "cve_id": "CVE-2021-44228",
                    "cvss_score": 10.0,
                    "cvss_severity": "CRITICAL",
                    "published": "2021-12-10",
                    "description": "Apache Log4j2 RCE",
                },
            ],
            "summary": {
                "cpes_found": 1,
                "total_unique_cves": 1,
                "critical_count": 1,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["apache log4j"])
            assert rc == 0

    def test_cli_json_output(self, capsys):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "test",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {"cpes": [], "total_results": 0},
            "unique_cves": [],
            "summary": {
                "cpes_found": 0,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["test", "--output", "json"])
            assert rc == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["keyword"] == "test"

    def test_cli_error_handling(self, capsys):
        from manus_agent.cli import _run_cpe_search

        mock_result = {"error": "NVD API failed", "summary": {}}
        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["test"])
            assert rc == 1
            captured = capsys.readouterr()
            assert "error" in captured.err.lower()

    def test_cli_no_cves_flag(self):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "test",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {
                "cpes": [
                    {
                        "cpe_uri": "cpe:2.3:a:test:test:1.0:*:*:*:*:*:*:*",
                        "vendor": "test",
                        "product": "test",
                        "version": "1.0",
                        "deprecated": False,
                        "title": "",
                    },
                ],
                "total_results": 1,
            },
            "cve_results": [],
            "unique_cves": [],
            "summary": {
                "cpes_found": 1,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = mock_result
            rc = _run_cpe_search(["test", "--no-cves"])
            assert rc == 0
            m.assert_called_once()
            assert m.call_args[1]["fetch_cves"] is False

    def test_cli_version_flag(self):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "test",
            "version": "2.0",
            "cpe_type": "a",
            "cpe_results": {"cpes": [], "total_results": 0},
            "unique_cves": [],
            "summary": {
                "cpes_found": 0,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = mock_result
            rc = _run_cpe_search(["test", "--version", "2.0"])
            assert rc == 0
            assert m.call_args[1]["version"] == "2.0"

    def test_cli_no_cpes_found(self, capsys):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "nonexistent",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {"cpes": [], "total_results": 0},
            "unique_cves": [],
            "summary": {
                "cpes_found": 0,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["nonexistent"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "No CPE matches" in captured.out

    def test_cli_no_cves_for_matched_cpes(self, capsys):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "test",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {
                "cpes": [
                    {
                        "cpe_uri": "cpe:2.3:a:test:test:1.0:*:*:*:*:*:*:*",
                        "vendor": "test",
                        "product": "test",
                        "version": "1.0",
                        "deprecated": False,
                        "title": "Test 1.0",
                    },
                ],
                "total_results": 1,
            },
            "unique_cves": [],
            "summary": {
                "cpes_found": 1,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["test"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "No CVEs found" in captured.out

    def test_cli_text_with_many_cves(self, capsys):
        from manus_agent.cli import _run_cpe_search

        cves = [
            {
                "cve_id": f"CVE-2024-{i:04d}",
                "cvss_score": 10.0 - i * 0.2,
                "cvss_severity": "CRITICAL" if i < 5 else "HIGH",
                "published": "2024-01-01",
                "description": f"Vuln {i}",
            }
            for i in range(35)
        ]

        mock_result = {
            "keyword": "test",
            "version": "",
            "cpe_type": "a",
            "cpe_results": {
                "cpes": [
                    {
                        "cpe_uri": "cpe:2.3:a:test:test:1.0:*:*:*:*:*:*:*",
                        "vendor": "test",
                        "product": "test",
                        "version": "1.0",
                        "deprecated": False,
                        "title": "",
                    },
                ],
                "total_results": 1,
            },
            "unique_cves": cves,
            "summary": {
                "cpes_found": 1,
                "total_unique_cves": 35,
                "critical_count": 5,
                "high_count": 30,
                "medium_count": 0,
                "low_count": 0,
            },
        }

        with mock.patch(
            "manus_agent.tools.search_cpe.search_cpe_and_cves",
            return_value=mock_result,
        ):
            rc = _run_cpe_search(["test"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "and 5 more" in captured.out

    def test_cli_os_type_flag(self):
        from manus_agent.cli import _run_cpe_search

        mock_result = {
            "keyword": "linux",
            "version": "",
            "cpe_type": "o",
            "cpe_results": {"cpes": [], "total_results": 0},
            "unique_cves": [],
            "summary": {
                "cpes_found": 0,
                "total_unique_cves": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
            },
        }
        with mock.patch("manus_agent.tools.search_cpe.search_cpe_and_cves") as m:
            m.return_value = mock_result
            rc = _run_cpe_search(["linux", "--type", "o"])
            assert rc == 0
            assert m.call_args[1]["cpe_type"] == "o"


# ---------------------------------------------------------------------------
# CLI dispatch integration test
# ---------------------------------------------------------------------------


class TestCpeSearchInSubcommands:
    """Verify cpe-search is wired into the CLI dispatch."""

    def test_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "cpe-search" in _SUBCOMMANDS

    def test_main_dispatches_cpe_search(self):
        from manus_agent.cli import main

        with mock.patch("manus_agent.cli._run_cpe_search", return_value=0) as m:
            with mock.patch("sys.argv", ["manus-agent", "cpe-search", "openssl"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
            m.assert_called_once_with(["openssl"])


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case coverage."""

    def test_search_cpes_zero_max_results(self):
        from manus_agent.tools.search_cpe import search_cpes

        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = {"totalResults": 0, "products": []}
        resp.raise_for_status.return_value = None
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("test", max_results=0)
            # Clamped to 1
            assert result["returned"] == 0  # No results but was clamped to 1

    def test_fetch_cves_zero_max_results(self):
        from manus_agent.tools.search_cpe import fetch_cves_for_cpe

        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = {"totalResults": 0, "vulnerabilities": []}
        resp.raise_for_status.return_value = None
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = fetch_cves_for_cpe("cpe:2.3:a:t:t:1:*:*:*:*:*:*:*", max_results=0)
            assert result["returned"] == 0

    def test_version_filter_case_insensitive(self):
        from manus_agent.tools.search_cpe import search_cpes

        data = {
            "totalResults": 1,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:vendor:product:2.0.BETA:*:*:*:*:*:*:*",
                        "titles": [],
                        "deprecated": False,
                        "lastModified": "",
                    }
                }
            ],
        }
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("vendor product", version="beta")
            assert len(result["cpes"]) == 1

    def test_combined_search_empty_keyword(self):
        from manus_agent.tools.search_cpe import search_cpe_and_cves

        result = search_cpe_and_cves("")
        assert "error" in result
        assert result["summary"]["cpes_found"] == 0

    def test_cpe_results_include_vendor_product(self):
        from manus_agent.tools.search_cpe import search_cpes

        data = {
            "totalResults": 1,
            "products": [
                {
                    "cpe": {
                        "cpeName": "cpe:2.3:a:apache:struts:2.5.30:*:*:*:*:*:*:*",
                        "titles": [{"lang": "en", "title": "Apache Struts 2.5.30"}],
                        "deprecated": False,
                        "lastModified": "2023-06-01T00:00:00.000",
                    }
                }
            ],
        }
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        with mock.patch("manus_agent.tools.search_cpe._nvd_get_with_retry", return_value=resp):
            result = search_cpes("apache struts")
            cpe = result["cpes"][0]
            assert cpe["vendor"] == "apache"
            assert cpe["product"] == "struts"
            assert cpe["version"] == "2.5.30"
            assert cpe["title"] == "Apache Struts 2.5.30"
            assert cpe["deprecated"] is False
