"""Tests for get_osv_data — OSV.dev CVE enrichment tool.

All HTTP calls are fully mocked; no real network requests are made. The
OSV_RETRY_BASE_DELAY env var is forced to "0" so retry loops complete
instantly without real sleeping.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools import get_osv_data as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tool_use(cve_id="CVE-2021-44228", tool_use_id="osv-test-001"):
    return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}


def _make_response(status_code=200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else {}
    if status_code < 400:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}", response=resp)
    return resp


def _cve_record_with_packages(cve_id="CVE-2021-44228"):
    """A CVE record that directly carries Maven package ranges."""
    return {
        "id": cve_id,
        "summary": "Log4Shell RCE",
        "aliases": ["GHSA-jfh8-c2jp-5v3q"],
        "modified": "2024-01-01T00:00:00Z",
        "published": "2021-12-10T00:00:00Z",
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
        "affected": [
            {
                "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "2.0-beta9"}, {"fixed": "2.15.0"}],
                    }
                ],
                "versions": ["2.0-beta9", "2.14.1"],
            }
        ],
        "references": [{"type": "WEB", "url": "https://logging.apache.org/log4j/"}],
    }


def _cve_record_no_packages(cve_id="CVE-2024-3094"):
    """A CVE record with a GHSA alias but no package-level affected data."""
    return {
        "id": cve_id,
        "details": "xz backdoor",
        "aliases": ["GHSA-rxwq-x6h5-x525"],
        "modified": "2024-04-01T00:00:00Z",
        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
        "affected": [],
        "references": [{"type": "ADVISORY", "url": "https://www.openwall.com/lists/oss-security"}],
    }


def _ghsa_record_with_packages(ghsa_id="GHSA-rxwq-x6h5-x525"):
    return {
        "id": ghsa_id,
        "summary": "Malicious code in xz",
        "aliases": ["CVE-2024-3094"],
        "affected": [
            {
                "package": {"ecosystem": "Alpine:v3.20", "name": "xz"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "5.6.0"}, {"fixed": "5.6.1-r2"}]}],
            }
        ],
        "references": [{"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094"}],
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC contract
# ---------------------------------------------------------------------------
class TestToolSpec:
    def test_name(self):
        assert mod.TOOL_SPEC["name"] == "get_osv_data"

    def test_required_input(self):
        schema = mod.TOOL_SPEC["inputSchema"]["json"]
        assert schema["required"] == ["cve_id"]
        assert "cve_id" in schema["properties"]

    def test_description_mentions_osv(self):
        assert "OSV" in mod.TOOL_SPEC["description"]

    def test_retryable_statuses(self):
        assert mod._OSV_RETRYABLE_STATUSES == frozenset({429, 500, 502, 503, 504})

    def test_config_env_defaults(self):
        assert mod._OSV_MAX_RETRIES >= 1
        assert mod._OSV_RETRY_BASE_DELAY >= 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
class TestParseAffected:
    def test_extracts_ranges(self):
        pkgs = mod._parse_affected(_cve_record_with_packages()["affected"])
        assert len(pkgs) == 1
        p = pkgs[0]
        assert p["ecosystem"] == "Maven"
        assert p["package"] == "org.apache.logging.log4j:log4j-core"
        assert p["introduced"] == ["2.0-beta9"]
        assert p["fixed"] == ["2.15.0"]
        assert p["affected_version_count"] == 2

    def test_skips_packageless_entry(self):
        pkgs = mod._parse_affected([{"ranges": [{"events": [{"introduced": "0"}]}]}])
        assert pkgs == []

    def test_handles_empty(self):
        assert mod._parse_affected([]) == []
        assert mod._parse_affected(None) == []

    def test_ignores_non_dict_entries(self):
        assert mod._parse_affected(["nope", 42, None]) == []

    def test_last_affected_captured(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "foo"},
                "ranges": [{"events": [{"introduced": "0"}, {"last_affected": "1.2.3"}]}],
            }
        ]
        pkgs = mod._parse_affected(affected)
        assert pkgs[0]["last_affected"] == ["1.2.3"]
        assert pkgs[0]["fixed"] == []

    def test_versions_sample_capped(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "bar"},
                "versions": [str(i) for i in range(50)],
            }
        ]
        pkgs = mod._parse_affected(affected)
        assert pkgs[0]["affected_version_count"] == 50
        assert len(pkgs[0]["affected_versions_sample"]) == 10


class TestParseSeverity:
    def test_extracts_cvss(self):
        sev = mod._parse_severity(_cve_record_with_packages())
        assert sev == [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}]

    def test_empty_when_missing(self):
        assert mod._parse_severity({}) == []

    def test_ignores_non_dict(self):
        assert mod._parse_severity({"severity": ["x", 1]}) == []


class TestSummariseRecord:
    def test_full_summary(self):
        s = mod._summarise_record(_cve_record_with_packages())
        assert s["osv_id"] == "CVE-2021-44228"
        assert s["affected_ecosystems"] == ["Maven"]
        assert s["aliases"] == ["GHSA-jfh8-c2jp-5v3q"]
        assert len(s["references"]) == 1

    def test_details_fallback_for_summary(self):
        s = mod._summarise_record(_cve_record_no_packages())
        assert s["summary"] == "xz backdoor"

    def test_references_capped_at_20(self):
        rec = {"id": "X", "references": [{"url": f"https://e/{i}"} for i in range(30)]}
        s = mod._summarise_record(rec)
        assert len(s["references"]) == 20


# ---------------------------------------------------------------------------
# _osv_get_with_retry — retry/back-off
# ---------------------------------------------------------------------------
class TestRetry:
    def test_success_first_try(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod.requests, "get", return_value=_make_response(200, {"id": "X"})) as g:
            resp = mod._osv_get_with_retry("CVE-2021-44228")
        assert resp.status_code == 200
        assert g.call_count == 1

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        monkeypatch.setattr(mod, "_OSV_MAX_RETRIES", 3)
        responses = [_make_response(429), _make_response(200, {"id": "X"})]
        with patch.object(mod.requests, "get", side_effect=responses) as g:
            with patch.object(mod.time, "sleep") as slp:
                resp = mod._osv_get_with_retry("CVE-1")
        assert resp.status_code == 200
        assert g.call_count == 2
        assert slp.call_count == 1

    def test_retries_on_503(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        monkeypatch.setattr(mod, "_OSV_MAX_RETRIES", 3)
        responses = [_make_response(503), _make_response(502), _make_response(200, {"id": "X"})]
        with patch.object(mod.requests, "get", side_effect=responses):
            with patch.object(mod.time, "sleep"):
                resp = mod._osv_get_with_retry("CVE-1")
        assert resp.status_code == 200

    def test_returns_last_retryable_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        monkeypatch.setattr(mod, "_OSV_MAX_RETRIES", 2)
        with patch.object(mod.requests, "get", return_value=_make_response(500)):
            with patch.object(mod.time, "sleep"):
                resp = mod._osv_get_with_retry("CVE-1")
        # After exhausting retries the final (still-500) response is returned.
        assert resp.status_code == 500

    def test_404_not_retried(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod.requests, "get", return_value=_make_response(404)) as g:
            resp = mod._osv_get_with_retry("CVE-NOPE")
        assert resp.status_code == 404
        assert g.call_count == 1

    def test_connection_error_retried(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        monkeypatch.setattr(mod, "_OSV_MAX_RETRIES", 3)
        side = [requests.exceptions.ConnectionError("boom"), _make_response(200, {"id": "X"})]
        with patch.object(mod.requests, "get", side_effect=side):
            with patch.object(mod.time, "sleep"):
                resp = mod._osv_get_with_retry("CVE-1")
        assert resp.status_code == 200

    def test_timeout_raised_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        monkeypatch.setattr(mod, "_OSV_MAX_RETRIES", 2)
        with patch.object(mod.requests, "get", side_effect=requests.exceptions.Timeout("t")):
            with patch.object(mod.time, "sleep"):
                with pytest.raises(requests.exceptions.Timeout):
                    mod._osv_get_with_retry("CVE-1")


# ---------------------------------------------------------------------------
# fetch_osv_data — end-to-end (mocked HTTP)
# ---------------------------------------------------------------------------
class TestFetchOsvData:
    def test_empty_cve_id(self):
        r = mod.fetch_osv_data("   ")
        assert r["found"] is False
        assert "Invalid CVE ID" in r["error"]

    def test_cve_with_direct_packages(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(
            mod, "_osv_get_with_retry", return_value=_make_response(200, _cve_record_with_packages())
        ) as g:
            r = mod.fetch_osv_data("CVE-2021-44228")
        # Only one call — no alias follow needed since packages present.
        assert g.call_count == 1
        assert r["found"] is True
        assert r["affected_ecosystems"] == ["Maven"]
        assert r["affected_package_count"] == 1
        assert "Maven" in r["message"]

    def test_follows_ghsa_alias_when_no_packages(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)

        def fake_get(osv_id):
            if osv_id == "CVE-2024-3094":
                return _make_response(200, _cve_record_no_packages())
            if osv_id == "GHSA-rxwq-x6h5-x525":
                return _make_response(200, _ghsa_record_with_packages())
            return _make_response(404)

        with patch.object(mod, "_osv_get_with_retry", side_effect=fake_get) as g:
            r = mod.fetch_osv_data("CVE-2024-3094")
        assert g.call_count == 2  # CVE record + GHSA alias
        assert r["found"] is True
        assert "Alpine:v3.20" in r["affected_ecosystems"]
        assert r["affected_package_count"] == 1
        # Both records merged, deduped.
        assert len(r["records"]) == 2

    def test_dedup_when_alias_returns_same_id(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        rec = _cve_record_no_packages()
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(200, rec)):
            r = mod.fetch_osv_data("CVE-2024-3094")
        # primary added; alias GHSA fetch returns same id => deduped
        ids = [rec["osv_id"] for rec in r["records"]]
        assert len(ids) == len(set(ids))

    def test_404_returns_not_found(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(404)):
            r = mod.fetch_osv_data("CVE-0000-0000")
        assert r["found"] is False
        assert "No OSV.dev record" in r["message"]

    def test_transport_failure_captured(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod, "_osv_get_with_retry", side_effect=requests.exceptions.Timeout("t")):
            r = mod.fetch_osv_data("CVE-1")
        assert r["found"] is False
        assert "failed" in r["message"].lower()

    def test_bad_json_captured(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        resp = _make_response(200)
        resp.json.side_effect = ValueError("bad json")
        with patch.object(mod, "_osv_get_with_retry", return_value=resp):
            r = mod.fetch_osv_data("CVE-1")
        assert r["found"] is False
        assert "unusable" in r["message"].lower()

    def test_alias_follow_best_effort_on_error(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)

        def fake_get(osv_id):
            if osv_id == "CVE-2024-3094":
                return _make_response(200, _cve_record_no_packages())
            raise requests.exceptions.ConnectionError("alias down")

        with patch.object(mod, "_osv_get_with_retry", side_effect=fake_get):
            r = mod.fetch_osv_data("CVE-2024-3094")
        # Primary still returned; alias failure swallowed.
        assert r["found"] is True
        assert r["affected_package_count"] == 0
        assert "no package-level" in r["message"]

    def test_record_no_packages_no_ghsa_alias(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        rec = {"id": "CVE-9", "aliases": ["CVE-other"], "affected": []}
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(200, rec)) as g:
            r = mod.fetch_osv_data("CVE-9")
        assert g.call_count == 1  # no GHSA alias to follow
        assert r["found"] is True
        assert r["affected_ecosystems"] == []


# ---------------------------------------------------------------------------
# get_osv_data — Strands tool entry point
# ---------------------------------------------------------------------------
class TestToolEntryPoint:
    def test_invalid_cve_id_error(self):
        res = mod.get_osv_data({"toolUseId": "t1", "input": {"cve_id": "  "}})
        assert res["status"] == "error"
        assert res["toolUseId"] == "t1"

    def test_missing_input_key(self):
        res = mod.get_osv_data({"toolUseId": "t2", "input": {}})
        assert res["status"] == "error"

    def test_success_status_and_json_content(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(200, _cve_record_with_packages())):
            res = mod.get_osv_data(_make_tool_use())
        assert res["status"] == "success"
        payload = json.loads(res["content"][0]["text"])
        assert payload["found"] is True
        assert payload["affected_ecosystems"] == ["Maven"]

    def test_not_found_is_error_status(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(404)):
            res = mod.get_osv_data(_make_tool_use("CVE-0000-0000"))
        assert res["status"] == "error"

    def test_content_is_valid_json(self, monkeypatch):
        monkeypatch.setattr(mod, "_OSV_RETRY_BASE_DELAY", 0)
        with patch.object(mod, "_osv_get_with_retry", return_value=_make_response(200, _cve_record_with_packages())):
            res = mod.get_osv_data(_make_tool_use())
        # Must round-trip through json.loads without error.
        json.loads(res["content"][0]["text"])


# ---------------------------------------------------------------------------
# VI agent wiring
# ---------------------------------------------------------------------------
class TestViAgentWiring:
    def test_tool_importable_from_vi_agent_path(self):
        from manus_agent.tools.get_osv_data import get_osv_data

        assert callable(get_osv_data)

    def test_vi_agent_source_registers_tool(self):
        import inspect

        from manus_agent.agents import vi_agent

        src = inspect.getsource(vi_agent)
        assert "get_osv_data" in src
        assert "OSV.dev" in src  # system-prompt step present
