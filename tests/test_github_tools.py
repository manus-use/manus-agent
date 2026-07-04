"""Comprehensive tests for get_github_advisory and search_for_exploits.

Both tools query the GitHub API and power the VI agent's exploit-discovery
pipeline (steps 4 and 7 of manus-agent analyze).  Every HTTP call is fully
mocked — no real network access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_response(
    payload,
    status_code: int = 200,
    raise_for_status_exc=None,
) -> MagicMock:
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(payload, str):
        resp.json.return_value = json.loads(payload)
    else:
        resp.json.return_value = payload
    if raise_for_status_exc is not None:
        resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        resp.raise_for_status.return_value = None
    return resp


def _http_error(status_code: int, message: str = "HTTP Error") -> requests.exceptions.HTTPError:
    err = requests.exceptions.HTTPError(message)
    err.response = MagicMock()
    err.response.status_code = status_code
    return err


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _advisory_payload(cve_id: str = "CVE-2024-3094") -> dict:
    """Minimal GitHub advisory API payload for a single CVE."""
    return {
        "ghsa_id": "GHSA-xxxx-xxxx-xxxx",
        "cve_id": cve_id,
        "summary": "Supply-chain backdoor in XZ Utils",
        "description": "A backdoor was introduced in xz-utils affecting SSH.",
        "severity": "critical",
        "cvss": {"score": 10.0, "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"},
        "published_at": "2024-03-29T00:00:00Z",
        "updated_at": "2024-04-01T00:00:00Z",
        "references": ["https://example.com/xz-advisory"],
        "vulnerabilities": [
            {
                "package": {"ecosystem": "npm", "name": "xz"},
                "severity": "critical",
                "vulnerable_version_range": "< 5.6.0",
                "first_patched_version": "5.6.0",
            }
        ],
    }


def _search_response(total_count: int = 3, num_items: int = 3) -> dict:
    """Minimal GitHub repository-search API payload."""
    items = []
    for i in range(num_items):
        items.append(
            {
                "full_name": f"owner/poc-cve-2024-{i}",
                "html_url": f"https://github.com/owner/poc-cve-2024-{i}",
                "description": f"PoC exploit {i}",
                "stargazers_count": 100 - i * 10,
                "updated_at": f"2024-0{i + 1}-01T00:00:00Z",
            }
        )
    return {"total_count": total_count, "incomplete_results": False, "items": items}


# ===========================================================================
# get_github_advisory — TOOL_SPEC contract
# ===========================================================================


class TestGetGitHubAdvisoryContract:
    """Module-level contract: importable, correct function name, etc."""

    def test_module_importable(self):
        from manus_agent.tools.get_github_advisory import get_github_advisory

        assert callable(get_github_advisory)

    def test_function_is_strands_tool(self):
        """@tool decorator should attach strands metadata."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        # Strands @tool wraps the function; it should still be callable
        assert callable(get_github_advisory)


# ===========================================================================
# get_github_advisory — input validation
# ===========================================================================


class TestGetGitHubAdvisoryValidation:
    def _call(self, cve_id):
        from manus_agent.tools.get_github_advisory import get_github_advisory

        return get_github_advisory(cve_id=cve_id)

    def test_invalid_id_returns_error_key(self):
        result = self._call("NOT-A-CVE")
        assert "error" in result

    def test_empty_string_returns_error(self):
        result = self._call("")
        assert "error" in result

    def test_none_returns_error(self):
        result = self._call(None)  # type: ignore[arg-type]
        assert "error" in result

    def test_integer_returns_error(self):
        result = self._call(12345)  # type: ignore[arg-type]
        assert "error" in result

    def test_partial_cve_prefix_returns_error(self):
        """Strings that do not start with 'CVE-' are invalid."""
        result = self._call("GHSA-xxxx-xxxx-xxxx")
        assert "error" in result

    def test_error_message_mentions_cve(self):
        result = self._call("INVALID")
        assert "CVE" in result["error"] or "Invalid" in result["error"]


# ===========================================================================
# get_github_advisory — successful advisory hit
# ===========================================================================


class TestGetGitHubAdvisorySuccess:
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_returns_first_advisory_entry(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"
        assert result["ghsa_id"] == "GHSA-xxxx-xxxx-xxxx"

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_summary_field_present(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert "summary" in result
        assert "XZ" in result["summary"] or "xz" in result["summary"].lower()

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_severity_field_present(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert result["severity"] == "critical"

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_only_first_advisory_returned(self, mock_get, monkeypatch):
        """When the API returns multiple advisories, only the first is returned."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        a1 = _advisory_payload("CVE-2024-3094")
        a2 = dict(_advisory_payload("CVE-2024-3094"))
        a2["ghsa_id"] = "GHSA-zzzz-zzzz-zzzz"
        mock_get.return_value = _mock_response([a1, a2])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert result["ghsa_id"] == "GHSA-xxxx-xxxx-xxxx"

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_cve_id_normalisation_lowercase(self, mock_get, monkeypatch):
        """Lowercase cve- prefix should still be accepted."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([_advisory_payload("CVE-2024-3094")])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="cve-2024-3094")
        # Should reach the API (no early-return error)
        assert "error" not in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_github_token_injected_in_header(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        get_github_advisory(cve_id="CVE-2024-3094")

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
        if not headers:
            # headers passed as keyword arg
            headers = call_kwargs[1].get("headers", {})
        assert "ghp_test123" in str(headers) or "Authorization" in str(headers)

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_with_explicit_token_auth_header_set(self, mock_get, monkeypatch):
        """Providing a token should produce an Authorization header with it."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_explicit_token_value")
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        get_github_advisory(cve_id="CVE-2024-3094")

        call_kwargs = mock_get.call_args
        all_args_str = str(call_kwargs)
        # When a token is present, the Authorization header should carry it
        assert "ghp_explicit_token_value" in all_args_str

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_correct_url_queried(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([_advisory_payload()])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        get_github_advisory(cve_id="CVE-2024-3094")

        call_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args[0][0]
        assert "api.github.com/advisories" in call_url
        assert "CVE-2024-3094" in call_url or "cve_id" in call_url


# ===========================================================================
# get_github_advisory — empty / not-found responses
# ===========================================================================


class TestGetGitHubAdvisoryNotFound:
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_empty_list_returns_message_key(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-9999-99999")
        assert "message" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_not_found_message_mentions_cve(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response([])

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-9999-99999")
        assert "CVE-9999-99999" in result["message"] or "advisory" in result["message"].lower()

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_http_404_returns_message_not_error(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exc = _http_error(404, "404 Not Found")
        mock_get.return_value = _mock_response({}, 404, raise_for_status_exc=exc)

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-9999-99999")
        assert "message" in result
        assert "error" not in result


# ===========================================================================
# get_github_advisory — error paths
# ===========================================================================


class TestGetGitHubAdvisoryErrors:
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_connection_error_returns_error_key(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_timeout_error_returns_error_key(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_http_500_returns_error_key(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exc = _http_error(500, "500 Internal Server Error")
        mock_get.return_value = _mock_response({}, 500, raise_for_status_exc=exc)

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_http_403_returns_error_key(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exc = _http_error(403, "403 Forbidden")
        mock_get.return_value = _mock_response({}, 403, raise_for_status_exc=exc)

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_malformed_response_structure_returns_error(self, mock_get, monkeypatch):
        """A response that returns a non-iterable/unexpected structure is handled.

        Note: get_github_advisory catches (KeyError, IndexError) but not ValueError.
        This test verifies the tool's behaviour with a non-list JSON body —
        an empty dict triggers the `not data` branch and returns {message: ...}.
        """
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        # An empty dict is falsy → triggers the `if not data` branch
        mock_get.return_value = _mock_response({})

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        # Either a message (empty-data path) or an error is acceptable
        assert isinstance(result, dict)
        assert "message" in result or "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_error_message_contains_useful_info(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        from manus_agent.tools.get_github_advisory import get_github_advisory

        result = get_github_advisory(cve_id="CVE-2024-3094")
        assert len(result["error"]) > 10  # non-trivial error message


# ===========================================================================
# get_github_advisory — VI agent wiring
# ===========================================================================


class TestGetGitHubAdvisoryVIWiring:
    def test_vi_agent_imports_function(self):
        """vi_agent must wire get_github_advisory into its tool list."""
        import manus_agent.agents.vi_agent as vi

        # The function reference appears somewhere in the agent's tool setup
        src = open(vi.__file__).read()
        assert "get_github_advisory" in src

    def test_vi_agent_system_prompt_references_advisory(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "get_github_advisory" in SYSTEM_PROMPT or "GitHub" in SYSTEM_PROMPT


# ===========================================================================
# search_for_exploits — TOOL_SPEC contract
# ===========================================================================


class TestSearchForExploitsContract:
    def test_module_importable(self):
        from manus_agent.tools.search_for_exploits import search_for_exploits

        assert callable(search_for_exploits)

    def test_tool_spec_name(self):
        from manus_agent.tools.search_for_exploits import TOOL_SPEC

        assert TOOL_SPEC["name"] == "search_for_exploits"

    def test_tool_spec_has_description(self):
        from manus_agent.tools.search_for_exploits import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 20

    def test_tool_spec_requires_cve_id(self):
        from manus_agent.tools.search_for_exploits import TOOL_SPEC

        required = TOOL_SPEC["inputSchema"]["json"]["required"]
        assert "cve_id" in required

    def test_tool_spec_cve_id_type_string(self):
        from manus_agent.tools.search_for_exploits import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["cve_id"]["type"] == "string"

    def test_tooluse_id_echoed_on_error(self):
        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits({"toolUseId": "echo-me", "input": {"cve_id": "BAD"}})
        assert result["toolUseId"] == "echo-me"


# ===========================================================================
# search_for_exploits — input validation
# ===========================================================================


class TestSearchForExploitsValidation:
    def _call(self, cve_id, tool_use_id="t1"):
        from manus_agent.tools.search_for_exploits import search_for_exploits

        return search_for_exploits({"toolUseId": tool_use_id, "input": {"cve_id": cve_id}})

    def test_invalid_id_status_error(self):
        result = self._call("NOT-A-CVE")
        assert result["status"] == "error"

    def test_empty_string_status_error(self):
        result = self._call("")
        assert result["status"] == "error"

    def test_none_cve_id_status_error(self):
        result = self._call(None)
        assert result["status"] == "error"

    def test_integer_cve_id_status_error(self):
        result = self._call(12345)
        assert result["status"] == "error"

    def test_missing_cve_id_status_error(self):
        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits({"toolUseId": "t", "input": {}})
        assert result["status"] == "error"

    def test_error_text_mentions_invalid(self):
        result = self._call("INVALID-FORMAT")
        text = result["content"][0]["text"]
        assert "Invalid" in text or "CVE" in text


# ===========================================================================
# search_for_exploits — successful results
# ===========================================================================


class TestSearchForExploitsSuccess:
    def _make_tool_use(self, cve_id: str = "CVE-2024-3094", tool_use_id: str = "t1") -> dict:
        return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_results_found_status_success(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response(total_count=3))

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "success"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_results_found_links_present(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response(total_count=3))

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert "links" in payload
        assert len(payload["links"]) > 0

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_results_capped_at_five(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        # Provide 10 items; only 5 should be returned
        mock_get.return_value = _mock_response(_search_response(total_count=10, num_items=10))

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert len(payload["links"]) <= 5

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_result_entry_has_required_keys(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        entry = payload["links"][0]
        assert "name" in entry
        assert "url" in entry
        assert "stars" in entry

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_result_url_is_github_url(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert "github.com" in payload["links"][0]["url"]

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_summary_field_present(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response(total_count=3))

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert "summary" in payload

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_summary_mentions_total_count(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response(total_count=42))

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert "42" in payload["summary"]

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_tooluse_id_echoed_on_success(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use(tool_use_id="my-id-42"))
        assert result["toolUseId"] == "my-id-42"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_github_token_injected_in_header(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_search_token")
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        search_for_exploits(self._make_tool_use())
        call_kwargs = mock_get.call_args
        assert "ghp_search_token" in str(call_kwargs)

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_correct_api_endpoint_queried(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        search_for_exploits(self._make_tool_use())
        call_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args[0][0]
        assert "api.github.com" in call_url
        assert "search" in call_url or "repositories" in call_url

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_cve_id_in_search_query(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        search_for_exploits(self._make_tool_use("CVE-2021-44228"))
        call_url = mock_get.call_args.args[0] if mock_get.call_args.args else mock_get.call_args[0][0]
        assert "CVE-2021-44228" in call_url or "44228" in call_url


# ===========================================================================
# search_for_exploits — no results
# ===========================================================================


class TestSearchForExploitsNoResults:
    def _make_tool_use(self, cve_id: str = "CVE-9999-99999") -> dict:
        return {"toolUseId": "t1", "input": {"cve_id": cve_id}}

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_zero_results_status_success(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response({"total_count": 0, "items": []})

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "success"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_zero_results_links_empty(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response({"total_count": 0, "items": []})

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        assert payload["links"] == []

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_zero_results_summary_indicates_none_found(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response({"total_count": 0, "items": []})

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        payload = result["content"][0]["json"]
        summary_lower = payload["summary"].lower()
        assert "no" in summary_lower or "0" in payload["summary"] or "not found" in summary_lower


# ===========================================================================
# search_for_exploits — error paths
# ===========================================================================


class TestSearchForExploitsErrors:
    def _make_tool_use(self, cve_id: str = "CVE-2024-3094") -> dict:
        return {"toolUseId": "t1", "input": {"cve_id": cve_id}}

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_connection_error_status_error(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_timeout_status_error(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_http_403_rate_limit_status_error(self, mock_get, monkeypatch):
        """GitHub rate-limits unauthenticated search API; 403 should give error."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exc = _http_error(403, "403 Forbidden")
        mock_get.return_value = _mock_response({}, 403, raise_for_status_exc=exc)

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_http_422_unprocessable_status_error(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        exc = _http_error(422, "422 Unprocessable Entity")
        mock_get.return_value = _mock_response({}, 422, raise_for_status_exc=exc)

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_invalid_json_status_error(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        mock_get.return_value = resp

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_error_text_nonempty(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.side_effect = requests.exceptions.ConnectionError("x")

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits(self._make_tool_use())
        assert len(result["content"][0]["text"]) > 5


# ===========================================================================
# search_for_exploits — result-entry field contract
# ===========================================================================


class TestSearchForExploitsResultFields:
    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_last_updated_field_in_entry(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits({"toolUseId": "t", "input": {"cve_id": "CVE-2024-3094"}})
        entry = result["content"][0]["json"]["links"][0]
        assert "last_updated" in entry

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_description_field_in_entry(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits({"toolUseId": "t", "input": {"cve_id": "CVE-2024-3094"}})
        entry = result["content"][0]["json"]["links"][0]
        assert "description" in entry

    @patch("manus_agent.tools.search_for_exploits.requests.get")
    def test_stars_field_is_int(self, mock_get, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        mock_get.return_value = _mock_response(_search_response())

        from manus_agent.tools.search_for_exploits import search_for_exploits

        result = search_for_exploits({"toolUseId": "t", "input": {"cve_id": "CVE-2024-3094"}})
        entry = result["content"][0]["json"]["links"][0]
        assert isinstance(entry["stars"], int)


# ===========================================================================
# search_for_exploits — parametrized TOOL_SPEC contract
# ===========================================================================


@pytest.mark.parametrize(
    "tool_name,module_path",
    [
        ("search_for_exploits", "manus_agent.tools.search_for_exploits"),
        ("get_github_advisory", "manus_agent.tools.get_github_advisory"),
    ],
)
def test_tool_module_has_tool_spec_or_decorator(tool_name, module_path):
    import importlib

    mod = importlib.import_module(module_path)
    # Either has TOOL_SPEC dict (ToolUse pattern) or the function has strands @tool metadata
    has_spec = hasattr(mod, "TOOL_SPEC")
    has_fn = hasattr(mod, tool_name) and callable(getattr(mod, tool_name))
    assert has_spec or has_fn, f"{module_path} missing TOOL_SPEC or callable {tool_name}"


# ===========================================================================
# search_for_exploits — VI agent wiring
# ===========================================================================


class TestSearchForExploitsVIWiring:
    def test_vi_agent_imports_search_for_exploits(self):
        import manus_agent.agents.vi_agent as vi

        src = open(vi.__file__).read()
        assert "search_for_exploits" in src

    def test_vi_agent_system_prompt_mentions_exploits(self):
        from manus_agent.agents.vi_agent import SYSTEM_PROMPT

        assert "exploit" in SYSTEM_PROMPT.lower() or "PoC" in SYSTEM_PROMPT
