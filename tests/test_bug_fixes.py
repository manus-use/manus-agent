"""Regression tests for two uncaught-exception bugs.

Bug 1 — submit_cves: raise ValueError for missing webhook URL was *outside* the
try/except block, so it propagated to the caller instead of being caught and
returned as {"status": "error"}.  Fix: moved URL resolution + raise inside the
existing try block so the broad ``except Exception`` handler catches it.

Bug 2 — get_github_advisory: the except clause only listed (KeyError, IndexError),
missing ValueError.  response.json() raises ValueError (/ JSONDecodeError, a
ValueError subclass) when the body is malformed — that escaped silently.  Fix:
add ValueError to the except tuple.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers shared across both sections
# ---------------------------------------------------------------------------


def _make_submit_tool_use(cves: list | None = None, tool_use_id: str = "tu-001") -> dict:
    cves = cves or [{"cve_id": "CVE-2024-1234", "priority": "CRITICAL"}]
    return {"toolUseId": tool_use_id, "input": {"cve_list": cves}}


# ===========================================================================
# Bug 1 — submit_cves: missing webhook URL must NOT raise, must return error
# ===========================================================================


class TestSubmitCvesMissingUrl:
    """submit_cves must return {status: error} when no webhook URL is configured."""

    def _call(self, cves=None):
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use(cves)
        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch.dict("os.environ", {}, clear=False),
        ):
            # Simulate no cve_submit_url in config
            cfg_instance = MagicMock()
            cfg_instance.webhooks = None
            mock_cfg.from_file.return_value = cfg_instance
            # Ensure env var is also absent
            import os

            os.environ.pop("CVE_SUBMIT_URL", None)
            return submit_cves(tool_use)

    def test_missing_url_returns_error_status(self):
        """Must return status='error', not raise ValueError."""
        result = self._call()
        assert result["status"] == "error", (
            "submit_cves raised instead of returning error — missing-URL ValueError escaped the try block"
        )

    def test_missing_url_does_not_raise(self):
        """Calling submit_cves with no URL must never raise to the caller."""
        try:
            self._call()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"submit_cves raised an exception to caller: {exc!r}")

    def test_missing_url_error_has_tool_use_id(self):
        result = self._call()
        assert result["toolUseId"] == "tu-001"

    def test_missing_url_error_content_mentions_url(self):
        """Error text should hint at what config to set."""
        result = self._call()
        content_text = result["content"][0]["text"].lower()
        assert "url" in content_text or "webhook" in content_text or "cve_submit" in content_text

    def test_missing_url_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # Confirm that the fix works end-to-end: when a URL *is* present the
    # happy path still calls requests.post and returns success.
    # ------------------------------------------------------------------

    def test_with_valid_url_calls_post(self):
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use([{"cve_id": "CVE-2024-9999", "priority": "HIGH"}])
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch("manus_agent.tools.submit_cves.requests.post", return_value=mock_response) as mock_post,
            patch("manus_agent.tools.submit_cves.analyze_affected_assets"),
            patch.dict("os.environ", {"CVE_SUBMIT_URL": "https://example.com/hook"}),
        ):
            cfg_instance = MagicMock()
            cfg_instance.webhooks = None
            mock_cfg.from_file.return_value = cfg_instance

            result = submit_cves(tool_use)

        assert result["status"] == "success"
        mock_post.assert_called_once()

    def test_with_valid_url_via_config(self):
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use([{"cve_id": "CVE-2024-8888", "priority": "MEDIUM"}])
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch("manus_agent.tools.submit_cves.requests.post", return_value=mock_response),
            patch("manus_agent.tools.submit_cves.analyze_affected_assets"),
        ):
            webhooks_mock = MagicMock()
            webhooks_mock.cve_submit_url = "https://config.example.com/hook"
            cfg_instance = MagicMock()
            cfg_instance.webhooks = webhooks_mock
            mock_cfg.from_file.return_value = cfg_instance

            result = submit_cves(tool_use)

        assert result["status"] == "success"

    def test_http_error_still_returns_error_status(self):
        """HTTPError from requests.post must be caught and returned, not raised."""
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use()
        http_err = requests.exceptions.HTTPError("500 Server Error")

        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch("manus_agent.tools.submit_cves.requests.post", side_effect=http_err),
            patch.dict("os.environ", {"CVE_SUBMIT_URL": "https://example.com/hook"}),
        ):
            cfg_instance = MagicMock()
            cfg_instance.webhooks = None
            mock_cfg.from_file.return_value = cfg_instance

            result = submit_cves(tool_use)

        assert result["status"] == "error"
        assert "HTTP error" in result["content"][0]["text"]

    def test_connection_error_returns_error_status(self):
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use()

        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch(
                "manus_agent.tools.submit_cves.requests.post",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ),
            patch.dict("os.environ", {"CVE_SUBMIT_URL": "https://example.com/hook"}),
        ):
            cfg_instance = MagicMock()
            cfg_instance.webhooks = None
            mock_cfg.from_file.return_value = cfg_instance

            result = submit_cves(tool_use)

        assert result["status"] == "error"

    def test_env_url_takes_precedence_when_config_has_none(self):
        """CVE_SUBMIT_URL env var is used when config.webhooks is None."""
        from manus_agent.tools.submit_cves import submit_cves

        tool_use = _make_submit_tool_use([{"cve_id": "CVE-2024-7777", "priority": "LOW"}])
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        with (
            patch("manus_agent.tools.submit_cves.Config") as mock_cfg,
            patch("manus_agent.tools.submit_cves.requests.post", return_value=mock_response) as mock_post,
            patch("manus_agent.tools.submit_cves.analyze_affected_assets"),
            patch.dict("os.environ", {"CVE_SUBMIT_URL": "https://env.example.com/hook"}),
        ):
            cfg_instance = MagicMock()
            cfg_instance.webhooks = None
            mock_cfg.from_file.return_value = cfg_instance

            result = submit_cves(tool_use)

        assert result["status"] == "success"
        call_kwargs = mock_post.call_args
        assert "env.example.com" in call_kwargs[0][0]


# ===========================================================================
# Bug 2 — get_github_advisory: ValueError from response.json() must be caught
# ===========================================================================


class TestGithubAdvisoryJsonError:
    """get_github_advisory must catch ValueError (malformed JSON) and return error dict."""

    def _make_bad_json_response(self, status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        return mock_resp

    def test_malformed_json_returns_error_dict(self):
        """ValueError from response.json() must be caught — not escape to caller."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=self._make_bad_json_response()),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-9999")

        assert isinstance(result, dict)
        assert "error" in result

    def test_malformed_json_does_not_raise(self):
        """Must never propagate ValueError to caller."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=self._make_bad_json_response()),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            try:
                get_github_advisory("CVE-2023-9999")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"get_github_advisory raised to caller: {exc!r}")

    def test_malformed_json_error_message_mentions_format(self):
        from manus_agent.tools.get_github_advisory import get_github_advisory

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=self._make_bad_json_response()),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-9999")

        assert "unexpected" in result["error"].lower() or "format" in result["error"].lower()

    def test_json_decode_error_subclass_also_caught(self):
        """json.JSONDecodeError is a subclass of ValueError — must also be caught."""
        import json

        from manus_agent.tools.get_github_advisory import get_github_advisory

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=mock_resp),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-5555")

        assert "error" in result
        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # Confirm existing error-path handlers still work after the change
    # ------------------------------------------------------------------

    def test_key_error_still_caught(self):
        """KeyError is listed in the except clause — a non-list response triggers IndexError."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        # Returning a non-subscriptable string-like object as json() output triggers
        # an IndexError when data[0] is accessed — verifying IndexError is still caught.
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        # json() returns an object whose [0] raises IndexError
        bad_data = MagicMock()
        bad_data.__bool__ = MagicMock(return_value=True)  # so ``if not data`` is False
        bad_data.__getitem__ = MagicMock(side_effect=IndexError("index out of range"))
        mock_resp.json.return_value = bad_data

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=mock_resp),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-1111")

        assert isinstance(result, dict)
        assert "error" in result

    def test_http_404_still_returns_not_found(self):
        from manus_agent.tools.get_github_advisory import get_github_advisory

        http_err = requests.exceptions.HTTPError(response=MagicMock(status_code=404))

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", side_effect=http_err),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-0000")

        assert "message" in result
        assert "No advisory" in result["message"]

    def test_valid_cve_success_path_unaffected(self):
        """The fix must not break the happy path."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        advisory = {"ghsa_id": "GHSA-abcd-1234", "cve_id": "CVE-2023-1234", "summary": "Test advisory"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [advisory]

        with (
            patch("manus_agent.tools.get_github_advisory.requests.get", return_value=mock_resp),
            patch("manus_agent.tools.get_github_advisory.Config") as mock_cfg,
            patch("manus_agent.tools.get_github_advisory.log_tool_output_size"),
        ):
            mock_cfg.from_file.side_effect = Exception("no config")
            result = get_github_advisory("CVE-2023-1234")

        assert result["ghsa_id"] == "GHSA-abcd-1234"

    def test_invalid_cve_id_still_returns_error_immediately(self):
        """Input validation short-circuit must not be affected."""
        from manus_agent.tools.get_github_advisory import get_github_advisory

        with patch("manus_agent.tools.get_github_advisory.log_tool_output_size"):
            result = get_github_advisory("NOT-A-CVE")

        assert "error" in result
        assert "Invalid" in result["error"]
