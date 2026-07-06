"""
Tests for submit_cves, tool_output_logger, and verify_exploit helpers.

All external I/O (requests, Docker, MCP) is mocked — no real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cve(
    cve_id: str = "CVE-2024-3094",
    priority: str = "CRITICAL",
    cvss_score: str = "Critical(9.8),CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    epss_score: str = "0.94",
    epss_percentile: str = "0.999",
    affected_products: str = "xz-utils",
    cisa_kev: bool = True,
    exploited: bool = True,
    cwe: str = "CWE-506",
) -> dict:
    return {
        "cve_id": cve_id,
        "priority": priority,
        "cvss_score": cvss_score,
        "epss_score": epss_score,
        "epss_percentile": epss_percentile,
        "affected_products": affected_products,
        "cisa_kev": cisa_kev,
        "exploited": exploited,
        "cwe": cwe,
    }


def _make_tool_use(cve_list: list | None = None) -> dict:
    if cve_list is None:
        cve_list = [_make_cve()]
    return {"toolUseId": "test-001", "input": {"cve_list": cve_list}}


def _make_ok_response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    return resp


def _make_http_error_response() -> MagicMock:
    import requests

    resp = MagicMock()
    err = requests.exceptions.HTTPError("500 Server Error")
    resp.raise_for_status.side_effect = err
    return resp


# ===========================================================================
# tool_output_logger — log_tool_output_size
# ===========================================================================


class TestLogToolOutputSize:
    def test_text_content_logs_without_exception(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {
            "content": [{"text": "hello world"}],
        }
        log_tool_output_size("my_tool", result)
        captured = capsys.readouterr()
        assert "my_tool" in captured.out
        assert "11" in captured.out  # len("hello world") == 11

    def test_json_content_logs_size(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {
            "content": [{"json": {"key": "value", "num": 42}}],
        }
        log_tool_output_size("json_tool", result)
        captured = capsys.readouterr()
        assert "json_tool" in captured.out

    def test_mixed_text_and_json_content(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {
            "content": [
                {"text": "abc"},
                {"json": {"k": "v"}},
            ],
        }
        log_tool_output_size("mixed_tool", result)
        captured = capsys.readouterr()
        assert "mixed_tool" in captured.out

    def test_empty_content_list_no_output(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": []}
        log_tool_output_size("empty_tool", result)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_content_key_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("notool", {})  # Should not raise

    def test_non_dict_result_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("notool", "bad result type")  # Should not raise

    def test_none_result_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("notool", None)  # Should not raise

    def test_multiple_text_items_sizes_summed(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {
            "content": [
                {"text": "abc"},  # 3 chars
                {"text": "defg"},  # 4 chars
            ],
        }
        log_tool_output_size("sum_tool", result)
        captured = capsys.readouterr()
        # Total is 7 chars
        assert "7" in captured.out

    def test_content_item_without_text_or_json_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"unknown_key": "value"}]}
        log_tool_output_size("notool", result)  # Should not raise

    def test_non_string_text_value_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"text": 12345}]}  # text is int, not str
        log_tool_output_size("notool", result)  # Should not raise

    def test_json_field_with_unserializable_value_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        # The implementation uses str(), which handles any object
        result = {"content": [{"json": object()}]}
        log_tool_output_size("notool", result)  # Should not raise

    def test_content_not_list_no_exception(self):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": "not a list"}
        log_tool_output_size("notool", result)  # Should not raise

    def test_tool_name_appears_in_output(self, capsys):
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"text": "x"}]}
        log_tool_output_size("verify_exploit", result)
        captured = capsys.readouterr()
        assert "verify_exploit" in captured.out


# ===========================================================================
# submit_cves — input validation
# ===========================================================================


class TestSubmitCvesInputValidation:
    def test_empty_cve_list_returns_error(self, monkeypatch):
        monkeypatch.delenv("CVE_SUBMIT_URL", raising=False)
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves({"toolUseId": "t1", "input": {"cve_list": []}})
        assert result["status"] == "error"
        assert "cve_list" in result["content"][0]["text"].lower() or "cannot be empty" in result["content"][0]["text"]

    def test_missing_cve_list_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("CVE_SUBMIT_URL", raising=False)
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves({"toolUseId": "t1", "input": {}})
        assert result["status"] == "error"

    def test_result_has_tool_use_id(self, monkeypatch):
        monkeypatch.delenv("CVE_SUBMIT_URL", raising=False)
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves({"toolUseId": "my-id-123", "input": {"cve_list": []}})
        assert result["toolUseId"] == "my-id-123"


# ===========================================================================
# submit_cves — missing URL (no config, no env)
# ===========================================================================


class TestSubmitCvesMissingUrl:
    def test_no_url_returns_error_status(self, monkeypatch):
        """When neither config nor env has a webhook URL, submit_cves returns an
        error-status dict instead of raising ValueError to the caller."""
        monkeypatch.delenv("CVE_SUBMIT_URL", raising=False)
        with patch("manus_agent.tools.submit_cves.Config") as mock_config:
            cfg = MagicMock()
            cfg.webhooks = None
            mock_config.from_file.return_value = cfg

            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use())
            assert result["status"] == "error"

    def test_no_url_error_message_mentions_config_toml(self, monkeypatch):
        """Error message should guide users to config.toml."""
        monkeypatch.delenv("CVE_SUBMIT_URL", raising=False)
        with patch("manus_agent.tools.submit_cves.Config") as mock_config:
            cfg = MagicMock()
            cfg.webhooks = None
            mock_config.from_file.return_value = cfg

            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use())
            assert result["status"] == "error"
            assert "config.toml" in result["content"][0]["text"]


# ===========================================================================
# submit_cves — successful submission
# ===========================================================================


class TestSubmitCvesSuccess:
    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_single_cve_submitted_successfully(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use([_make_cve()]))
        assert result["status"] == "success"

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_success_message_contains_cve_id(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use([_make_cve("CVE-2021-44228")]))
        assert "CVE-2021-44228" in result["content"][0]["text"]

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_multiple_cves_submitted(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()
        cves = [_make_cve("CVE-2024-3094"), _make_cve("CVE-2021-44228", priority="HIGH")]
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use(cves))
        assert result["status"] == "success"
        assert mock_post.call_count == 2

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_success_count_in_message(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()
        cves = [_make_cve("CVE-2024-3094"), _make_cve("CVE-2021-44228", priority="HIGH")]
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use(cves))
        assert "2" in result["content"][0]["text"]

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_post_called_with_correct_url(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://test-webhook.example.com/submit")
        mock_post.return_value = _make_ok_response()
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            submit_cves(_make_tool_use())
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://test-webhook.example.com/submit"

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_tool_use_id_preserved_on_success(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()
        with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves({"toolUseId": "id-abc", "input": {"cve_list": [_make_cve()]}})
        assert result["toolUseId"] == "id-abc"


# ===========================================================================
# submit_cves — HTTP error handling
# ===========================================================================


class TestSubmitCvesHttpErrors:
    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_http_error_returns_error_status(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_http_error_response()
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves(_make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_http_error_message_contains_hint(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_http_error_response()
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves(_make_tool_use())
        assert "error" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_connection_error_returns_error_status(self, mock_post, monkeypatch):
        import requests as req_lib

        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.side_effect = req_lib.exceptions.ConnectionError("network down")
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves(_make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_unexpected_exception_returns_error_status(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.side_effect = RuntimeError("unexpected internal error")
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves(_make_tool_use())
        assert result["status"] == "error"

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_tool_use_id_preserved_on_error(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_http_error_response()
        from manus_agent.tools.submit_cves import submit_cves

        result = submit_cves({"toolUseId": "err-id", "input": {"cve_list": [_make_cve()]}})
        assert result["toolUseId"] == "err-id"


# ===========================================================================
# submit_cves — critical CVE filtering (analyze_affected_assets)
# ===========================================================================


class TestSubmitCvesCriticalFiltering:
    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_only_critical_and_high_passed_to_analyze(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()

        critical = _make_cve("CVE-2024-3094", priority="CRITICAL")
        high = _make_cve("CVE-2021-44228", priority="HIGH")
        medium = _make_cve("CVE-2022-12345", priority="MEDIUM")
        low = _make_cve("CVE-2023-99999", priority="LOW")

        captured_cves: list = []

        def fake_analyze(cves):
            captured_cves.extend(cves)

        with patch("manus_agent.tools.submit_cves.analyze_affected_assets", side_effect=fake_analyze):
            from manus_agent.tools.submit_cves import submit_cves

            result = submit_cves(_make_tool_use([critical, high, medium, low]))

        assert result["status"] == "success"
        assert "CVE-2024-3094" in captured_cves
        assert "CVE-2021-44228" in captured_cves
        assert "CVE-2022-12345" not in captured_cves
        assert "CVE-2023-99999" not in captured_cves

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_no_critical_or_high_analyze_called_with_empty(self, mock_post, monkeypatch):
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://example.com/webhook")
        mock_post.return_value = _make_ok_response()

        medium = _make_cve("CVE-2022-12345", priority="MEDIUM")

        captured_cves: list = []

        def fake_analyze(cves):
            captured_cves.extend(cves)

        with patch("manus_agent.tools.submit_cves.analyze_affected_assets", side_effect=fake_analyze):
            from manus_agent.tools.submit_cves import submit_cves

            submit_cves(_make_tool_use([medium]))

        assert captured_cves == []

    @patch("manus_agent.tools.submit_cves.requests.post")
    def test_env_url_used_when_config_has_no_webhook(self, mock_post, monkeypatch):
        """CVE_SUBMIT_URL env var should be used when config has no webhook URL."""
        monkeypatch.setenv("CVE_SUBMIT_URL", "https://env-url.example.com/hook")
        mock_post.return_value = _make_ok_response()

        with patch("manus_agent.tools.submit_cves.Config") as mock_config:
            cfg = MagicMock()
            cfg.webhooks = None
            mock_config.from_file.return_value = cfg
            with patch("manus_agent.tools.submit_cves.analyze_affected_assets"):
                from manus_agent.tools.submit_cves import submit_cves

                result = submit_cves(_make_tool_use())

        assert result["status"] == "success"
        assert mock_post.call_args[0][0] == "https://env-url.example.com/hook"


# ===========================================================================
# submit_cves — TOOL_SPEC contract
# ===========================================================================


def test_submit_cves_tool_spec_name():
    from manus_agent.tools.submit_cves import TOOL_SPEC

    assert TOOL_SPEC["name"] == "submit_cves"


def test_submit_cves_tool_spec_has_input_schema():
    from manus_agent.tools.submit_cves import TOOL_SPEC

    assert "inputSchema" in TOOL_SPEC
    assert "json" in TOOL_SPEC["inputSchema"]


def test_submit_cves_callable():
    from manus_agent.tools.submit_cves import submit_cves

    assert callable(submit_cves)


def test_submit_cves_tool_spec_requires_cve_list():
    from manus_agent.tools.submit_cves import TOOL_SPEC

    required = TOOL_SPEC["inputSchema"]["json"].get("required", [])
    assert "cve_list" in required


# ===========================================================================
# verify_exploit — _truncate_text helper
# ===========================================================================


class TestTruncateText:
    def test_short_text_unchanged(self):
        from manus_agent.tools.verify_exploit import _truncate_text

        text = "hello\nworld"
        result = _truncate_text("label", text)
        assert result == text

    def test_empty_text_returned_unchanged(self):
        from manus_agent.tools.verify_exploit import _truncate_text

        result = _truncate_text("label", "")
        assert result == ""

    def test_line_truncation_at_max_lines(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        monkeypatch.setattr(ve_mod, "MAX_LOG_LINES", 3)
        monkeypatch.setattr(ve_mod, "MAX_LOG_CHARS", 1_000_000)

        lines = "\n".join(f"line{i}" for i in range(10))
        result = ve_mod._truncate_text("label", lines)
        assert result.startswith("[truncated")
        # Should keep last 3 lines: line7, line8, line9
        assert "line9" in result
        assert "line7" in result

    def test_char_truncation_at_max_chars(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        monkeypatch.setattr(ve_mod, "MAX_LOG_LINES", 10_000)
        monkeypatch.setattr(ve_mod, "MAX_LOG_CHARS", 5)

        text = "abcdefghij"  # 10 chars
        result = ve_mod._truncate_text("label", text)
        assert result.startswith("[truncated")
        assert "fghij" in result  # last 5 chars

    def test_text_within_limits_has_no_truncation_prefix(self):
        from manus_agent.tools.verify_exploit import _truncate_text

        text = "normal log output"
        result = _truncate_text("label", text)
        assert not result.startswith("[truncated")

    def test_exactly_at_line_limit_not_truncated(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        monkeypatch.setattr(ve_mod, "MAX_LOG_LINES", 5)
        monkeypatch.setattr(ve_mod, "MAX_LOG_CHARS", 1_000_000)

        text = "\n".join(f"line{i}" for i in range(5))
        result = ve_mod._truncate_text("label", text)
        assert not result.startswith("[truncated")

    def test_one_over_line_limit_triggers_truncation(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        monkeypatch.setattr(ve_mod, "MAX_LOG_LINES", 5)
        monkeypatch.setattr(ve_mod, "MAX_LOG_CHARS", 1_000_000)

        text = "\n".join(f"line{i}" for i in range(6))
        result = ve_mod._truncate_text("label", text)
        assert result.startswith("[truncated")


# ===========================================================================
# verify_exploit — _result helper
# ===========================================================================


class TestVerifyExploitResultHelper:
    def test_result_has_correct_structure(self):
        from manus_agent.tools.verify_exploit import _result

        r = _result(
            tool_use_id="tu-1",
            status="build_error",
            error_msg="build failed",
            build_log="some log",
            elapsed=1.5,
        )
        assert r["toolUseId"] == "tu-1"
        assert r["status"] == "success"  # outer always "success"
        payload = r["content"][0]["json"]
        assert payload["verification_status"] == "build_error"
        assert "build failed" in payload["summary"]
        assert payload["execution_time_seconds"] == 1.5

    def test_result_exploit_output_exit_code_is_minus_one(self):
        from manus_agent.tools.verify_exploit import _result

        r = _result("tu", "target_error", "msg", "log", 2.0)
        assert r["content"][0]["json"]["exploit_output"]["exit_code"] == -1

    def test_result_target_logs_default_empty(self):
        from manus_agent.tools.verify_exploit import _result

        r = _result("tu", "build_error", "msg", "log", 0.1)
        assert r["content"][0]["json"]["target_logs"] == ""

    def test_result_target_logs_passed_through(self):
        from manus_agent.tools.verify_exploit import _result

        r = _result("tu", "build_error", "msg", "log", 0.1, target_logs="some target output")
        assert r["content"][0]["json"]["target_logs"] == "some target output"

    def test_result_error_field_default_none(self):
        from manus_agent.tools.verify_exploit import _result

        r = _result("tu", "build_error", "msg", "log", 0.1)
        assert r["content"][0]["json"]["error"] is None

    def test_result_error_field_populated_when_provided(self):
        from manus_agent.tools.verify_exploit import _result

        err = {"category": "infra", "stage": "docker_preflight", "retryable": True, "message": "oops"}
        r = _result("tu", "infra_error", "msg", "log", 0.1, error=err)
        assert r["content"][0]["json"]["error"] == err


# ===========================================================================
# verify_exploit — _error_obj helper
# ===========================================================================


class TestVerifyExploitErrorObj:
    def test_error_obj_structure(self):
        from manus_agent.tools.verify_exploit import _error_obj

        exc = ValueError("bad input")
        obj = _error_obj(category="infra", stage="preflight", exc=exc, retryable=False)
        assert obj["category"] == "infra"
        assert obj["stage"] == "preflight"
        assert obj["retryable"] is False
        assert "ValueError" in obj["exception_type"]
        assert "bad input" in obj["message"]

    def test_error_obj_retryable_true(self):
        from manus_agent.tools.verify_exploit import _error_obj

        exc = RuntimeError("transient")
        obj = _error_obj(category="target", stage="run", exc=exc, retryable=True)
        assert obj["retryable"] is True

    def test_error_obj_module_in_exception_type(self):
        from manus_agent.tools.verify_exploit import _error_obj

        exc = OSError("disk full")
        obj = _error_obj(category="infra", stage="build", exc=exc, retryable=False)
        assert "." in obj["exception_type"]


# ===========================================================================
# verify_exploit — missing required inputs
# ===========================================================================


class TestVerifyExploitInputValidation:
    def test_missing_dockerfile_returns_error(self):
        from manus_agent.tools.verify_exploit import verify_exploit

        tool_use = {
            "toolUseId": "t1",
            "input": {
                "dockerfile_content": "",
                "exploit_code": "print('hi')",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = verify_exploit(tool_use)
        assert result["status"] == "error"
        assert "dockerfile_content" in result["content"][0]["text"] or "required" in result["content"][0]["text"]

    def test_missing_exploit_code_returns_error(self):
        from manus_agent.tools.verify_exploit import verify_exploit

        tool_use = {
            "toolUseId": "t1",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04",
                "exploit_code": "",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = verify_exploit(tool_use)
        assert result["status"] == "error"


# ===========================================================================
# verify_exploit — Docker preflight failure
# ===========================================================================


class TestVerifyExploitDockerPreflight:
    def test_docker_connection_error_returns_infra_error(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod
        from manus_agent.utils.docker_client import DockerConnectionError

        class FakeSandbox:
            build_log = ""

            def get_docker_ps_all(self):
                return ""

            def get_target_logs(self):
                return ""

            def get_target_exit_code(self):
                return None

            def cleanup(self):
                pass

        def fake_get_docker_client():
            raise DockerConnectionError(
                message="No Docker daemon",
                diagnosis="Docker is not running",
                remediation="Start Docker",
            )

        monkeypatch.setattr(ve_mod, "ExploitSandbox", lambda **_: FakeSandbox())
        monkeypatch.setattr(ve_mod, "get_docker_client", fake_get_docker_client)
        monkeypatch.setattr(ve_mod, "log_tool_output_size", lambda *a, **k: None)

        tool_use = {
            "toolUseId": "t1",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04",
                "exploit_code": "print('x')",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = ve_mod.verify_exploit(tool_use)
        payload = result["content"][0]["json"]
        assert payload["verification_status"] == "infra_error"

    def test_generic_exception_in_docker_check_returns_infra_error(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        class FakeSandbox:
            build_log = ""

            def get_docker_ps_all(self):
                return ""

            def get_target_logs(self):
                return ""

            def get_target_exit_code(self):
                return None

            def cleanup(self):
                pass

        def fake_get_docker_client():
            raise RuntimeError("unexpected docker error")

        monkeypatch.setattr(ve_mod, "ExploitSandbox", lambda **_: FakeSandbox())
        monkeypatch.setattr(ve_mod, "get_docker_client", fake_get_docker_client)
        monkeypatch.setattr(ve_mod, "log_tool_output_size", lambda *a, **k: None)

        tool_use = {
            "toolUseId": "t2",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04",
                "exploit_code": "print('x')",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = ve_mod.verify_exploit(tool_use)
        payload = result["content"][0]["json"]
        assert payload["verification_status"] == "infra_error"

    def test_infra_error_has_error_metadata(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod
        from manus_agent.utils.docker_client import DockerConnectionError

        class FakeSandbox:
            build_log = ""

            def get_docker_ps_all(self):
                return ""

            def cleanup(self):
                pass

        def fake_get_docker_client():
            raise DockerConnectionError(
                message="No Docker daemon",
                diagnosis="not running",
                remediation="start it",
            )

        monkeypatch.setattr(ve_mod, "ExploitSandbox", lambda **_: FakeSandbox())
        monkeypatch.setattr(ve_mod, "get_docker_client", fake_get_docker_client)
        monkeypatch.setattr(ve_mod, "log_tool_output_size", lambda *a, **k: None)

        tool_use = {
            "toolUseId": "t3",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04",
                "exploit_code": "x",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = ve_mod.verify_exploit(tool_use)
        payload = result["content"][0]["json"]
        assert payload["error"] is not None
        assert payload["error"]["category"] == "infra"


# ===========================================================================
# verify_exploit — build failure
# ===========================================================================


class TestVerifyExploitBuildFailure:
    def _make_fake_sandbox(self, build_raise=None):
        class FakeSandbox:
            build_log = "ERROR: failed to build"

            def build_target(self, dockerfile):
                if build_raise:
                    raise build_raise
                return "image-id-123"

            def get_docker_ps_all(self):
                return ""

            def get_target_logs(self):
                return ""

            def get_target_exit_code(self):
                return 1

            def cleanup(self):
                pass

        return FakeSandbox()

    def _make_fake_docker_client(self):
        client = MagicMock()
        client.ping.return_value = True
        client.close.return_value = None
        return client

    def test_build_failure_returns_build_error_status(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        sandbox = self._make_fake_sandbox(build_raise=RuntimeError("Dockerfile build failed"))
        monkeypatch.setattr(ve_mod, "ExploitSandbox", lambda **_: sandbox)
        monkeypatch.setattr(ve_mod, "get_docker_client", lambda: self._make_fake_docker_client())
        monkeypatch.setattr(ve_mod, "log_tool_output_size", lambda *a, **k: None)

        tool_use = {
            "toolUseId": "t4",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04\nINVALID",
                "exploit_code": "print('x')",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = ve_mod.verify_exploit(tool_use)
        payload = result["content"][0]["json"]
        assert payload["verification_status"] == "build_error"

    def test_build_failure_includes_build_log(self, monkeypatch):
        import manus_agent.tools.verify_exploit as ve_mod

        sandbox = self._make_fake_sandbox(build_raise=RuntimeError("failed"))
        monkeypatch.setattr(ve_mod, "ExploitSandbox", lambda **_: sandbox)
        monkeypatch.setattr(ve_mod, "get_docker_client", lambda: self._make_fake_docker_client())
        monkeypatch.setattr(ve_mod, "log_tool_output_size", lambda *a, **k: None)

        tool_use = {
            "toolUseId": "t5",
            "input": {
                "dockerfile_content": "FROM ubuntu:22.04",
                "exploit_code": "x",
                "exploit_language": "python",
                "cve_id": "CVE-2024-3094",
                "target_info": {
                    "affected_software": "xz",
                    "affected_versions": "5.6.x",
                    "vulnerability_type": "RCE",
                },
            },
        }
        result = ve_mod.verify_exploit(tool_use)
        payload = result["content"][0]["json"]
        assert "build_log" in payload


# ===========================================================================
# verify_exploit — module contract
# ===========================================================================


def test_verify_exploit_module_importable():
    from manus_agent.tools.verify_exploit import TOOL_SPEC, verify_exploit

    assert callable(verify_exploit)
    assert TOOL_SPEC["name"] == "verify_exploit"


def test_verify_exploit_tool_spec_has_required_fields():
    from manus_agent.tools.verify_exploit import TOOL_SPEC

    required = TOOL_SPEC["inputSchema"]["json"].get("required", [])
    assert "dockerfile_content" in required
    assert "exploit_code" in required
    assert "cve_id" in required


def test_verify_exploit_max_log_constants_positive():
    from manus_agent.tools.verify_exploit import MAX_LOG_CHARS, MAX_LOG_LINES

    assert MAX_LOG_LINES > 0
    assert MAX_LOG_CHARS > 0


def test_tool_output_logger_importable():
    from manus_agent.tools.tool_output_logger import log_tool_output_size

    assert callable(log_tool_output_size)
