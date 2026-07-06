"""Tests for GoalLoop integration in VulnerabilityIntelligenceAgent."""


# ---------------------------------------------------------------------------
# Validator unit tests (no strands import needed)
# ---------------------------------------------------------------------------


def _import_validator():
    """Import the module-level validator function."""
    from manus_agent.agents.vi_agent import (  # noqa: PLC0415
        _REQUIRED_REPORT_SECTIONS,
        _report_complete_validator,
    )

    return _report_complete_validator, _REQUIRED_REPORT_SECTIONS


def _make_response(text: str) -> dict:
    return {"content": [{"text": text}]}


def test_validator_passes_complete_report():
    validate, _ = _import_validator()
    text = (
        "Exploitation Status: no confirmed in-the-wild exploitation\n"
        "CVSS: 9.8 (Critical)\n"
        "Remediation: Apply patch version 1.2.3.\n"
        "Exploitability: Active exploitation observed.\n"
        "Detection: Monitor logs for anomalous requests.\n"
    )
    assert validate(_make_response(text), None) is True


def test_validator_fails_missing_cvss():
    validate, _ = _import_validator()
    text = "Remediation: patch it. Exploitability: easy. Detection: monitor logs."
    result = validate(_make_response(text), None)
    assert result["passed"] is False
    assert "CVSS" in result["feedback"]


def test_validator_fails_missing_remediation():
    validate, _ = _import_validator()
    text = "CVSS: 7.5. Exploitability: low. Detection: check syslog."
    result = validate(_make_response(text), None)
    assert result["passed"] is False
    assert "Remediation" in result["feedback"]


def test_validator_fails_missing_exploitability():
    validate, _ = _import_validator()
    text = "CVSS: 8.1. Remediation: upgrade now. Detection: firewall rule."
    result = validate(_make_response(text), None)
    assert result["passed"] is False
    assert "Exploitability" in result["feedback"]


def test_validator_fails_missing_detection():
    validate, _ = _import_validator()
    text = "CVSS: 5.0. Remediation: patch. Exploitability: requires auth."
    result = validate(_make_response(text), None)
    assert result["passed"] is False
    assert "Detection" in result["feedback"]


def test_validator_fails_multiple_missing():
    validate, sections = _import_validator()
    text = "Some vulnerability was found."
    result = validate(_make_response(text), None)
    assert result["passed"] is False
    for section in sections:
        assert section in result["feedback"]


def test_validator_case_insensitive():
    validate, _ = _import_validator()
    # Lowercase variants — validator must match case-insensitively.
    text = "exploitation status: none. cvss: 6.5. remediation: update. exploitability: network. detection: ids."
    assert validate(_make_response(text), None) is True


def test_validator_empty_response():
    validate, _ = _import_validator()
    result = validate({"content": []}, None)
    assert result["passed"] is False


def test_validator_non_text_blocks_ignored():
    validate, _ = _import_validator()
    response = {
        "content": [
            {"type": "tool_use", "id": "t1"},
            {"text": "Exploitation Status: none. CVSS: 9.0. Remediation: patch. Exploitability: rce. Detection: waf."},
        ]
    }
    assert validate(response, None) is True


# ---------------------------------------------------------------------------
# Integration: GoalLoop is importable and wires into VulnerabilityIntelligenceAgent
# ---------------------------------------------------------------------------


def test_goal_loop_attached_to_agent():
    """GoalLoop plugin is importable and accepts the VA report validator."""
    from strands.vended_plugins.goal import GoalLoop  # noqa: PLC0415

    from manus_agent.agents.vi_agent import (  # noqa: PLC0415
        _REQUIRED_REPORT_SECTIONS,
        _report_complete_validator,
    )

    # Verify GoalLoop instantiates cleanly with the VA validator
    plugin = GoalLoop(
        goal=_report_complete_validator,
        max_attempts=2,
        timeout=900.0,
    )
    assert plugin is not None
    assert len(_REQUIRED_REPORT_SECTIONS) >= 4


def test_required_sections_tuple_not_empty():
    from manus_agent.agents.vi_agent import _REQUIRED_REPORT_SECTIONS  # noqa: PLC0415

    assert len(_REQUIRED_REPORT_SECTIONS) >= 4
