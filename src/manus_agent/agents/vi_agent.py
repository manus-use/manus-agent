"""Vulnerability Intelligence agent.

This module provides :class:`VulnerabilityIntelligenceAgent`, a Strands-based
agent that produces a comprehensive, actionable vulnerability report for a given
CVE identifier using free, public data sources (NVD, CISA KEV, OTX, GitHub
advisories, Exploit-DB, PacketStorm, …) and optional Docker-based exploit
verification.

The module is written so it can be *imported* without the optional heavy
dependencies (``strands``, ``strands_tools``, ``browser_use``) being installed:
all such imports are deferred into :meth:`VulnerabilityIntelligenceAgent.__init__`
and guarded with ``try/except ImportError``. Only constructing the agent
requires those dependencies.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Any

from manus_agent.config import Config

__all__ = ["VulnerabilityIntelligenceAgent", "DEFAULT_MODEL_ID"]

# Sensible default used only when no model can be resolved from ``Config``.
# Kept as a single named constant rather than scattered literals.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Repository root (…/manus-agent) — used to locate bundled skills.
_REPO_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = (files("manus_agent.agents") / "prompts" / "vi_system_prompt.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GoalLoop validator
# ---------------------------------------------------------------------------

# Sections the final VA report must contain.  Using a programmatic validator
# avoids spawning an LLM judge on every successful run (zero extra cost).
_REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "Exploitation Status",
    "CVSS",
    "Exploitability",
    "Detection",
    "Remediation",
)


def _report_complete_validator(
    response: dict,  # last assistant message from the agent
    agent: Any,  # host agent instance (unused but required by the interface)
) -> bool | dict:
    """Return True when the response contains all required VA report sections.

    On failure returns a dict with ``passed=False`` and ``feedback`` listing
    the missing sections so the agent can complete them in the next attempt.
    """
    text = " ".join(block.get("text", "") for block in response.get("content", []) if isinstance(block, dict))
    missing = [s for s in _REQUIRED_REPORT_SECTIONS if s.lower() not in text.lower()]
    if not missing:
        return True
    return {
        "passed": False,
        "feedback": (
            f"Report is incomplete. Missing required sections: {missing}. "
            "Please complete those sections and regenerate the full report."
        ),
    }


class VulnerabilityIntelligenceAgent:
    """Agent that performs vulnerability analysis via a sequential, tool-based workflow.

    The agent wires together the free-source vulnerability-intelligence tools
    bundled with ManusUse (NVD, CISA KEV, OTX, CWE, Exploit-DB, PacketStorm,
    GitHub advisories) plus optional Docker-based exploit verification, and
    drives them with a detailed system prompt.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        """Build the underlying Strands agent.

        Args:
            config: ManusUse configuration. Loaded from disk when omitted.
            model: A pre-built Strands model instance. Takes precedence over
                ``model_name`` and over the model resolved from ``config``.
            model_name: Explicit model id to use instead of resolving one from
                ``config``. Falls back to :data:`DEFAULT_MODEL_ID`.

        Raises:
            ImportError: if the optional ``strands`` / ``strands_tools``
                dependencies required to run the agent are not installed.
        """
        self.config = config or Config.from_file()
        self.system_prompt = SYSTEM_PROMPT
        self._local_chromium_browser = None

        try:
            # Heavy / optional dependencies are imported lazily so the module
            # can be imported (and unit-tested) without them present.
            os.environ.setdefault("BYPASS_TOOL_CONSENT", "True")

            from strands import Agent
            from strands_tools import current_time

            from manus_agent.tools.get_dependency_blast_radius import get_dependency_blast_radius
            from manus_agent.tools.get_epss_trend import get_epss_trend
            from manus_agent.tools.get_github_advisory import get_github_advisory
            from manus_agent.tools.get_osv_data import get_osv_data
            from manus_agent.tools.get_patch_diff import get_patch_diff
            from manus_agent.tools.get_poc_week import get_poc_week
            from manus_agent.tools.get_trickest_pocs import get_trickest_pocs
            from manus_agent.tools.get_vulncheck_data import get_vulncheck_data
            from manus_agent.tools.score_exploit_complexity import score_exploit_complexity
            from manus_agent.tools.search_poc_sources import search_poc_sources
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "VulnerabilityIntelligenceAgent requires the optional 'strands' "
                "and 'strands_tools' dependencies. Install them to run analyses."
            ) from exc

        # Best-effort browser patch + tool; never fatal if unavailable.
        use_browser = self._resolve_use_browser()

        model_obj = model if model is not None else self._resolve_model(model_name)

        # Use agentic context management: the model monitors its own token
        # usage and decides when/what to compress via summarize_context,
        # truncate_context, and pin_context tools.  This is better than a
        # fixed SlidingWindow for the VI pipeline because the model can
        # distinguish "NVD dump I already parsed" from "PoC code I still need".
        # A ContextOffloader (inline threshold 8 000 tokens) is added
        # automatically; SlidingWindowConversationManager is kept as a
        # safety-net fallback inside agentic mode.
        context_manager: str = "agentic"

        # GoalLoop: ensure the final response contains all required report
        # sections before returning.  Uses a fast programmatic validator (no
        # judge-agent invocation) so there is no extra LLM cost on success.
        from strands.vended_plugins.goal import GoalLoop

        goal_loop = GoalLoop(
            goal=_report_complete_validator,
            max_attempts=2,
            timeout=900.0,
        )

        tools: list[Any] = [
            "manus_agent.tools.http_request",
            "manus_agent.tools.python_repl",
            current_time,
            "manus_agent.tools.create_lark_document",
            "manus_agent.tools.get_nvd_data",
            get_trickest_pocs,
            get_poc_week,
            "manus_agent.tools.search_for_exploits",
            "manus_agent.tools.get_cwe_details",
            "manus_agent.tools.search_exploit_db",
            "manus_agent.tools.search_packetstorm",
            "manus_agent.tools.check_cisa_kev",
            "manus_agent.tools.get_otx_cve_details",
            "manus_agent.tools.query_threat_intelligence_feeds",
            get_github_advisory,
            "manus_agent.tools.verify_exploit",
            get_epss_trend,
            get_osv_data,
            get_patch_diff,
            score_exploit_complexity,
            get_vulncheck_data,
            search_poc_sources,
            get_dependency_blast_radius,
        ]
        if use_browser is not None:
            tools.append(use_browser)

        agent_kwargs: dict[str, Any] = dict(
            context_manager=context_manager,
            model=model_obj,
            system_prompt=self.system_prompt,
            tools=tools,
        )

        # Attach the bundled verify-exploit skill when AgentSkills is available.
        plugin = self._resolve_skills_plugin()
        plugins: list[Any] = [goal_loop]
        if plugin is not None:
            plugins.append(plugin)
        agent_kwargs["plugins"] = plugins

        self.agent = Agent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_use_browser(self):
        """Return a ``use_browser`` tool, or ``None`` if unavailable."""
        try:
            from manus_agent.tools.patches.use_browser_patch import (
                apply_comprehensive_patch,
            )

            apply_comprehensive_patch()
        except Exception:  # pragma: no cover - patch is best-effort
            pass

        try:
            from strands_tools import use_browser

            return use_browser
        except Exception:
            pass

        try:  # pragma: no cover - fallback path
            from strands_tools.browser import LocalChromiumBrowser

            self._local_chromium_browser = LocalChromiumBrowser()
            return self._local_chromium_browser.browser
        except Exception:
            return None

    def _resolve_model(self, model_name: str | None) -> Any:
        """Resolve a model instance from an explicit name or from config.

        Avoids hardcoded model ids in the hot path: prefers ``config.get_model``
        and only falls back to a named id when configuration cannot produce one.
        """
        if model_name:
            return model_name

        try:
            return self.config.get_model()
        except Exception:
            # Fall back to a bare model id string; Strands accepts these.
            return DEFAULT_MODEL_ID

    def _resolve_skills_plugin(self):
        """Return an AgentSkills plugin for the verify-exploit skill, if present."""
        skills_dir = _REPO_ROOT / "skills" / "verify-exploit"
        if not skills_dir.exists():
            return None
        try:
            from strands import AgentSkills

            return AgentSkills(skills=[str(skills_dir)])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def build_request(cve_id: str, *, verify: bool = False) -> str:
        """Build the natural-language analysis request for a CVE.

        Args:
            cve_id: The CVE identifier (e.g. ``"CVE-2025-6554"``).
            verify: When ``True``, also instruct the agent to develop and verify
                exploit code in Docker via the ``verify-exploit`` skill.
        """
        if verify:
            return (
                f"Please perform a comprehensive vulnerability intelligence analysis "
                f"for {cve_id}.\n"
                "Follow your sequential process and create a Lark document with the "
                "final report.\n"
                "Additionally, activate the `verify-exploit` skill to develop and "
                "verify exploit code in Docker."
            )
        return (
            f"Please perform a comprehensive vulnerability intelligence analysis "
            f"for {cve_id}.\n"
            "Follow your sequential process and create a Lark document with the "
            "final report.\n"
            "Do NOT perform exploit verification."
        )

    def handle_request(self, request: str) -> str:
        """Run the agent on a request string and return its response.

        Ensures any local Chromium browser spawned for page rendering is
        cleaned up afterwards.
        """
        try:
            # NOTE: Strands ``Agent.__call__`` has no ``timeout`` parameter; any
            # extra kwargs are folded into the (deprecated) event-loop
            # ``invocation_state`` and silently ignored, so passing
            # ``timeout=600`` here enforced nothing. The real wall-clock cap is
            # applied by ``GoalLoop(timeout=900.0)`` in ``_build_agent``.
            return self.agent(request)
        finally:
            cleanup = getattr(self._local_chromium_browser, "_cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"WARNING: Browser cleanup failed: {exc}")

    def analyze(self, cve_id: str, *, verify: bool = False) -> str:
        """Convenience wrapper: build the request for ``cve_id`` and run it."""
        return self.handle_request(self.build_request(cve_id, verify=verify))
