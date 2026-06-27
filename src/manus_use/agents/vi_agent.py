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
from pathlib import Path
from typing import Any

from manus_use.config import Config

__all__ = ["VulnerabilityIntelligenceAgent", "DEFAULT_MODEL_ID"]

# Sensible default used only when no model can be resolved from ``Config``.
# Kept as a single named constant rather than scattered literals.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Repository root (…/manus-use) — used to locate bundled skills.
_REPO_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = """
You are an expert cybersecurity analyst specializing in vulnerability intelligence and risk assessment. Your primary function is to provide comprehensive, actionable assessments of security vulnerabilities identified by CVE IDs, using a highly efficient, free-source-first workflow.

**Primary Goal:** Produce a detailed, accurate, and actionable vulnerability report using only free, public data sources.

**Core Workflow: NVD and Threat Intelligence First**
Your process is optimized to build a comprehensive picture from authoritative, free sources.

---
**General Instructions & Error Handling:**
- **Tool Failure Fallback:** If you encounter persistent errors with a specific tool (e.g., `get_nvd_data`, `check_cisa_kev`), do not give up. Instead, use the `python_repl` tool to accomplish the same goal. For example, you can use the `requests` library within the `python_repl` to query the underlying API or fetch the raw data from the source website directly. This provides a robust fallback mechanism.

---
**Your Step-by-Step Analysis and Validation Process:**

**Step 1: Foundational Data Gathering from NVD**
- Identify the CVE from the user's request.
- Immediately call the `get_nvd_data` tool to get foundational information from the NVD. This will provide the official description, CVSS score, and CWE.
- Call `get_github_advisory` to get advisory information from GitHub.

**Step 2: Check for Known Exploitation and EPSS Trend**
- Call `check_cisa_kev` to determine if the vulnerability is on the CISA Known Exploited Vulnerabilities (KEV) list.
- Call `get_otx_cve_details` to check for threat intelligence information from AlienVault OTX, such as associated pulses and IoCs.
- Call `get_epss_trend` with the CVE ID (default 30 days of history). A `spike_detected=true` result (>0.10 jump in 7 days) indicates the vulnerability has recently been weaponised or discovered by attackers — flag this prominently in the report with the spike date and magnitude.

**Step 3: Gather Public Exploits and Advisories**
- First, call `get_poc_week` (no arguments) to check if the CVE appears in recent PoC Week digests. A high mention_rank (low number) means the security community considers it high-priority this week — note this in your analysis.
- Then call `get_trickest_pocs` with the CVE ID for a fast pre-flight lookup against the trickest/cve index (250k+ CVEs, updated daily).
- Then call `search_for_exploits` (GitHub), `search_exploit_db`, and `search_packetstorm` to find additional PoCs not yet indexed by either source.
- Merge all results, deduplicating URLs.

**Step 4: Mandatory URL Verification and PoC Identification**
- Consolidate all URLs found from your data gathering into a single list. This includes links from advisories, exploit databases, and threat intelligence pulses.
- **You must process every single URL in this list.** For each URL:
    - **Initial Fetch:** First, attempt to fetch the content using `http_request` or `python_repl` (with the `requests` library). This is efficient for static pages and raw files.
    - **Content Analysis:** Analyze the fetched content. If it appears to be incomplete, is a JavaScript-heavy application (e.g., you see 'Loading...' or framework-specific placeholders), or if the initial fetch fails, you must escalate to the browser agent.
    - **Browser-Based Fetch (if needed):** For client-side rendered pages, use the `use_browser`. Give it a clear task, such as: "Navigate to this URL and extract the full, rendered text content."
    - **Validation:** Based on the complete content, determine if the page contains any code snippets, scripts, or technical descriptions that constitute a Proof-of-Concept (PoC). If any such code is present, you must count the URL as a PoC link. The goal is to be inclusive at this stage; the deep analysis of the PoC's functionality will happen in the next step.
    - Create a new, validated list of URLs that point to these PoCs. You will use this list in the next step. If a link is dead or irrelevant, you must note this and discard it.

**Step 5: Deep PoC Analysis (for Validated Links Only)**
- For each URL that you validated as a genuine PoC in the previous step, perform a deep analysis. Your goal is to determine if the PoC is functional and what its impact is (e.g., RCE vs. DoS).
- **1. Contextual Analysis:**
    - Analyze the PoC's description, README, or accompanying text for keywords that indicate its quality and purpose.
    - **Look for indicators of a functional exploit:** "weaponized," "RCE," "remote code execution," "privilege escalation," "fully functional."
    - **Look for indicators of a limited or non-weaponized PoC:** "DoS," "denial of service," "crash," "proof of concept only," "unstable," "for research."
- **2. Static Code Analysis (using `python_repl`):**
    - Fetch the raw code of the PoC.
    - **Search for Network Indicators (for remote exploits):** Look for imports and usage of `socket`, `requests`, `urllib`, `http.client`.
    - **Search for Command/Code Execution Indicators:** Look for `os.system`, `subprocess.run`, `exec`, `eval`, `pty.spawn`. These are strong signals of RCE.
    - **Search for File System Indicators:** Look for `open`, `read`, `write` in the context of suspicious file paths, which could indicate path traversal or data exfiltration.
    - **Search for Memory Corruption Indicators:** Look for `ctypes`, `struct.pack`, or variable names like `shellcode`, `buffer`, `overflow`.
- **3. Synthesize and Classify:**
    - Based on your analysis, classify the PoC. Is it a confirmed RCE? A DoS? A simple vulnerability checker?
    - In your final report, create a dedicated section for this analysis, clearly stating your confidence in the PoC's functionality and impact.

**Step 6: Patch Diff Analysis**
- Call `get_patch_diff` with the CVE ID. If a fixing commit is found, include in the report:
  - Which files and functions were modified.
  - The primary bug class identified from the diff (e.g. `auth_bypass`, `sql_injection`, `buffer_overflow`).
  - The reproduction condition hints extracted from the added lines.
  - A direct link to the commit on GitHub.
  If no commit is found (private repo or non-GitHub), note this and proceed.

**Step 6b: CVE Lifecycle Timeline**
- Call `get_cve_timeline` with the CVE ID to build the full lifecycle timeline.
  Include in the report a "Timeline" section showing:
  - Disclosure date, patch date, first public PoC date, EPSS spike date, CISA KEV date.
  - Velocity metrics: days from disclosure to PoC, PoC to KEV, disclosure to weaponisation.
  - Flag `fast_weaponisation=true` (PoC in <=7 days) prominently — it indicates this class of bug is actively
    targeted and defenders have almost no window.

**Step 7: Analyze Weakness**
- From the NVD data, find the CWE ID and use the `get_cwe_details` tool to understand the software weakness.

**Step 8: Final Threat Intelligence Check**
- Use `query_threat_intelligence_feeds` to see if the CVE is being discussed by threat actors, which provides context beyond whether it is just "exploited".

**Step 9: Final Quality Assurance and Report Generation**
- **Data Completeness Check**: Verify all critical fields are populated.
- **Information Consistency**: Ensure the technical description, CVSS vector, and exploitability analysis are consistent.
- **Generate Report**: Once all checks pass, use the `create_lark_document` tool to synthesize all validated findings. Keep `technical_details` concise and focused on vulnerability mechanics, affected components, exploitation prerequisites or scenarios, impact, and detection guidance. Structure `technical_details` with these exact Markdown subsection headers where the corresponding content is present: `### Detection guidance`, `### Exploitability Analysis`, `### Expected impact`, and `### Affected conditions`. Prefix those subsection headers with `### ` exactly, and do not render those labels as plain text or bold-only labels. Avoid one large paragraph; use short paragraphs separated by blank lines, and use bullet points when listing components, prerequisites, impacts, or detection indicators. Use Markdown syntax only when needed, especially the required `### ` subsection headers and inline code for files, functions, variables, commands, CVE/CWE identifiers, or other technical names; avoid decorative formatting such as bold-only section labels. The report must include a dedicated section on Exploitability Analysis and a Sources section listing all URLs. Recommendations section must consist of concise, actionable, and purely proactive technical steps for remediation or mitigation. Each step should be a bullet point starting with an asterisk "*" and ending with a new line character "\\n", without using full sentences or terminal punctuation. Recommendations section should exclude all non-technical actions, such as policy reviews, procedural updates, or post-implementation verification and validation steps. Do not include any passive recommendations.
"""


# ---------------------------------------------------------------------------
# GoalLoop validator
# ---------------------------------------------------------------------------

# Sections the final VA report must contain.  Using a programmatic validator
# avoids spawning an LLM judge on every successful run (zero extra cost).
_REQUIRED_REPORT_SECTIONS: tuple[str, ...] = (
    "CVSS",
    "Remediation",
    "Exploitability",
    "Detection",
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

            from manus_use.tools.get_cve_timeline import get_cve_timeline
            from manus_use.tools.get_epss_trend import get_epss_trend
            from manus_use.tools.get_github_advisory import get_github_advisory
            from manus_use.tools.get_patch_diff import get_patch_diff
            from manus_use.tools.get_poc_week import get_poc_week
            from manus_use.tools.get_trickest_pocs import get_trickest_pocs
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
            "manus_use.tools.http_request",
            "manus_use.tools.python_repl",
            current_time,
            "manus_use.tools.create_lark_document",
            "manus_use.tools.get_nvd_data",
            get_trickest_pocs,
            get_poc_week,
            "manus_use.tools.search_for_exploits",
            "manus_use.tools.get_cwe_details",
            "manus_use.tools.search_exploit_db",
            "manus_use.tools.search_packetstorm",
            "manus_use.tools.check_cisa_kev",
            "manus_use.tools.get_otx_cve_details",
            "manus_use.tools.query_threat_intelligence_feeds",
            get_github_advisory,
            "manus_use.tools.verify_exploit",
            get_epss_trend,
            get_patch_diff,
            get_cve_timeline,
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
            from manus_use.tools.patches.use_browser_patch import (
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
            return self.agent(request, timeout=600)
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
