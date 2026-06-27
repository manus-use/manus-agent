#!/usr/bin/env python3
"""Demo: WorkflowAgent with Lark document output and vulnerability assessment.

Uses workflow_tool + create_lark_document to coordinate a multi-step CVE
analysis and publish the result as a Lark document.

Usage::

    python examples/workflow_lark_demo.py [CVE-ID]
    python examples/workflow_lark_demo.py CVE-2025-6545
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from strands import Agent  # noqa: E402
from strands_tools import stop, think  # noqa: E402

import manus_use.tools.create_lark_document as create_lark_document  # noqa: E402
import manus_use.tools.workflow_tool as workflow_tool  # noqa: E402
from manus_use.config import Config  # noqa: E402

SYSTEM_PROMPT = (
    "You are a cybersecurity and vulnerability intelligence expert and a Workflow "
    "Management Agent. Coordinate multi-step CVE analysis tasks and publish results "
    "as Lark documents."
)


class WorkflowAgent:
    """Agent that orchestrates CVE intelligence workflows with Lark output."""

    def __init__(self, model_name: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        self.agent = Agent(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            tools=[workflow_tool, think, create_lark_document, stop],
        )

    def handle_request(self, request: str) -> str:
        return self.agent(request)


def main() -> None:
    cve_id = sys.argv[1] if len(sys.argv) > 1 else "CVE-2025-6545"

    model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    try:
        config = Config.from_file()
        if hasattr(config, "llm") and config.llm.provider == "bedrock":
            print("Using AWS Bedrock configuration")
    except Exception as exc:
        print(f"Config not found ({exc}), using default model.")

    agent = WorkflowAgent(model_name=model_name)

    print(f"=== Workflow + Lark Demo — {cve_id} ===\n")
    result = agent.handle_request(
        f"Assess {cve_id} with detailed information, find PoCs if available, "
        "and create a Lark document for the vulnerability assessment."
    )
    print(result)


if __name__ == "__main__":
    main()
