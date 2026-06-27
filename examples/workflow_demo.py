#!/usr/bin/env python3
"""Demo: WorkflowAgent — coordinate multi-step tasks across specialised agents.

The WorkflowAgent uses the ``manus_workflow`` tool to create, start, and
monitor workflows composed of tasks delegated to different agent types
(manus, browser, data_analysis, mcp).

Usage::

    python examples/workflow_demo.py
"""

import os
from typing import Any

from strands import Agent

from manus_use.config import Config
from manus_use.tools import manus_workflow
from manus_use.tools.manus_workflow import WORKFLOW_DIR

SYSTEM_PROMPT = """You are a Workflow Management Agent that coordinates complex
multi-step tasks using different specialised agents:

1. **manus**         — general computation, file operations, Python execution
2. **browser**       — web browsing and scraping
3. **data_analysis** — data processing, analysis, and visualisation
4. **mcp**           — Model Context Protocol tool servers

Use the manus_workflow tool to create, start, and monitor workflows.

Tool calling examples
---------------------
List workflows:
  {"action": "list"}

Create a workflow:
  {
    "action": "create",
    "workflow_id": "my-workflow",
    "tasks": [
      {"task_id": "t1", "description": "Fetch data", "agent_type": "browser",
       "priority": 1, "dependencies": []},
      {"task_id": "t2", "description": "Analyse data", "agent_type": "data_analysis",
       "priority": 1, "dependencies": ["t1"]}
    ]
  }

Start a workflow:
  {"action": "start", "workflow_id": "my-workflow"}

Check status:
  {"action": "status", "workflow_id": "my-workflow"}

Delete a workflow:
  {"action": "delete", "workflow_id": "my-workflow"}
"""


class WorkflowAgent:
    """Agent that manages complex workflows using multiple agent types."""

    def __init__(self, model_name: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        self.agent = Agent(model=model_name, system_prompt=SYSTEM_PROMPT, tools=[manus_workflow])

    def handle_request(self, request: str) -> str:
        return self.agent(request)


def start_workflow(workflow_id: str) -> dict[str, Any]:
    """Helper: start a workflow by ID."""
    tool_use = {"toolUseId": f"start-{workflow_id}", "input": {"action": "start", "workflow_id": workflow_id}}
    return manus_workflow(tool_use)


def main() -> None:
    print("=== Workflow Agent Demo ===")
    print(f"Workflow directory: {WORKFLOW_DIR}")
    os.makedirs(WORKFLOW_DIR, exist_ok=True)

    # Auto-select model from config when available
    model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    try:
        config = Config.from_file()
        if hasattr(config, "llm") and config.llm.provider == "bedrock":
            print("Using AWS Bedrock configuration")
    except Exception as exc:
        print(f"Config not found ({exc}), using default model.")

    agent = WorkflowAgent(model_name=model_name)

    print("\n--- What tools do you have? ---")
    print(agent.handle_request("What tools do you have available?"))

    print("\n--- Vulnerability research workflow ---")
    print(
        agent.handle_request(
            "Assess the 2 most recent high-severity vulnerabilities. Use the most appropriate agent type for each step."
        )
    )


if __name__ == "__main__":
    main()
