"""Custom workflow tool that supports ManusUse agent types."""

import copy
import logging
import uuid
from typing import Any

from strands.types.tools import ToolResult, ToolUse
from strands_tools.workflow import WorkflowManager
from strands_tools.workflow import workflow as _base_workflow_tool

from ..agents import BrowserUseAgent, DataAnalysisAgent, ManusAgent, MCPAgent
from ..config import Config

logger = logging.getLogger(__name__)

# Build TOOL_SPEC from the base workflow tool's spec (strands_tools removed the
# standalone TOOL_SPEC export; the canonical source is now workflow.tool_spec).
_BASE_TOOL_SPEC = _base_workflow_tool.tool_spec
TOOL_SPEC = copy.deepcopy(_BASE_TOOL_SPEC)
TOOL_SPEC["name"] = "manus_workflow"
TOOL_SPEC["description"] = (
    _BASE_TOOL_SPEC["description"]
    + """
This version supports ManusUse agent types:
- manus: General computation and file operations
- browser: Web browsing using browser-use
- data_analysis: Data analysis and visualization
- mcp: Model Context Protocol tools
"""
)

# Add agent_type to the task schema
TOOL_SPEC["inputSchema"]["json"]["properties"]["tasks"]["items"]["properties"] = {
    **TOOL_SPEC["inputSchema"]["json"]["properties"]["tasks"]["items"].get("properties", {}),
    "agent_type": {
        "type": "string",
        "enum": ["manus", "browser", "data_analysis", "mcp"],
        "description": "Agent type to use for executing this task (defaults to 'manus')",
        "default": "manus",
    },
}


class ManusWorkflowManager(WorkflowManager):
    """Extended workflow manager that supports ManusUse agent types."""

    def __init__(self, tool_context: dict[str, Any]):
        """Initialize with agent registry."""
        super().__init__(tool_context)

        # Get config from tool context
        self.config = Config.from_file()

        # Initialize agent registry
        self.agent_registry = {
            "manus": ManusAgent,
            "browser": BrowserUseAgent,
            "data_analysis": DataAnalysisAgent,
            "mcp": MCPAgent,
        }

        # Cache for agent instances
        self.agent_instances = {}

    def get_agent_for_task(self, task: dict) -> Any:
        """Get the appropriate agent for a task based on agent_type."""
        agent_type = task.get("agent_type", "manus")

        # Check cache first
        if agent_type in self.agent_instances:
            return self.agent_instances[agent_type]

        # Create new agent instance
        agent_class = self.agent_registry.get(agent_type)
        if not agent_class:
            # logger.warning(f"Unknown agent type: {agent_type}, using ManusAgent")
            agent_class = ManusAgent

        # Create agent with system prompt if provided
        system_prompt = task.get("system_prompt")
        if system_prompt:
            agent = agent_class(config=self.config, system_prompt=system_prompt)
        else:
            agent = agent_class(config=self.config)

        # Cache the instance
        self.agent_instances[agent_type] = agent
        return agent

    def execute_task(self, task: dict, workflow: dict, tool_use_id: str) -> dict:
        """Execute a single task using the appropriate agent."""
        try:
            # Build context from dependent tasks
            context = []
            if task.get("dependencies"):
                for dep_id in task["dependencies"]:
                    dep_result = workflow["task_results"].get(dep_id, {})
                    if dep_result.get("status") == "completed" and dep_result.get("result"):
                        # Format the dependency results
                        dep_content = [msg.get("text", "") for msg in dep_result["result"]]
                        context.append(f"Results from {dep_id}:\n" + "\n".join(dep_content))

            # Build comprehensive task prompt with context
            task_prompt = task["description"]
            if context:
                task_prompt = "Previous task results:\n" + "\n\n".join(context) + "\n\nTask:\n" + task_prompt

            # Get the appropriate agent for this task
            agent = self.get_agent_for_task(task)

            # Execute task using the agent
            result = agent(task_prompt)

            # Extract response content - handle both dict and custom object return types
            try:
                # If result is a dict or has .get() method
                content = result.get("content", [])
            except AttributeError:
                # If result is an object with .content attribute
                content = getattr(result, "content", [])

            # Extract stop_reason - handle both dict and custom object return types
            try:
                # If result is a dict or has .get() method
                stop_reason = result.get("stop_reason", "")
            except AttributeError:
                # If result is an object with .stop_reason attribute
                stop_reason = getattr(result, "stop_reason", "")

            # Update task status
            status = "success" if stop_reason != "error" else "error"
            return {
                "toolUseId": tool_use_id,
                "status": status,
                "content": content,
            }

        except Exception as e:
            error_msg = f"Error executing task {task['task_id']}: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}


def manus_workflow(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """ManusUse-aware workflow tool implementation.

    This tool extends the base workflow tool with support for ManusUse agent types:
    - manus: General computation and file operations
    - browser: Web browsing using browser-use
    - data_analysis: Data analysis and visualization
    - mcp: Model Context Protocol tools

    Each task can specify an agent_type to route to the appropriate agent.
    """
    system_prompt = kwargs.get("system_prompt")
    inference_config = kwargs.get("inference_config")
    messages = kwargs.get("messages")
    tool_config = kwargs.get("tool_config")

    try:
        tool_use_id = tool.get("toolUseId", str(uuid.uuid4()))
        tool_input = tool.get("input", {})
        action = tool_input.get("action")

        print("==========================")
        print(tool_input)

        # Initialize workflow manager
        manager = ManusWorkflowManager(
            {
                "system_prompt": system_prompt,
                "inference_config": inference_config,
                "messages": messages,
                "tool_config": tool_config,
            }
        )

        if action == "create":
            workflow_id = tool_input.get("workflow_id", str(uuid.uuid4()))
            if not tool_input.get("tasks"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "Tasks are required for create action"}],
                }

            result = manager.create_workflow(workflow_id, tool_input["tasks"], tool_use_id)

        elif action == "start":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for start action"}],
                }

            result = manager.start_workflow(tool_input["workflow_id"], tool_use_id)

        elif action == "list":
            result = manager.list_workflows(tool_use_id)

        elif action == "status":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for status action"}],
                }

            result = manager.get_workflow_status(tool_input["workflow_id"], tool_use_id)

        elif action == "delete":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for delete action"}],
                }

            result = manager.delete_workflow(tool_input["workflow_id"], tool_use_id)

        else:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

        return {
            "toolUseId": tool_use_id,
            "status": result["status"],
            "content": result["content"],
        }

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(f"\nError in workflow tool: {error_msg}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
