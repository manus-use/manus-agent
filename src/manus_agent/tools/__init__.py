"""Tools for ManusUse agents."""

from typing import Any

from strands.types.tools import AgentTool
from strands_tools import (
    calculator,
    current_time,
    editor,
    environment,
    file_read,
    file_write,
    generate_image,
    shell,
)

from manus_agent.tools.http_request import http_request
from manus_agent.tools.python_repl import python_repl
from manus_agent.tools.dep_audit import dep_audit

# Collect all tools
ALL_TOOLS = {
    "file_read": file_read,
    "file_write": file_write,
    "python_repl": python_repl,
    "shell": shell,
    "http_request": http_request,
    "editor": editor,
    "environment": environment,
    # "web_search": retrieve.retrieve,  # Using retrieve for web search
    "generate_image": generate_image,
    "current_time": current_time,
    "calculator": calculator,
    "dep_audit": dep_audit,
}


def get_tools_by_names(names: list[str], config: Any | None = None) -> list[AgentTool]:
    """Get tools by their names.

    Args:
        names: List of tool names to retrieve
        config: Optional configuration object

    Returns:
        List of tool instances
    """
    tools = []
    for name in names:
        if name in ALL_TOOLS:
            tool = ALL_TOOLS[name]
            # Some tools might need configuration
            if hasattr(tool, "set_config") and config:
                tool.set_config(config)
            tools.append(tool)
    return tools


__all__ = [
    "file_read",
    "file_write",
    "python_repl",
    "shell",
    "http_request",
    "editor",
    "environment",
    # "web_search": retrieve.retrieve,  # Using retrieve for web search
    "generate_image",
    "current_time",
    "calculator",
]
