"""Demo: connect to an MCP server over SSE transport and run an agent.

Usage::

    # Set the server URL in config.toml:
    #   [mcp]
    #   server_url = "http://localhost:8000/sse"
    #
    # Or via environment variable:
    #   export MCP_SSE_URL=http://localhost:8000/sse

    python examples/mcp_sse_demo.py
"""

import os

from mcp.client.sse import sse_client
from strands import Agent
from strands.tools.mcp import MCPClient

from manus_use.config import Config

# Resolve SSE URL: config file takes precedence, env var as fallback
config = Config.from_file()
sse_url = getattr(getattr(config, "mcp", None), "server_url", None) or os.environ.get("MCP_SSE_URL")

if not sse_url:
    raise ValueError(
        "MCP SSE server URL not configured. Set [mcp] server_url in config.toml or export MCP_SSE_URL=<url>."
    )

# Connect to the MCP server and run a quick sanity task
sse_mcp_client = MCPClient(lambda: sse_client(sse_url))
sse_mcp_client.start()

tools = sse_mcp_client.list_tools_sync()
print(f"Available MCP tools: {[t.name for t in tools]}")

agent = Agent(tools=tools)
result = agent("print(100 * 100)")
print(result)
