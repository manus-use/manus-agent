import asyncio

try:
    from browser_use.agent.service import Agent

    # from browser_use import Agent, ActionResult
    from browser_use.browser.browser import BrowserProfile, BrowserSession
    from browser_use.controller.service import Controller
    from browser_use.llm import ChatAnthropicBedrock
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "This module requires the optional 'browser' extra. Install it with: pip install 'manus-agent[browser]'"
    ) from exc
from pydantic import BaseModel, Field


# Define the structured output format for the browser agent
class BrowserTaskResult(BaseModel):
    """Structured output for the browser agent."""

    task_completed: bool = Field(description="Whether the assigned tasks are successfully completed.")
    summary: str = Field(description="A high-level summary of what was found or accomplished.")
    result: str = Field(description="Result data, or error messages from the tasks.")


"""
controller = Controller()
@controller.registry.action('Done with task', param_model=BrowserTaskResult)
async def done(params: BrowserTaskResult):
	result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
	print(result)
	# NOTE: this is clearly wrong - to demonstrate the validator
	return 'blablabla'
"""


class BrowserAgentRunner:
    def __init__(self, headless: bool = True, keep_alive: bool = True):
        self.browser_session = BrowserSession(browser_profile=BrowserProfile(keep_alive=keep_alive, headless=headless))

    async def start_browser(self):
        await self.browser_session.start()

    async def run_browser_task(self, task: str, structured_output: BaseModel) -> str:
        """
        Initializes and runs a browser_use agent for a given task, returning a JSON string.
        Args:
            task: The detailed instruction for the browser agent.
        Returns:
            A JSON string representing the structured result from the agent.
        """
        print(f"Browser agent starting task: {task}")
        output_model = structured_output
        if not output_model:
            output_model = BrowserTaskResult

        llm = ChatAnthropicBedrock(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            aws_region="us-west-2",
        )
        controller = Controller(output_model=output_model)
        agent = Agent(
            task=task, llm=llm, controller=controller, browser_session=self.browser_session, validate_output=True
        )
        result_model = None
        history = await agent.run(max_steps=300)
        result_model = history.final_result()
        if result_model:
            result = output_model.model_validate_json(result_model)
            print("=========\n" + result.result)
            return result_model
        else:
            error_result = BrowserTaskResult(
                task_completed=False,
                summary="Agent failed to produce a result.",
                result="The agent did not return a valid BrowserTaskResult model.",
            )
            return error_result.model_dump_json(indent=2)

    async def close_browser(self):
        # Defensive: close browser if it exists in session
        await self.browser_session.kill()

    @classmethod
    async def cli_entry(cls):
        import sys

        if len(sys.argv) > 1:
            tasks_to_run = sys.argv[1:]
            runner = cls()
            await runner.start_browser()
            for task in tasks_to_run:
                result_model = await runner.run_browser_task(task, AssetMatch)
                result = AssetMatch.model_validate_json(result_model)
                print(f"{result.precisely_matched_assets}/{result.fuzzy_matched_asset}")
                print(f"--- BROWSER AGENT FINAL RESULT for task: {task} ---")
                print(result)
            await runner.close_browser()
        else:
            print("Please provide one or more tasks as command-line arguments for direct testing.")


class AssetMatch(BaseModel):
    result: str = Field(description="Result data, or error messages from the tasks.")
    precisely_matched_assets: int = Field(description="The number of Assets Precisely Matched")
    fuzzy_matched_asset: int = Field(description="The number of Assets Fuzzy Matched (Name + Version)")


# MCP server implementation to provide tools using MCP official Python SDK

from mcp.server import MCPServer, tool  # noqa: E402
from mcp.types import ToolInput, ToolOutput  # noqa: E402


class AssetMatchInput(ToolInput):
    task: str


class AssetMatchOutput(ToolOutput):
    result: str
    precisely_matched_assets: int
    fuzzy_matched_asset: int


class BrowserAgentTools:
    def __init__(self) -> None:
        self.runner = BrowserAgentRunner(headless=True)

    @tool(
        name="asset_match",
        description="Run the browser agent to match assets for a given task.",
        input_model=AssetMatchInput,
        output_model=AssetMatchOutput,
    )
    async def asset_match(self, input: AssetMatchInput) -> AssetMatchOutput:
        result_model = await self.runner.run_browser_task(input.task, AssetMatch)
        result = AssetMatch.model_validate_json(result_model)
        return AssetMatchOutput(
            result=result.result,
            precisely_matched_assets=result.precisely_matched_assets,
            fuzzy_matched_asset=result.fuzzy_matched_asset,
        )


async def main():
    browser = BrowserAgentTools()
    await browser.runner.start_browser()
    server = MCPServer(tools=browser)
    server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
