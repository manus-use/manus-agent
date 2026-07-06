import asyncio

# Logging config at module level
import logging
import sys

import requests

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
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


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
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.browser_session = BrowserSession(
            browser_profile=BrowserProfile(
                keep_alive=keep_alive,
                # user_data_dir='~/Library/Application Support/Google/Chrome/Default',
                user_data_dir="~/.config/browseruse/profiles/default",
                headless=headless,
            )
        )

    async def receive(self, cve):
        await self.task_queue.put(cve)

    async def consumer(self):
        while True:
            try:
                cve = await asyncio.wait_for(self.task_queue.get(), timeout=10)
                logging.info(f"🔄 Processing: {cve}")
                try:
                    await self.run_browser_task_by_cve(cve, AssetMatch)
                    logging.info(f"✅ Completed: {cve}")
                except Exception as task_error:
                    logging.error(f"Task processing error for {cve}: {task_error}")
                finally:
                    self.task_queue.task_done()  # Always mark task done
            except asyncio.TimeoutError:
                # No tasks in queue, continue waiting
                continue
            except Exception as e:
                logging.error(f"Consumer error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

    async def start_browser(self):
        await self.browser_session.start()
        self.consumer_task = asyncio.create_task(self.consumer())
        # Add error handler to log if consumer task fails
        self.consumer_task.add_done_callback(self._consumer_done_callback)

    def _consumer_done_callback(self, task):
        """Callback to handle consumer task completion/failure"""
        try:
            task.result()
        except asyncio.CancelledError:
            logging.info("Consumer task was cancelled")
        except Exception as e:
            logging.error(f"Consumer task failed with exception: {e}")

    async def _execute_browser_task(self, task: str, structured_output: BaseModel, post_process_result=None) -> str:
        """
        Common method to execute browser tasks with the given task and structured output model.

        Args:
            task: The detailed instruction for the browser agent.
            structured_output: The output model class to use for validation.
            post_process_result: Optional callback function to process the result before returning.
                               Should take (result_model, output_model) as parameters.

        Returns:
            A JSON string representing the structured result from the agent.
        """
        logging.info(f"Browser agent starting task: {task}")

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
        # Redirect stdout to stderr only during agent.run
        history = await agent.run(max_steps=300)
        result_model = history.final_result()

        if result_model:
            result = output_model.model_validate_json(result_model)
            logging.info("=========:\n" + result.result)

            # Apply post-processing if provided
            if post_process_result:
                post_process_result(result, output_model)

            return result_model
        else:
            # Return a standard error result for any failure
            error_result = BrowserTaskResult(
                task_completed=False,
                summary="Agent failed to produce a result.",
                result="The agent did not return a valid result.",
            )
            return error_result.model_dump_json()

    async def run_browser_task_by_cve(self, cve: str, structured_output: BaseModel) -> str:
        """
        Initializes and runs a browser_use agent for a given CVE task, returning a JSON string.
        Args:
            cve: The CVE ID to analyze.
            structured_output: The output model class to use.
        Returns:
            A JSON string representing the structured result from the agent.
        """
        task = f"Please visit {cve}, only focusing on the Impacted Component table, calculate the total number of impacted components that are Precisely Matched and Fuzzy Matched (Name + Version). If the **Impacted Component** table has multiple pages with pagination controls, when pagedowning, select and keep maximum components/page on the pagination to review all pages, sum the counts of each impacted component row on each page to calculate the total. **Make sure all pages are reviewed**"

        def post_process_cve_result(result, output_model):

            print(f"{result.precisely_matched_assets}/{result.fuzzy_matched_asset}")
            cve_analyzed = {
                "cve_id": cve,
                "affected_asset_result": result.result,
                "precisely_matched_assets": result.precisely_matched_assets,
                "fuzzy_matched_asset": result.fuzzy_matched_asset,
            }
            headers = {"Content-Type": "application/json"}
            url = ""
            response = requests.post(url, headers=headers, json=cve_analyzed)
            response.raise_for_status()

        return await self._execute_browser_task(task, structured_output, post_process_cve_result)

    async def run_browser_task(self, task: str, structured_output: BaseModel) -> str:
        """
        Initializes and runs a browser_use agent for a given task, returning a JSON string.
        Args:
            task: The detailed instruction for the browser agent.
            structured_output: The output model class to use.
        Returns:
            A JSON string representing the structured result from the agent.
        """
        return await self._execute_browser_task(task, structured_output)

    async def close_browser(self):
        # Cancel consumer task if running
        if hasattr(self, "consumer_task") and not self.consumer_task.done():
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
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
                logging.info(f"{result.precisely_matched_assets}/{result.fuzzy_matched_asset}")
                logging.info(f"--- BROWSER AGENT FINAL RESULT for task: {task} ---")
                logging.info(result)
            await runner.close_browser()
        else:
            logging.info("Please provide one or more tasks as command-line arguments for direct testing.")


class AssetMatch(BaseModel):
    result: str = Field(description="Result data, or error messages from the tasks.")
    precisely_matched_assets: int = Field(description="The number of Assets Precisely Matched")
    fuzzy_matched_asset: int = Field(description="The number of Assets Fuzzy Matched (Name + Version)")


# MCP server implementation to provide tools using MCP official Python SDK

mcp = FastMCP(name="BrowserMCP", port=3001)

runner = BrowserAgentRunner(headless=False)


@mcp.tool()
async def asset_match_by_cve(cve: str) -> dict:
    """Match assets for a given CVE.
    Args:
        cve: The unique identifier for the vulnerability (e.g., 'CVE-2025-12345').
    """
    await runner.receive(cve)
    # Return a valid AssetMatch object
    # return AssetMatch(
    return {
        "result": f"The asset match task for {cve} successfully added to the task queue",
        # precisely_matched_assets=0,
        # fuzzy_matched_asset=0
    }
    # Ensure we return a Pydantic model, not a dict or JSON string
    # return AssetMatch.model_validate_json(result_model)


@mcp.tool()
async def browser(task: str) -> dict:
    """Performing a task through the browser.
    Args:
        task: The specific task.
    """
    result = await runner.recerun_browser_task(task)
    # Return a valid AssetMatch object
    # return AssetMatch(
    return {
        "result": f"{result.result}",
        # precisely_matched_assets=0,
        # fuzzy_matched_asset=0
    }
    # Ensure we return a Pydantic model, not a dict or JSON string
    # return AssetMatch.model_validate_json(result_model)


async def main():
    await runner.start_browser()
    try:
        await mcp.run_streamable_http_async()  # Use HTTP transport instead of stdio
    finally:
        await runner.close_browser()


if __name__ == "__main__":
    asyncio.run(main())
