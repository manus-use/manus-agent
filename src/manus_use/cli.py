"""Command-line interface for ManusUse."""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

import toml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import __version__
from .config import Config
from .multi_agents import Orchestrator

console = Console()


# ---------------------------------------------------------------------------
# Complexity heuristic
# ---------------------------------------------------------------------------


def is_complex_task(user_input: str) -> bool:
    """Determine if a task requires multi-agent orchestration."""
    complex_indicators = [
        r"\band\b.*\band\b",
        r"\bthen\b",
        r"\bafter\b",
        r"\banalyze\b.*\b(create|generate|build)",
        r"\bcompare\b.*\bsummarize",
        r"\bmultiple\b",
        r"\bsteps?\b",
        r"\bworkflow\b",
        r"(first|second|third|finally)",
        r"\b(visuali[sz]e|chart|graph)\b.*\b(analyze|data)",
        r"\bbrowse\b.*\b(extract|analyze)",
        r"\bresearch\b.*\b(implement|create)",
    ]

    if len(user_input.split()) > 30:
        return True

    if len(re.split(r"[.;]", user_input)) > 2:
        return True

    for pattern in complex_indicators:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True

    return False


# ---------------------------------------------------------------------------
# Execution plan display
# ---------------------------------------------------------------------------


def display_task_plan(tasks) -> None:
    """Display the execution plan in a formatted table."""
    table = Table(title="Execution Plan", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Task", style="white")
    table.add_column("Agent", style="yellow")
    table.add_column("Dependencies", style="blue")

    for task in tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "None"
        table.add_row(
            task.task_id,
            task.description[:50] + "..." if len(task.description) > 50 else task.description,
            task.agent_type.value,
            deps,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_agent(agent_type: str, config: Config, **agent_kwargs):
    """Instantiate the requested agent type.

    Extra keyword arguments (e.g. *callback_handler*) are forwarded to the
    agent constructor so callers can inject streaming handlers.
    """
    if agent_type == "browser":
        from .agents import BrowserUseAgent

        return BrowserUseAgent(config=config, **agent_kwargs)
    if agent_type == "data":
        from .agents import DataAnalysisAgent

        return DataAnalysisAgent(config=config, **agent_kwargs)
    if agent_type == "mcp":
        from .agents import MCPAgent

        return MCPAgent(config=config, **agent_kwargs)
    # default: manus
    from .agents import ManusAgent

    return ManusAgent(config=config, **agent_kwargs)


# ---------------------------------------------------------------------------
# Single-shot and interactive runners
# ---------------------------------------------------------------------------


def _build_analyze_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `analyze` subcommand."""
    parser = argparse.ArgumentParser(
        prog="manus-use analyze",
        description="Run a vulnerability intelligence analysis for a CVE.",
    )
    parser.add_argument(
        "cve_id",
        metavar="CVE-ID",
        help="CVE identifier to analyze (e.g. CVE-2025-6554)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable Docker-based exploit verification (default: off)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "lark"],
        default="text",
        help="Report output format (default: text)",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to a config.toml file (overrides default search paths)",
    )
    return parser


def _run_analyze(
    *,
    cve_id: str,
    verify: bool,
    output: str,
    config: Config,
) -> int:
    """Run the vulnerability intelligence analysis for a CVE and report it.

    Returns an exit code (0 = success, 1 = failure).
    """
    try:
        from .agents import VulnerabilityIntelligenceAgent
    except ImportError as exc:
        console.print(f"[red]\u2717 Vulnerability intelligence agent unavailable: {exc}[/red]")
        return 1

    console.print(f"[bold blue]Analyzing {cve_id}[/bold blue]")
    if verify:
        console.print("[dim]Exploit verification: ENABLED[/dim]")

    try:
        agent = VulnerabilityIntelligenceAgent(config=config)
    except Exception as exc:
        console.print(f"[red]\u2717 Failed to initialise agent: {exc}[/red]")
        return 1

    request = agent.build_request(cve_id, verify=verify)

    try:
        with console.status(f"Running analysis for {cve_id}\u2026", spinner="dots"):
            result = agent.handle_request(request)
    except Exception as exc:
        console.print(f"[red]\u2717 Analysis failed: {exc}[/red]")
        return 1

    result_text = str(result)

    if output == "json":
        import json

        console.print_json(json.dumps({"cve": cve_id, "report": result_text}))
    elif output == "lark":
        console.print("[dim]Report delivered to Lark (see create_lark_document output above).[/dim]")
        console.print(Panel(result_text, title=f"[bold green]{cve_id}[/bold green]", border_style="green"))
    else:
        console.print(Panel(result_text, title=f"[bold green]{cve_id}[/bold green]", border_style="green"))

    return 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

_HISTORY_PATH = Path.home() / ".manus-use" / "history.jsonl"


def _append_history(
    task: str,
    result: str,
    *,
    agent_type: str,
    mode: str,
    success: bool,
    format: str = "text",
) -> None:
    """Append a single-shot run record to the history log.

    The log lives at ``~/.manus-use/history.jsonl`` (one JSON object per line)
    so it can be streamed/grepped without loading the whole file.

    Each record has:
    - ``timestamp``: ISO-8601 UTC timestamp
    - ``task``: the user's task string
    - ``agent``: agent type used
    - ``mode``: execution mode (auto/single/multi)
    - ``format``: output format requested
    - ``success``: boolean
    - ``result``: the result text (or error message)
    """
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": task,
        "agent": agent_type,
        "mode": mode,
        "format": format,
        "success": success,
        "result": result,
    }
    with _HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# manus-use discover
# ---------------------------------------------------------------------------

# manus-use discover
# ---------------------------------------------------------------------------


def _build_discover_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `discover` subcommand."""
    parser = argparse.ArgumentParser(
        prog="manus-use discover",
        description="Discover recent high-EPSS CVEs and submit them for tracking.",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        default=None,
        help="Start date for the discovery window (default: 4 weeks ago)",
    )
    parser.add_argument(
        "--min-epss",
        metavar="SCORE",
        type=float,
        default=0.5,
        help="Minimum EPSS score threshold, 0.0–1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Report output format (default: text)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover CVEs but do not submit them (default: off)",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to a config.toml file (overrides default search paths)",
    )
    return parser


def _run_discover(
    *,
    since: str | None,
    min_epss: float,
    output: str,
    dry_run: bool,
    config: Config,
) -> int:
    """Run the vulnerability discovery workflow and report the results.

    Returns an exit code (0 = success, 1 = failure).
    """
    try:
        from .agents import VulnerabilityDiscoveryAgent
    except ImportError as exc:
        console.print(f"[red]\u2717 Vulnerability discovery agent unavailable: {exc}[/red]")
        return 1

    # Validate min_epss range
    if not 0.0 <= min_epss <= 1.0:
        console.print("[red]\u2717 --min-epss must be between 0.0 and 1.0[/red]")
        return 1

    since_display = since or "4 weeks ago"
    console.print(f"[bold blue]Discovering CVEs[/bold blue] since [cyan]{since_display}[/cyan] (min-epss={min_epss})")
    if dry_run:
        console.print("[dim]Dry-run mode: CVEs will NOT be submitted.[/dim]")

    try:
        agent = VulnerabilityDiscoveryAgent(config=config)
    except Exception as exc:
        console.print(f"[red]\u2717 Failed to initialise agent: {exc}[/red]")
        return 1

    request = VulnerabilityDiscoveryAgent.build_request(
        since=since,
        min_epss=min_epss,
        dry_run=dry_run,
    )

    try:
        with console.status("Running discovery workflow\u2026", spinner="dots"):
            result = agent.handle_request(request)
    except Exception as exc:
        console.print(f"[red]\u2717 Discovery failed: {exc}[/red]")
        return 1

    result_text = str(result)

    if output == "json":
        import json

        console.print_json(
            json.dumps(
                {
                    "since": since,
                    "min_epss": min_epss,
                    "dry_run": dry_run,
                    "result": result_text,
                }
            )
        )
    else:
        console.print(
            Panel(
                result_text,
                title="[bold green]Discovery Results[/bold green]",
                border_style="green",
            )
        )

    return 0


# ---------------------------------------------------------------------------
# manus-use remediate
# ---------------------------------------------------------------------------


def _build_remediate_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``remediate`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="manus-use remediate",
        description="Generate actionable remediation guidance for a CVE.",
    )
    parser.add_argument(
        "cve_id",
        metavar="CVE-ID",
        help="CVE identifier to remediate (e.g. CVE-2024-3094)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Report output format (default: text)",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to a config.toml file (overrides default search paths)",
    )
    return parser


def _run_remediate(
    *,
    cve_id: str,
    output: str,
    config: Config,
) -> int:
    """Run the remediation workflow for a CVE and print the report.

    Returns an exit code (0 = success, 1 = failure).
    """
    try:
        from .agents import RemediationAgent
    except ImportError as exc:
        console.print(f"[red]\u2717 Remediation agent unavailable: {exc}[/red]")
        return 1

    console.print(f"[bold blue]Remediating {cve_id}[/bold blue]")

    try:
        agent = RemediationAgent(config=config)
    except Exception as exc:
        console.print(f"[red]\u2717 Failed to initialise agent: {exc}[/red]")
        return 1

    request = RemediationAgent.build_request(cve_id, output=output)

    try:
        with console.status(f"Generating remediation for {cve_id}\u2026", spinner="dots"):
            result = agent.handle_request(request)
    except Exception as exc:
        console.print(f"[red]\u2717 Remediation failed: {exc}[/red]")
        return 1

    result_text = str(result)

    if output == "json":
        import json as _json

        console.print_json(_json.dumps({"cve": cve_id, "report": result_text}))
    else:
        console.print(
            Panel(
                result_text,
                title=f"[bold green]{cve_id} Remediation[/bold green]",
                border_style="green",
            )
        )

    return 0


def _run_single_shot(
    task: str,
    *,
    mode: str,
    agent_type: str,
    show_plan: bool,
    output: Path | None,
    fmt: str,
    no_history: bool,
    config: Config,
    stream: bool = False,
) -> int:
    """Execute a single task non-interactively and exit.

    Returns an exit code (0 = success, 1 = failure).
    """
    try:
        agent = _make_agent(agent_type, config)
    except Exception as exc:
        console.print(f"[red]✗ Failed to initialise agent: {exc}[/red]")
        return 1

    use_multi_agent = mode == "multi" or (mode == "auto" and is_complex_task(task))

    result_text: str
    if use_multi_agent:
        console.print("[dim]Detected complex task – using multi-agent orchestration[/dim]")
        orchestrator = Orchestrator(config=config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            _ptask = progress.add_task("Running workflow…", total=None)
            workflow_result = orchestrator.run(task)
            progress.update(_ptask, completed=True)

        if not workflow_result.success:
            error_msg = f"Task failed: {workflow_result.error}"
            console.print(
                Panel(
                    error_msg,
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )
            if not no_history:
                _append_history(
                    task,
                    error_msg,
                    agent_type=agent_type,
                    mode=mode,
                    success=False,
                    format=fmt,
                )
            return 1
        result_text = workflow_result.output
    else:
        if stream:
            # --stream + --format json are incompatible: warn and fall back.
            if fmt == "json":
                sys.stderr.write(
                    "[warn] --stream is incompatible with --format json; falling back to buffered JSON output\n"
                )

            # Try to use PrintingCallbackHandler for real-time output.
            _streamed = False
            try:
                from strands.handlers import PrintingCallbackHandler

                stream_agent = _make_agent(
                    agent_type,
                    config,
                    callback_handler=PrintingCallbackHandler(),
                )
                response = stream_agent(task)
                # PrintingCallbackHandler already flushed tokens to stdout.
                # str(response) extracts text content from AgentResult.
                result_text = str(response)
                _streamed = True
            except (ImportError, TypeError):
                pass

            if not _streamed:
                # Fallback: call normally, check if result is a generator.
                response = agent(task)
                import types

                if isinstance(response, types.GeneratorType):
                    chunks = []
                    for chunk in response:
                        chunk_str = str(chunk)
                        sys.stdout.write(chunk_str)
                        sys.stdout.flush()
                        chunks.append(chunk_str)
                    sys.stdout.write("\n")
                    result_text = "".join(chunks)
                else:
                    sys.stderr.write(
                        "[warn] streaming not supported by this agent/model, falling back to buffered output\n"
                    )
                    result_text = str(response)
        else:
            with console.status("Running…", spinner="dots"):
                response = agent(task)
            result_text = str(response)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    if stream and fmt != "json" and not use_multi_agent:
        # Streaming path already printed to stdout; only handle file save + history.
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result_text, encoding="utf-8")
            console.print(f"[dim]Output saved to {output}[/dim]")
        if not no_history:
            _append_history(
                task,
                result_text,
                agent_type=agent_type,
                mode=mode,
                success=True,
                format=fmt,
            )
        return 0

    if fmt == "json":
        payload = json.dumps(
            {
                "task": task,
                "agent": agent_type,
                "mode": "multi" if use_multi_agent else "single",
                "result": result_text,
            },
            ensure_ascii=False,
            indent=2,
        )
        # Print raw JSON to stdout so it can be piped to other tools.
        # Rich's print_json adds colours that break pipe consumers; use sys.stdout.
        sys.stdout.write(payload + "\n")
    else:
        console.print(Panel(result_text, title="[bold green]Result[/bold green]", border_style="green"))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            output.write_text(payload, encoding="utf-8")
        else:
            output.write_text(result_text, encoding="utf-8")
        console.print(f"[dim]Output saved to {output}[/dim]")

    if not no_history:
        _append_history(
            task,
            result_text,
            agent_type=agent_type,
            mode=mode,
            success=True,
            format=fmt,
        )

    return 0


def _run_interactive(
    *,
    mode: str,
    agent_type: str,
    show_plan: bool,
    config: Config,
) -> None:
    """Run the interactive REPL loop."""
    console.print("[bold blue]Welcome to ManusUse![/bold blue]")
    console.print("An advanced AI agent framework powered by Strands SDK")
    console.print(f"Mode: [cyan]{mode}[/cyan]  Agent: [cyan]{agent_type}[/cyan]\n")

    console.print("Initialising agents…", style="dim")
    try:
        agent = _make_agent(agent_type, config)
        orchestrator = None
        if mode in ("auto", "multi"):
            orchestrator = Orchestrator(config=config)
        console.print("✓ Agents initialised\n", style="green")
    except Exception as exc:
        console.print(f"[red]✗ Failed to initialise agents: {exc}[/red]")
        sys.exit(1)

    console.print("Type your requests below (type 'exit' to quit):\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if user_input.lower() in ("exit", "quit", "bye"):
                console.print("\n[bold blue]Goodbye![/bold blue]")
                break

            use_multi_agent = mode == "multi" or (mode == "auto" and is_complex_task(user_input))

            if use_multi_agent and orchestrator:
                console.print("\n[bold green]Orchestrator[/bold green]: Planning execution…\n")

                if show_plan:
                    planner = orchestrator.agents.get("planner")
                    if planner:
                        tasks = planner.create_plan(user_input)
                        display_task_plan(tasks)
                        console.print()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    _ptask = progress.add_task("Executing multi-agent workflow…", total=None)
                    result = orchestrator.run(user_input)
                    progress.update(_ptask, completed=True)

                if result.success:
                    console.print(
                        Panel(
                            result.output,
                            title="[bold green]Result[/bold green]",
                            border_style="green",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"Task failed: {result.error}",
                            title="[bold red]Error[/bold red]",
                            border_style="red",
                        )
                    )
            else:
                console.print("\n[bold green]Agent[/bold green]: ", end="")
                with console.status("Thinking…", spinner="dots"):
                    response = agent(user_input)
                console.print(response)

        except KeyboardInterrupt:
            console.print("\n\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]")


# ---------------------------------------------------------------------------
# manus-use init
# ---------------------------------------------------------------------------

# Provider metadata: (display_name, default_model, env_var_for_api_key, needs_api_key)
_PROVIDERS = {
    "openai": ("OpenAI", "gpt-4o", "OPENAI_API_KEY", True),
    "anthropic": ("Anthropic", "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY", True),
    "bedrock": ("AWS Bedrock", "us.anthropic.claude-3-5-sonnet-20241022-v2:0", None, False),
    "ollama": ("Ollama (local)", "llama3.2", None, False),
}

_DEFAULT_CONFIG_PATH = Path.home() / ".manus-use" / "config.toml"


def _cmd_init(args: argparse.Namespace) -> int:  # noqa: C901
    """Guided interactive config generator."""
    console.print(
        Panel(
            "[bold]Welcome to [cyan]manus-use init[/cyan]![/bold]\n"
            "This wizard will create a [yellow]config.toml[/yellow] for you.",
            border_style="blue",
        )
    )

    # Destination path
    dest: Path = args.output or _DEFAULT_CONFIG_PATH
    if dest.exists() and not args.force:
        overwrite = Confirm.ask(f"[yellow]{dest}[/yellow] already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[dim]Aborted – existing config left unchanged.[/dim]")
            return 0

    # Provider selection
    console.print("\n[bold]Choose a provider:[/bold]")
    provider_keys = list(_PROVIDERS.keys())
    for i, key in enumerate(provider_keys, 1):
        display, default_model, env_var, needs_key = _PROVIDERS[key]
        env_hint = f" (env: [dim]{env_var}[/dim])" if env_var else ""
        console.print(f"  [cyan]{i}[/cyan]. {display}{env_hint}")

    while True:
        choice = Prompt.ask(
            "Provider",
            default="1",
            choices=[str(i) for i in range(1, len(provider_keys) + 1)],
            show_choices=False,
        )
        provider = provider_keys[int(choice) - 1]
        break

    display_name, default_model, env_var, needs_key = _PROVIDERS[provider]

    # Model
    model = Prompt.ask(
        f"\n[bold]Model ID[/bold] for {display_name}",
        default=default_model,
    )

    # API key / region
    api_key: str | None = None
    aws_region: str | None = None
    base_url: str | None = None

    if needs_key:
        env_val = os.environ.get(env_var, "") if env_var else ""
        if env_val:
            console.print(
                f"[green]✓[/green] Found [dim]{env_var}[/dim] in environment "
                "(will be used at runtime – not stored in config)."
            )
            store_key = Confirm.ask("Store API key in config file anyway?", default=False)
        else:
            console.print(f"[yellow]![/yellow] [dim]{env_var}[/dim] not set in environment.")
            store_key = Confirm.ask("Enter API key now and store in config?", default=True)

        if store_key:
            api_key = Prompt.ask(f"[bold]{env_var}[/bold]", password=True)

    elif provider == "bedrock":
        aws_region = Prompt.ask(
            "\n[bold]AWS region[/bold]",
            default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    elif provider == "ollama":
        base_url = Prompt.ask(
            "\n[bold]Ollama base URL[/bold]",
            default="http://localhost:11434",
        )

    # Sandbox
    sandbox_enabled = Confirm.ask("\n[bold]Enable Docker sandbox?[/bold]", default=True)

    # Assemble config dict
    llm_section: dict = {"provider": provider, "model": model}
    if api_key:
        llm_section["api_key"] = api_key
    if aws_region:
        llm_section["aws_region"] = aws_region
    if base_url:
        llm_section["base_url"] = base_url

    config_data: dict = {
        "llm": llm_section,
        "sandbox": {"enabled": sandbox_enabled},
    }

    # Write
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        toml.dump(config_data, fh)

    console.print(
        Panel(
            f"[green]✓ Config written to[/green] [bold]{dest}[/bold]\n\n"
            f"Run [cyan]manus-use doctor[/cyan] to verify your setup.",
            border_style="green",
            title="[bold green]Done![/bold green]",
        )
    )
    return 0


# ---------------------------------------------------------------------------
# manus-use doctor
# ---------------------------------------------------------------------------

# (package_import, pip_extra, description)
_PROVIDER_PACKAGES: dict[str, list[tuple[str, str, str]]] = {
    "openai": [("openai", "openai", "OpenAI SDK")],
    "anthropic": [("anthropic", "anthropic", "Anthropic SDK")],
    "bedrock": [("boto3", "boto3", "AWS boto3")],
    "ollama": [("ollama", "ollama", "Ollama client")],
}

_ALWAYS_CHECK: list[tuple[str, str, str]] = [
    ("strands", "strands-agents", "Strands Agents SDK"),
    ("rich", "rich", "Rich (terminal UI)"),
    ("toml", "toml", "TOML parser"),
    ("pydantic", "pydantic", "Pydantic"),
]


def _check_import(package: str) -> bool:
    """Return True if *package* is importable."""
    import importlib

    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check packages, config, and environment variables."""
    console.print(
        Panel(
            "[bold cyan]manus-use doctor[/bold cyan] – environment diagnostics",
            border_style="blue",
        )
    )

    issues: list[str] = []

    # ------------------------------------------------------------------
    # 1. Core packages
    # ------------------------------------------------------------------
    console.print("\n[bold]Core packages[/bold]")
    for pkg, pip_name, desc in _ALWAYS_CHECK:
        ok = _check_import(pkg)
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        note = "" if ok else f"  → [dim]pip install {pip_name}[/dim]"
        console.print(f"  {status} {desc} ([dim]{pkg}[/dim]){note}")
        if not ok:
            issues.append(f"Missing package: {pip_name}")

    # ------------------------------------------------------------------
    # 2. Config file
    # ------------------------------------------------------------------
    console.print("\n[bold]Configuration[/bold]")
    config_path: Path | None = args.config

    if config_path is None:
        search_paths = [
            Path("config.toml"),
            Path("config/config.toml"),
            _DEFAULT_CONFIG_PATH,
        ]
        for p in search_paths:
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        console.print(f"  [green]✓[/green] Config found: [bold]{config_path}[/bold]")
        try:
            config = Config.from_file(config_path)
        except Exception as exc:
            console.print(f"  [red]✗[/red] Failed to parse config: {exc}")
            issues.append(f"Config parse error: {exc}")
            config = Config()
    else:
        console.print("  [yellow]![/yellow] No config file found (run [cyan]manus-use init[/cyan] to create one)")
        config = Config()

    # ------------------------------------------------------------------
    # 3. Provider packages + env vars
    # ------------------------------------------------------------------
    provider = (config.llm.provider or "openai").lower()
    console.print(f"\n[bold]Provider[/bold]: [cyan]{provider}[/cyan]  model: [cyan]{config.llm.model}[/cyan]")

    pkg_checks = _PROVIDER_PACKAGES.get(provider, [])
    for pkg, pip_name, desc in pkg_checks:
        ok = _check_import(pkg)
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        note = "" if ok else f"  → [dim]pip install {pip_name}[/dim]"
        console.print(f"  {status} {desc} ([dim]{pkg}[/dim]){note}")
        if not ok:
            issues.append(f"Missing provider package: {pip_name}")

    # Env var checks
    provider_envs: dict[str, list[tuple[str, bool]]] = {
        "openai": [("OPENAI_API_KEY", True)],
        "anthropic": [("ANTHROPIC_API_KEY", True)],
        "bedrock": [
            ("AWS_ACCESS_KEY_ID", False),
            ("AWS_SECRET_ACCESS_KEY", False),
            ("AWS_DEFAULT_REGION", False),
        ],
        "ollama": [],
    }
    for env_var, required in provider_envs.get(provider, []):
        val = os.environ.get(env_var)
        in_config = False
        if provider == "openai" and env_var == "OPENAI_API_KEY":
            in_config = bool(config.llm.api_key)
        elif provider == "anthropic" and env_var == "ANTHROPIC_API_KEY":
            in_config = bool(config.llm.api_key)

        if val or in_config:
            src = "config" if (not val and in_config) else "env"
            console.print(f"  [green]✓[/green] {env_var} ([dim]{src}[/dim])")
        elif required:
            console.print(f"  [red]✗[/red] {env_var} not set (required – set it or store in config.toml)")
            issues.append(f"Missing env var: {env_var}")
        else:
            console.print(f"  [yellow]![/yellow] {env_var} not set (optional for Bedrock)")

    # ------------------------------------------------------------------
    # 4. Optional tools
    # ------------------------------------------------------------------
    console.print("\n[bold]Optional tools[/bold]")
    _optional: list[tuple[str, str, str]] = [
        ("docker", "docker", "Docker (sandbox execution)"),
        ("playwright", "playwright", "Playwright (browser agent)"),
        ("pandas", "pandas", "Pandas (data analysis agent)"),
    ]
    for pkg, pip_name, desc in _optional:
        ok = _check_import(pkg)
        status = "[green]✓[/green]" if ok else "[dim]-[/dim]"
        note = "" if ok else f"  [dim](optional – pip install {pip_name})[/dim]"
        console.print(f"  {status} {desc}{note}")

    # Docker daemon reachable?
    if shutil.which("docker"):
        import subprocess

        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if result.returncode == 0:
            console.print("  [green]✓[/green] Docker daemon reachable")
        else:
            console.print("  [yellow]![/yellow] Docker installed but daemon not running")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.print()
    if not issues:
        console.print(
            Panel(
                "[bold green]All checks passed![/bold green] Your environment looks ready.",
                border_style="green",
            )
        )
        return 0
    else:
        issue_list = "\n".join(f"  • {i}" for i in issues)
        console.print(
            Panel(
                f"[bold red]{len(issues)} issue(s) found:[/bold red]\n{issue_list}",
                border_style="red",
            )
        )
        return 1


def _build_variants_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manus-use variants",
        description="CVE variant analysis — find similar bugs in related codebases",
        add_help=True,
    )
    p.add_argument("cve_id", metavar="CVE-ID", help="CVE identifier, e.g. CVE-2024-3094")
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return p


def _run_variants(argv: list[str]) -> int:
    parser = _build_variants_parser()
    args = parser.parse_args(argv)
    cve_id = args.cve_id.strip()
    if not cve_id:
        parser.error("CVE-ID is required")

    try:
        from manus_use.agents.variant_agent import VariantAnalysisAgent
    except ImportError as exc:
        print(f"[error] missing dependencies: {exc}", file=sys.stderr)
        return 1

    agent = VariantAnalysisAgent()
    result = agent.analyze_variants(cve_id)

    if args.output == "json":
        import json

        print(json.dumps({"cve_id": cve_id, "report": result}))
    else:
        print(result)
    return 0


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    "init",
    "doctor",
    "analyze",
    "history",
    "discover",
    "remediate",
    "variants",
    "epss-trend",
    "patch-diff",
    "compare",
    "silent-patches",
}


# ---------------------------------------------------------------------------
# epss-trend subcommand
# ---------------------------------------------------------------------------


def _build_epss_trend_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manus-use epss-trend",
        description=(
            "Fetch EPSS (Exploit Prediction Scoring System) score history for a CVE \n"
            "and flag significant spikes that indicate new exploitation activity."
        ),
        add_help=True,
    )
    p.add_argument("cve_id", metavar="CVE-ID", help="CVE identifier, e.g. CVE-2024-3094")
    p.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Days of EPSS history to retrieve (default: 30, max: 365)",
    )
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return p


def _run_epss_trend(argv: list[str]) -> int:
    parser = _build_epss_trend_parser()
    args = parser.parse_args(argv)
    cve_id = args.cve_id.strip()
    if not cve_id:
        parser.error("CVE-ID is required")

    try:
        from manus_use.tools.get_epss_trend import _analyse_series, _fetch_epss_time_series
    except ImportError as exc:  # pragma: no cover
        print(f"[error] missing dependencies: {exc}", file=sys.stderr)
        return 1

    try:
        raw = _fetch_epss_time_series(cve_id, args.days)
    except Exception as exc:
        print(f"[error] EPSS API request failed: {exc}", file=sys.stderr)
        return 1

    data_entries = raw.get("data", [])
    if not data_entries:
        print(f"[error] No EPSS data found for {cve_id.upper()}", file=sys.stderr)
        return 1

    entry = data_entries[0]
    time_series = list(entry.get("time-series", []))
    current_point = {
        "date": entry.get("date", ""),
        "epss": entry.get("epss", "0"),
        "percentile": entry.get("percentile", "0"),
    }
    if current_point["date"] and not any(p["date"] == current_point["date"] for p in time_series):
        time_series = [current_point] + time_series

    analysis = _analyse_series(time_series)

    if args.output == "json":
        import json

        print(json.dumps({"cve_id": cve_id.upper(), "analysis": analysis}, indent=2))
        return 0

    # Text output
    spike_flag = "\u26a0\ufe0f  SPIKE DETECTED" if analysis["spike_detected"] else "\u2705  No significant spike"
    current_pct = float(analysis.get("current_percentile", 0))
    print(f"EPSS trend for {cve_id.upper()}  ({len(analysis['points'])} days)")
    print(f"  Current score   : {analysis.get('current_epss', 'N/A'):.4f}  (percentile {current_pct:.1%})")
    print(f"  Oldest score    : {analysis.get('oldest_epss', 'N/A'):.4f}  (date: {analysis.get('oldest_date', 'N/A')})")
    print(f"  Latest date     : {analysis.get('latest_date', 'N/A')}")
    print(f"  Trend           : {analysis['trend']}")
    print(
        f"  Max 7-day jump  : {analysis['max_7d_jump']:.4f}"
        + (f"  (peaked {analysis['max_7d_jump_end_date']})" if analysis["max_7d_jump_end_date"] else "")
    )
    print(f"  {spike_flag}")
    if analysis["points"]:
        print()
        print("  Date         EPSS     Percentile")
        print("  ----------   ------   ----------")
        for pt in analysis["points"][-15:]:  # show last 15 rows
            print(f"  {pt['date']}   {pt['epss']:.4f}   {pt['percentile']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# patch-diff subcommand
# ---------------------------------------------------------------------------


def _build_patch_diff_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manus-use patch-diff",
        description=(
            "Fetch the fixing-commit diff for a CVE from GitHub and produce a\n"
            "structured summary: files changed, functions touched, bug class,\n"
            "and reproduction condition hints."
        ),
        add_help=True,
    )
    p.add_argument("cve_id", metavar="CVE-ID", help="CVE identifier, e.g. CVE-2024-3094")
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return p


def _run_patch_diff(argv: list[str]) -> int:
    parser = _build_patch_diff_parser()
    args = parser.parse_args(argv)
    cve_id = args.cve_id.strip()
    if not cve_id:
        parser.error("CVE-ID is required")

    try:
        from manus_use.tools.get_patch_diff import fetch_and_summarise
    except ImportError as exc:
        print(f"[error] missing dependencies: {exc}", file=sys.stderr)
        return 1

    payload = fetch_and_summarise(cve_id)

    if args.output == "json":
        import json

        print(json.dumps(payload, indent=2))
        return 0

    # --- text output ---
    print(payload["message"])
    if payload["not_found"]:
        return 0

    for idx, s in enumerate(payload["commit_summaries"], 1):
        print(f"\nCommit {idx}: {s['commit_url']}")
        if s["files_changed"]:
            print(f"  Files changed    : {', '.join(s['files_changed'][:8])}")
        if s["functions_touched"]:
            print(f"  Functions touched: {', '.join(s['functions_touched'][:8])}")
        print(f"  Lines +{s['added_lines']} / -{s['removed_lines']}")
        print(f"  Primary bug class: {s['primary_bug_class']}")
        if len(s["matched_bug_classes"]) > 1:
            print(f"  All bug classes  : {', '.join(s['matched_bug_classes'])}")
        if s["reproduction_condition_hints"]:
            print("  Reproduction hints:")
            for hint in s["reproduction_condition_hints"]:
                print(f"    • {hint}")
    return 0


# ---------------------------------------------------------------------------
# silent-patches subcommand
# ---------------------------------------------------------------------------


def _build_silent_patches_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manus-use silent-patches",
        description=(
            "Scan a GitHub repository's commit history for potential silent security\n"
            "fixes — commits that look like security patches (based on keywords in\n"
            "the commit message or diff) but have no associated CVE/GHSA reference.\n"
            "\n"
            "Useful for finding vulnerabilities that were quietly fixed without a CVE,\n"
            "or that the vendor chose not to disclose publicly."
        ),
        add_help=True,
    )
    p.add_argument(
        "repo",
        metavar="OWNER/REPO",
        help="GitHub repository to scan, e.g. 'django/django' or 'curl/curl'",
    )
    p.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Scan commits after this date (default: 90 days ago)",
    )
    p.add_argument(
        "--until",
        default=None,
        metavar="YYYY-MM-DD",
        help="Scan commits before this date (default: today)",
    )
    p.add_argument(
        "--max-commits",
        type=int,
        default=200,
        metavar="N",
        help="Maximum number of commits to inspect (default: 200, max: 500)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Skip diff scan; rely only on commit-message keywords (faster, less precise)",
    )
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return p


def _run_silent_patches(argv: list[str]) -> int:
    parser = _build_silent_patches_parser()
    args = parser.parse_args(argv)

    try:
        from manus_use.tools.find_silent_patches import (
            find_silent_patches_impl,
            render_silent_patches_text,
        )
    except ImportError as exc:  # pragma: no cover
        print(f"[error] missing dependencies: {exc}", file=sys.stderr)
        return 1

    result = find_silent_patches_impl(
        repo=args.repo,
        since=args.since,
        until=args.until,
        max_commits=args.max_commits,
        fast=args.fast,
    )

    if "error" in result:
        print(f"[error] {result['error']}", file=sys.stderr)
        return 1

    if args.output == "json":
        import json

        print(json.dumps(result, indent=2))
        return 0

    print(render_silent_patches_text(result))
    return 0


# compare subcommand
# ---------------------------------------------------------------------------


def _build_compare_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manus-use compare",
        description=(
            "Compare two CVEs side-by-side across CVSS, EPSS, CISA KEV, CWE,\n"
            "attack vector, and other dimensions.  Outputs a prioritisation\n"
            "recommendation: which CVE poses the greater immediate risk and why."
        ),
        add_help=True,
    )
    p.add_argument("cve_id_a", metavar="CVE-ID-A", help="First CVE identifier, e.g. CVE-2024-3094")
    p.add_argument("cve_id_b", metavar="CVE-ID-B", help="Second CVE identifier, e.g. CVE-2021-44228")
    p.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    return p


def _run_compare(argv: list[str]) -> int:
    parser = _build_compare_parser()
    args = parser.parse_args(argv)

    cve_id_a = args.cve_id_a.strip()
    cve_id_b = args.cve_id_b.strip()

    for cid in (cve_id_a, cve_id_b):
        if not cid.upper().startswith("CVE-"):
            print(f"[error] Invalid CVE ID '{cid}'. Must be like 'CVE-YYYY-NNNN'.", file=sys.stderr)
            return 1

    try:
        from manus_use.tools.compare_cves import (
            _build_comparison,
            _build_cve_profile,
            _fetch_kev,
            _render_text,
        )
    except ImportError as exc:  # pragma: no cover
        print(f"[error] missing dependencies: {exc}", file=sys.stderr)
        return 1

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_a = executor.submit(_build_cve_profile, cve_id_a)
        future_b = executor.submit(_build_cve_profile, cve_id_b)
        future_kev_a = executor.submit(_fetch_kev, cve_id_a)
        future_kev_b = executor.submit(_fetch_kev, cve_id_b)
        profile_a = future_a.result()
        profile_b = future_b.result()
        kev_a = future_kev_a.result()
        kev_b = future_kev_b.result()

    comparison = _build_comparison(profile_a, kev_a, profile_b, kev_b)

    if args.output == "json":
        import json

        print(json.dumps(comparison, indent=2))
        return 0

    # Text output
    print(_render_text(comparison))
    return 0


def _build_run_parser() -> argparse.ArgumentParser:
    """Build the top-level run/interactive parser."""
    parser = argparse.ArgumentParser(
        prog="manus-use",
        description="ManusUse – Advanced AI Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single-shot (non-interactive)\n"
            '  manus-use "Create a factorial function in Python"\n'
            '  manus-use --agent browser "Find the top 5 trending GitHub repos today"\n'
            '  manus-use --output result.txt "Summarise the latest AI news"\n'
            '  manus-use --format json "List prime numbers up to 50"\n'
            '  manus-use --format json "task" | jq .result\n\n'
            "  # Interactive REPL\n"
            "  manus-use\n"
            "  manus-use --mode multi\n\n"
            "  # Setup helpers\n"
            "  manus-use init           # create ~/.manus-use/config.toml interactively\n"
            "  manus-use doctor         # check packages, config, and API keys\n"
            "  manus-use history        # show recent runs (use --help for filters)\n"
            "\n"
            "  # EPSS trend analysis\n"
            "  manus-use epss-trend CVE-2024-3094\n"
            "  manus-use epss-trend CVE-2024-3094 --days 90 --output json\n"
            "\n"
            "  # Patch diff summariser (fixing-commit analysis)\n"
            "  manus-use patch-diff CVE-2024-3094\n"
            "  manus-use patch-diff CVE-2024-3094 --output json\n"
            "\n"
            "  # Vulnerability intelligence analysis\n"
            "  manus-use analyze CVE-2025-6554\n"
            "  manus-use analyze CVE-2024-3094 --verify --output json\n"
            "  \n"
            "  # CVE discovery\n"
            "  manus-use discover\n"
            "  manus-use discover --since 2025-06-01 --min-epss 0.7 --output json\n"
            "  manus-use discover --dry-run\n"
            "  \n"
            "  # CVE remediation guidance\n"
            "  manus-use remediate CVE-2024-3094\n"
            "  manus-use remediate CVE-2024-3094 --output json\n"
            "  \n"
            "  # Silent patch detection\n"
            "  manus-use silent-patches django/django\n"
            "  manus-use silent-patches curl/curl --since 2024-01-01 --output json\n"
            "  manus-use silent-patches owner/repo --fast --max-commits 100\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"manus-use {__version__}",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task to execute (omit for interactive mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "multi"],
        default="auto",
        help="Execution mode: auto (detect complexity), single, or multi (default: auto)",
    )
    parser.add_argument(
        "--agent",
        choices=["manus", "browser", "data", "mcp"],
        default="manus",
        dest="agent_type",
        help="Agent type to use for single-agent execution (default: manus)",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Show the multi-agent execution plan before running",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help="Save the result to FILE (single-shot mode only)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="fmt",
        help="Output format for single-shot mode: text (default) or json",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        dest="no_history",
        help=f"Do not record this run in the history log ({_HISTORY_PATH})",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to a config.toml file (overrides default search paths)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream output tokens in real time (single-shot mode only)",
    )
    return parser


def _build_init_parser() -> argparse.ArgumentParser:
    """Build the `init` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="manus-use init",
        description="Guided wizard to create a manus-use configuration file.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help=f"Where to write config (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config without prompting",
    )
    return parser


def _build_doctor_parser() -> argparse.ArgumentParser:
    """Build the `doctor` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="manus-use doctor",
        description="Diagnose your manus-use installation: packages, config file, API keys.",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=Path,
        default=None,
        help="Path to a config.toml file to validate (overrides default search paths)",
    )
    return parser


def _build_history_parser() -> argparse.ArgumentParser:
    """Build the `history` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="manus-use history",
        description="Show past single-shot task runs recorded in the history log.",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=20,
        help="Maximum number of entries to show (default: 20, 0 = all)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="fmt",
        help="Display format: text (default) or json",
    )
    parser.add_argument(
        "--grep",
        metavar="PATTERN",
        default=None,
        help="Filter entries whose task contains PATTERN (case-insensitive substring)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all history entries (irreversible)",
    )
    return parser


def _cmd_history(args: argparse.Namespace) -> int:
    """Display (or clear) the single-shot run history log."""
    if args.clear:
        if _HISTORY_PATH.exists():
            _HISTORY_PATH.unlink()
        console.print("[dim]History cleared.[/dim]")
        return 0

    if not _HISTORY_PATH.exists():
        console.print("[dim]No history yet – run a task with [cyan]manus-use 'task...'[/cyan] first.[/dim]")
        return 0

    records = []
    with _HISTORY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

    if args.grep:
        pattern = args.grep.lower()
        records = [r for r in records if pattern in r.get("task", "").lower()]

    # Most-recent entries first, then apply limit.
    records = list(reversed(records))
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        console.print("[dim]No matching history entries.[/dim]")
        return 0

    if args.fmt == "json":
        sys.stdout.write(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        return 0

    # Text table
    from rich.table import Table

    table = Table(
        title=f"Run history ({len(records)} entries)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Timestamp", style="cyan", width=22)
    table.add_column("Agent", style="yellow", width=8)
    table.add_column("Fmt", style="blue", width=5)
    table.add_column("OK", width=3)
    table.add_column("Task", style="white")

    for i, rec in enumerate(records, 1):
        ts = rec.get("timestamp", "-")
        # Trim timezone indicator for brevity
        ts = ts[:19].replace("T", " ")
        ok = "[green]✓[/green]" if rec.get("success") else "[red]✗[/red]"
        task_text = rec.get("task", "")[:80]
        if len(rec.get("task", "")) > 80:
            task_text += "…"
        table.add_row(
            str(i),
            ts,
            rec.get("agent", "-"),
            rec.get("format", "text"),
            ok,
            task_text,
        )

    console.print(table)
    console.print(f"[dim]History file: {_HISTORY_PATH}[/dim]")
    return 0


def main() -> None:
    """Main CLI entry point."""
    argv = sys.argv[1:]

    # Check whether the first non-flag token is a known subcommand.
    first_positional = next((a for a in argv if not a.startswith("-")), None)

    if first_positional == "init":
        idx = argv.index("init")
        args = _build_init_parser().parse_args(argv[idx + 1 :])
        sys.exit(_cmd_init(args))

    if first_positional == "doctor":
        idx = argv.index("doctor")
        args = _build_doctor_parser().parse_args(argv[idx + 1 :])
        sys.exit(_cmd_doctor(args))

    if first_positional == "analyze":
        idx = argv.index("analyze")
        analyze_args = _build_analyze_parser().parse_args(argv[idx + 1 :])
        config = Config.from_file(analyze_args.config)
        sys.exit(
            _run_analyze(
                cve_id=analyze_args.cve_id,
                verify=analyze_args.verify,
                output=analyze_args.output,
                config=config,
            )
        )

    if first_positional == "history":
        idx = argv.index("history")
        history_args = _build_history_parser().parse_args(argv[idx + 1 :])
        sys.exit(_cmd_history(history_args))

    if first_positional == "variants":
        idx = argv.index("variants")
        sys.exit(_run_variants(argv[idx + 1 :]))

    if first_positional == "epss-trend":
        idx = argv.index("epss-trend")
        sys.exit(_run_epss_trend(argv[idx + 1 :]))

    if first_positional == "patch-diff":
        idx = argv.index("patch-diff")
        sys.exit(_run_patch_diff(argv[idx + 1 :]))

    if first_positional == "compare":
        idx = argv.index("compare")
        sys.exit(_run_compare(argv[idx + 1 :]))

    if first_positional == "silent-patches":
        idx = argv.index("silent-patches")
        sys.exit(_run_silent_patches(argv[idx + 1 :]))

    if first_positional == "discover":
        idx = argv.index("discover")
        discover_args = _build_discover_parser().parse_args(argv[idx + 1 :])
        config = Config.from_file(discover_args.config)
        sys.exit(
            _run_discover(
                since=discover_args.since,
                min_epss=discover_args.min_epss,
                output=discover_args.output,
                dry_run=discover_args.dry_run,
                config=config,
            )
        )

    if first_positional == "remediate":
        idx = argv.index("remediate")
        remediate_args = _build_remediate_parser().parse_args(argv[idx + 1 :])
        config = Config.from_file(remediate_args.config)
        sys.exit(
            _run_remediate(
                cve_id=remediate_args.cve_id,
                output=remediate_args.output,
                config=config,
            )
        )

    # Default: run / interactive
    run_parser = _build_run_parser()
    args = run_parser.parse_args(argv)

    config = Config.from_file(args.config)

    if args.task is not None:
        # Non-interactive single-shot mode
        exit_code = _run_single_shot(
            args.task,
            mode=args.mode,
            agent_type=args.agent_type,
            show_plan=args.show_plan,
            output=args.output,
            fmt=args.fmt,
            no_history=args.no_history,
            config=config,
            stream=args.stream,
        )
        sys.exit(exit_code)
    else:
        if args.output is not None:
            run_parser.error("--output requires a task argument")
        if args.fmt != "text":
            run_parser.error("--format requires a task argument")
        _run_interactive(
            mode=args.mode,
            agent_type=args.agent_type,
            show_plan=args.show_plan,
            config=config,
        )


if __name__ == "__main__":
    main()
