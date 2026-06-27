# ManusUse

[![PyPI version](https://img.shields.io/pypi/v/manus-use.svg)](https://pypi.org/project/manus-use/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/manus-use/manus-use/actions/workflows/test.yml/badge.svg)](https://github.com/manus-use/manus-use/actions)

A powerful, extensible framework for building AI agents with comprehensive tool support, multi-agent orchestration, and advanced web automation capabilities.

## Overview

ManusUse empowers developers to create sophisticated AI agents that can:

- Execute code in secure Docker sandboxes
- Automate web browsing and data extraction
- Analyze data and generate visualizations
- Coordinate multiple specialized agents for complex tasks
- Integrate with various LLM providers seamlessly
- Perform deep vulnerability intelligence analysis

Built on [Strands SDK](https://github.com/strands-agents/sdk-python) and integrated with [browser-use](https://github.com/browser-use/browser-use), ManusUse provides a production-ready foundation for AI agent development.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Security & Vulnerability Intelligence](#security--vulnerability-intelligence)
- [Development](#development)

---

## Installation

```bash
# Basic installation
pip install manus-use

# With browser automation support
pip install manus-use[browser]
playwright install chromium

# Full installation with all optional features
pip install manus-use[browser,search,visualization]
```

---

## Quick Start

### 1. Initialize your configuration

```bash
manus-use init
```

The interactive wizard creates `~/.manus-use/config.toml` with your LLM provider credentials.

### 2. Check your environment

```bash
manus-use doctor
```

Verifies installed packages, configuration, and API key accessibility.

### 3. Run your first task

```bash
# Single-shot (non-interactive)
manus-use "Write a Python script that fetches the current Bitcoin price"

# Interactive REPL
manus-use
```

---

## CLI Reference

ManusUse ships a single `manus-use` entry point with several subcommands.

### `manus-use [task]` — Run a task

```bash
# Single-shot task (prints result, then exits)
manus-use "Create a factorial function in Python"

# Use a specific agent type
manus-use --agent browser "Find the top 5 trending GitHub repos today"

# Force multi-agent orchestration
manus-use --mode multi "Research quantum computing and create a presentation"

# Save output to a file
manus-use --output result.txt "Summarise the latest AI news"

# JSON output for piping into other tools
manus-use --format json "List the first 10 prime numbers" | jq .result

# Stream output tokens in real time
manus-use --stream "Write a short story about a robot"

# Interactive REPL (omit the task argument)
manus-use
manus-use --mode multi
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {auto,single,multi}` | `auto` | Execution mode; `auto` detects task complexity |
| `--agent {manus,browser,data,mcp}` | `manus` | Agent type for single-agent execution |
| `--show-plan` | off | Print the multi-agent plan before running |
| `--output FILE` | — | Write result to FILE (single-shot only) |
| `--format {text,json}` | `text` | Output format; `json` is scriptable |
| `--stream` | off | Stream output tokens in real time (single-shot only) |
| `--no-history` | off | Skip recording this run in the history log |
| `--config FILE` | — | Override default config file search path |
| `--version` | — | Print version and exit |

### `manus-use init` — Configure credentials

```bash
manus-use init                       # write to ~/.manus-use/config.toml
manus-use init --output ./my.toml   # write to a custom location
manus-use init --force               # overwrite without prompting
```

### `manus-use doctor` — Diagnose your environment

```bash
manus-use doctor
manus-use doctor --config ./custom.toml
```

Checks Python packages, config file validity, and whether API keys are accessible.

### `manus-use analyze <CVE-ID>` — Vulnerability intelligence

```bash
# Deep CVE analysis (NVD · CISA KEV · OTX · PoC search · CWE · threat feeds)
manus-use analyze CVE-2025-6554

# With Docker-based exploit verification
manus-use analyze CVE-2024-3094 --verify

# Machine-readable output
manus-use analyze CVE-2025-6554 --output json

# Generate a Lark document report
manus-use analyze CVE-2025-6554 --output lark
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--verify` | off | Run exploit in a Docker sandbox to confirm exploitability |
| `--output {text,json,lark}` | `text` | Report format |
| `--config FILE` | — | Override config |

### `manus-use remediate <CVE-ID>` — Remediation guidance

```bash
# Generate actionable remediation steps for a CVE
manus-use remediate CVE-2024-3094

# Machine-readable output
manus-use remediate CVE-2024-3094 --output json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Report format |
| `--config FILE` | — | Override config |

### `manus-use epss-trend <CVE-ID>` — EPSS score history

```bash
# Show 30-day EPSS score history and detect exploitation spikes
manus-use epss-trend CVE-2024-3094

# Show 90 days of history
manus-use epss-trend CVE-2024-3094 --days 90

# Machine-readable output
manus-use epss-trend CVE-2024-3094 --output json | jq .analysis.spike_detected
```

Fetches daily EPSS (Exploit Prediction Scoring System) scores from the
[FIRST.org API](https://www.first.org/epss/) and detects significant jumps.
A spike of ≥ 0.10 in a 7-day window indicates the vulnerability has recently
been weaponised or picked up by active threat actors.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--days N` | `30` | Days of EPSS history to retrieve (max 365) |
| `--output {text,json}` | `text` | Output format; `json` includes the full time-series |

### `manus-use patch-diff <CVE-ID>` — Patch diff summariser

```bash
# Fetch the fixing commit(s) for a CVE and summarise what changed
manus-use patch-diff CVE-2024-3094

# Machine-readable output
manus-use patch-diff CVE-2024-3094 --output json | jq .commit_summaries
```

Finds the GitHub fixing commit(s) for a CVE (via the
[GitHub Security Advisory database](https://github.com/advisories) and NVD
reference links), fetches the raw unified diff, and produces a structured
summary:
- **Files and functions changed** — where in the codebase the fix landed
- **Bug class** — detected from diff keywords (e.g. `auth_bypass`, `sql_injection`,
  `buffer_overflow`, `use_after_free`, `input_validation`, …)
- **Reproduction condition hints** — added guard/validation lines that reveal the
  minimal condition required to trigger the vulnerability
- **Commit URL** — direct link to the fixing commit on GitHub

Useful for understanding *how* a vulnerability was introduced and fixed without
having to read the raw diff yourself. Composable with `analyze` and `epss-trend`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format; `json` includes the full commit summary |

### `manus-use variants <CVE-ID>` — Variant analysis

```bash
# Find similar bugs in related codebases
manus-use variants CVE-2024-3094

# Machine-readable output
manus-use variants CVE-2024-3094 --output json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Report format |

### `manus-use compare <CVE-A> <CVE-B>` — Side-by-side CVE comparison

```bash
# Compare two CVEs and get a prioritisation recommendation
manus-use compare CVE-2024-3094 CVE-2021-44228

# Machine-readable output
manus-use compare CVE-2024-3094 CVE-2021-44228 --output json | jq .higher_priority
```

Fetches NVD, EPSS, and CISA KEV data for both CVEs in parallel and produces a
structured side-by-side comparison across:

- **CVSS score and severity** (v3.1 preferred, falls back to v3.0 then v2)
- **EPSS exploitation probability** (current score and percentile)
- **CISA KEV membership** (confirmed active exploitation)
- **CWE weakness class**
- **Attack vector, privileges required, user interaction** (exploitability factors)
- **Affected vendor / product**

Each CVE is assigned a composite priority score using a weighted rubric (KEV
membership: +10, Critical CVSS: +8, high EPSS: +8, network attack vector: +3,
etc.) and the output includes a plain-English recommendation with confidence
level: *strong*, *moderate*, or *weak*.

Useful for triage: quickly answer "should I patch A or B first?" without manually
cross-referencing three data sources.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format; `json` includes the full comparison |

### `manus-use discover` — CVE discovery

```bash
# Discover recent high-EPSS CVEs and submit them for tracking
manus-use discover

# Narrow the date window and raise the EPSS threshold
manus-use discover --since 2025-06-01 --min-epss 0.7

# Preview without submitting
manus-use discover --dry-run

# JSON output
manus-use discover --output json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--since YYYY-MM-DD` | 4 weeks ago | Start date for the discovery window |
| `--min-epss SCORE` | `0.5` | Minimum EPSS score threshold (0.0–1.0) |
| `--output {text,json}` | `text` | Report format |
| `--dry-run` | off | Discover CVEs but do not submit them |
| `--config FILE` | — | Override config |

### `manus-use history` — Browse past runs

```bash
manus-use history                        # last 20 runs
manus-use history --limit 50             # last 50 runs
manus-use history --grep "bitcoin"       # filter by task text
manus-use history --format json | jq .   # all history as JSON
manus-use history --clear                # delete all history
```

History is stored at `~/.manus-use/history.jsonl`.

---

### `manus-use silent-patches <owner/repo>` — Silent patch detector

```bash
# Scan the last 90 days of commits for silent security fixes
manus-use silent-patches django/django

# Narrow the date window
manus-use silent-patches curl/curl --since 2024-01-01 --until 2024-06-01

# Machine-readable output for scripting / SIEM ingestion
manus-use silent-patches owner/repo --output json | jq '.candidates[] | select(.score >= 30)'

# Fast mode: message keywords only (no diff fetch — much faster)
manus-use silent-patches torvalds/linux --fast --max-commits 500
```

Scans a public GitHub repository's commit history for potential silent security
fixes — commits that look like security patches (based on keywords in the commit
message and/or diff) but have **no associated CVE or GHSA reference**.

Silent patches are a major blind spot in CVE-based vulnerability management.
Vendors sometimes quietly fix security bugs without filing a CVE — either
deliberately (to limit exposure) or because the fix predates the CVE assignment.

**How it works:**

1. Fetches the commit list from the GitHub REST API (paginates up to `--max-commits`).
2. Skips any commit whose message already contains `CVE-XXXX-YYYY` or `GHSA-…` (overt disclosures).
3. Scores remaining commits on security keyword frequency in the commit message
   (high-signal: `vulnerability`, `overflow`, `injection`, `auth bypass`, `RCE`;
   lower-signal: `fix`, `sanitize`, `escape`, `validate`).
4. For commits that pass the message threshold, fetches the unified diff and scores
   it on security-sensitive code patterns (`html.escape`, `check_permission`,
   `is_authenticated`, `parameterized`, `shell=False`, …).
5. Infers the most likely bug class from the diff (`xss`, `sql_injection`,
   `auth_bypass`, `buffer_overflow`, `deserialization`, …).
6. Returns candidates sorted by suspicion score (0–100), descending.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--since YYYY-MM-DD` | 90 days ago | Start date for the scan window |
| `--until YYYY-MM-DD` | today | End date for the scan window |
| `--max-commits N` | `200` | Maximum commits to inspect (max 500) |
| `--fast` | off | Skip diff scan; rely only on commit-message keywords |
| `--output {text,json}` | `text` | Output format; `json` includes the full candidate list |

---

## Configuration

Create `~/.manus-use/config.toml` (or run `manus-use init`):

```toml
[llm]
provider = "bedrock"          # "openai" | "anthropic" | "bedrock" | "ollama"
model = "us.anthropic.claude-sonnet-4-20250514-v1:0"

[sandbox]
enabled = true
docker_image = "python:3.12-slim"

[tools]
enabled = ["file_ops", "code_execute", "web_search"]

[agent]
# "none" | "sliding_window" | "agentic" (model-managed, recommended for long tasks)
context_manager = "agentic"
```

See [config/config.example.toml](config/config.example.toml) for all available options.

### Provider-specific examples

**AWS Bedrock:**
```toml
[llm]
provider = "bedrock"
model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
# Uses ~/.aws/credentials or IAM role automatically
```

**OpenAI:**
```toml
[llm]
provider = "openai"
model = "gpt-4o"
api_key = "sk-..."    # or set OPENAI_API_KEY env var
```

**Anthropic:**
```toml
[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
api_key = "sk-ant-..."    # or set ANTHROPIC_API_KEY env var
```

**Ollama (local):**
```toml
[llm]
provider = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434"
```

---

## Python API

### Basic usage

```python
from manus_use import ManusAgent

agent = ManusAgent()

result = agent("Write a Python script that fetches weather data and saves it to CSV")
print(result)
```

### Browser automation

```python
from manus_use.agents import BrowserUseAgent

agent = BrowserUseAgent()
result = agent("Go to GitHub and find the top 5 trending Python repositories today")
print(result)
```

### Data analysis

```python
from manus_use.agents import DataAnalysisAgent

agent = DataAnalysisAgent()
result = agent("Load sales.csv, compute monthly revenue, and plot a bar chart")
print(result)
```

### Multi-agent orchestration

```python
from manus_use.multi_agents import WorkflowAgent

workflow = WorkflowAgent()
result = workflow.handle_request("""
    1. Search the web for recent AI research papers
    2. Analyse the trends and create visualizations
    3. Generate a comprehensive report with insights
""")
print(result)
```

### Vulnerability intelligence

```python
from manus_use.agents import VulnerabilityIntelligenceAgent
from manus_use.config import Config

config = Config.from_file()
agent = VulnerabilityIntelligenceAgent(config=config)

result = agent.handle_request("Analyse CVE-2025-6554 and create a comprehensive report")
print(result)
```

### Remediation guidance

```python
from manus_use.agents import RemediationAgent
from manus_use.config import Config

config = Config.from_file()
agent = RemediationAgent(config=config)

result = agent.handle_request("Generate remediation steps for CVE-2024-3094")
print(result)
```

### Variant analysis

```python
from manus_use.agents import VariantAnalysisAgent
from manus_use.config import Config

config = Config.from_file()
agent = VariantAnalysisAgent(config=config)

result = agent.handle_request("Find variants of CVE-2024-3094 in related codebases")
print(result)
```

### Custom configuration

```python
from manus_use import ManusAgent
from manus_use.config import Config

config = Config.from_file("path/to/config.toml")
agent = ManusAgent(config=config)

result = agent("Create a factorial function")
print(result)
```

---

## Security & Vulnerability Intelligence

ManusUse includes a multi-source vulnerability intelligence pipeline accessible via the CLI or Python API.

### How it works

The `manus-use analyze` command runs an 8-step pipeline:

1. **NVD + GitHub Advisory** — official CVE metadata, CVSS, CWE
2. **CISA KEV** — known-exploited-vulnerabilities catalogue
3. **AlienVault OTX** — threat intelligence pulses and IoCs
4. **PoC discovery** — PoC Week trending digest, Trickest/CVE index, Exploit-DB, PacketStorm, GitHub
5. **URL verification** — every candidate URL is fetched and validated
6. **Static analysis** — code-level analysis of confirmed PoCs (network calls, payload patterns)
7. **CWE correlation** — weakness classification and remediation hints
8. **Report generation** — structured text, JSON, or Lark document output

### Examples

```bash
# Analyse a CVE and print a structured text report
manus-use analyze CVE-2024-3094

# Get JSON output and extract the CVSS score
manus-use analyze CVE-2024-3094 --output json | jq .cvss_score

# Verify exploitability in a Docker sandbox, then write a Lark report
manus-use analyze CVE-2025-6554 --verify --output lark

# Check 30-day EPSS trend and detect exploitation spikes
manus-use epss-trend CVE-2024-3094

# Check 90-day EPSS history in JSON
manus-use epss-trend CVE-2025-6554 --days 90 --output json

# Summarise the fixing commit (files changed, bug class, reproduction hints)
manus-use patch-diff CVE-2024-3094
manus-use patch-diff CVE-2024-3094 --output json | jq .commit_summaries

# Generate remediation steps
manus-use remediate CVE-2024-3094

# Compare two CVEs side-by-side for triage prioritisation
manus-use compare CVE-2024-3094 CVE-2021-44228
manus-use compare CVE-2024-3094 CVE-2021-44228 --output json | jq .higher_priority

# Discover new high-EPSS CVEs from the last 2 weeks
manus-use discover --since 2025-06-12 --min-epss 0.6

# Scan a repo for silent security fixes (no CVE reference)
manus-use silent-patches django/django
manus-use silent-patches curl/curl --since 2024-01-01 --output json | jq '.candidates[] | select(.score >= 25)'
```

> **Important:** These tools are designed for defensive security purposes only. Use them for legitimate security research, vulnerability management, and defence.

---

## Key Features

### 🤖 Agent types

| Agent | Class | Best for |
|-------|-------|----------|
| General | `ManusAgent` | File ops, code execution, reasoning |
| Browser automation | `BrowserUseAgent` | JS-heavy sites, form filling, scraping |
| Lightweight browser | `BrowserAgent` | Static pages, simple navigation |
| Data analysis | `DataAnalysisAgent` | CSV/JSON processing, charts |
| MCP | `MCPAgent` | Model Context Protocol tool servers |
| Multi-agent | `WorkflowAgent` | Complex tasks needing multiple specialists |
| Vulnerability intel | `VulnerabilityIntelligenceAgent` | CVE analysis, threat intelligence |
| Remediation | `RemediationAgent` | Actionable fix guidance for CVEs |
| Variant analysis | `VariantAnalysisAgent` | Finding similar bugs in related codebases |
| CVE discovery | `VulnerabilityDiscoveryAgent` | High-EPSS CVE triage and tracking |

### 🛠️ Built-in tools

- File operations (read, write, edit, delete)
- Code execution in Docker sandboxes
- Web search (DuckDuckGo, configurable)
- Browser automation (click, type, extract, screenshot)
- Data visualization (charts, plots, reports)
- Security tools (NVD, CISA KEV, OTX, Exploit-DB, Trickest, PoC Week)
- HTTP requests with content extraction
- Python REPL with persistent state

### 🔌 LLM providers

- **AWS Bedrock** (Claude, Titan, …)
- **OpenAI** (GPT-4o, GPT-4-turbo, …)
- **Anthropic** (Claude 3.5 Sonnet, Opus, …)
- **Ollama** (Llama, Mistral, … running locally)

---

## Development

### Set up a development environment

```bash
git clone https://github.com/manus-use/manus-use.git
cd manus-use
pip install -e ".[dev,browser,search,visualization]"
```

### Run tests

```bash
# All non-integration tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=manus_use --cov-report=html

# Or via hatch
hatch run test
hatch run test-cov
```

### Lint and format

```bash
ruff check src/ tests/
ruff format src/ tests/

# Or via hatch
hatch run lint
hatch run format
```

### Project layout

```
manus-use/
├── src/manus_use/
│   ├── agents/          # Agent implementations
│   │   ├── base.py      # BaseManusAgent (all agents inherit from this)
│   │   ├── manus.py     # ManusAgent (general purpose)
│   │   ├── browser.py   # BrowserAgent (lightweight)
│   │   ├── browser_use_agent.py  # BrowserUseAgent (full JS automation)
│   │   ├── data_analysis.py      # DataAnalysisAgent
│   │   ├── mcp.py       # MCPAgent
│   │   ├── vi_agent.py  # VulnerabilityIntelligenceAgent
│   │   ├── remediation_agent.py  # RemediationAgent
│   │   ├── variant_agent.py      # VariantAnalysisAgent
│   │   └── vulnerability_discovery_agent.py  # VulnerabilityDiscoveryAgent
│   ├── multi_agents/    # Multi-agent orchestration
│   │   └── workflow_agent.py
│   ├── tools/           # Individual tool implementations
│   ├── cli.py           # manus-use CLI entry point
│   ├── config.py        # Config model (TOML-backed)
│   └── __init__.py
├── tests/               # pytest test suite (375+ tests)
├── config/
│   └── config.example.toml
├── examples/            # Runnable usage examples
└── pyproject.toml
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Examples

Explore the [examples/](examples/) directory for runnable scripts:

- [`basic_usage.py`](examples/basic_usage.py) — simple ManusAgent task
- [`browser_use_demo.py`](examples/browser_use_demo.py) — browser automation
- [`multi_agent_flow.py`](examples/multi_agent_flow.py) — multi-agent orchestration

---

## Support

- 📖 [Documentation](https://github.com/manus-use/manus-use/wiki)
- 🐛 [Issue Tracker](https://github.com/manus-use/manus-use/issues)
- 💬 [Discussions](https://github.com/manus-use/manus-use/discussions)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Strands SDK](https://github.com/strands-agents/sdk-python) — a powerful Python SDK for building AI agents
- Browser automation powered by [browser-use](https://github.com/browser-use/browser-use) — framework for AI-driven web automation
- Inspired by Anthropic's computer use demonstrations
