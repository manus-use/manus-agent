# manus-agent

[![PyPI version](https://img.shields.io/pypi/v/manus-use.svg)](https://pypi.org/project/manus-use/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/manus-use/manus-agent/actions/workflows/test.yml/badge.svg)](https://github.com/manus-use/manus-agent/actions)

A powerful, extensible framework for building AI agents with comprehensive tool support, multi-agent orchestration, and advanced vulnerability intelligence.

Built on [Strands SDK](https://github.com/strands-agents/sdk-python) and integrated with [browser-use](https://github.com/browser-use/browser-use).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Run a task](#manus-agent-task--run-a-task)
  - [init / doctor / history](#manus-agent-init--configure-credentials)
  - [analyze](#manus-agent-analyze-cve-id--vulnerability-intelligence)
  - [remediate](#manus-agent-remediate-cve-id--remediation-guidance)
  - [discover](#manus-agent-discover--cve-discovery)
  - [epss-trend](#manus-agent-epss-trend-cve-id--epss-score-history)
  - [patch-diff](#manus-agent-patch-diff-cve-id--patch-diff-summariser)
  - [variants](#manus-agent-variants-cve-id--variant-analysis)
  - [compare](#manus-agent-compare-cve-a-cve-b--side-by-side-comparison)
  - [exploit-complexity](#manus-agent-exploit-complexity-cve-id--exploit-complexity-scorer)
  - [poc-search](#manus-agent-poc-search-cve-id--multi-source-poc-aggregator)
  - [blast-radius](#manus-agent-blast-radius-spec--dependency-blast-radius)
  - [vendor-response](#manus-agent-vendor-response-cve-id--vendor-response-tracker)
  - [silent-patches](#manus-agent-silent-patches-ownerrepo--silent-patch-detector)
  - [cve-timeline](#manus-agent-cve-timeline-cve-id--cve-timeline)
  - [version-range](#manus-agent-version-range-cve-id--affected-version-ranges)
  - [poc-freshness](#manus-agent-poc-freshness-cve-id--poc-freshness-checker)
  - [sbom-scan](#manus-agent-sbom-scan-bomfile--sbom-scanner)
  - [temporal-priority](#manus-agent-temporal-priority-cve-id--temporal-priority-scorer)
  - [cluster-variants](#manus-agent-cluster-variants-cve-id--cve-variant-clustering)
  - [changelog](#manus-agent-changelog--manage-project-changelog)
- [Vulnerability Analysis Workflow](#vulnerability-analysis-workflow)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Security & Vulnerability Intelligence](#security--vulnerability-intelligence)
- [Development](#development)

---

## Installation

```bash
# Basic installation
pip install manus-agent

# With browser automation support
pip install manus-agent[browser]
playwright install chromium

# Full installation with all optional features
pip install manus-agent[browser,search,visualization]
```

---

## Quick Start

```bash
# 1. Set up credentials
manus-agent init

# 2. Verify your environment
manus-agent doctor

# 3. Run a task
manus-agent "Write a Python script that fetches the current Bitcoin price"

# Or start the interactive REPL
manus-agent
```

---

## CLI Reference

### `manus-agent [task]` — Run a task

```bash
# Single-shot (prints result, then exits)
manus-agent "Create a factorial function in Python"

# Use a specific agent type
manus-agent --agent browser "Find the top 5 trending GitHub repos today"

# Force multi-agent orchestration
manus-agent --mode multi "Research quantum computing and create a presentation"

# JSON output for piping
manus-agent --format json "List the first 10 prime numbers" | jq .result

# Stream output tokens in real time
manus-agent --stream "Write a short story about a robot"

# Interactive REPL
manus-agent
manus-agent --mode multi
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {auto,single,multi}` | `auto` | Execution mode; `auto` detects task complexity |
| `--agent {manus,browser,data,mcp}` | `manus` | Agent type for single-agent mode |
| `--show-plan` | off | Print the multi-agent plan before running |
| `--output FILE` | — | Write result to FILE (single-shot only) |
| `--format {text,json}` | `text` | Output format |
| `--stream` | off | Stream output tokens in real time |
| `--no-history` | off | Skip recording this run in the history log |
| `--config FILE` | — | Override config file path |
| `--version` | — | Print version and exit |

---

### `manus-agent init` — Configure credentials

```bash
manus-agent init                        # write to ~/.manus-agent/config.toml
manus-agent init --output ./my.toml    # write to a custom path
manus-agent init --force                # overwrite without prompting
```

### `manus-agent doctor` — Diagnose your environment

```bash
manus-agent doctor
manus-agent doctor --config ./custom.toml
```

Checks Python packages, config file validity, and API key accessibility.

### `manus-agent history` — Browse past runs

```bash
manus-agent history                        # last 20 runs
manus-agent history --limit 50            # last 50 runs
manus-agent history --grep "bitcoin"      # filter by task text
manus-agent history --format json | jq .  # all history as JSON
manus-agent history --clear               # delete all history
```

History is stored at `~/.manus-agent/history.jsonl`.

---

### `manus-agent analyze <CVE-ID>` — Vulnerability intelligence

```bash
manus-agent analyze CVE-2025-6554
manus-agent analyze CVE-2024-3094 --verify
manus-agent analyze CVE-2025-6554 --output json
manus-agent analyze CVE-2025-6554 --output lark
```

Runs an 8-step intelligence pipeline — see [Security & Vulnerability Intelligence](#security--vulnerability-intelligence) for full details.

| Flag | Default | Description |
|------|---------|-------------|
| `--verify` | off | Run exploit in a Docker sandbox to confirm exploitability |
| `--output {text,json,lark}` | `text` | Report format |
| `--config FILE` | — | Override config |

---

### `manus-agent remediate <CVE-ID>` — Remediation guidance

```bash
manus-agent remediate CVE-2024-3094
manus-agent remediate CVE-2024-3094 --output json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Report format |
| `--config FILE` | — | Override config |

---

### `manus-agent discover` — CVE discovery

```bash
manus-agent discover
manus-agent discover --since 2025-06-01 --min-epss 0.7
manus-agent discover --dry-run
manus-agent discover --output json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--since YYYY-MM-DD` | 4 weeks ago | Start date for the discovery window |
| `--min-epss SCORE` | `0.5` | Minimum EPSS score (0.0–1.0) |
| `--output {text,json}` | `text` | Report format |
| `--dry-run` | off | Discover CVEs but do not submit them |
| `--config FILE` | — | Override config |

---

### `manus-agent epss-trend <CVE-ID>` — EPSS score history

```bash
manus-agent epss-trend CVE-2024-3094
manus-agent epss-trend CVE-2024-3094 --days 90
manus-agent epss-trend CVE-2024-3094 --output json | jq .analysis.spike_detected
```

Fetches daily EPSS scores from the [FIRST.org API](https://www.first.org/epss/) and detects exploitation spikes (≥ 0.10 jump in a 7-day window).

| Flag | Default | Description |
|------|---------|-------------|
| `--days N` | `30` | Days of history (max 365) |
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent patch-diff <CVE-ID>` — Patch diff summariser

```bash
manus-agent patch-diff CVE-2024-3094
manus-agent patch-diff CVE-2024-3094 --output json | jq .commit_summaries
```

Finds the fixing commit(s) via GHSA + NVD, fetches the raw unified diff, and produces a structured summary: files/functions changed, bug class (14 categories), reproduction condition hints, and commit URL.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent variants <CVE-ID>` — Variant analysis

```bash
manus-agent variants CVE-2024-3094
manus-agent variants CVE-2024-3094 --output json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Report format |

---

### `manus-agent compare <CVE-A> <CVE-B>` — Side-by-side comparison

```bash
manus-agent compare CVE-2024-3094 CVE-2021-44228
manus-agent compare CVE-2024-3094 CVE-2021-44228 --output json | jq .higher_priority
```

Fetches NVD, EPSS, and CISA KEV data for both CVEs in parallel and produces a side-by-side comparison across CVSS, EPSS, KEV membership, CWE, and exploitability factors. Outputs a prioritisation recommendation with confidence level (*strong / moderate / weak*).

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent exploit-complexity <CVE-ID>` — Exploit complexity scorer

```bash
manus-agent exploit-complexity CVE-2024-3094
manus-agent exploit-complexity CVE-2024-3094 --output json | jq .attacker_friendly
```

Scores practical attacker effort on a 1–5 scale across five dimensions:

| Dimension | What it measures |
|-----------|------------------|
| Lines of code | How much exploit code must be written/adapted |
| Authentication | Credentials or privilege level required |
| Network hops | How many services the exploit must reach |
| OS/platform deps | Platform-specific syscalls, structs, gadgets |
| Exploit chain length | Number of discrete attack stages |

Outputs a `complexity_score` (1–5), a label (*trivial / low / moderate / high / very_high*), and an `attacker_friendly` boolean (true when score ≤ 2.5).

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent poc-search <CVE-ID>` — Multi-source PoC aggregator

```bash
manus-agent poc-search CVE-2024-3094
manus-agent poc-search CVE-2024-3094 --sources trickest,exploitdb,github
manus-agent poc-search CVE-2024-3094 --output json | jq .exploited_in_wild
```

Queries five PoC sources **in parallel**, deduplicates results, and sorts by exploited-in-wild status and publication date:

| Source | What it provides |
|--------|-----------------|
| `trickest` | [trickest/cve](https://github.com/trickest/cve) — 250k+ CVE PoC index |
| `vulncheck_kev` | [VulnCheck KEV](https://vulncheck.com) — exploited-in-wild signal from 100+ intel sources (requires `VULNCHECK_API_KEY`) |
| `exploitdb` | [Exploit-DB](https://www.exploit-db.com) CSV — cached 24 h |
| `github` | GitHub repo search for repositories mentioning the CVE |
| `nvd` | NVD references filtered for GitHub / Exploit-DB / PacketStorm URLs |

| Flag | Default | Description |
|------|---------|-------------|
| `--sources LIST` | all | Comma-separated subset: `trickest,vulncheck_kev,exploitdb,github,nvd` |
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent blast-radius <SPEC>` — Dependency blast radius

```bash
manus-agent blast-radius requests@2.28.0
manus-agent blast-radius pypi:urllib3@1.26.5
manus-agent blast-radius npm:lodash@4.17.20
manus-agent blast-radius CVE-2021-44228
manus-agent blast-radius CVE-2021-44228 --output json | jq .summary
```

Estimates how broadly a vulnerability propagates downstream. Resolves affected packages from NVD CPE + OSV.dev + GHSA, then enriches each with real download/dependent stats:

| Metric | Source |
|--------|--------|
| npm dependent packages | npm registry search API |
| npm weekly/monthly downloads | npm downloads API |
| PyPI downloads | PyPI JSON API + pypistats.org |
| Maven artifact metadata | Maven Central Solr search |

Blast-radius labels per package:

| Label | Threshold |
|-------|-----------|
| **CRITICAL** | ≥ 5 M weekly downloads or ≥ 50 K npm dependents |
| **HIGH** | ≥ 500 K downloads or ≥ 5 K dependents |
| **MEDIUM** | ≥ 50 K downloads or ≥ 500 dependents |
| **LOW** | any measurable signal |
| **UNKNOWN** | no data available |

| Flag | Default | Description |
|------|---------|-------------|
| `--max-packages N` | `10` | Max affected packages to enrich |
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent silent-patches <owner/repo>` — Silent patch detector

```bash
manus-agent silent-patches torvalds/linux
manus-agent silent-patches torvalds/linux --since 2025-01-01
manus-agent silent-patches torvalds/linux --output json | jq .[].classification
```

Scans a repository's commit history for security fixes that were never assigned a CVE. Uses two-stage heuristic scoring: commit message keywords then diff keywords. Each candidate commit is labelled with one of 14 bug classes (e.g. `auth_bypass`, `buffer_overflow`, `use_after_free`).

| Flag | Default | Description |
|------|---------|-------------|
| `--since YYYY-MM-DD` | 90 days ago | Start date for commit scan |
| `--until YYYY-MM-DD` | today | End date |
| `--max-commits N` | `500` | Hard limit on commits fetched |
| `--fast` | off | Skip diff scoring (message keywords only) |
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent cve-timeline <CVE-ID>` — CVE timeline

```bash
manus-agent cve-timeline CVE-2021-44228
manus-agent cve-timeline CVE-2021-44228 --output json
```

Reconstructs the full event timeline for a CVE: NVD publish date → EPSS history → CISA KEV add date → patch release date. Useful for understanding how quickly a vulnerability was weaponised and fixed.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent version-range <CVE-ID>` — Affected version ranges

```bash
manus-agent version-range CVE-2021-44228
manus-agent version-range CVE-2021-44228 --ecosystem pypi
manus-agent version-range CVE-2021-44228 --output json | jq .first_patched_version
```

Walks NVD CPE configurations and cross-references PyPI / npm / Maven to produce structured vulnerable semver ranges, a list of affected releases, and the first patched release.

| Flag | Default | Description |
|------|---------|-------------|
| `--ecosystem {auto,pypi,npm,maven}` | `auto` | Force a specific ecosystem |
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent vendor-response <CVE-ID>` — Vendor response tracker

```bash
manus-agent vendor-response CVE-2024-3094
manus-agent vendor-response CVE-2024-3094 --output json | jq .vendor_response_state
```

Queries NVD reference tags, CISA KEV required-action field, and VulnCheck KEV (when `VULNCHECK_API_KEY` is set) to classify the vendor's patch/response status into one of six states:

| State | Meaning |
|-------|---------|
| `patch_available` | A confirmed fix/patch has been released |
| `patch_pending` | Vendor acknowledged; fix in progress or announced |
| `workaround_only` | Vendor published a mitigation but no patch yet |
| `investigating` | Vendor acknowledged but response status is unclear |
| `no_patch_expected` | Vendor will not fix (EoL, won't-fix, or disputed) |
| `unknown` | Insufficient data to classify |

Confidence is rated `high (≥0.9) / medium (≥0.6) / low (<0.6)`. A VulnCheck KEV hit upgrades confidence and can push state to `patch_available` or `investigating`.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent poc-freshness <CVE-ID>` — PoC freshness checker

```bash
manus-agent poc-freshness CVE-2024-3094
manus-agent poc-freshness CVE-2024-3094 --output json | jq .freshness_score
```

Measures how recently PoC activity occurred: last commit recency in known PoC repos, recently-starred repositories, new Exploit-DB entries. A high freshness score means attacker interest is ongoing.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent sbom-scan <bom-file>` — SBOM scanner

```bash
manus-agent sbom-scan bom.json
manus-agent sbom-scan sbom.spdx.json --output json | jq .critical_count
```

Parses CycloneDX or SPDX SBOMs, queries OSV.dev in batch for all components, enriches each finding with EPSS and CISA KEV status, and ranks results by KEV membership then EPSS score.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent temporal-priority <CVE-ID>` — Temporal priority scorer

```bash
manus-agent temporal-priority CVE-2024-3094
manus-agent temporal-priority CVE-2024-3094 --output json | jq .score
```

Produces a 0–100 urgency score combining CVSS base score, current EPSS, EPSS spike recency, CISA KEV membership, patch availability, and CVE age. Designed to answer: *"given everything I know today, how urgent is this?"*

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent cluster-variants <CVE-ID>` — CVE variant clustering

```bash
manus-agent cluster-variants CVE-2021-44228
manus-agent cluster-variants CVE-2021-44228 --output json | jq .clusters
```

Groups CVEs related to the input across three cluster dimensions: same component/vendor, same CWE weakness class, and same researcher/disclosure domain. Useful for finding the full attack surface when one CVE is confirmed exploited.

| Flag | Default | Description |
|------|---------|-------------|
| `--output {text,json}` | `text` | Output format |

---

### `manus-agent changelog` — Manage project changelog

```bash
manus-agent changelog                       # show full CHANGELOG.md
manus-agent changelog --version 0.1.0      # show section for a specific version
manus-agent changelog --generate           # preview next release notes from commits
manus-agent changelog --generate --output json
```

Parses [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.) and groups them into `Added`, `Fixed`, and `Changed` sections.

| Flag | Default | Description |
|------|---------|-------------|
| `--version X.Y.Z` | — | Filter output to a specific release section |
| `--generate` | off | Preview next release notes from commits since last tag |
| `--output {text,json}` | `text` | Output format |

---

## Vulnerability Analysis Workflow

The CLI tools form a layered analysis pipeline. Each tool answers one question;
together they build a complete risk picture for any CVE.

```
Discover & Triage
  └─ manus-agent discover              # find high-EPSS CVEs in your window
  └─ manus-agent analyze <CVE>         # 8-step full intelligence report

Exploitability
  └─ manus-agent epss-trend <CVE>      # EPSS history + spike detection
  └─ manus-agent exploit-complexity <CVE>  # attacker effort score (1–5)
  └─ manus-agent poc-search <CVE>      # multi-source PoC aggregation

Reach
  └─ manus-agent blast-radius <CVE>    # downstream package/download stats
  └─ manus-agent version-range <CVE>   # affected semver ranges
  └─ manus-agent cluster-variants <CVE> # related CVEs by component/CWE/researcher

Remediation
  └─ manus-agent patch-diff <CVE>      # what changed in the fix commit
  └─ manus-agent vendor-response <CVE> # patch_available / investigating / unknown
  └─ manus-agent remediate <CVE>       # actionable fix guidance

Situation awareness
  └─ manus-agent compare <CVE-A> <CVE-B>  # side-by-side prioritisation
  └─ manus-agent cve-timeline <CVE>    # disclosure → exploit → patch timeline
  └─ manus-agent sbom-scan bom.json    # scan your dependency tree
```

### Worked example — CVE-2021-44228 (Log4Shell)

This example walks through a complete analyst workflow using Log4Shell as the
canonical fixture. Every command below exits 0 on a patched environment.

**Step 1 — How exploitable is it?**

```bash
# EPSS is near 1.0 and has been since Dec 2021 — highest urgency
$ manus-agent epss-trend CVE-2021-44228 --days 30
EPSS Trend: CVE-2021-44228
  Latest score : 0.9762 (97.6th percentile)
  7-day spike  : ⚡ YES — jump of +0.02 detected

# Trivial to exploit: AC=LOW, no auth, no user interaction
$ manus-agent exploit-complexity CVE-2021-44228
Exploit Complexity: CVE-2021-44228
  Score: 1.4 / 5 — trivial
  Attacker-friendly: yes
```

**Step 2 — How broadly does it spread?**

```bash
$ manus-agent blast-radius CVE-2021-44228
Dependency Blast Radius — CVE-2021-44228
  [1] log4j-core  (Maven)
      Blast radius:     CRITICAL
      Weekly downloads: 18,432,107
      Vulnerable range: >=2.0.0, <2.16.0

# Confirm patch is available for the specific version in use
$ manus-agent vendor-response CVE-2021-44228
Vendor Response: CVE-2021-44228
  Status     : ✅ PATCH AVAILABLE
  Confidence : high (0.95)
  Signals:
    NVD references  : 42
    CISA KEV        : yes
    VulnCheck KEV   : yes
```

**Step 3 — What exactly changed in the fix?**

```bash
$ manus-agent patch-diff CVE-2021-44228
Patch Diff: CVE-2021-44228
  Commit  : rel/2.17.1 (apache/logging-log4j2)
  Bug class: remote_code_execution
  Files changed: JndiLookup.java, Log4j2.xml (3 files total)
  Key change: JNDI lookup disabled by default; allowlist added for permitted protocols
```

**Step 4 — Are there related CVEs you should track?**

```bash
# Log4Shell spawned a cluster: CVE-2021-45046, CVE-2021-45105, CVE-2021-44832
$ manus-agent cluster-variants CVE-2021-44228
Variant Clusters: CVE-2021-44228
  Same component: CVE-2021-45046, CVE-2021-45105, CVE-2021-44832  (log4j2)
  Same CWE-917: 8 CVEs with Improper Neutralization of JNDI
```

**Step 5 — Compare with another CVE to prioritise work**

```bash
$ manus-agent compare CVE-2021-44228 CVE-2024-3094
Comparison: CVE-2021-44228 vs CVE-2024-3094
  CVSS    : 10.0 vs 10.0        (tie)
  EPSS    : 0.976 vs 0.781      → CVE-2021-44228 higher
  KEV     : yes vs yes          (both actively exploited)
  Higher priority: CVE-2021-44228 (strong confidence)
  Reason: significantly higher EPSS; Log4Shell has broader blast radius
```

**JSON pipeline — feed tools into each other**

```bash
# Extract the highest blast-radius label, then only act if CRITICAL
BLAST=$(manus-agent blast-radius CVE-2021-44228 --output json | jq -r '.summary.highest_blast_radius')
if [ "$BLAST" = "CRITICAL" ]; then
  manus-agent remediate CVE-2021-44228 --output json > remediation.json
fi

# Get vendor_response_state for a CI gate
STATE=$(manus-agent vendor-response CVE-2021-44228 --output json | jq -r '.vendor_response_state')
if [ "$STATE" != "patch_available" ]; then
  echo "WARNING: no confirmed patch for CVE-2021-44228" && exit 1
fi

# Score exploit complexity and fail CI if attacker_friendly is true
manus-agent exploit-complexity CVE-2021-44228 --output json \
  | jq -e '.attacker_friendly == false'
```

---

## Configuration

Create `~/.manus-agent/config.toml` (or run `manus-agent init`):

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

See [config/config.example.toml](config/config.example.toml) for all options.

### Provider examples

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
api_key = "***"    # or OPENAI_API_KEY env var
```

**Anthropic:**
```toml
[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
api_key = "***"    # or ANTHROPIC_API_KEY env var
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

```python
# General-purpose agent
from manus_agent import ManusAgent
agent = ManusAgent()
result = agent("Write a Python script that fetches weather data and saves it to CSV")

# Browser automation
from manus_agent.agents import BrowserUseAgent
agent = BrowserUseAgent()
result = agent("Go to GitHub and find the top 5 trending Python repositories today")

# Data analysis
from manus_agent.agents import DataAnalysisAgent
agent = DataAnalysisAgent()
result = agent("Load sales.csv, compute monthly revenue, and plot a bar chart")

# Multi-agent orchestration
from manus_agent.multi_agents import WorkflowAgent
workflow = WorkflowAgent()
result = workflow.handle_request("""
    1. Search the web for recent AI research papers
    2. Analyse the trends and create visualisations
    3. Generate a comprehensive report
""")

# Vulnerability intelligence
from manus_agent.agents import VulnerabilityIntelligenceAgent
from manus_agent.config import Config
agent = VulnerabilityIntelligenceAgent(config=Config.from_file())
result = agent.handle_request("Analyse CVE-2025-6554")
```

---

## Security & Vulnerability Intelligence

### Pipeline

The `manus-agent analyze` command runs an 8-step pipeline:

1. **NVD + GHSA** — official CVE metadata, CVSS, CWE
2. **CISA KEV** — known-exploited-vulnerabilities catalogue; **VulnCheck KEV** when `VULNCHECK_API_KEY` is set (100+ intel sources, ransomware flag)
3. **AlienVault OTX** — threat intelligence pulses and IoCs
4. **PoC discovery** — Trickest/CVE index, Exploit-DB, PacketStorm, GitHub, VulnCheck KEV (with API key)
5. **URL verification** — every candidate URL is fetched and validated
6. **Deep analysis** — patch diff summary · exploit complexity score · version range resolution · vendor response status · PoC freshness
7. **CWE correlation** — weakness classification and remediation hints
8. **Report generation** — structured text, JSON, or Lark document

### Quick reference

```bash
# Full intelligence report
manus-agent analyze CVE-2024-3094

# How exploitable is it right now?
manus-agent exploit-complexity CVE-2024-3094
manus-agent epss-trend CVE-2024-3094

# What changed in the fix?
manus-agent patch-diff CVE-2024-3094

# Am I affected?
manus-agent version-range CVE-2024-3094
manus-agent sbom-scan bom.json

# Triage two CVEs
manus-agent compare CVE-2024-3094 CVE-2021-44228
manus-agent temporal-priority CVE-2024-3094

# Find related exposure
manus-agent blast-radius CVE-2021-44228
manus-agent cluster-variants CVE-2021-44228
manus-agent silent-patches apache/log4j
```

### VulnCheck enrichment (optional)

Set `VULNCHECK_API_KEY` to unlock two additional intel sources:

| Index | What it adds |
|-------|--------------|
| `vulncheck-kev` | Exploitation status from 100+ sources (FBI Flash, CERT advisories, …); prints 🚨 **ACTIVELY EXPLOITED** banner when confirmed; ⚠️ **RANSOMWARE ASSOCIATED** when applicable |
| `nist-nvd2` | Enriched CPE matching with additional CVSS metadata and version ranges |

> **Note:** These tools are for defensive security purposes only.

---

## Key Features

### Agent types

| Agent | Class | Best for |
|-------|-------|----------|
| General | `ManusAgent` | File ops, code execution, reasoning |
| Browser (full JS) | `BrowserUseAgent` | JS-heavy sites, form filling, scraping |
| Browser (lightweight) | `BrowserAgent` | Static pages, simple navigation |
| Data analysis | `DataAnalysisAgent` | CSV/JSON processing, charts |
| MCP | `MCPAgent` | Model Context Protocol tool servers |
| Multi-agent | `WorkflowAgent` | Complex tasks needing multiple specialists |
| Vulnerability intel | `VulnerabilityIntelligenceAgent` | CVE analysis, threat intelligence |
| Remediation | `RemediationAgent` | Actionable fix guidance |
| Variant analysis | `VariantAnalysisAgent` | Finding similar bugs in related codebases |
| CVE discovery | `VulnerabilityDiscoveryAgent` | High-EPSS CVE triage and tracking |

### LLM providers

- **AWS Bedrock** — Claude, Titan, …
- **OpenAI** — GPT-4o, GPT-4-turbo, …
- **Anthropic** — Claude 3.5 Sonnet, Opus, …
- **Ollama** — Llama, Mistral, … (local)

---

## Development

```bash
git clone https://github.com/manus-use/manus-agent.git
cd manus-agent
pip install -e ".[dev,browser,search,visualization]"
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ --cov=manus_agent --cov-report=html

# Via hatch
hatch run test
hatch run test-cov
```

900+ tests, all HTTP calls mocked.

### Lint and format

```bash
ruff check src/ tests/
ruff format src/ tests/

# Via hatch
hatch run lint
hatch run format
```

### Project layout

```
manus-agent/
├── src/manus_agent/
│   ├── agents/          # Agent implementations
│   │   ├── base.py
│   │   ├── manus.py
│   │   ├── browser.py / browser_use_agent.py
│   │   ├── data_analysis.py
│   │   ├── mcp.py
│   │   ├── vi_agent.py          # VulnerabilityIntelligenceAgent
│   │   ├── remediation_agent.py
│   │   ├── variant_agent.py
│   │   └── vulnerability_discovery_agent.py
│   ├── multi_agents/
│   │   └── workflow_agent.py
│   ├── tools/           # Strands tool implementations (one file per tool)
│   ├── cli.py           # manus-agent CLI entry point
│   ├── config.py
│   └── __init__.py
├── tests/               # pytest test suite (900+ tests, all mocked)
├── config/
│   └── config.example.toml
├── examples/
├── CHANGELOG.md
└── pyproject.toml
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Support

- 🐛 [Issue Tracker](https://github.com/manus-use/manus-agent/issues)
- 💬 [Discussions](https://github.com/manus-use/manus-agent/discussions)

---

## License

MIT — see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

---

## Acknowledgments

- Built with [Strands SDK](https://github.com/strands-agents/sdk-python)
- Browser automation powered by [browser-use](https://github.com/browser-use/browser-use)
