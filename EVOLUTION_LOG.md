
## PR #111 — OSV per-package CVE lookup in get_dependency_blast_radius

**Branch:** `feat/osv-package-cve-lookup`
**Date:** 2026-07-09
**Status:** Open

### Problem

`get_dependency_blast_radius` showed blast-radius exposure metrics (downloads,
dependents) for direct package queries but never asked: *does this package have
known CVEs?*  The OSV `/v1/query` endpoint existed and was used elsewhere in the
codebase (for CVE→affected-packages lookups) but was never called in the direct
package path.

### Solution

New helper `_fetch_osv_package_vulns(name, ecosystem, version, max_vulns)`:
- POSTs to `https://api.osv.dev/v1/query` with package + optional version
- `_ECOSYSTEM_TO_OSV` dict maps 31 input aliases → canonical OSV ecosystem strings
  (covers PyPI, npm, Maven, Go, crates.io, RubyGems, NuGet, Packagist, Hex, Pub, Conda)
- Parses id, summary (≤120 chars), aliases, fixed version from range events
- Graceful empty-list return on all errors

`get_dependency_blast_radius` direct package path:
- Calls `_fetch_osv_package_vulns` post-enrichment; version-scoped when version given
- New output section: `Known CVEs (OSV) affecting @X.Y.Z: N` with per-advisory bullets
- Summary line: `Known CVEs from OSV affecting version X.Y.Z: N`
- CVE input path unchanged (function never called there)

### Tests

35 new tests across `TestEcosystemToOsvMapping`, `TestFetchOsvPackageVulns`,
`TestBlastRadiusOsvIntegration`. Suite: 1193 passed (was 1158).

### Suggested next contributions

- **Swift Package Index enricher** — `_enrich_swift`: API at
  `swiftpackageindex.com/api/packages/{owner}/{package}`; complication is needing
  owner/repo format; map to OSV ecosystem `SwiftURL`
- **Conda enricher** — `_enrich_conda` is already in PR #110; check if merged
- **Extend `_ECOSYSTEM_TO_OSV`** to cover Conda/Bioconda once PR #110 lands
- **`first_release_date` for crates.io** (extend PR #103 analogue)

---

## Session N+4 — feat/enrich-cran (PR #113)

### Contribution

Added `_enrich_cran()` enricher to `get_dependency_blast_radius.py`, extending
the blast-radius tool with native **CRAN (R)** package support.

### Ecosystem selection rationale

- Checked all open PRs (#103–#112): none covered CRAN.
- OSV ecosystem list confirms `CRAN` as a valid, active ecosystem with real
  advisories (e.g., `jsonlite`, `openssl`, `httr` packages).
- Swift Package Index API returned 401 (auth required) — ruled out.
- ConanCenter API returned HTML/404 for REST queries — ruled out.
- crandb and cranlogs: both free, unauthenticated, JSON, confirmed working.

### Implementation summary

**Constants**: `_CRANDB_URL`, `_CRANLOGS_URL`

**New helper**: `_parse_cran_timestamp(ts)` — normalises crandb ISO-8601
timestamps (`"2014-10-21T08:27:55+00:00"` or `"...Z"`) to `"YYYY-MM-DD"` by
slicing the first 10 characters.

**New enricher**: `_enrich_cran(name)`:
- Call 1: `crandb.r-pkg.org/{name}/all` → latest version, title (max 120 chars),
  archived flag, `revdeps` (→ `dependent_packages_count`), version timeline count,
  first-release date, age in years, CRAN page URL.
- Call 2: `cranlogs.r-pkg.org/downloads/total/last-month/{name}` → monthly
  downloads; `weekly_downloads = monthly // 4` (integer floor division).
- Independent try/except per call for graceful degradation.
- Both-fail fallback: `{ecosystem: "CRAN", package_name: name}` without raising.

**Dispatcher**: `eco_lower in ("cran", "r")` in `_enrich_package` — covers
`cran:openssl` and `r:openssl` input forms.

**Output block**: title, latest version, total versions, reverse deps, monthly
downloads, first release + age, archived status, CRAN page URL.  Fixed npm
dependents label to show only for npm ecosystem (previously shown for any
ecosystem with `dependent_packages_count`).

### Tests

30 new tests (6 classes, all HTTP mocked):
`TestParseCranTimestamp` (6), `TestEnrichCranHappyPath` (16),
`TestEnrichCranDegradation` (7), `TestEnrichPackageCranDispatch` (5),
`TestBlastScoreWithCranData` (6), `TestGetDependencyBlastRadiusCran` (9).

Suite: **1206 passed** (was 1158 before this session series; +30 this PR).
3 deselected, 3 warnings — all pre-existing, zero regressions.

### Suggested next contributions

- **Hackage enricher** (`_enrich_hackage`) — Haskell package registry; OSV
  ecosystem `Hackage`; API: `hackage.haskell.org/package/{name}.json` (free,
  unauthenticated). Revdeps via `hackage.haskell.org/package/{name}/reverse`.
- **opam enricher** (`_enrich_opam`) — OCaml package registry; OSV ecosystem
  `OSS-Fuzz`/`crates.io` alignment; API: `opam.ocaml.org/pkg/{name}/latest`
  (JSON, no auth).
- **OSV `_ECOSYSTEM_TO_OSV` coverage** — extend the mapping to include CRAN
  once this PR merges.
- **Per-version CVE integration for CRAN** — `manus-agent blast-radius
  cran:openssl@2.1.1` should call `_fetch_osv_package_vulns("openssl", "CRAN", "2.1.1")`.

---

## PR #158 — Shared test conftest.py with reusable fixtures (2026-07-29)

### What
Added `tests/conftest.py` with shared pytest fixtures and `tests/test_conftest_fixtures.py`
(50 tests) that validates and documents them.

### Why
30+ test files independently define their own mock factories (tool_use dicts,
HTTP responses, NVD/EPSS/KEV payloads). This leads to duplicated helpers,
inconsistent mock shapes, and higher friction when writing new tests. A shared
conftest.py is the standard pytest solution.

### Fixtures provided
- `make_tool_use` — builds Strands tool_use dicts with auto-generated IDs
- `mock_http_response` — creates mock requests.Response with raise_for_status()
- `nvd_cve_factory` / `nvd_api_response` — builds NVD CVE payloads
- `epss_data_factory` — builds EPSS API responses
- `kev_catalog_factory` — builds CISA KEV catalog payloads
- `osv_record_factory` — builds OSV.dev records
- Config fixtures (default, bedrock, openai, anthropic, ollama)
- `tmp_history_file` / `tmp_config_file` — temp filesystem helpers
- `env_override` — clean monkeypatch wrapper
- Module constants: `SAMPLE_CVE_LOG4SHELL`, `SAMPLE_CVE_XZ`

### Results
- 50 new fixture validation tests pass
- All 1208 existing tests unchanged / passing
- Ruff clean

### Suggested next contributions

- **`_run_changelog_generate` test coverage** — the changelog CLI tests
  (`test_changelog_cli.py`) are comprehensive but `_run_changelog_generate`
  could use edge-case tests for merge commits, non-conventional commits mixed
  in, and empty repos.
- **BrowserUseAgent.stream_async tests** — `stream_async()` uses asyncio.Queue
  with step/done callbacks; no tests cover the streaming yield/queue pattern.
- **WorkflowAgent / Orchestrator tests** — `Orchestrator.run()` currently just
  delegates to WorkflowAgent; tests for the `TaskPlan`, `OrchestratorResult`,
  and `AgentType` data structures + error handling.
- **Refactor existing tests to use conftest fixtures** — once PR #158 merges,
  existing test files can be simplified by replacing local helpers with shared
  fixtures.

---

## Contribution 6 — BrowserUseAgent.stream_async tests

**PR:** #159
**Branch:** `feat/test-stream-async-browser-agent`
**File:** `tests/test_browser_use_stream_async.py` (+1548 lines)

### Rationale

`stream_async()` is a 165-line async generator with callbacks, queue management,
error handling, and cleanup logic. Existing tests only covered the fallback path
(2 tests when BrowserUse is None), leaving zero coverage of the real streaming
path: step_callback, done_callback, queue draining, background task, keep_alive
timeout, and finally-block cleanup.

### What was added

45 tests in 7 classes:
- Fallback path (5): BrowserUse/BrowserProfile/Controller unavailable
- True streaming path (11): callbacks, queue flow, extracted content, output_model
- Error handling (7): constructor/LLM/callback failures, post-done exceptions
- Browser cleanup (7): close(), keep_alive timeout, error resilience
- Input handling (4): string/list/empty task parsing
- Task cancellation (2): background task lifecycle
- Configuration (4): headless, output_model, enable_memory, patch config
- Edge cases (8): missing attributes, mixed types, validate_output

### Results
- 45 new tests pass
- All 1203 tests pass (up from 1158), no regressions
- Ruff clean

### Suggested next contributions

- **WorkflowAgent / Orchestrator tests** — `Orchestrator.run()` delegates to
  WorkflowAgent; tests for `TaskPlan`, `OrchestratorResult`, `AgentType` data
  structures + error handling.
- **`_run_browser_task` tests** — the non-streaming browser task method has
  complex try/finally with keep_alive/timeout logic worth covering.
- **CLI `vendor-response` / `verify-exploit` subcommand tests** — several open
  PRs add new subcommands but many lack unit tests for argument parsing.
- **Refactor existing tests to use conftest fixtures** — once PR #158 merges,
  existing test files can be simplified.

---

## Session: 2026-08-01 00:00 UTC — PR #165

### Contribution: test_check_cisa_kev.py — CISA KEV tool test suite

**Branch:** `test/check-cisa-kev-suite`
**PR:** https://github.com/manus-use/manus-agent/pull/165
**Tests added:** 43

### Coverage areas
- TOOL_SPEC schema validation (5 tests)
- `_get_kev_data` caching logic: fresh/stale/missing cache, network errors,
  cache write, malformed cache, missing timestamp (8 tests)
- `check_cisa_kev` main function: CVE found/not-found, case-insensitive,
  exact match, multiple CVEs, empty catalog, missing key, invalid inputs,
  tool_use_id propagation (15 tests)
- `log_tool_output_size` integration — every exit path (4 tests)
- Edge cases: single-entry catalog, mixed-case, missing key, content structure (8 tests)
- Module constants validation (3 tests)

### Discovered issue
`_get_kev_data()` does not handle corrupted cache files gracefully —
`json.loads()` on a malformed `.cisa_kev_cache.json` propagates
`JSONDecodeError` instead of falling through to a fresh API fetch.
Documented in test.

### Results
- 43 new tests pass
- Full suite: 1201 passed, 0 failures, 0 regressions
- Ruff clean (lint + format)

### Suggested next contributions
- **`get_cwe_details` tool tests** — another core tool with zero test coverage;
  similar structure (HTTP fetch + HTML parsing + error paths)
- **`obtain_cves` tool tests** — complex multi-source aggregation pipeline
  (NVD + GitHub + EPSS + KEV + webhook) with zero unit tests
- **`search_poc_sources` tool tests** — parallel 5-source aggregator with
  dedup/sort logic; existing `test_search_poc_sources.py` only covers basic cases
- **Cache robustness fix PR** — wrap `json.loads()` in `_get_kev_data` with
  try/except to gracefully handle corrupted cache

## 2026-08-01 — PR #112 (updated): decode-cvss vector decoder/explainer tool

**What:** New `decode_cvss_vector` tool + `decode-cvss` CLI subcommand that parses CVSS v3.0/v3.1 vector strings and produces structured breakdowns with:
- Per-metric human-readable explanations
- Computed base score (exact CVSS v3.1 formula)
- Severity classification
- Natural-language attack summary
- Remediation priority with escalation factors

**Why:** The project collects CVSS vectors from multiple sources (NVD, VulnCheck, patch-diff) but never explains what they mean. This fills the gap between raw scores and actionable security context.

**Tests:** +70 new tests (1228 total passing, up from 1158 baseline)
**Branch:** feat/cvss-vector-decoder
**PR:** https://github.com/manus-use/manus-agent/pull/112

## 2026-08-02 — PR #170: cve-enrich tool + CLI subcommand

**Branch:** `feat/cve-enrich`
**What:** Lightweight multi-source CVE enrichment tool that fetches NVD, EPSS, CISA KEV, OSV.dev, and VulnCheck data in parallel. Returns unified risk snapshot with composite scoring. No LLM agent required.
**CLI:** `manus-agent enrich CVE-XXXX-YYYY [--output json|text] [--no-vulncheck]`
**Tests:** +43 new tests (all fetchers, risk computation, CLI modes, Strands interface)
**Suggested next:** test suite for `_run_discover` CLI execution (VulnerabilityDiscoveryAgent mocking), or `Config.from_file()` edge cases, or `obtain_cves.py` unit tests

## 2026-08-03 — PR #172: Config._apply_env_overrides() + from_file() tests

**Branch:** `feat/test-config-env-overrides`
**What:** Comprehensive test suite for the untested environment-variable → config mapping pipeline. Tests `_apply_env_overrides()` (Pydantic model validator), `Config.from_file()` search-path logic, and `_load_dotenv()` behavior.
**Tests:** +66 new tests covering:
- MANUS_LLM_* always-override semantics (provider, model, base_url, temperature, max_tokens)
- API key fill-when-None semantics (OPENAI_API_KEY, ANTHROPIC_API_KEY per provider)
- AWS region resolution priority (MANUS_AWS_REGION > AWS_DEFAULT_REGION > AWS_REGION)
- Integration env vars (OTX, GitHub, Lark, Webhooks, MCP)
- from_file() search paths (CWD, config/, ~/.manus-agent/)
- _load_dotenv() search, priority, override=False, missing-dotenv graceful no-op
- Combined: full env-only, partial override, empty-string-as-unset
**Suggested next:** test suite for `_cmd_doctor` execution logic (check_import, provider checks, docker/playwright probing), or `_run_discover` CLI execution (VulnerabilityDiscoveryAgent mocking), or `browser_agent_tool` wrapper tests
