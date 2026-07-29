
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
