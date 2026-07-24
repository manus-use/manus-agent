
---

## 2026-07-24 — PR #146: `verify-exploit` CLI subcommand

**Branch:** `feat/cli-verify-exploit`
**PR:** https://github.com/manus-use/manus-agent/pull/146

### What was done

Wired the existing `ExploitSandbox` (`src/manus_agent/sandbox/exploit_sandbox.py`)
as a standalone CLI command: `manus-agent verify-exploit`.

**Implementation:**
- `_build_verify_exploit_parser()` — argparse with full help/epilog
- `_run_verify_exploit(argv)` — main logic: Docker preflight → build → start → exploit → result
- `_verify_exploit_output(data, fmt, status)` — text/JSON formatter
- Dispatch entry in CLI router

**Features:**
- Remote mode (default): exploit in separate container over isolated network
- Local mode: exploit inside target container (LPE/parsing bugs)
- Docker preflight with diagnostic messages
- Text + JSON output formats
- Structured statuses: verified, failed, build_error, target_error, infra_error, exploit_error
- `--port`, `--timeout`, `--env`, `--software`, `--versions`, `--vuln-type`

**Files modified:**
- `src/manus_agent/cli.py` (+435 lines)
- `tests/test_cli_verify_exploit.py` (new, 797 lines)

39 new tests (5 classes, all mocked):
`TestVerifyExploitParser` (9), `TestVerifyExploitRemote` (11),
`TestVerifyExploitLocal` (5), `TestVerifyExploitImportErrors` (1),
`TestVerifyExploitOutput` (8), `TestVerifyExploitDispatch` (5).

Suite: **1197 passed** (3 deselected, 3 warnings — pre-existing, zero regressions).

### Suggested next contributions

- **Hackage enricher** (`_enrich_hackage`) — Haskell package registry
- **opam enricher** (`_enrich_opam`) — OCaml package registry
- **CLI subcommand: `generate-exploit`** — wire the exploit generation tool
- **CLI subcommand: `sandbox-status`** — Docker health check + cleanup orphans
