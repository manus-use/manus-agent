
## 2026-08-24 — PR #193: resolve_advisory_aliases tool + CLI subcommand

**Branch:** `feat/advisory-aliases`
**What:** New `resolve_advisory_aliases` tool + `manus-agent advisory-aliases` CLI subcommand that maps a CVE to all its cross-referenced vendor-specific advisory IDs (GHSA, RHSA, DSA, USN, ALAS, PYSEC, RUSTSEC, GO, HSEC, MAL, etc.) using OSV.dev alias graph + NVD reference URL parsing + optional VulnCheck.
**CLI:** `manus-agent advisory-aliases CVE-XXXX-YYYY [--output json|text] [--no-urls]`
**Why:** The project fetches data from many sources but never unifies advisory cross-references. Knowing all vendor IDs for a CVE enables targeted patching (e.g., finding the DSA for Debian, the RHSA for RHEL, the GHSA for GitHub Dependabot).
**Tests:** +77 new tests (1235 total passing, up from 1158 baseline). All mocked.
**Suggested next:** Add `search_vulns` tool for free-text vulnerability search across OSV.dev (by package name + ecosystem), or a `monitor-cve` background watcher that alerts on new alias/advisory appearances.
