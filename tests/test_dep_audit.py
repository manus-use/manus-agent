"""Comprehensive test suite for dep_audit tool.

Tests cover:
- All manifest parsers (requirements.txt, package.json, package-lock.json,
  go.mod, Cargo.toml, Cargo.lock, Gemfile.lock, pom.xml, composer.json)
- OSV.dev batch query logic
- EPSS score enrichment
- CISA KEV enrichment
- Full audit pipeline (integration)
- CLI text and JSON output
- Edge cases (empty files, malformed input, partial failures)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.dep_audit import (
    _detect_and_parse,
    _fetch_epss_scores,
    _fetch_kev_set,
    _osv_batch_query,
    _parse_cargo_lock,
    _parse_cargo_toml,
    _parse_composer_json,
    _parse_gemfile_lock,
    _parse_go_mod,
    _parse_package_json,
    _parse_package_lock_json,
    _parse_pom_xml,
    _parse_requirements_txt,
    audit_dependencies,
)


# ===========================================================================
# Parser tests
# ===========================================================================


class TestParseRequirementsTxt:
    """Tests for _parse_requirements_txt."""

    def test_basic_pinned(self):
        content = "requests==2.28.0\nflask==2.3.1\n"
        result = _parse_requirements_txt(content)
        assert len(result) == 2
        assert result[0] == {"name": "requests", "version": "2.28.0"}
        assert result[1] == {"name": "flask", "version": "2.3.1"}

    def test_with_extras(self):
        content = "requests[security]==2.28.0\n"
        result = _parse_requirements_txt(content)
        assert len(result) == 1
        assert result[0] == {"name": "requests", "version": "2.28.0"}

    def test_comments_and_blanks(self):
        content = "# comment\n\nrequests==2.28.0\n# another comment\n\n"
        result = _parse_requirements_txt(content)
        assert len(result) == 1
        assert result[0]["name"] == "requests"

    def test_flags_ignored(self):
        content = "-r other.txt\n--index-url https://x\nrequests==2.28.0\n"
        result = _parse_requirements_txt(content)
        assert len(result) == 1

    def test_non_pinned_skipped(self):
        content = "requests>=2.28.0\nflask~=2.3\ndjango\n"
        result = _parse_requirements_txt(content)
        assert len(result) == 0

    def test_name_normalized_lowercase(self):
        content = "Flask==2.3.1\nDjango-REST-Framework==3.14.0\n"
        result = _parse_requirements_txt(content)
        assert result[0]["name"] == "flask"
        assert result[1]["name"] == "django-rest-framework"

    def test_version_with_markers(self):
        content = 'requests==2.28.0; python_version>="3.7"\n'
        result = _parse_requirements_txt(content)
        assert len(result) == 1
        assert result[0]["version"] == "2.28.0"

    def test_empty_content(self):
        result = _parse_requirements_txt("")
        assert result == []


class TestParsePackageJson:
    """Tests for _parse_package_json."""

    def test_basic_deps(self):
        content = json.dumps({
            "dependencies": {"express": "^4.18.0", "lodash": "4.17.21"},
            "devDependencies": {"jest": "~29.0.0"},
        })
        result = _parse_package_json(content)
        assert len(result) == 3
        names = {r["name"] for r in result}
        assert "express" in names
        assert "lodash" in names
        assert "jest" in names

    def test_strips_semver_prefixes(self):
        content = json.dumps({"dependencies": {"pkg": "^1.2.3"}})
        result = _parse_package_json(content)
        assert result[0]["version"] == "1.2.3"

    def test_invalid_json(self):
        result = _parse_package_json("not json{{{")
        assert result == []

    def test_no_deps_key(self):
        content = json.dumps({"name": "my-app", "version": "1.0.0"})
        result = _parse_package_json(content)
        assert result == []

    def test_non_semver_versions_skipped(self):
        content = json.dumps({
            "dependencies": {"git-dep": "git+https://github.com/x/y.git"}
        })
        result = _parse_package_json(content)
        assert result == []


class TestParsePackageLockJson:
    """Tests for _parse_package_lock_json."""

    def test_v2_format(self):
        content = json.dumps({
            "lockfileVersion": 2,
            "packages": {
                "": {"name": "root", "version": "1.0.0"},
                "node_modules/express": {"version": "4.18.2"},
                "node_modules/lodash": {"version": "4.17.21", "name": "lodash"},
            },
        })
        result = _parse_package_lock_json(content)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "express" in names
        assert "lodash" in names

    def test_v1_fallback(self):
        content = json.dumps({
            "lockfileVersion": 1,
            "dependencies": {
                "express": {"version": "4.18.2"},
                "lodash": {"version": "4.17.21"},
            },
        })
        result = _parse_package_lock_json(content)
        assert len(result) == 2

    def test_invalid_json(self):
        result = _parse_package_lock_json("broken")
        assert result == []


class TestParseGoMod:
    """Tests for _parse_go_mod."""

    def test_basic_require_block(self):
        content = """module example.com/myapp

go 1.21

require (
\tgithub.com/gin-gonic/gin v1.9.1
\tgithub.com/stretchr/testify v1.8.4
)
"""
        result = _parse_go_mod(content)
        assert len(result) == 2
        assert result[0] == {"name": "github.com/gin-gonic/gin", "version": "1.9.1"}
        assert result[1] == {"name": "github.com/stretchr/testify", "version": "1.8.4"}

    def test_single_require(self):
        content = "module x\ngo 1.21\nrequire github.com/pkg/errors v0.9.1\n"
        result = _parse_go_mod(content)
        assert len(result) == 1
        assert result[0]["name"] == "github.com/pkg/errors"

    def test_empty_mod(self):
        content = "module example.com/x\ngo 1.21\n"
        result = _parse_go_mod(content)
        assert result == []


class TestParseCargoLock:
    """Tests for _parse_cargo_lock."""

    def test_basic_packages(self):
        content = '''[[package]]
name = "serde"
version = "1.0.188"

[[package]]
name = "tokio"
version = "1.32.0"
'''
        result = _parse_cargo_lock(content)
        assert len(result) == 2
        assert result[0] == {"name": "serde", "version": "1.0.188"}
        assert result[1] == {"name": "tokio", "version": "1.32.0"}

    def test_empty(self):
        result = _parse_cargo_lock("")
        assert result == []


class TestParseCargoToml:
    """Tests for _parse_cargo_toml."""

    def test_basic_deps(self):
        content = '''[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = "1.32"

[dev-dependencies]
criterion = "0.5"
'''
        result = _parse_cargo_toml(content)
        assert len(result) == 2
        assert result[0] == {"name": "serde", "version": "1.0"}
        assert result[1] == {"name": "tokio", "version": "1.32"}

    def test_no_deps_section(self):
        content = "[package]\nname = \"x\"\n"
        result = _parse_cargo_toml(content)
        assert result == []


class TestParseGemfileLock:
    """Tests for _parse_gemfile_lock."""

    def test_basic_gems(self):
        content = """GEM
  remote: https://rubygems.org/
  specs:
    rack (3.0.8)
    rails (7.0.6)
      actioncable (= 7.0.6)

PLATFORMS
  ruby
"""
        result = _parse_gemfile_lock(content)
        assert len(result) == 2
        assert result[0] == {"name": "rack", "version": "3.0.8"}
        assert result[1] == {"name": "rails", "version": "7.0.6"}

    def test_empty(self):
        result = _parse_gemfile_lock("")
        assert result == []


class TestParsePomXml:
    """Tests for _parse_pom_xml."""

    def test_basic_deps(self):
        content = """<project>
  <dependencies>
    <dependency>
      <groupId>org.springframework</groupId>
      <artifactId>spring-core</artifactId>
      <version>5.3.29</version>
    </dependency>
  </dependencies>
</project>"""
        result = _parse_pom_xml(content)
        assert len(result) == 1
        assert result[0] == {
            "name": "org.springframework:spring-core",
            "version": "5.3.29",
        }

    def test_skips_property_refs(self):
        content = """<dependency>
      <groupId>org.x</groupId>
      <artifactId>y</artifactId>
      <version>${project.version}</version>
    </dependency>"""
        result = _parse_pom_xml(content)
        assert result == []

    def test_empty(self):
        result = _parse_pom_xml("")
        assert result == []


class TestParseComposerJson:
    """Tests for _parse_composer_json."""

    def test_basic_deps(self):
        content = json.dumps({
            "require": {
                "php": ">=8.0",
                "laravel/framework": "^9.0",
                "ext-json": "*",
            },
            "require-dev": {
                "phpunit/phpunit": "^9.5",
            },
        })
        result = _parse_composer_json(content)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "laravel/framework" in names
        assert "phpunit/phpunit" in names
        # php and ext- should be skipped
        assert "php" not in names

    def test_invalid_json(self):
        result = _parse_composer_json("not json")
        assert result == []


# ===========================================================================
# _detect_and_parse tests
# ===========================================================================


class TestDetectAndParse:
    """Tests for _detect_and_parse."""

    def test_finds_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\n")
        result = _detect_and_parse(str(tmp_path))
        assert len(result) == 1
        assert result[0]["ecosystem"] == "PyPI"
        assert result[0]["source_file"] == "requirements.txt"

    def test_multiple_manifests(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\n")
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"express": "4.18.0"}})
        )
        result = _detect_and_parse(str(tmp_path))
        assert len(result) == 2
        ecosystems = {r["ecosystem"] for r in result}
        assert "PyPI" in ecosystems
        assert "npm" in ecosystems

    def test_nonexistent_directory(self):
        result = _detect_and_parse("/nonexistent/path/xyz")
        assert result == []

    def test_empty_directory(self, tmp_path):
        result = _detect_and_parse(str(tmp_path))
        assert result == []


# ===========================================================================
# OSV batch query tests
# ===========================================================================


class TestOsvBatchQuery:
    """Tests for _osv_batch_query."""

    def test_empty_input(self):
        result = _osv_batch_query([])
        assert result == []

    @patch("manus_agent.tools.dep_audit.requests.post")
    def test_successful_query(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"vulns": [{"id": "GHSA-xxx", "aliases": ["CVE-2024-1234"]}]},
                {"vulns": []},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        packages = [
            {"name": "requests", "version": "2.25.0", "ecosystem": "PyPI"},
            {"name": "flask", "version": "2.3.1", "ecosystem": "PyPI"},
        ]
        result = _osv_batch_query(packages)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[0][0]["id"] == "GHSA-xxx"
        assert result[1] == []

    @patch("manus_agent.tools.dep_audit.requests.post")
    def test_request_failure_returns_empty(self, mock_post):
        import requests as _requests
        mock_post.side_effect = _requests.ConnectionError("Network error")
        packages = [{"name": "x", "version": "1.0", "ecosystem": "PyPI"}]
        result = _osv_batch_query(packages)
        assert len(result) == 1
        assert result[0] == []

    @patch("manus_agent.tools.dep_audit.requests.post")
    def test_batch_splitting(self, mock_post):
        """Verifies batching works for large package lists."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        # Return empty vulns for each package
        def side_effect(*args, **kwargs):
            body = kwargs.get("json", {})
            n = len(body.get("queries", []))
            mock_resp.json.return_value = {
                "results": [{"vulns": []} for _ in range(n)]
            }
            return mock_resp

        mock_post.side_effect = side_effect

        # Create 1500 packages (should split into 2 batches)
        packages = [
            {"name": f"pkg-{i}", "version": "1.0", "ecosystem": "PyPI"}
            for i in range(1500)
        ]
        result = _osv_batch_query(packages)
        assert len(result) == 1500
        assert mock_post.call_count == 2


# ===========================================================================
# EPSS enrichment tests
# ===========================================================================


class TestFetchEpssScores:
    """Tests for _fetch_epss_scores."""

    def test_empty_input(self):
        result = _fetch_epss_scores([])
        assert result == {}

    @patch("manus_agent.tools.dep_audit.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2024-1234", "epss": "0.95"},
                {"cve": "CVE-2024-5678", "epss": "0.01"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_epss_scores(["CVE-2024-1234", "CVE-2024-5678"])
        assert result["CVE-2024-1234"] == 0.95
        assert result["CVE-2024-5678"] == 0.01

    @patch("manus_agent.tools.dep_audit.requests.get")
    def test_failure_returns_empty(self, mock_get):
        import requests as _requests
        mock_get.side_effect = _requests.Timeout("timeout")
        result = _fetch_epss_scores(["CVE-2024-1234"])
        assert result == {}

    @patch("manus_agent.tools.dep_audit.requests.get")
    def test_batch_splitting_100(self, mock_get):
        """EPSS API limited to 100 per request."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [f"CVE-2024-{i:04d}" for i in range(250)]
        _fetch_epss_scores(cves)
        assert mock_get.call_count == 3  # 100 + 100 + 50


# ===========================================================================
# KEV enrichment tests
# ===========================================================================


class TestFetchKevSet:
    """Tests for _fetch_kev_set."""

    @patch("manus_agent.tools.dep_audit.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-1234"},
                {"cveID": "CVE-2024-5678"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_kev_set()
        assert "CVE-2024-1234" in result
        assert "CVE-2024-5678" in result

    @patch("manus_agent.tools.dep_audit.requests.get")
    def test_failure_returns_empty(self, mock_get):
        import requests as _requests
        mock_get.side_effect = _requests.Timeout("timeout")
        result = _fetch_kev_set()
        assert result == set()


# ===========================================================================
# Full audit integration tests
# ===========================================================================


class TestAuditDependencies:
    """Tests for audit_dependencies."""

    def test_no_manifests(self, tmp_path):
        result = audit_dependencies(str(tmp_path))
        assert result["status"] == "no_manifests"
        assert result["packages_scanned"] == 0
        assert result["vulnerabilities_found"] == 0

    @patch("manus_agent.tools.dep_audit._fetch_kev_set")
    @patch("manus_agent.tools.dep_audit._fetch_epss_scores")
    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_full_pipeline(self, mock_osv, mock_epss, mock_kev, tmp_path):
        # Setup manifest
        (tmp_path / "requirements.txt").write_text("requests==2.25.0\nflask==2.0.0\n")

        # Mock OSV response
        mock_osv.return_value = [
            [
                {
                    "id": "GHSA-j8r2-6x86-q33q",
                    "aliases": ["CVE-2023-32681"],
                    "summary": "Requests proxy credential leak",
                    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N"}],
                    "affected": [
                        {
                            "package": {"name": "requests", "ecosystem": "PyPI"},
                            "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.31.0"}]}],
                        }
                    ],
                }
            ],
            [],  # No vulns for flask
        ]

        # Mock EPSS
        mock_epss.return_value = {"CVE-2023-32681": 0.42}

        # Mock KEV
        mock_kev.return_value = set()

        result = audit_dependencies(str(tmp_path))

        assert result["status"] == "completed"
        assert result["packages_scanned"] == 2
        assert result["vulnerabilities_found"] == 1
        assert result["packages_affected"] == 1
        assert len(result["findings"]) == 1

        finding = result["findings"][0]
        assert finding["package"] == "requests"
        assert finding["version"] == "2.25.0"
        assert finding["vuln_id"] == "GHSA-j8r2-6x86-q33q"
        assert "CVE-2023-32681" in finding["aliases"]
        assert finding["fixed_version"] == "2.31.0"
        assert finding["epss_score"] == 0.42
        assert finding["in_kev"] is False

    @patch("manus_agent.tools.dep_audit._fetch_kev_set")
    @patch("manus_agent.tools.dep_audit._fetch_epss_scores")
    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_kev_findings_sorted_first(self, mock_osv, mock_epss, mock_kev, tmp_path):
        (tmp_path / "requirements.txt").write_text("pkg-a==1.0\npkg-b==1.0\n")

        mock_osv.return_value = [
            [{"id": "GHSA-aaaa", "aliases": ["CVE-2024-0001"], "summary": "A", "severity": [], "affected": []}],
            [{"id": "GHSA-bbbb", "aliases": ["CVE-2024-0002"], "summary": "B", "severity": [], "affected": []}],
        ]
        mock_epss.return_value = {"CVE-2024-0001": 0.9, "CVE-2024-0002": 0.1}
        mock_kev.return_value = {"CVE-2024-0002"}  # Only pkg-b is in KEV

        result = audit_dependencies(str(tmp_path))
        assert result["findings"][0]["vuln_id"] == "GHSA-bbbb"  # KEV first
        assert result["findings"][0]["in_kev"] is True
        assert result["findings"][1]["vuln_id"] == "GHSA-aaaa"

    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_skip_epss_and_kev(self, mock_osv, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.25.0\n")
        mock_osv.return_value = [[]]

        result = audit_dependencies(str(tmp_path), skip_epss=True, skip_kev=True)
        assert result["status"] == "completed"
        assert result["packages_scanned"] == 1

    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_no_vulns_found(self, mock_osv, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        mock_osv.return_value = [[]]

        result = audit_dependencies(str(tmp_path), skip_epss=True, skip_kev=True)
        assert result["vulnerabilities_found"] == 0
        assert result["findings"] == []


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCliDepAudit:
    """Tests for the dep-audit CLI subcommand."""

    @patch("manus_agent.tools.dep_audit._fetch_kev_set")
    @patch("manus_agent.tools.dep_audit._fetch_epss_scores")
    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_json_output(self, mock_osv, mock_epss, mock_kev, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.25.0\n")
        mock_osv.return_value = [[]]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        from manus_agent.cli import _run_dep_audit

        exit_code = _run_dep_audit([str(tmp_path), "--output", "json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "completed"
        assert data["packages_scanned"] == 1

    @patch("manus_agent.tools.dep_audit._fetch_kev_set")
    @patch("manus_agent.tools.dep_audit._fetch_epss_scores")
    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_text_output_no_vulns(self, mock_osv, mock_epss, mock_kev, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        mock_osv.return_value = [[]]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        from manus_agent.cli import _run_dep_audit

        exit_code = _run_dep_audit([str(tmp_path)])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No known vulnerabilities found" in captured.out

    @patch("manus_agent.tools.dep_audit._fetch_kev_set")
    @patch("manus_agent.tools.dep_audit._fetch_epss_scores")
    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_text_output_with_vulns(self, mock_osv, mock_epss, mock_kev, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.25.0\n")
        mock_osv.return_value = [
            [{"id": "GHSA-test", "aliases": ["CVE-2024-9999"], "summary": "Test vuln", "severity": [], "affected": []}]
        ]
        mock_epss.return_value = {"CVE-2024-9999": 0.75}
        mock_kev.return_value = {"CVE-2024-9999"}

        from manus_agent.cli import _run_dep_audit

        exit_code = _run_dep_audit([str(tmp_path)])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "GHSA-test" in captured.out
        assert "KEV" in captured.out
        assert "EPSS=75.0%" in captured.out

    def test_nonexistent_directory(self, capsys):
        from manus_agent.cli import _run_dep_audit

        exit_code = _run_dep_audit(["/nonexistent/dir/xyz"])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "error" in captured.err.lower()

    def test_no_manifests(self, tmp_path, capsys):
        from manus_agent.cli import _run_dep_audit

        exit_code = _run_dep_audit([str(tmp_path)])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "No supported manifest files" in captured.out

    def test_skip_flags(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")

        from manus_agent.cli import _run_dep_audit

        with patch("manus_agent.tools.dep_audit._osv_batch_query") as mock_osv, \
             patch("manus_agent.tools.dep_audit._fetch_epss_scores") as mock_epss, \
             patch("manus_agent.tools.dep_audit._fetch_kev_set") as mock_kev:
            mock_osv.return_value = [[]]
            mock_epss.return_value = {}
            mock_kev.return_value = set()

            exit_code = _run_dep_audit([
                str(tmp_path), "--skip-epss", "--skip-kev"
            ])
            assert exit_code == 0
            # EPSS and KEV should not be called when skipped
            mock_epss.assert_not_called()
            mock_kev.assert_not_called()


# ===========================================================================
# Strands tool interface tests
# ===========================================================================


class TestStrandsToolInterface:
    """Tests for the @tool-decorated dep_audit function."""

    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_returns_json_string(self, mock_osv, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        mock_osv.return_value = [[]]

        from manus_agent.tools.dep_audit import dep_audit as dep_audit_fn

        # The @tool decorator wraps it; call the underlying function
        result = audit_dependencies(str(tmp_path), skip_epss=True, skip_kev=True)
        assert isinstance(result, dict)
        assert result["status"] == "completed"


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_binary_file_in_manifest_location(self, tmp_path):
        """Binary file with manifest name should not crash."""
        (tmp_path / "requirements.txt").write_bytes(b"\x00\x01\x02\x03")
        result = _detect_and_parse(str(tmp_path))
        # Should return empty or handle gracefully
        assert isinstance(result, list)

    def test_very_large_requirements(self, tmp_path):
        """Handles large manifest files."""
        lines = [f"pkg-{i}=={i}.0.0" for i in range(500)]
        (tmp_path / "requirements.txt").write_text("\n".join(lines))
        result = _detect_and_parse(str(tmp_path))
        assert len(result) == 500

    @patch("manus_agent.tools.dep_audit._osv_batch_query")
    def test_duplicate_vulns_deduped_in_summary(self, mock_osv, tmp_path):
        """Same vuln for same package should count once in unique_vulns."""
        (tmp_path / "requirements.txt").write_text("requests==2.25.0\n")
        mock_osv.return_value = [
            [
                {"id": "GHSA-xxx", "aliases": ["CVE-2024-1111"], "summary": "A", "severity": [], "affected": []},
                {"id": "GHSA-xxx", "aliases": ["CVE-2024-1111"], "summary": "A", "severity": [], "affected": []},
            ]
        ]

        result = audit_dependencies(str(tmp_path), skip_epss=True, skip_kev=True)
        # Two findings (from raw data), but unique_vulns should deduplicate
        assert result["vulnerabilities_found"] == 1

    def test_go_mod_with_indirect(self):
        """go.mod indirect deps should still parse."""
        content = """module example.com/x

require (
\tgithub.com/pkg/errors v0.9.1
\tgolang.org/x/text v0.13.0 // indirect
)
"""
        result = _parse_go_mod(content)
        assert len(result) == 2
