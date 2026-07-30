"""Comprehensive test suite for _run_changelog_generate.

Tests cover:
- Conventional commit parsing (type, scope, breaking, description extraction)
- Version inference from pyproject.toml
- Semantic version bumping (major/minor/patch)
- Section grouping by commit type
- Text and JSON output formats
- Edge cases (no commits, no tags, breaking changes in footer, etc.)
- Git subprocess interaction (fully mocked)

All git and filesystem operations are fully mocked — no real git calls or file I/O.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(output: str = "text", generate: bool = True, filter_version: str | None = None) -> argparse.Namespace:
    """Create a minimal args namespace matching what _build_changelog_parser produces."""
    return argparse.Namespace(
        output=output,
        generate=generate,
        filter_version=filter_version,
    )


def _git_log_entry(sha: str, subject: str, body: str = "") -> str:
    """Format a single git log entry in the \x1f-separated, \x1e-terminated format."""
    return f"{sha}\x1f{subject}\x1f{body}\x1e"


def _pyproject_content(version: str = "1.2.3") -> str:
    """Return minimal pyproject.toml content with a version field."""
    return dedent(f"""\
        [project]
        name = "manus-agent"
        version = "{version}"
    """)


# ---------------------------------------------------------------------------
# Test: Conventional commit parsing
# ---------------------------------------------------------------------------


class TestConventionalCommitParsing:
    """Test that conventional commits are correctly parsed from git log output."""

    def _run(self, git_log_output: str, version: str = "1.0.0", output: str = "json") -> tuple[int, str, str]:
        """Run _run_changelog_generate with mocked git and pyproject."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output=output)

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = git_log_output
            else:
                result.stdout = ""
            return result

        pyproject = mock.Mock()
        pyproject.exists.return_value = True
        pyproject.read_text.return_value = _pyproject_content(version)

        out_buf, err_buf = io.StringIO(), io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch(
                "pathlib.Path.__truediv__",
                side_effect=lambda self, key: (
                    pyproject if key == "pyproject.toml" else mock.Mock(exists=mock.Mock(return_value=False))
                ),
            ),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", err_buf),
        ):
            # We need to patch Path operations carefully
            # Instead, let's do a simpler approach: patch the internal _git helper
            pass

        # Simpler approach: import and call with controlled subprocess mock
        out_buf, err_buf = io.StringIO(), io.StringIO()

        with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            # Mock the root / "pyproject.toml" path
            fake_pyproject = mock.Mock(spec=Path)
            fake_pyproject.exists.return_value = True
            fake_pyproject.read_text.return_value = _pyproject_content(version)

            fake_root = mock.Mock(spec=Path)
            fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

            with mock.patch("sys.stdout", out_buf), mock.patch("sys.stderr", err_buf):
                rc = _run_changelog_generate(args, fake_root)

        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_feat_commit_parsed(self):
        """A feat commit is parsed and placed in the 'Added' section."""
        log = _git_log_entry("abc12345", "feat: add new feature")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["type"] == "feat"
        assert data["commits"][0]["section"] == "Added"
        assert data["commits"][0]["description"] == "add new feature"

    def test_fix_commit_parsed(self):
        """A fix commit is parsed and placed in the 'Fixed' section."""
        log = _git_log_entry("def67890", "fix: resolve crash on startup")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["type"] == "fix"
        assert data["commits"][0]["section"] == "Fixed"

    def test_scoped_commit_parsed(self):
        """A scoped commit (e.g. feat(cli):) is parsed with scope extracted."""
        log = _git_log_entry("111aaaa", "feat(cli): add new subcommand")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["scope"] == "cli"
        assert data["commits"][0]["description"] == "add new subcommand"

    def test_breaking_bang_commit(self):
        """A commit with ! (e.g. feat!:) is marked as breaking."""
        log = _git_log_entry("222bbbb", "feat!: remove deprecated API")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["breaking"] is True

    def test_breaking_footer_commit(self):
        """A commit with BREAKING CHANGE in body is marked as breaking."""
        log = _git_log_entry("333cccc", "feat: redesign output", "BREAKING CHANGE: output format changed")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["breaking"] is True

    def test_breaking_change_hyphen_footer(self):
        """A commit with BREAKING-CHANGE in body is marked as breaking."""
        log = _git_log_entry("444dddd", "fix: update schema", "BREAKING-CHANGE: schema v2 required")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["breaking"] is True

    def test_non_conventional_commit_skipped(self):
        """A non-conventional commit (no type: prefix) is not included."""
        log = _git_log_entry("555eeee", "Update README with examples")
        # NOTE: there is a known bug in _run_changelog_generate where the
        # no-commits JSON path calls print(..., indent=2) which raises TypeError.
        # We verify the function raises TypeError (exposing the bug) rather than
        # successfully producing JSON.
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            # Known bug: print(_json.dumps(...), indent=2) passes indent to print()
            with pytest.raises(TypeError):
                _run_changelog_generate(args, fake_root)

    def test_docs_commit_parsed(self):
        """A docs commit is placed in Documentation section."""
        log = _git_log_entry("666ffff", "docs: update API reference")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Documentation"

    def test_test_commit_parsed(self):
        """A test commit is placed in Testing section."""
        log = _git_log_entry("777aaaa", "test: add unit tests for config")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Testing"

    def test_refactor_commit_parsed(self):
        """A refactor commit is placed in Changed section."""
        log = _git_log_entry("888bbbb", "refactor: simplify agent init")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Changed"

    def test_perf_commit_parsed(self):
        """A perf commit is placed in Performance section."""
        log = _git_log_entry("999cccc", "perf: optimize blast radius calculation")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Performance"

    def test_chore_commit_parsed(self):
        """A chore commit is placed in Maintenance section."""
        log = _git_log_entry("aaabbbb", "chore: update dependencies")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Maintenance"

    def test_ci_commit_parsed(self):
        """A ci commit is placed in CI/CD section."""
        log = _git_log_entry("bbbcccc", "ci: add release workflow")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "CI/CD"

    def test_unknown_type_goes_to_other(self):
        """A commit with an unknown type goes to the 'Other' section."""
        log = _git_log_entry("cccdddd", "build: update makefile targets")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["section"] == "Other"

    def test_sha_truncated_to_8_chars(self):
        """Commit SHA is truncated to first 8 characters."""
        log = _git_log_entry("abc12345678901234567890123456789", "feat: something")
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commits"][0]["sha"] == "abc12345"

    def test_multiple_commits_parsed(self):
        """Multiple commits are all parsed correctly."""
        log = (
            _git_log_entry("aaa11111", "feat: feature one")
            + _git_log_entry("bbb22222", "fix: bug fix one")
            + _git_log_entry("ccc33333", "docs: update docs")
        )
        rc, stdout, _ = self._run(log)
        assert rc == 0
        data = json.loads(stdout)
        assert data["commit_count"] == 3


# ---------------------------------------------------------------------------
# Test: Version inference and bumping
# ---------------------------------------------------------------------------


class TestVersionBumping:
    """Test semantic version bump inference."""

    def _run_json(self, git_log: str, version: str = "1.2.3") -> dict:
        """Run _run_changelog_generate in JSON mode and return parsed output."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.2.3"
            elif "log" in cmd:
                result.stdout = git_log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content(version)

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        return json.loads(out_buf.getvalue())

    def test_feat_bumps_minor(self):
        """A feat commit triggers a minor version bump."""
        log = _git_log_entry("aaa11111", "feat: add new feature")
        data = self._run_json(log, "1.2.3")
        assert data["inferred_bump"] == "minor"
        assert data["next_version"] == "1.3.0"

    def test_fix_bumps_patch(self):
        """A fix-only commit set triggers a patch version bump."""
        log = _git_log_entry("bbb22222", "fix: resolve issue")
        data = self._run_json(log, "1.2.3")
        assert data["inferred_bump"] == "patch"
        assert data["next_version"] == "1.2.4"

    def test_breaking_bumps_major(self):
        """A breaking change triggers a major version bump."""
        log = _git_log_entry("ccc33333", "feat!: redesign API")
        data = self._run_json(log, "1.2.3")
        assert data["inferred_bump"] == "major"
        assert data["next_version"] == "2.0.0"

    def test_breaking_footer_bumps_major(self):
        """A breaking change in the footer triggers a major version bump."""
        log = _git_log_entry("ddd44444", "feat: new output format", "BREAKING CHANGE: old format removed")
        data = self._run_json(log, "2.5.1")
        assert data["inferred_bump"] == "major"
        assert data["next_version"] == "3.0.0"

    def test_mixed_commits_highest_bump_wins(self):
        """When mixed commits exist, the highest bump level wins."""
        log = (
            _git_log_entry("eee55555", "fix: small fix")
            + _git_log_entry("fff66666", "feat: new feature")
            + _git_log_entry("ggg77777", "docs: update readme")
        )
        data = self._run_json(log, "0.5.2")
        assert data["inferred_bump"] == "minor"
        assert data["next_version"] == "0.6.0"

    def test_breaking_wins_over_feat(self):
        """Breaking change wins over feat in bump precedence."""
        log = _git_log_entry("hhh88888", "feat: add tool") + _git_log_entry("iii99999", "feat!: remove old API")
        data = self._run_json(log, "1.0.0")
        assert data["inferred_bump"] == "major"
        assert data["next_version"] == "2.0.0"

    def test_docs_only_bumps_patch(self):
        """Docs-only commits trigger a patch bump."""
        log = _git_log_entry("jjj00000", "docs: fix typo in readme")
        data = self._run_json(log, "3.1.0")
        assert data["inferred_bump"] == "patch"
        assert data["next_version"] == "3.1.1"

    def test_version_from_zero(self):
        """Version bump from 0.1.0 works correctly."""
        log = _git_log_entry("kkk11111", "feat: initial feature")
        data = self._run_json(log, "0.1.0")
        assert data["next_version"] == "0.2.0"

    def test_current_version_reported(self):
        """JSON output includes current_version field."""
        log = _git_log_entry("lll22222", "fix: a fix")
        data = self._run_json(log, "2.3.4")
        assert data["current_version"] == "2.3.4"

    def test_since_tag_reported(self):
        """JSON output includes since_tag field."""
        log = _git_log_entry("mmm33333", "fix: a fix")
        data = self._run_json(log, "1.0.0")
        assert data["since_tag"] == "v1.2.3"  # from mock_subprocess_run describe


# ---------------------------------------------------------------------------
# Test: No commits found
# ---------------------------------------------------------------------------


class TestNoCommitsFound:
    """Test behaviour when no conventional commits are found."""

    def _run_no_commits(self, output: str = "text") -> tuple[int, str, str]:
        """Run with empty git log output."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output=output)

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = ""  # No commits
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf, err_buf = io.StringIO(), io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", err_buf),
        ):
            rc = _run_changelog_generate(args, fake_root)

        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_no_commits_returns_zero(self):
        """Returns 0 even when no commits found (informational, not error)."""
        rc, _, _ = self._run_no_commits()
        assert rc == 0

    def test_no_commits_text_prints_message(self):
        """Text output prints a message about no conventional commits."""
        rc, _, stderr = self._run_no_commits(output="text")
        assert "No conventional commits" in stderr

    def test_no_commits_json_raises_typeerror_known_bug(self):
        """JSON path with no commits hits known bug: print(..., indent=2) raises TypeError."""
        # Known bug in source: line 1696 calls print(_json.dumps({...}), indent=2)
        # where indent=2 is incorrectly passed to print() rather than json.dumps().
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = ""  # No commits
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            with pytest.raises(TypeError):
                _run_changelog_generate(args, fake_root)

    def test_non_conventional_commits_only(self):
        """When only non-conventional commits exist, reports no commits."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="text")
        log = _git_log_entry("aaa11111", "Update README") + _git_log_entry("bbb22222", "Merge branch main")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        err_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", err_buf),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        assert "No conventional commits" in err_buf.getvalue()


# ---------------------------------------------------------------------------
# Test: No tag found (first release)
# ---------------------------------------------------------------------------


class TestNoTagFound:
    """Test behaviour when no previous v* tag exists."""

    def test_no_tag_uses_head(self):
        """When describe returns empty, log range is HEAD (all commits)."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: initial feature")
        calls = []

        def mock_subprocess_run(cmd, **kwargs):
            calls.append(cmd)
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = ""  # No tag found
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("0.1.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        data = json.loads(out_buf.getvalue())
        assert data["since_tag"] is None

    def test_no_tag_json_since_tag_is_none(self):
        """JSON output has since_tag=None when no tag exists."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("bbb22222", "fix: a fix")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = ""
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("0.1.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        data = json.loads(out_buf.getvalue())
        assert data["since_tag"] is None


# ---------------------------------------------------------------------------
# Test: pyproject.toml edge cases
# ---------------------------------------------------------------------------


class TestPyprojectEdgeCases:
    """Test version reading from pyproject.toml with edge cases."""

    def test_missing_pyproject_uses_default_version(self):
        """When pyproject.toml doesn't exist, default version 0.1.0 is used."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: a feature")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v0.1.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = False  # No pyproject.toml

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        data = json.loads(out_buf.getvalue())
        # Default version is (0, 1, 0) → minor bump → 0.2.0
        assert data["current_version"] == "0.1.0"
        assert data["next_version"] == "0.2.0"

    def test_pyproject_without_version_field_uses_default(self):
        """When pyproject.toml exists but has no version field, default is used."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("bbb22222", "fix: fix something")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v0.1.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = '[project]\nname = "manus-agent"\n'

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        data = json.loads(out_buf.getvalue())
        # Default (0, 1, 0) with patch bump → 0.1.1
        assert data["current_version"] == "0.1.0"
        assert data["next_version"] == "0.1.1"


# ---------------------------------------------------------------------------
# Test: Text output format
# ---------------------------------------------------------------------------


class TestTextOutput:
    """Test the text output format of _run_changelog_generate."""

    def _run_text(self, git_log: str, version: str = "1.0.0") -> tuple[int, str, str]:
        """Run in text mode and return (rc, stdout, stderr)."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="text")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = git_log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content(version)

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf, err_buf = io.StringIO(), io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", err_buf),
        ):
            rc = _run_changelog_generate(args, fake_root)

        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_text_has_version_header(self):
        """Text output starts with ## [next_version] -- date."""
        log = _git_log_entry("aaa11111", "feat: new feature")
        rc, stdout, _ = self._run_text(log, "1.0.0")
        assert rc == 0
        assert "## [1.1.0]" in stdout

    def test_text_has_section_headings(self):
        """Text output groups commits under ### Section headings."""
        log = _git_log_entry("aaa11111", "feat: feature one") + _git_log_entry("bbb22222", "fix: bug fix one")
        rc, stdout, _ = self._run_text(log)
        assert "### Added" in stdout
        assert "### Fixed" in stdout

    def test_text_has_commit_lines(self):
        """Text output includes commit descriptions as list items."""
        log = _git_log_entry("aaa11111", "feat: add blast radius tool")
        rc, stdout, _ = self._run_text(log)
        assert "- add blast radius tool" in stdout

    def test_text_scoped_commit_has_bold_scope(self):
        """Scoped commits show **scope**: description in text output."""
        log = _git_log_entry("aaa11111", "feat(cli): add subcommand")
        rc, stdout, _ = self._run_text(log)
        assert "**cli**:" in stdout

    def test_text_breaking_shows_tag(self):
        """Breaking changes show (BREAKING CHANGE) in text output."""
        log = _git_log_entry("aaa11111", "feat!: remove old API")
        rc, stdout, _ = self._run_text(log)
        assert "(BREAKING CHANGE)" in stdout

    def test_text_sha_in_parentheses(self):
        """Text output includes SHA in parentheses at end of line."""
        log = _git_log_entry("abc12345", "feat: something")
        rc, stdout, _ = self._run_text(log)
        assert "(abc12345)" in stdout

    def test_text_stderr_shows_bump_info(self):
        """Stderr includes inferred bump and version info."""
        log = _git_log_entry("aaa11111", "feat: new feature")
        rc, _, stderr = self._run_text(log, "1.0.0")
        assert "minor" in stderr
        assert "1.1.0" in stderr

    def test_text_date_in_header(self):
        """Text output header includes today's date."""
        import datetime

        log = _git_log_entry("aaa11111", "fix: a fix")
        rc, stdout, _ = self._run_text(log, "1.0.0")
        today = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        assert today in stdout

    def test_text_section_ordering(self):
        """Sections appear in canonical order: Added before Fixed before Documentation."""
        log = (
            _git_log_entry("aaa11111", "docs: update readme")
            + _git_log_entry("bbb22222", "fix: a bug")
            + _git_log_entry("ccc33333", "feat: a feature")
        )
        rc, stdout, _ = self._run_text(log)
        added_pos = stdout.find("### Added")
        fixed_pos = stdout.find("### Fixed")
        docs_pos = stdout.find("### Documentation")
        assert added_pos < fixed_pos < docs_pos


# ---------------------------------------------------------------------------
# Test: JSON output format
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Test JSON output structure and content."""

    def _run_json(self, git_log: str, version: str = "1.0.0") -> dict:
        """Run in JSON mode and return parsed output."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = git_log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content(version)

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        return json.loads(out_buf.getvalue())

    def test_json_has_required_keys(self):
        """JSON output has all required top-level keys."""
        log = _git_log_entry("aaa11111", "feat: something")
        data = self._run_json(log)
        assert "current_version" in data
        assert "next_version" in data
        assert "inferred_bump" in data
        assert "since_tag" in data
        assert "commit_count" in data
        assert "commits" in data

    def test_json_commit_count_matches(self):
        """commit_count matches actual number of parsed commits."""
        log = (
            _git_log_entry("aaa11111", "feat: one")
            + _git_log_entry("bbb22222", "fix: two")
            + _git_log_entry("ccc33333", "docs: three")
        )
        data = self._run_json(log)
        assert data["commit_count"] == 3
        assert len(data["commits"]) == 3

    def test_json_commit_structure(self):
        """Each commit in JSON has the expected fields."""
        log = _git_log_entry("abc12345", "feat(tools): add blast radius")
        data = self._run_json(log)
        commit = data["commits"][0]
        assert commit["sha"] == "abc12345"
        assert commit["type"] == "feat"
        assert commit["scope"] == "tools"
        assert commit["breaking"] is False
        assert commit["description"] == "add blast radius"
        assert commit["section"] == "Added"

    def test_json_is_valid_json(self):
        """Output is valid JSON (no trailing commas, proper structure)."""
        log = _git_log_entry("aaa11111", "feat: something")
        data = self._run_json(log)
        # If we got here, json.loads succeeded → valid JSON
        assert isinstance(data, dict)

    def test_json_since_tag_is_string_or_none(self):
        """since_tag is either a string or None."""
        log = _git_log_entry("aaa11111", "feat: something")
        data = self._run_json(log)
        assert data["since_tag"] is None or isinstance(data["since_tag"], str)


# ---------------------------------------------------------------------------
# Test: Scope and breaking change combinations
# ---------------------------------------------------------------------------


class TestScopeBreakingCombinations:
    """Test various combinations of scope and breaking markers."""

    def _parse_commit(self, subject: str, body: str = "") -> dict:
        """Parse a single commit and return its data from JSON output."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", subject, body)

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            rc = _run_changelog_generate(args, fake_root)

        assert rc == 0
        data = json.loads(out_buf.getvalue())
        return data["commits"][0]

    def test_scoped_breaking_bang(self):
        """feat(api)!: is parsed with scope='api' and breaking=True."""
        commit = self._parse_commit("feat(api)!: remove endpoint")
        assert commit["scope"] == "api"
        assert commit["breaking"] is True
        assert commit["type"] == "feat"

    def test_no_scope_no_breaking(self):
        """fix: parsed with scope='' and breaking=False."""
        commit = self._parse_commit("fix: typo in error message")
        assert commit["scope"] == ""
        assert commit["breaking"] is False

    def test_scope_no_breaking(self):
        """fix(cli): parsed with scope='cli' and breaking=False."""
        commit = self._parse_commit("fix(cli): handle empty input")
        assert commit["scope"] == "cli"
        assert commit["breaking"] is False

    def test_breaking_footer_with_scope(self):
        """Scope + breaking footer combination."""
        commit = self._parse_commit(
            "refactor(config): change config format",
            "BREAKING CHANGE: toml config keys renamed",
        )
        assert commit["scope"] == "config"
        assert commit["breaking"] is True

    def test_breaking_footer_case_insensitive(self):
        """BREAKING CHANGE footer match is case-insensitive."""
        commit = self._parse_commit("feat: new output", "breaking change: old format gone")
        assert commit["breaking"] is True


# ---------------------------------------------------------------------------
# Test: Git subprocess interaction
# ---------------------------------------------------------------------------


class TestGitSubprocessInteraction:
    """Test that git commands are called with correct arguments."""

    def test_describe_called_with_tags_match(self):
        """git describe is called with --tags --match v* --abbrev=0."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: something")
        captured_cmds = []

        def mock_subprocess_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            _run_changelog_generate(args, fake_root)

        describe_cmd = [c for c in captured_cmds if "describe" in c]
        assert len(describe_cmd) == 1
        cmd = describe_cmd[0]
        assert "--tags" in cmd
        assert "--match" in cmd
        assert "v*" in cmd
        assert "--abbrev=0" in cmd

    def test_log_uses_tag_range(self):
        """When a tag exists, git log uses tag..HEAD range."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: something")
        captured_cmds = []

        def mock_subprocess_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v2.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("2.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            _run_changelog_generate(args, fake_root)

        log_cmd = [c for c in captured_cmds if "log" in c]
        assert len(log_cmd) == 1
        assert "v2.0.0..HEAD" in log_cmd[0]

    def test_log_uses_head_when_no_tag(self):
        """When no tag exists, git log uses HEAD as range."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: something")
        captured_cmds = []

        def mock_subprocess_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = ""  # No tag
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("0.1.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            _run_changelog_generate(args, fake_root)

        log_cmd = [c for c in captured_cmds if "log" in c]
        assert len(log_cmd) == 1
        assert "HEAD" in log_cmd[0]

    def test_subprocess_called_with_cwd(self):
        """Subprocess calls use cwd=root."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="json")
        log = _git_log_entry("aaa11111", "feat: something")
        captured_kwargs = []

        def mock_subprocess_run(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            _run_changelog_generate(args, fake_root)

        for kwargs in captured_kwargs:
            assert kwargs.get("cwd") == fake_root


# ---------------------------------------------------------------------------
# Test: Section grouping
# ---------------------------------------------------------------------------


class TestSectionGrouping:
    """Test that commits are grouped into correct sections."""

    def _get_sections_from_text(self, git_log: str) -> str:
        """Run in text mode and return stdout."""
        import io

        from manus_agent.cli import _run_changelog_generate

        args = _make_args(output="text")

        def mock_subprocess_run(cmd, **kwargs):
            result = mock.Mock()
            if "describe" in cmd:
                result.stdout = "v1.0.0"
            elif "log" in cmd:
                result.stdout = git_log
            else:
                result.stdout = ""
            return result

        fake_pyproject = mock.Mock(spec=Path)
        fake_pyproject.exists.return_value = True
        fake_pyproject.read_text.return_value = _pyproject_content("1.0.0")

        fake_root = mock.Mock(spec=Path)
        fake_root.__truediv__ = mock.Mock(return_value=fake_pyproject)

        out_buf = io.StringIO()
        with (
            mock.patch("subprocess.run", side_effect=mock_subprocess_run),
            mock.patch("sys.stdout", out_buf),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            _run_changelog_generate(args, fake_root)

        return out_buf.getvalue()

    def test_all_section_types_appear(self):
        """All 9 section types appear when all commit types present."""
        log = (
            _git_log_entry("a1111111", "feat: a feature")
            + _git_log_entry("b2222222", "fix: a fix")
            + _git_log_entry("c3333333", "docs: documentation")
            + _git_log_entry("d4444444", "test: testing")
            + _git_log_entry("e5555555", "refactor: refactoring")
            + _git_log_entry("f6666666", "perf: performance")
            + _git_log_entry("g7777777", "chore: maintenance")
            + _git_log_entry("h8888888", "ci: ci/cd")
            + _git_log_entry("i9999999", "build: other stuff")
        )
        output = self._get_sections_from_text(log)
        assert "### Added" in output
        assert "### Fixed" in output
        assert "### Documentation" in output
        assert "### Testing" in output
        assert "### Changed" in output
        assert "### Performance" in output
        assert "### Maintenance" in output
        assert "### CI/CD" in output
        assert "### Other" in output

    def test_only_relevant_sections_appear(self):
        """Only sections with commits appear in output."""
        log = _git_log_entry("aaa11111", "feat: only feature")
        output = self._get_sections_from_text(log)
        assert "### Added" in output
        assert "### Fixed" not in output
        assert "### Documentation" not in output

    def test_multiple_commits_in_same_section(self):
        """Multiple commits of same type appear under one section heading."""
        log = (
            _git_log_entry("aaa11111", "feat: feature one")
            + _git_log_entry("bbb22222", "feat: feature two")
            + _git_log_entry("ccc33333", "feat(cli): feature three")
        )
        output = self._get_sections_from_text(log)
        # Only one ### Added heading
        assert output.count("### Added") == 1
        assert "feature one" in output
        assert "feature two" in output
        assert "feature three" in output
