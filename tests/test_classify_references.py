"""Comprehensive test suite for classify_references module.

Tests cover:
- URL classification logic (path patterns, domain rules, NVD tags, fallbacks)
- CVE reference fetching and classification pipeline
- Strands tool interface (valid/invalid input, error handling)
- CLI subcommand (parser, text/json output, category filtering, error paths)
- Edge cases (empty refs, unknown domains, mixed confidence)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.classify_references import (
    CATEGORY_ADVISORY,
    CATEGORY_EXPLOIT,
    CATEGORY_ISSUE_TRACKER,
    CATEGORY_MAILING_LIST,
    CATEGORY_MEDIA,
    CATEGORY_OTHER,
    CATEGORY_PATCH,
    CATEGORY_VENDOR_NOTICE,
    classify_cve_references,
    classify_references,
    classify_url,
)

# ===========================================================================
# classify_url — Path pattern rules (highest priority)
# ===========================================================================


class TestClassifyUrlPathPatterns:
    """Test URL classification via path-based pattern matching."""

    def test_github_commit_classified_as_patch(self):
        result = classify_url("https://github.com/apache/httpd/commit/abc123def456")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"
        assert result["source"] == "github.com"

    def test_github_pull_request_classified_as_patch(self):
        result = classify_url("https://github.com/torvalds/linux/pull/42")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"

    def test_github_release_tag_classified_as_patch(self):
        result = classify_url("https://github.com/nodejs/node/releases/tag/v20.11.1")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"

    def test_gitlab_commit_classified_as_patch(self):
        result = classify_url("https://gitlab.com/gnome/glib/-/commit/abc123")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"

    def test_github_security_advisory_classified_as_advisory(self):
        result = classify_url("https://github.com/advisories/GHSA-abcd-1234-efgh")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_github_repo_security_advisory_classified_as_advisory(self):
        result = classify_url("https://github.com/django/django/security/advisories/GHSA-xxxx-yyyy-zzzz")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_github_issue_classified_as_issue_tracker(self):
        result = classify_url("https://github.com/curl/curl/issues/12345")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "high"

    def test_bugzilla_classified_as_issue_tracker(self):
        result = classify_url("https://bugzilla.mozilla.org/show_bug.cgi?id=123456")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "high"

    def test_redhat_bugzilla_classified_as_issue_tracker(self):
        result = classify_url("https://bugzilla.redhat.com/show_bug.cgi?id=2222222")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "high"

    def test_jira_classified_as_issue_tracker(self):
        result = classify_url("https://jira.atlassian.com/browse/PROJ-1234")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "high"

    def test_exploitdb_path_classified_as_exploit(self):
        result = classify_url("https://www.exploit-db.com/exploits/51234")
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "high"

    def test_packetstorm_path_classified_as_exploit(self):
        result = classify_url("https://packetstormsecurity.com/files/176543/vuln.txt")
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "high"

    def test_debian_dsa_classified_as_advisory(self):
        result = classify_url("https://www.debian.org/security/2024/dsa-5678")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_debian_dla_classified_as_advisory(self):
        result = classify_url("https://www.debian.org/security/2024/dla-1234")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"


# ===========================================================================
# classify_url — Domain rules (second priority)
# ===========================================================================


class TestClassifyUrlDomainRules:
    """Test URL classification via domain-based heuristics."""

    def test_exploitdb_domain_classified_as_exploit(self):
        result = classify_url("https://www.exploit-db.com/search?q=test")
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "high"

    def test_packetstorm_domain_classified_as_exploit(self):
        result = classify_url("https://packetstormsecurity.com/news/")
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "high"

    def test_vuldb_classified_as_exploit(self):
        result = classify_url("https://vuldb.com/?id.270000")
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "high"

    def test_seclists_classified_as_mailing_list(self):
        result = classify_url("https://seclists.org/fulldisclosure/2024/Jan/1")
        assert result["category"] == CATEGORY_MAILING_LIST
        assert result["confidence"] == "high"

    def test_openwall_lists_classified_as_mailing_list(self):
        result = classify_url("https://www.openwall.com/lists/oss-security/2024/01/01/1")
        assert result["category"] == CATEGORY_MAILING_LIST
        assert result["confidence"] == "high"

    def test_gentoo_security_classified_as_advisory(self):
        result = classify_url("https://security.gentoo.org/glsa/202401-01")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_ubuntu_usn_classified_as_advisory(self):
        result = classify_url("https://ubuntu.com/security/notices/USN-6543-1")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_redhat_errata_classified_as_advisory(self):
        result = classify_url("https://access.redhat.com/errata/RHSA-2024:0001")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_redhat_security_classified_as_advisory(self):
        result = classify_url("https://access.redhat.com/security/cve/CVE-2024-0001")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_debian_security_tracker_classified_as_advisory(self):
        result = classify_url("https://security-tracker.debian.org/tracker/CVE-2024-0001")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_cert_org_classified_as_advisory(self):
        result = classify_url("https://www.cert.org/advisories/2024-001")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_cisa_gov_classified_as_advisory(self):
        result = classify_url("https://www.cisa.gov/known-exploited-vulnerabilities")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "high"

    def test_hackernews_classified_as_media(self):
        result = classify_url("https://thehackernews.com/2024/01/critical-vuln.html")
        assert result["category"] == CATEGORY_MEDIA
        assert result["confidence"] == "high"

    def test_bleepingcomputer_classified_as_media(self):
        result = classify_url("https://www.bleepingcomputer.com/news/security/vuln/")
        assert result["category"] == CATEGORY_MEDIA
        assert result["confidence"] == "high"

    def test_lists_debian_classified_as_mailing_list(self):
        result = classify_url("https://lists.debian.org/debian-security-announce/2024/")
        assert result["category"] == CATEGORY_MAILING_LIST
        assert result["confidence"] == "high"

    def test_lists_fedora_classified_as_mailing_list(self):
        result = classify_url("https://lists.fedoraproject.org/archives/list/")
        assert result["category"] == CATEGORY_MAILING_LIST
        assert result["confidence"] == "high"


# ===========================================================================
# classify_url — NVD tag-based classification (medium confidence)
# ===========================================================================


class TestClassifyUrlNvdTags:
    """Test URL classification via NVD-provided tags."""

    def test_patch_tag_classifies_as_patch(self):
        result = classify_url(
            "https://unknown-vendor.com/fix/v1.2.3",
            nvd_tags=["Patch"],
        )
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "medium"

    def test_exploit_tag_classifies_as_exploit(self):
        result = classify_url(
            "https://some-site.com/poc/demo",
            nvd_tags=["Exploit"],
        )
        assert result["category"] == CATEGORY_EXPLOIT
        assert result["confidence"] == "medium"

    def test_third_party_advisory_tag(self):
        result = classify_url(
            "https://unknown-advisory-site.com/vuln/123",
            nvd_tags=["Third Party Advisory"],
        )
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "medium"

    def test_vendor_advisory_tag(self):
        result = classify_url(
            "https://vendor.com/security/notice",
            nvd_tags=["Vendor Advisory"],
        )
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "medium"

    def test_mailing_list_tag(self):
        result = classify_url(
            "https://unknown-list.org/archive/msg001",
            nvd_tags=["Mailing List"],
        )
        assert result["category"] == CATEGORY_MAILING_LIST
        assert result["confidence"] == "medium"

    def test_issue_tracking_tag(self):
        result = classify_url(
            "https://tracker.vendor.com/issue/999",
            nvd_tags=["Issue Tracking"],
        )
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "medium"

    def test_release_notes_tag(self):
        result = classify_url(
            "https://vendor.com/docs/releases",
            nvd_tags=["Release Notes"],
        )
        assert result["category"] == CATEGORY_VENDOR_NOTICE
        assert result["confidence"] == "medium"

    def test_vdb_entry_tag_classifies_as_other(self):
        result = classify_url(
            "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
            nvd_tags=["VDB Entry"],
        )
        assert result["category"] == CATEGORY_OTHER
        assert result["confidence"] == "medium"

    def test_first_matching_tag_wins(self):
        """When multiple tags present, first matching one determines category."""
        result = classify_url(
            "https://unknown.com/some/path",
            nvd_tags=["Patch", "Exploit"],
        )
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "medium"

    def test_path_pattern_takes_priority_over_tags(self):
        """Path patterns should override NVD tags."""
        result = classify_url(
            "https://github.com/org/repo/commit/abc123",
            nvd_tags=["Exploit"],  # Wrong tag, but path wins
        )
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"


# ===========================================================================
# classify_url — Fallback heuristics (low confidence)
# ===========================================================================


class TestClassifyUrlFallbacks:
    """Test low-confidence fallback classification."""

    def test_commit_in_path_classified_as_patch(self):
        result = classify_url("https://git.vendor.com/project/commit/abc123")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "low"

    def test_releases_in_path_classified_as_patch(self):
        result = classify_url("https://vendor.com/releases/v2.0.0")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "low"

    def test_changelog_in_path_classified_as_patch(self):
        result = classify_url("https://vendor.com/changelog")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "low"

    def test_advisory_in_path_classified_as_advisory(self):
        result = classify_url("https://vendor.com/advisory/2024-001")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "low"

    def test_security_in_path_classified_as_advisory(self):
        result = classify_url("https://vendor.com/security/notice")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "low"

    def test_issues_in_path_classified_as_issue_tracker(self):
        result = classify_url("https://tracker.vendor.com/issues/1234")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "low"

    def test_bug_in_path_classified_as_issue_tracker(self):
        result = classify_url("https://tracker.vendor.com/bug/5678")
        assert result["category"] == CATEGORY_ISSUE_TRACKER
        assert result["confidence"] == "low"

    def test_unknown_url_classified_as_other(self):
        result = classify_url("https://completely-random-site.com/page.html")
        assert result["category"] == CATEGORY_OTHER
        assert result["confidence"] == "low"

    def test_empty_tags_dont_affect_fallback(self):
        result = classify_url("https://random.com/page", nvd_tags=[])
        assert result["category"] == CATEGORY_OTHER


# ===========================================================================
# classify_url — General behavior
# ===========================================================================


class TestClassifyUrlGeneral:
    """Test general classify_url behavior."""

    def test_result_structure(self):
        result = classify_url("https://example.com/test")
        assert "url" in result
        assert "category" in result
        assert "source" in result
        assert "confidence" in result
        assert "nvd_tags" in result

    def test_url_preserved_in_result(self):
        url = "https://github.com/org/repo/commit/abc"
        result = classify_url(url)
        assert result["url"] == url

    def test_nvd_tags_preserved_in_result(self):
        tags = ["Patch", "Third Party Advisory"]
        result = classify_url("https://example.com", nvd_tags=tags)
        assert result["nvd_tags"] == tags

    def test_none_nvd_tags_becomes_empty_list(self):
        result = classify_url("https://example.com", nvd_tags=None)
        assert result["nvd_tags"] == []

    def test_www_prefix_stripped_from_source(self):
        result = classify_url("https://www.example.com/page")
        assert result["source"] == "example.com"

    def test_source_extracted_correctly(self):
        result = classify_url("https://security-tracker.debian.org/tracker/CVE-2024-0001")
        assert result["source"] == "security-tracker.debian.org"


# ===========================================================================
# classify_cve_references — Integration tests (mocked HTTP)
# ===========================================================================


class TestClassifyCveReferences:
    """Test the full CVE reference classification pipeline."""

    def _mock_nvd_response(self, references):
        """Build a mock NVD API response with given references."""
        return {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-1234",
                        "references": references,
                    }
                }
            ]
        }

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_basic_classification_pipeline(self, mock_get):
        refs = [
            {"url": "https://github.com/org/repo/commit/abc123", "tags": ["Patch"]},
            {"url": "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz", "tags": ["Third Party Advisory"]},
            {"url": "https://www.exploit-db.com/exploits/51234", "tags": ["Exploit"]},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response(refs)
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-1234")

        assert result["cve_id"] == "CVE-2024-1234"
        assert result["total"] == 3
        assert len(result["references"]) == 3
        assert result["summary"]["patch"] == 1
        assert result["summary"]["advisory"] == 1
        assert result["summary"]["exploit"] == 1

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_actionable_references_extracted(self, mock_get):
        refs = [
            {"url": "https://github.com/org/repo/commit/abc123", "tags": ["Patch"]},
            {"url": "https://random.com/page", "tags": []},
            {"url": "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz", "tags": []},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response(refs)
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-1234")

        # actionable = high/medium confidence patches + advisories
        assert len(result["actionable"]) == 2
        categories = {r["category"] for r in result["actionable"]}
        assert CATEGORY_PATCH in categories
        assert CATEGORY_ADVISORY in categories

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_empty_references(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response([])
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-0000")

        assert result["total"] == 0
        assert result["references"] == []
        assert result["summary"] == {}
        assert result["actionable"] == []

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_cve_id_uppercased(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response([])
        mock_get.return_value = mock_resp

        result = classify_cve_references("cve-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_no_vulnerabilities_returns_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-9999")
        assert "error" in result
        assert "No NVD data found" in result["error"]

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_network_error_returns_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = classify_cve_references("CVE-2024-1234")
        assert "error" in result
        assert "NVD API request failed" in result["error"]

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_refs_with_empty_url_skipped(self, mock_get):
        refs = [
            {"url": "", "tags": ["Patch"]},
            {"url": "https://github.com/org/repo/commit/abc", "tags": []},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response(refs)
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-1234")
        assert result["total"] == 1

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_summary_counts_all_categories(self, mock_get):
        refs = [
            {"url": "https://github.com/org/repo/commit/a1", "tags": []},
            {"url": "https://github.com/org/repo/commit/a2", "tags": []},
            {"url": "https://seclists.org/post/1", "tags": []},
            {"url": "https://random.com/page", "tags": []},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_nvd_response(refs)
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-1234")
        assert result["summary"]["patch"] == 2
        assert result["summary"]["mailing_list"] == 1
        assert result["summary"]["other"] == 1


# ===========================================================================
# Strands tool interface
# ===========================================================================


class TestClassifyReferencesToolInterface:
    """Test the Strands tool entry point."""

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_valid_cve_returns_success(self, mock_classify):
        mock_classify.return_value = {
            "cve_id": "CVE-2024-1234",
            "total": 2,
            "references": [
                {
                    "url": "https://example.com/commit/abc",
                    "category": "patch",
                    "source": "example.com",
                    "confidence": "high",
                    "nvd_tags": [],
                },
            ],
            "summary": {"patch": 1},
            "actionable": [],
        }

        tool_use = {"toolUseId": "test-123", "input": {"cve_id": "CVE-2024-1234"}}
        result = classify_references(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert result["content"][0]["json"]["cve_id"] == "CVE-2024-1234"

    def test_invalid_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-456", "input": {"cve_id": "not-a-cve"}}
        result = classify_references(tool_use)

        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-789", "input": {"cve_id": ""}}
        result = classify_references(tool_use)

        assert result["status"] == "error"

    def test_missing_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-000", "input": {}}
        result = classify_references(tool_use)

        assert result["status"] == "error"

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_api_error_returns_error_status(self, mock_classify):
        mock_classify.return_value = {"error": "NVD API request failed: timeout", "cve_id": "CVE-2024-1234"}

        tool_use = {"toolUseId": "test-err", "input": {"cve_id": "CVE-2024-1234"}}
        result = classify_references(tool_use)

        assert result["status"] == "error"
        assert "NVD API request failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_lowercase_cve_accepted(self, mock_classify):
        mock_classify.return_value = {
            "cve_id": "CVE-2024-5678",
            "total": 0,
            "references": [],
            "summary": {},
            "actionable": [],
        }

        tool_use = {"toolUseId": "test-lc", "input": {"cve_id": "cve-2024-5678"}}
        result = classify_references(tool_use)

        assert result["status"] == "success"


# ===========================================================================
# CLI subcommand tests
# ===========================================================================


class TestClassifyRefsCli:
    """Test the classify-refs CLI subcommand."""

    def test_classify_refs_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_classify_refs_parser

        parser = _build_classify_refs_parser()
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.cve_id == "CVE-2024-1234"
        assert args.output == "text"
        assert args.category is None

    def test_classify_refs_parser_output_json(self):
        from manus_agent.cli import _build_classify_refs_parser

        parser = _build_classify_refs_parser()
        args = parser.parse_args(["CVE-2024-1234", "--output", "json"])
        assert args.output == "json"

    def test_classify_refs_parser_category_filter(self):
        from manus_agent.cli import _build_classify_refs_parser

        parser = _build_classify_refs_parser()
        args = parser.parse_args(["CVE-2024-1234", "--category", "patch"])
        assert args.category == "patch"

    def test_classify_refs_parser_rejects_missing_cve_id(self):
        from manus_agent.cli import _build_classify_refs_parser

        parser = _build_classify_refs_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_classify_refs_parser_rejects_invalid_output(self):
        from manus_agent.cli import _build_classify_refs_parser

        parser = _build_classify_refs_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2024-1234", "--output", "xml"])

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_run_classify_refs_success_text(self, mock_classify):
        from manus_agent.cli import _run_classify_refs

        mock_classify.return_value = {
            "cve_id": "CVE-2024-1234",
            "total": 2,
            "references": [
                {
                    "url": "https://github.com/org/repo/commit/abc",
                    "category": "patch",
                    "source": "github.com",
                    "confidence": "high",
                    "nvd_tags": ["Patch"],
                },
                {
                    "url": "https://random.com/page",
                    "category": "other",
                    "source": "random.com",
                    "confidence": "low",
                    "nvd_tags": [],
                },
            ],
            "summary": {"patch": 1, "other": 1},
            "actionable": [
                {
                    "url": "https://github.com/org/repo/commit/abc",
                    "category": "patch",
                    "source": "github.com",
                    "confidence": "high",
                    "nvd_tags": ["Patch"],
                },
            ],
        }

        rc = _run_classify_refs(["CVE-2024-1234"])
        assert rc == 0

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_run_classify_refs_success_json(self, mock_classify):
        from manus_agent.cli import _run_classify_refs

        mock_classify.return_value = {
            "cve_id": "CVE-2024-1234",
            "total": 1,
            "references": [
                {
                    "url": "https://github.com/org/repo/commit/abc",
                    "category": "patch",
                    "source": "github.com",
                    "confidence": "high",
                    "nvd_tags": [],
                },
            ],
            "summary": {"patch": 1},
            "actionable": [],
        }

        rc = _run_classify_refs(["CVE-2024-1234", "--output", "json"])
        assert rc == 0

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_run_classify_refs_category_filter(self, mock_classify):
        from manus_agent.cli import _run_classify_refs

        mock_classify.return_value = {
            "cve_id": "CVE-2024-1234",
            "total": 3,
            "references": [
                {
                    "url": "https://github.com/org/repo/commit/abc",
                    "category": "patch",
                    "source": "github.com",
                    "confidence": "high",
                    "nvd_tags": [],
                },
                {
                    "url": "https://random.com/page",
                    "category": "other",
                    "source": "random.com",
                    "confidence": "low",
                    "nvd_tags": [],
                },
                {
                    "url": "https://seclists.org/post/1",
                    "category": "mailing_list",
                    "source": "seclists.org",
                    "confidence": "high",
                    "nvd_tags": [],
                },
            ],
            "summary": {"patch": 1, "other": 1, "mailing_list": 1},
            "actionable": [],
        }

        # With category filter, only matching references are shown
        rc = _run_classify_refs(["CVE-2024-1234", "--category", "patch"])
        assert rc == 0

    @patch("manus_agent.tools.classify_references.classify_cve_references")
    def test_run_classify_refs_error_from_api(self, mock_classify):
        from manus_agent.cli import _run_classify_refs

        mock_classify.return_value = {"error": "NVD API request failed: timeout", "cve_id": "CVE-2024-1234"}

        rc = _run_classify_refs(["CVE-2024-1234"])
        assert rc == 1

    def test_run_classify_refs_invalid_cve_format(self):
        from manus_agent.cli import _run_classify_refs

        rc = _run_classify_refs(["NOT-A-CVE"])
        assert rc == 1

    def test_classify_refs_registered_in_main(self):
        """classify-refs dispatches to _run_classify_refs in main()."""
        # Verify the subcommand is recognized by checking source
        import inspect

        from manus_agent import cli

        source = inspect.getsource(cli.main)
        assert "classify-refs" in source

    def test_classify_refs_dispatch_calls_run(self):
        """main() dispatches classify-refs to _run_classify_refs."""
        from manus_agent import cli

        with patch.object(cli, "_run_classify_refs", return_value=0) as mock_run:
            with patch("sys.argv", ["manus-agent", "classify-refs", "CVE-2024-1234"]):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
            assert exc_info.value.code == 0
            mock_run.assert_called_once_with(["CVE-2024-1234"])


# ===========================================================================
# Edge cases
# ===========================================================================


class TestClassifyReferencesEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_url_with_query_params(self):
        result = classify_url("https://github.com/org/repo/commit/abc?w=1")
        assert result["category"] == CATEGORY_PATCH

    def test_url_with_fragment(self):
        result = classify_url("https://github.com/org/repo/commit/abc#diff")
        assert result["category"] == CATEGORY_PATCH

    def test_url_with_port(self):
        result = classify_url("https://bugzilla.vendor.com:8080/show_bug.cgi?id=123")
        assert result["category"] == CATEGORY_ISSUE_TRACKER

    def test_ftp_url_classified_as_other(self):
        result = classify_url("ftp://ftp.vendor.com/patches/fix.tar.gz")
        # Path has no matching patterns, classified as patch via /patches/ → no, that's not in fallback
        # Actually /patches/ is not in the fallback keywords. So it should be other.
        assert result["category"] == CATEGORY_OTHER

    def test_multiple_categories_path_wins_over_domain(self):
        """GitHub commit should be patch even though github.com has no domain rule."""
        result = classify_url("https://github.com/org/repo/commit/abc123")
        assert result["category"] == CATEGORY_PATCH
        assert result["confidence"] == "high"

    def test_very_long_url(self):
        long_path = "/a" * 500
        result = classify_url(f"https://example.com{long_path}")
        assert result["category"] == CATEGORY_OTHER

    def test_url_with_unicode(self):
        result = classify_url("https://example.com/advisory/日本語")
        assert result["category"] == CATEGORY_ADVISORY
        assert result["confidence"] == "low"

    @patch("manus_agent.tools.classify_references._nvd_get_with_retry")
    def test_many_references_all_classified(self, mock_get):
        """All references get classified even with many entries."""
        refs = [{"url": f"https://example.com/ref/{i}", "tags": []} for i in range(50)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cve": {"id": "CVE-2024-1234", "references": refs}}]}
        mock_get.return_value = mock_resp

        result = classify_cve_references("CVE-2024-1234")
        assert result["total"] == 50
        assert len(result["references"]) == 50
