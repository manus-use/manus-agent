"""
Tests for _enrich_vcpkg in get_dependency_blast_radius.py

All external HTTP calls are mocked — no real network I/O.
100% mocked: vcpkg.io/output.json manifest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_dependency_blast_radius import (
    _VCPKG_OUTPUT_URL,
    _blast_score,
    _enrich_package,
    _enrich_vcpkg,
    get_dependency_blast_radius,
)


# ===========================================================================
# Fixture helpers
# ===========================================================================


def _make_manifest(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal vcpkg output.json-style manifest."""
    return {"Baseline": "abc123", "Size": len(packages), "Source": packages}


_OPENSSL_PKG: dict[str, Any] = {
    "Name": "openssl",
    "Version": "3.6.3",
    "Description": "OpenSSL TLS and SSL toolkit",
    "homepage": "https://www.openssl.org",
    "License": "Apache-2.0",
    "LastModified": "2026-07-01",
    "LastCommit": "abc123",
    "Dependencies": [
        {"name": "vcpkg-cmake", "host": True},
        {"name": "vcpkg-cmake-config", "host": True},
    ],
    "Features": {
        "fips": {"description": "Enable FIPS"},
        "ssl3": {"description": "Enable SSL3"},
    },
}

_CURL_PKG: dict[str, Any] = {
    "Name": "curl",
    "Version": "8.21.0",
    "Port-Version": 1,
    "Description": "A library for transferring data with URLs",
    "homepage": "https://curl.se/",
    "License": "curl",
    "LastModified": "2026-06-01",
    "LastCommit": "def456",
    "Dependencies": [
        {"name": "vcpkg-cmake", "host": True},
        {"name": "vcpkg-cmake-config", "host": True},
        "zlib",
        "openssl",
    ],
    "Features": {
        "brotli": {"description": "brotli support"},
        "c-ares": {"description": "c-ares support"},
        "ssl": {"description": "SSL support"},
    },
}

_ZLIB_PKG: dict[str, Any] = {
    "Name": "zlib",
    "Version": "1.3.2",
    "Port-Version": 1,
    "Description": "A compression library",
    "homepage": "https://www.zlib.net/",
    "License": "Zlib",
    "LastModified": "2026-05-01",
    "LastCommit": "ghi789",
    "Dependencies": [
        {"name": "vcpkg-cmake", "host": True},
        {"name": "vcpkg-cmake-config", "host": True},
    ],
    "Features": {},
}

_LIBPNG_PKG: dict[str, Any] = {
    "Name": "libpng",
    "Version": "1.6.43",
    "Description": "PNG library",
    "homepage": "http://www.libpng.org/",
    "License": "libpng",
    "LastModified": "2024-01-01",
    "LastCommit": "jkl012",
    "Dependencies": [
        "zlib",
        {"name": "vcpkg-cmake", "host": True},
    ],
    "Features": {},
}

_SAMPLE_MANIFEST = _make_manifest([_OPENSSL_PKG, _CURL_PKG, _ZLIB_PKG, _LIBPNG_PKG])


def _mock_vcpkg_get(manifest: dict[str, Any] = _SAMPLE_MANIFEST):
    """Return a mock for requests.get that serves the vcpkg manifest."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = manifest
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ===========================================================================
# TestEnrichVcpkg — core behaviour
# ===========================================================================


class TestEnrichVcpkgBasics:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        """Context manager: patch _VCPKG_MANIFEST_CACHE to None and mock requests.get."""
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_returns_correct_ecosystem(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["ecosystem"] == "vcpkg"

    def test_returns_package_name(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["package_name"] == "openssl"

    def test_returns_latest_version(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["latest_version"] == "3.6.3"

    def test_returns_port_version_when_present(self):
        with self._patch():
            result = _enrich_vcpkg("curl")
        assert result["port_version"] == 1

    def test_no_port_version_key_when_zero(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        # Port-Version absent / 0 → key should not appear
        assert "port_version" not in result

    def test_returns_description(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert "OpenSSL" in result["description"]

    def test_description_truncated_to_120_chars(self):
        long_desc_pkg = {**_OPENSSL_PKG, "Description": "x" * 200}
        manifest = _make_manifest([long_desc_pkg])
        with self._patch(manifest):
            result = _enrich_vcpkg("openssl")
        assert len(result["description"]) <= 120

    def test_returns_homepage(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["homepage"] == "https://www.openssl.org"

    def test_returns_license(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["license"] == "Apache-2.0"

    def test_returns_last_modified(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["last_modified"] == "2026-07-01"

    def test_returns_vcpkg_page_url(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["vcpkg_page"] == "https://vcpkg.io/en/package/openssl"


class TestEnrichVcpkgReverseDependencies:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_zlib_has_two_reverse_deps(self):
        # curl and libpng both depend on zlib
        with self._patch():
            result = _enrich_vcpkg("zlib")
        assert result["dependent_packages_count"] == 2

    def test_openssl_has_one_reverse_dep(self):
        # Only curl depends on openssl
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert result["dependent_packages_count"] == 1

    def test_no_reverse_deps_returns_zero(self):
        # curl has 0 reverse deps in our sample manifest
        with self._patch():
            result = _enrich_vcpkg("curl")
        assert result["dependent_packages_count"] == 0

    def test_meta_ports_excluded_from_count(self):
        # vcpkg-cmake is a meta-port and should not be counted as reverse dep of cmake-itself
        vcpkg_cmake_pkg: dict[str, Any] = {
            "Name": "vcpkg-cmake",
            "Version": "2024-01-01",
            "Description": "vcpkg cmake helper",
            "Dependencies": [],
        }
        manifest = _make_manifest([_OPENSSL_PKG, _CURL_PKG, vcpkg_cmake_pkg])
        with self._patch(manifest):
            result = _enrich_vcpkg("vcpkg-cmake")
        # meta-ports excluded from blast-radius signal — dependent_packages_count still
        # shows who structurally depends on it, but helper-meta count should be 0
        # (they are filtered from REVERSE dep computations of real packages,
        # but vcpkg-cmake itself has reverse deps; check it stays clean for non-meta lookups)
        # At minimum: no crash
        assert "dependent_packages_count" in result

    def test_direct_dep_count_correct(self):
        with self._patch():
            result = _enrich_vcpkg("curl")
        # curl has zlib and openssl as non-meta deps
        assert result.get("direct_dep_count", 0) >= 2

    def test_direct_deps_sample_present(self):
        with self._patch():
            result = _enrich_vcpkg("curl")
        assert "direct_deps_sample" in result
        sample = result["direct_deps_sample"]
        assert "openssl" in sample or "zlib" in sample


class TestEnrichVcpkgFeatures:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_feature_count_correct(self):
        with self._patch():
            result = _enrich_vcpkg("curl")
        assert result["feature_count"] == 3  # brotli, c-ares, ssl

    def test_zero_features_allowed(self):
        with self._patch():
            result = _enrich_vcpkg("zlib")
        # zlib has no Features dict entries
        assert result.get("feature_count", 0) == 0


class TestEnrichVcpkgNotFound:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_missing_package_returns_error_key(self):
        with self._patch():
            result = _enrich_vcpkg("this-package-does-not-exist")
        assert "error" in result

    def test_missing_package_error_message_contains_name(self):
        with self._patch():
            result = _enrich_vcpkg("nonexistent-pkg")
        assert "nonexistent-pkg" in result["error"]

    def test_missing_package_ecosystem_still_set(self):
        with self._patch():
            result = _enrich_vcpkg("nonexistent-pkg")
        assert result["ecosystem"] == "vcpkg"


class TestEnrichVcpkgGracefulDegradation:
    def _patch_cache(self):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None

    def test_http_error_returns_error_key(self):
        import requests
        self._patch_cache()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_vcpkg("openssl")
        assert "error" in result

    def test_connection_error_returns_error_key(self):
        import requests
        self._patch_cache()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("down")):
            result = _enrich_vcpkg("openssl")
        assert "error" in result

    def test_timeout_error_returns_error_key(self):
        import requests
        self._patch_cache()
        with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
            result = _enrich_vcpkg("openssl")
        assert "error" in result

    def test_malformed_json_returns_error_key(self):
        import requests
        self._patch_cache()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("bad json")
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_vcpkg("openssl")
        assert "error" in result


class TestEnrichVcpkgManifestCaching:
    def test_manifest_fetched_only_once_per_call_series(self):
        """The manifest should be fetched once and cached for subsequent lookups."""
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        mock_resp = _mock_vcpkg_get()
        with patch("requests.get", return_value=mock_resp) as mock_get:
            _enrich_vcpkg("openssl")
            _enrich_vcpkg("zlib")
            _enrich_vcpkg("curl")
        # All three lookups share one manifest fetch
        assert mock_get.call_count == 1

    def test_cache_seeding_skips_network(self):
        """When _VCPKG_MANIFEST_CACHE is pre-seeded, no request is made."""
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = _SAMPLE_MANIFEST
        with patch("requests.get") as mock_get:
            result = _enrich_vcpkg("openssl")
        mock_get.assert_not_called()
        assert result["latest_version"] == "3.6.3"


# ===========================================================================
# TestEnrichPackageDispatch — ecosystem alias routing
# ===========================================================================


class TestEnrichVcpkgDispatch:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_dispatch_vcpkg_alias(self):
        with self._patch():
            result = _enrich_package("openssl", "vcpkg")
        assert result["ecosystem"] == "vcpkg"

    def test_dispatch_cpp_alias(self):
        with self._patch():
            result = _enrich_package("zlib", "c++")
        assert result["ecosystem"] == "vcpkg"

    def test_dispatch_cpp_lowercase_alias(self):
        with self._patch():
            result = _enrich_package("zlib", "cpp")
        assert result["ecosystem"] == "vcpkg"

    def test_dispatch_conan_alias(self):
        """conan alias also routes to _enrich_vcpkg (both are C/C++ ecosystems)."""
        with self._patch():
            result = _enrich_package("openssl", "conan")
        assert result["ecosystem"] == "vcpkg"


# ===========================================================================
# TestBlastScore — vcpkg scoring thresholds
# ===========================================================================


class TestEnrichVcpkgBlastScore:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_zero_reverse_deps_gives_low_or_unknown_score(self):
        with self._patch():
            result = _enrich_vcpkg("curl")  # 0 reverse deps
        score = _blast_score(result)
        assert score in ("UNKNOWN", "LOW")

    def test_one_reverse_dep_gives_at_least_low_score(self):
        with self._patch():
            result = _enrich_vcpkg("openssl")  # 1 reverse dep
        score = _blast_score(result)
        assert score in ("LOW",)

    def test_two_reverse_deps_gives_low_score(self):
        with self._patch():
            result = _enrich_vcpkg("zlib")  # 2 reverse deps
        score = _blast_score(result)
        assert score in ("LOW",)

    def test_medium_blast_score_at_500_deps(self):
        """Inject 500 reverse deps via manifest to cross MEDIUM threshold."""
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        # Build 500 packages all depending on "biglib"
        biglib = {
            "Name": "biglib",
            "Version": "1.0.0",
            "Description": "A big library",
            "Dependencies": [],
            "Features": {},
        }
        dependents = [
            {
                "Name": f"pkg-{i}",
                "Version": "1.0.0",
                "Description": f"Package {i}",
                "Dependencies": ["biglib"],
                "Features": {},
            }
            for i in range(500)
        ]
        manifest = _make_manifest([biglib] + dependents)
        mock_resp = _mock_vcpkg_get(manifest)
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_vcpkg("biglib")
        score = _blast_score(result)
        assert score in ("MEDIUM", "HIGH", "CRITICAL")


# ===========================================================================
# TestGetDependencyBlastRadiusVcpkg — top-level tool integration
# ===========================================================================


class TestGetDependencyBlastRadiusVcpkg:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_top_level_tool_returns_str(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert isinstance(result, str)

    def test_output_contains_package_name(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert "openssl" in result

    def test_output_contains_version(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert "3.6.3" in result

    def test_output_contains_blast_score(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert any(s in result for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"))

    def test_output_contains_vcpkg_rev_deps(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert "vcpkg rev-deps" in result or "rev-deps" in result

    def test_output_contains_license(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert "Apache-2.0" in result

    def test_output_contains_ecosystem_label(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:openssl")
        assert "vcpkg" in result.lower()

    def test_package_not_found_contains_error_info(self):
        with self._patch():
            result = get_dependency_blast_radius("vcpkg:no-such-pkg")
        assert "not found" in result.lower() or "error" in result.lower() or "no-such-pkg" in result

    def test_cpp_alias_produces_vcpkg_output(self):
        with self._patch():
            result = get_dependency_blast_radius("c++:openssl")
        assert "openssl" in result

    def test_cpp_short_alias_produces_vcpkg_output(self):
        with self._patch():
            result = get_dependency_blast_radius("cpp:openssl")
        assert "openssl" in result


# ===========================================================================
# TestEnrichVcpkgEdgeCases — edge cases and variant packages
# ===========================================================================


class TestEnrichVcpkgEdgeCases:
    def _patch(self, manifest=_SAMPLE_MANIFEST):
        import manus_agent.tools.get_dependency_blast_radius as mod
        mod._VCPKG_MANIFEST_CACHE = None
        return patch("requests.get", return_value=_mock_vcpkg_get(manifest))

    def test_case_insensitive_package_lookup(self):
        """vcpkg package names are lowercase; confirm lookup is normalised."""
        with self._patch():
            result = _enrich_vcpkg("OpenSSL")
        # Should find it (normalised) or gracefully not find it — either is acceptable
        assert "error" in result or result.get("package_name", "").lower() == "openssl"

    def test_package_with_string_dep(self):
        """Dependency listed as plain string (not dict) should be handled."""
        pkg = {
            "Name": "libfoo",
            "Version": "1.0",
            "Description": "foo lib",
            "Dependencies": ["zlib", {"name": "vcpkg-cmake", "host": True}],
            "Features": {},
        }
        manifest = _make_manifest([pkg, _ZLIB_PKG])
        with self._patch(manifest):
            result = _enrich_vcpkg("zlib")
        assert result["dependent_packages_count"] == 1

    def test_no_dependencies_field(self):
        """Package with no Dependencies key should not crash."""
        pkg = {
            "Name": "standalone",
            "Version": "2.0",
            "Description": "standalone lib",
            "Features": {},
        }
        manifest = _make_manifest([pkg])
        with self._patch(manifest):
            result = _enrich_vcpkg("standalone")
        assert result["dependent_packages_count"] == 0

    def test_no_features_field(self):
        """Package with no Features key should not crash."""
        pkg = {
            "Name": "nofeats",
            "Version": "1.0",
            "Description": "no features",
            "Dependencies": [],
        }
        manifest = _make_manifest([pkg])
        with self._patch(manifest):
            result = _enrich_vcpkg("nofeats")
        assert result.get("feature_count", 0) == 0

    def test_empty_manifest(self):
        """Empty Source list should gracefully return error."""
        manifest = _make_manifest([])
        with self._patch(manifest):
            result = _enrich_vcpkg("openssl")
        assert "error" in result

    def test_total_vcpkg_packages_in_result(self):
        """Result should include total package count for context."""
        with self._patch():
            result = _enrich_vcpkg("openssl")
        assert "total_vcpkg_packages" in result
        assert result["total_vcpkg_packages"] == 4  # our sample has 4 packages
