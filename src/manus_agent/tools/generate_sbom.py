#!/usr/bin/env python3
"""
Tool for generating CycloneDX 1.5 SBOM JSON from common lockfile formats.

Supported lockfile formats:
- package-lock.json (npm v2/v3)
- yarn.lock (v1)
- pnpm-lock.yaml (v5+)
- poetry.lock (Python/Poetry)
- requirements.txt (Python/pip — pinned versions only)
- go.sum (Go modules)
- Cargo.lock (Rust)
- Gemfile.lock (Ruby)

This tool bridges the gap between raw dependency lockfiles and structured
SBOM tooling.  It produces CycloneDX 1.5 JSON suitable for ingestion by
vulnerability scanners (e.g. manus-agent sbom-scan, Grype, Trivy).

Design decisions:
- Zero external dependencies beyond stdlib + requests (already in project)
- Deterministic output (sorted components, stable UUIDs via namespace hash)
- Package URL (purl) generation per ecosystem for accurate OSV/Grype matching
- Graceful handling of malformed/partial lockfiles (best-effort extraction)
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from strands.types.tools import ToolResult, ToolUse

# ---------------------------------------------------------------------------
# CycloneDX 1.5 constants
# ---------------------------------------------------------------------------
_CYCLONEDX_SPEC_VERSION = "1.5"
_CYCLONEDX_FORMAT = "CycloneDX"
_CYCLONEDX_VERSION = 1

# Namespace UUID for deterministic serial number generation
_SBOM_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# ---------------------------------------------------------------------------
# Strands TOOL_SPEC
# ---------------------------------------------------------------------------
TOOL_SPEC = {
    "name": "generate_sbom",
    "description": (
        "Generates a CycloneDX 1.5 SBOM (Software Bill of Materials) in JSON format "
        "from a project lockfile. Supports package-lock.json (npm), yarn.lock, "
        "pnpm-lock.yaml, poetry.lock, requirements.txt (pinned), go.sum, Cargo.lock, "
        "and Gemfile.lock. Returns structured JSON suitable for vulnerability scanning "
        "with tools like sbom-scan, Grype, or Trivy. Each component includes a Package URL "
        "(purl) for accurate ecosystem-specific vulnerability matching."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "lockfile_path": {
                    "type": "string",
                    "description": (
                        "Path to the lockfile to parse. The format is auto-detected "
                        "from the filename (e.g. 'package-lock.json', 'poetry.lock')."
                    ),
                },
                "lockfile_content": {
                    "type": "string",
                    "description": (
                        "Raw lockfile content as a string. Use this when the file "
                        "is not on disk. Must also provide lockfile_type."
                    ),
                },
                "lockfile_type": {
                    "type": "string",
                    "description": (
                        "Explicit lockfile type when using lockfile_content. One of: "
                        "npm, yarn, pnpm, poetry, requirements, gosum, cargo, gemfile."
                    ),
                    "enum": [
                        "npm",
                        "yarn",
                        "pnpm",
                        "poetry",
                        "requirements",
                        "gosum",
                        "cargo",
                        "gemfile",
                    ],
                },
                "project_name": {
                    "type": "string",
                    "description": (
                        "Optional project name for the SBOM metadata component. "
                        "Defaults to the lockfile's parent directory name."
                    ),
                },
            },
            "required": [],
        }
    },
}


# ---------------------------------------------------------------------------
# Purl helpers
# ---------------------------------------------------------------------------
def _purl_encode(s: str) -> str:
    """Percent-encode special characters in a purl segment."""
    return s.replace("%", "%25").replace("/", "%2F").replace("@", "%40").replace(":", "%3A")


def _make_purl(ecosystem: str, name: str, version: str, namespace: str = "") -> str:
    """Build a Package URL string."""
    purl_type_map = {
        "npm": "npm",
        "yarn": "npm",
        "pnpm": "npm",
        "poetry": "pypi",
        "requirements": "pypi",
        "gosum": "golang",
        "cargo": "cargo",
        "gemfile": "gem",
    }
    purl_type = purl_type_map.get(ecosystem, ecosystem)

    if namespace:
        return f"pkg:{purl_type}/{_purl_encode(namespace)}/{_purl_encode(name)}@{_purl_encode(version)}"
    return f"pkg:{purl_type}/{_purl_encode(name)}@{_purl_encode(version)}"


# ---------------------------------------------------------------------------
# Lockfile parsers — each returns list of (name, version, purl, ecosystem)
# ---------------------------------------------------------------------------

Component = tuple[str, str, str, str]  # (name, version, purl, ecosystem)


def _parse_npm(content: str) -> list[Component]:
    """Parse package-lock.json (npm v2/v3 format)."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []

    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # npm v2/v3: packages dict (preferred)
    packages = data.get("packages", {})
    if packages:
        for pkg_path, pkg_info in packages.items():
            if not pkg_path:  # root package
                continue
            # Extract package name from path (node_modules/name or node_modules/@scope/name)
            name = pkg_info.get("name", "")
            if not name:
                # Derive from path
                parts = pkg_path.split("node_modules/")
                if parts:
                    name = parts[-1]
            version = pkg_info.get("version", "")
            if name and version and (name, version) not in seen:
                seen.add((name, version))
                # Handle scoped packages for purl
                namespace = ""
                pkg_name = name
                if name.startswith("@") and "/" in name:
                    namespace, pkg_name = name.split("/", 1)
                purl = _make_purl("npm", pkg_name, version, namespace=namespace)
                components.append((name, version, purl, "npm"))
        return components

    # npm v1 fallback: dependencies dict
    deps = data.get("dependencies", {})
    _parse_npm_deps_recursive(deps, components, seen)
    return components


def _parse_npm_deps_recursive(
    deps: dict[str, Any],
    components: list[Component],
    seen: set[tuple[str, str]],
) -> None:
    """Recursively parse npm v1 nested dependencies."""
    for name, info in deps.items():
        if not isinstance(info, dict):
            continue
        version = info.get("version", "")
        if name and version and (name, version) not in seen:
            seen.add((name, version))
            namespace = ""
            pkg_name = name
            if name.startswith("@") and "/" in name:
                namespace, pkg_name = name.split("/", 1)
            purl = _make_purl("npm", pkg_name, version, namespace=namespace)
            components.append((name, version, purl, "npm"))
        # Recurse into nested dependencies
        nested = info.get("dependencies", {})
        if nested:
            _parse_npm_deps_recursive(nested, components, seen)


def _parse_yarn(content: str) -> list[Component]:
    """Parse yarn.lock v1 format."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # yarn.lock v1 uses a custom format:
    # "package@^version":
    #   version "1.2.3"
    # Header lines may list multiple specs separated by commas:
    # express@^4.18.0, express@^4.17.0:
    current_name: str | None = None
    for line in content.splitlines():
        # Match header lines like: "lodash@^4.17.0", lodash@^4.17.0:
        # or "@scope/pkg@^1.0.0":
        if not line.startswith(" ") and not line.startswith("#") and line.strip():
            stripped = line.strip().rstrip(":")
            # May have multiple comma-separated specs; take the first one
            first_spec = stripped.split(",")[0].strip()
            # Remove surrounding quotes
            first_spec = first_spec.strip('"')
            # Extract package name (everything before the last @version-spec)
            # Handle scoped packages: @scope/name@^version
            if first_spec.startswith("@"):
                # Scoped: find second @ which is the version separator
                at_idx = first_spec.index("@", 1) if "@" in first_spec[1:] else -1
                if at_idx > 0:
                    current_name = first_spec[:at_idx]
                else:
                    current_name = None
            else:
                at_idx = first_spec.rfind("@")
                if at_idx > 0:
                    current_name = first_spec[:at_idx]
                else:
                    current_name = None
        elif line.strip().startswith("version ") and current_name:
            # Extract version value
            match = re.match(r'\s+version\s+"([^"]+)"', line)
            if match:
                version = match.group(1)
                if (current_name, version) not in seen:
                    seen.add((current_name, version))
                    namespace = ""
                    pkg_name = current_name
                    if current_name.startswith("@") and "/" in current_name:
                        namespace, pkg_name = current_name.split("/", 1)
                    purl = _make_purl("yarn", pkg_name, version, namespace=namespace)
                    components.append((current_name, version, purl, "npm"))
            current_name = None

    return components


def _parse_pnpm(content: str) -> list[Component]:
    """Parse pnpm-lock.yaml (v5+ format)."""
    # Minimal YAML parsing to avoid adding PyYAML dependency
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # pnpm-lock.yaml v6+ uses packages: section with entries like:
    #   /@scope/name@version:
    # or
    #   /name@version:
    # pnpm-lock.yaml v9+ (lockfileVersion: '9.0') uses snapshots/packages:
    #   name@version:
    in_packages = False
    for line in content.splitlines():
        stripped = line.strip()

        # Detect packages or snapshots section
        if stripped in ("packages:", "snapshots:"):
            in_packages = True
            continue
        # Exit section on non-indented non-empty line that's a new top-level key
        if in_packages and stripped and not line.startswith(" ") and stripped.endswith(":"):
            if stripped not in ("packages:", "snapshots:"):
                in_packages = False
                continue

        if not in_packages:
            continue

        # Match package entries:
        # v5/v6:  /@scope/name@version:  or  /name@version:
        # v9:     name@version:  or  @scope/name@version:
        if not stripped.endswith(":"):
            continue

        entry = stripped.rstrip(":").strip("'\"")

        # v5/v6 format: starts with /
        if entry.startswith("/"):
            entry = entry[1:]

        # Now parse: @scope/name@version or name@version
        if entry.startswith("@"):
            # Scoped package
            # Find the version separator (second @)
            second_at = entry.index("@", 1) if "@" in entry[1:] else -1
            if second_at <= 0:
                continue
            name = entry[:second_at]
            version = entry[second_at + 1 :]
        else:
            at_idx = entry.rfind("@")
            if at_idx <= 0:
                continue
            name = entry[:at_idx]
            version = entry[at_idx + 1 :]

        # Strip any trailing parenthetical qualifiers e.g. (react@18.2.0)
        version = re.sub(r"\(.*\)$", "", version).strip()

        if not name or not version:
            continue
        if (name, version) in seen:
            continue
        seen.add((name, version))

        namespace = ""
        pkg_name = name
        if name.startswith("@") and "/" in name:
            namespace, pkg_name = name.split("/", 1)
        purl = _make_purl("pnpm", pkg_name, version, namespace=namespace)
        components.append((name, version, purl, "npm"))

    return components


def _parse_poetry(content: str) -> list[Component]:
    """Parse poetry.lock (TOML-ish format, minimal parser)."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # poetry.lock has [[package]] sections with name and version fields
    current_name: str | None = None
    current_version: str | None = None

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            # Save previous package
            if current_name and current_version and (current_name, current_version) not in seen:
                seen.add((current_name, current_version))
                # PyPI normalizes names to lowercase with hyphens
                normalized = current_name.lower().replace("_", "-")
                purl = _make_purl("poetry", normalized, current_version)
                components.append((current_name, current_version, purl, "pypi"))
            current_name = None
            current_version = None
        elif stripped.startswith("name"):
            match = re.match(r'^name\s*=\s*"([^"]+)"', stripped)
            if match:
                current_name = match.group(1)
        elif stripped.startswith("version"):
            match = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
            if match:
                current_version = match.group(1)

    # Don't forget last package
    if current_name and current_version and (current_name, current_version) not in seen:
        seen.add((current_name, current_version))
        normalized = current_name.lower().replace("_", "-")
        purl = _make_purl("poetry", normalized, current_version)
        components.append((current_name, current_version, purl, "pypi"))

    return components


def _parse_requirements(content: str) -> list[Component]:
    """Parse requirements.txt (pinned versions only: name==version)."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    for line in content.splitlines():
        stripped = line.strip()
        # Skip comments and empty lines
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        # Match pinned: name==version (with optional extras)
        match = re.match(r"^([A-Za-z0-9._-]+)(?:\[.*?\])?\s*==\s*([^\s;#]+)", stripped)
        if match:
            name = match.group(1)
            version = match.group(2)
            if (name, version) not in seen:
                seen.add((name, version))
                normalized = name.lower().replace("_", "-")
                purl = _make_purl("requirements", normalized, version)
                components.append((name, version, purl, "pypi"))

    return components


def _parse_gosum(content: str) -> list[Component]:
    """Parse go.sum (Go modules)."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # go.sum format: module version hash
        # e.g.: github.com/pkg/errors v0.9.1 h1:FEBLx1...
        parts = stripped.split()
        if len(parts) < 3:
            continue
        module = parts[0]
        version_raw = parts[1]
        # Strip /go.mod suffix from version
        version = version_raw.split("/go.mod")[0]
        # Strip v prefix for purl
        version_clean = version.lstrip("v")
        if not module or not version_clean:
            continue
        if (module, version_clean) in seen:
            continue
        seen.add((module, version_clean))
        # Go purl uses the full module path as namespace/name
        # pkg:golang/github.com/pkg/errors@0.9.1
        purl = f"pkg:golang/{module}@{_purl_encode(version_clean)}"
        components.append((module, version_clean, purl, "golang"))

    return components


def _parse_cargo(content: str) -> list[Component]:
    """Parse Cargo.lock (Rust/Cargo)."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # Cargo.lock uses [[package]] sections similar to poetry.lock
    current_name: str | None = None
    current_version: str | None = None

    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            if current_name and current_version and (current_name, current_version) not in seen:
                seen.add((current_name, current_version))
                purl = _make_purl("cargo", current_name, current_version)
                components.append((current_name, current_version, purl, "crates.io"))
            current_name = None
            current_version = None
        elif stripped.startswith("name"):
            match = re.match(r'^name\s*=\s*"([^"]+)"', stripped)
            if match:
                current_name = match.group(1)
        elif stripped.startswith("version"):
            match = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
            if match:
                current_version = match.group(1)

    # Last package
    if current_name and current_version and (current_name, current_version) not in seen:
        seen.add((current_name, current_version))
        purl = _make_purl("cargo", current_name, current_version)
        components.append((current_name, current_version, purl, "crates.io"))

    return components


def _parse_gemfile(content: str) -> list[Component]:
    """Parse Gemfile.lock (Ruby/Bundler)."""
    components: list[Component] = []
    seen: set[tuple[str, str]] = set()

    # Gemfile.lock has a SPECS section with indented gem entries:
    #     gem-name (version)
    in_specs = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "specs:":
            in_specs = True
            continue
        if in_specs:
            # End of specs section on non-indented line
            if not line.startswith(" ") and stripped:
                in_specs = False
                continue
            # Match gem entries: "    name (version)"
            match = re.match(r"^\s{4}(\S+)\s+\(([^)]+)\)", line)
            if match:
                name = match.group(1)
                version = match.group(2)
                if (name, version) not in seen:
                    seen.add((name, version))
                    purl = _make_purl("gemfile", name, version)
                    components.append((name, version, purl, "rubygems"))

    return components


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
_FILENAME_TO_TYPE: dict[str, str] = {
    "package-lock.json": "npm",
    "npm-shrinkwrap.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-lock.yaml": "pnpm",
    "poetry.lock": "poetry",
    "requirements.txt": "requirements",
    "go.sum": "gosum",
    "Cargo.lock": "cargo",
    "Gemfile.lock": "gemfile",
}

_TYPE_TO_PARSER: dict[str, Any] = {
    "npm": _parse_npm,
    "yarn": _parse_yarn,
    "pnpm": _parse_pnpm,
    "poetry": _parse_poetry,
    "requirements": _parse_requirements,
    "gosum": _parse_gosum,
    "cargo": _parse_cargo,
    "gemfile": _parse_gemfile,
}


def detect_lockfile_type(filename: str) -> str | None:
    """Detect lockfile type from filename."""
    basename = Path(filename).name
    return _FILENAME_TO_TYPE.get(basename)


# ---------------------------------------------------------------------------
# CycloneDX SBOM generation
# ---------------------------------------------------------------------------
def _build_cyclonedx(
    components: list[Component],
    project_name: str = "unknown",
    lockfile_path: str = "",
) -> dict[str, Any]:
    """Build a CycloneDX 1.5 JSON document from parsed components."""
    # Deterministic serial number based on content hash
    content_hash = hashlib.sha256(
        json.dumps([(c[0], c[1]) for c in sorted(components)], sort_keys=True).encode()
    ).hexdigest()
    serial = str(uuid.uuid5(_SBOM_UUID_NAMESPACE, content_hash))

    # Sort components for deterministic output
    sorted_components = sorted(components, key=lambda c: (c[0].lower(), c[1]))

    cdx_components = []
    for name, version, purl, ecosystem in sorted_components:
        comp: dict[str, Any] = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": purl,
        }
        # Add ecosystem as a property
        comp["properties"] = [{"name": "cdx:npm:package:ecosystem", "value": ecosystem}]
        cdx_components.append(comp)

    sbom: dict[str, Any] = {
        "bomFormat": _CYCLONEDX_FORMAT,
        "specVersion": _CYCLONEDX_SPEC_VERSION,
        "serialNumber": f"urn:uuid:{serial}",
        "version": _CYCLONEDX_VERSION,
        "metadata": {
            "component": {
                "type": "application",
                "name": project_name,
            },
            "tools": [
                {
                    "vendor": "manus-agent",
                    "name": "generate_sbom",
                    "version": "1.0.0",
                }
            ],
        },
        "components": cdx_components,
    }

    if lockfile_path:
        sbom["metadata"]["component"]["properties"] = [{"name": "manus-agent:lockfile", "value": lockfile_path}]

    return sbom


# ---------------------------------------------------------------------------
# Main entry point (Strands tool handler)
# ---------------------------------------------------------------------------
def generate_sbom(
    tool: ToolUse,
    lockfile_path: str = "",
    lockfile_content: str = "",
    lockfile_type: str = "",
    project_name: str = "",
) -> ToolResult:
    """Generate a CycloneDX 1.5 SBOM from a lockfile.

    Either ``lockfile_path`` (reads from disk) or ``lockfile_content`` +
    ``lockfile_type`` (in-memory) must be provided.
    """
    # Resolve content
    content: str = ""
    detected_type: str = ""
    resolved_path: str = lockfile_path

    if lockfile_path and not lockfile_content:
        path = Path(lockfile_path).expanduser().resolve()
        if not path.exists():
            return {
                "toolUseId": tool["toolUseId"],
                "status": "error",
                "content": [{"text": f"File not found: {lockfile_path}"}],
            }
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {
                "toolUseId": tool["toolUseId"],
                "status": "error",
                "content": [{"text": f"Failed to read file: {exc}"}],
            }
        detected_type = detect_lockfile_type(path.name) or ""
        resolved_path = str(path)
    elif lockfile_content:
        content = lockfile_content
    else:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "Either lockfile_path or lockfile_content must be provided."}],
        }

    # Determine parser
    final_type = lockfile_type or detected_type
    if not final_type:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [
                {
                    "text": (
                        "Cannot detect lockfile type. Provide lockfile_type explicitly. "
                        f"Supported types: {', '.join(sorted(_TYPE_TO_PARSER.keys()))}"
                    )
                }
            ],
        }

    parser = _TYPE_TO_PARSER.get(final_type)
    if not parser:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [
                {
                    "text": (
                        f"Unsupported lockfile type: {final_type}. "
                        f"Supported: {', '.join(sorted(_TYPE_TO_PARSER.keys()))}"
                    )
                }
            ],
        }

    # Parse
    components = parser(content)

    # Determine project name
    if not project_name:
        if lockfile_path:
            project_name = Path(lockfile_path).resolve().parent.name
        else:
            project_name = "unknown"

    # Build SBOM
    sbom = _build_cyclonedx(components, project_name=project_name, lockfile_path=resolved_path)

    result_text = json.dumps(sbom, indent=2)
    summary = (
        f"Generated CycloneDX {_CYCLONEDX_SPEC_VERSION} SBOM: {len(components)} components from {final_type} lockfile"
    )

    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": f"{summary}\n\n{result_text}"}],
    }
