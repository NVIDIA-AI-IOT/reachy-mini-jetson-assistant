#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit exact Jetson wheels and emit an inventory plus an SPDX 2.3 SBOM."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ELF_MAGIC = b"\x7fELF"
AR_MAGIC = b"!<arch>\n"
NVIDIA_LIBRARY_PATTERN = re.compile(
    r"^lib(?:cuda|cudart|cudnn|cublaslt|cublas|cufft|curand|cusolver|cusparse|"
    r"nvrtc|nvjitlink|nvtoolsext|npp|nccl|culibos)(?:[._-]|$)",
    re.IGNORECASE,
)
NVIDIA_SONAME_PATTERN = re.compile(
    r"\blib(?:cuda|cudart|cudnn|cublaslt|cublas|cufft|curand|cusolver|"
    r"cusparse|nvrtc|nvjitlink|nvtoolsext|npp|nccl|culibos)"
    r"[A-Za-z0-9_.+-]*\.so(?:\.[0-9]+)*\b",
    re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-.")
    return normalized or "item"


def classify_member(name: str, data: bytes) -> str:
    lower = name.lower()
    if data.startswith(ELF_MAGIC):
        return "elf"
    if data.startswith(AR_MAGIC) or lower.endswith((".a", ".lib")):
        return "static-library"
    if lower.endswith((".o", ".obj")):
        return "object"
    if lower.endswith((".so", ".dll", ".dylib")) or ".so." in lower:
        return "shared-library-non-elf"
    if "license" in lower or "notice" in lower or lower.endswith("copying"):
        return "license-or-notice"
    if lower.endswith((".dist-info/metadata", ".dist-info/wheel", ".dist-info/record")):
        return "wheel-metadata"
    if lower.endswith(".py"):
        return "python"
    return "other"


def parse_header(output: str) -> dict[str, str]:
    wanted = {"Class", "Data", "Type", "Machine"}
    result: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in wanted:
            result[key.lower()] = value.strip()
    return result


def parse_dynamic(output: str) -> dict[str, Any]:
    result: dict[str, Any] = {"needed": []}
    for line in output.splitlines():
        match = re.search(r"\((NEEDED|SONAME|RPATH|RUNPATH)\).*?\[(.*?)\]", line)
        if not match:
            continue
        tag, value = match.groups()
        if tag == "NEEDED":
            result["needed"].append(value)
        else:
            result[tag.lower()] = value
    result["needed"].sort()
    return result


def parse_build_id(output: str) -> str | None:
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", output)
    return match.group(1).lower() if match else None


def parse_ldd(output: str, wheel_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=> not found" in line:
            name = line.split("=>", 1)[0].strip()
            unresolved.append(name)
            entries.append({"name": name, "resolved": None, "raw": line})
            continue
        match = re.match(r"(\S+)\s+=>\s+(\S+)\s+\(", line)
        if match:
            name, resolved = match.groups()
            resolved_path = Path(resolved).resolve()
            entries.append(
                {
                    "name": name,
                    "resolved": str(resolved_path),
                    "inside_wheel": wheel_root in resolved_path.parents,
                    "raw": line,
                }
            )
            continue
        match = re.match(r"(/\S+)\s+\(", line)
        if match:
            entries.append(
                {
                    "name": Path(match.group(1)).name,
                    "resolved": str(Path(match.group(1)).resolve()),
                    "inside_wheel": False,
                    "raw": line,
                }
            )
            continue
        entries.append({"name": line.split()[0], "resolved": None, "raw": line})
    return {"entries": entries, "unresolved": sorted(unresolved)}


def dpkg_owner(path: str | None) -> str | None:
    if not path or not path.startswith("/"):
        return None
    candidates = [path, str(Path(path).resolve())]
    for candidate in dict.fromkeys(candidates):
        result = run(["dpkg-query", "-S", candidate])
        if result.returncode == 0 and ":" in result.stdout:
            return result.stdout.split(":", 1)[0].strip()
    return None


def inspect_elf(path: Path, wheel_root: Path) -> dict[str, Any]:
    header = run(["readelf", "--file-header", str(path)])
    dynamic = run(["readelf", "--dynamic", str(path)])
    notes = run(["readelf", "--notes", str(path)])
    ldd = run(["ldd", str(path)])
    symbols = run(["nm", "-a", str(path)])
    string_table = run(["strings", "-a", str(path)])
    dependency_resolution = parse_ldd(ldd.stdout, wheel_root)
    for entry in dependency_resolution["entries"]:
        entry["dpkg_owner"] = dpkg_owner(entry.get("resolved"))
    culibos_symbols = sorted(
        {
            token
            for token in re.findall(r"\bculibos[A-Za-z0-9_]*\b", symbols.stdout)
        }
    )
    result = {
        "header": parse_header(header.stdout),
        "dynamic": parse_dynamic(dynamic.stdout),
        "build_id": parse_build_id(notes.stdout),
        "dependency_resolution": dependency_resolution,
        "culibos_symbols": culibos_symbols,
        "nvidia_library_strings": sorted(
            set(NVIDIA_SONAME_PATTERN.findall(string_table.stdout))
        ),
    }
    return result


def parse_wheel_metadata(archive: zipfile.ZipFile) -> dict[str, Any]:
    metadata_names = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    wheel_names = [
        name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
    ]
    if len(metadata_names) != 1 or len(wheel_names) != 1:
        raise ValueError("wheel must contain exactly one METADATA and one WHEEL file")
    message = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
    wheel_text = archive.read(wheel_names[0]).decode("utf-8", errors="replace")
    return {
        "name": message.get("Name"),
        "version": message.get("Version"),
        "license": message.get("License"),
        "requires_python": message.get("Requires-Python"),
        "requires_dist": sorted(message.get_all("Requires-Dist", [])),
        "license_files": sorted(message.get_all("License-File", [])),
        "wheel_tags": sorted(
            line.split(":", 1)[1].strip()
            for line in wheel_text.splitlines()
            if line.startswith("Tag:")
        ),
        "generator": next(
            (
                line.split(":", 1)[1].strip()
                for line in wheel_text.splitlines()
                if line.startswith("Generator:")
            ),
            None,
        ),
    }


def inspect_wheel(
    key: str,
    spec: dict[str, Any],
    wheel_path: Path,
    extraction_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    actual_sha256 = sha256_file(wheel_path)
    actual_size = wheel_path.stat().st_size
    if actual_sha256 != spec["sha256"]:
        violations.append(f"{key}: wheel SHA-256 does not match the manifest")
    if actual_size != spec["size"]:
        violations.append(f"{key}: wheel size does not match the manifest")

    wheel_root = extraction_root / key
    wheel_root.mkdir(parents=True)
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            violations.append(f"{key}: wheel contains duplicate member paths")
        metadata = parse_wheel_metadata(archive)
        archive.extractall(wheel_root)
        members: list[dict[str, Any]] = []
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            data = archive.read(info)
            member_type = classify_member(info.filename, data)
            member: dict[str, Any] = {
                "path": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "sha256": sha256_bytes(data),
                "type": member_type,
                "unix_mode": oct((info.external_attr >> 16) & 0o7777),
            }
            if member_type == "elf":
                member["elf"] = inspect_elf(wheel_root / info.filename, wheel_root)
            if member_type in {"static-library", "object"}:
                member["archive_or_object"] = True
            basename = Path(info.filename).name
            if member_type in {"elf", "static-library", "shared-library-non-elf"}:
                if NVIDIA_LIBRARY_PATTERN.match(basename):
                    violations.append(
                        f"{key}: bundled NVIDIA library candidate: {info.filename}"
                    )
            members.append(member)

        member_hashes = {item["path"]: item["sha256"] for item in members}
        for required_path, required_hash in spec.get(
            "required_member_sha256", {}
        ).items():
            if member_hashes.get(required_path) != required_hash:
                violations.append(
                    f"{key}: required license/notice is missing or changed: "
                    f"{required_path}"
                )

    expected_name = re.sub(r"[-_.]+", "-", spec["distribution"]).lower()
    actual_name = re.sub(r"[-_.]+", "-", metadata["name"] or "").lower()
    if expected_name != actual_name:
        violations.append(f"{key}: distribution metadata does not match manifest")
    if metadata["version"] != spec["version"]:
        violations.append(f"{key}: version metadata does not match manifest")

    nvidia_needed: list[dict[str, Any]] = []
    unresolved: list[str] = []
    culibos_symbols: list[str] = []
    for member in members:
        elf = member.get("elf")
        if not elf:
            continue
        unresolved.extend(elf["dependency_resolution"]["unresolved"])
        culibos_symbols.extend(elf["culibos_symbols"])
        resolution_by_name = {
            item["name"]: item for item in elf["dependency_resolution"]["entries"]
        }
        for needed in elf["dynamic"]["needed"]:
            if NVIDIA_LIBRARY_PATTERN.match(needed):
                resolution = resolution_by_name.get(needed, {})
                nvidia_needed.append(
                    {
                        "member": member["path"],
                        "needed": needed,
                        "resolved": resolution.get("resolved"),
                        "inside_wheel": resolution.get("inside_wheel"),
                        "dpkg_owner": resolution.get("dpkg_owner"),
                    }
                )
                if not resolution.get("resolved"):
                    violations.append(
                        f"{key}: NVIDIA dependency is unresolved: {needed}"
                    )
                if resolution.get("inside_wheel"):
                    violations.append(
                        f"{key}: NVIDIA dependency resolves inside wheel: {needed}"
                    )
    if unresolved:
        violations.append(
            f"{key}: unresolved ELF dependencies: {', '.join(sorted(set(unresolved)))}"
        )
    if culibos_symbols:
        violations.append(
            f"{key}: libculibos symbols detected: "
            f"{', '.join(sorted(set(culibos_symbols)))}"
        )

    native_members = [item for item in members if item["type"] == "elf"]
    archives = [
        item
        for item in members
        if item["type"] in {"static-library", "object"}
    ]
    result = {
        "key": key,
        "path": str(wheel_path),
        "filename": wheel_path.name,
        "size": actual_size,
        "sha256": actual_sha256,
        "metadata": metadata,
        "license_declared": spec.get("license", "NOASSERTION"),
        "source": spec.get("source", {}),
        "member_count": len(members),
        "native_member_count": len(native_members),
        "static_library_or_object_count": len(archives),
        "nvidia_runtime_dependencies": sorted(
            nvidia_needed, key=lambda item: (item["member"], item["needed"])
        ),
        "nvidia_library_reference_strings": sorted(
            {
                reference
                for member in members
                for reference in member.get("elf", {}).get(
                    "nvidia_library_strings", []
                )
            }
        ),
        "bundled_nvidia_library_count": sum(
            1
            for item in members
            if item["type"] in {"elf", "static-library", "shared-library-non-elf"}
            and NVIDIA_LIBRARY_PATTERN.match(Path(item["path"]).name)
        ),
        "members": members,
    }
    return result, violations


def load_source_dependencies(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["components"]


def make_spdx(
    audit: dict[str, Any],
    source_dependencies: list[dict[str, Any]],
    created: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        "".join(wheel["sha256"] for wheel in audit["wheels"]).encode()
    ).hexdigest()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"reachy-jetson-wheels-{audit['platform_id']}",
        "documentNamespace": (
            "https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant/"
            f"sbom/{digest}"
        ),
        "creationInfo": {
            "created": created,
            "creators": ["Tool: scripts/audit_jetson_wheels.py"],
        },
        "packages": [],
        "files": [],
        "relationships": [],
        "documentDescribes": [],
        "hasExtractedLicensingInfos": [
            {
                "licenseId": "LicenseRef-NVIDIA-EULA",
                "name": "NVIDIA software license terms",
                "extractedText": (
                    "The applicable NVIDIA software is not included in these "
                    "wheels. It is supplied by JetPack on the target system."
                ),
                "seeAlsos": [
                    "https://www.nvidia.com/en-us/drivers/nvidia-license/"
                ],
            }
        ],
    }
    package_ids: dict[str, str] = {}
    for wheel in audit["wheels"]:
        package_id = f"SPDXRef-Package-{safe_id(wheel['key'])}"
        package_ids[wheel["key"]] = package_id
        source = wheel.get("source", {})
        package: dict[str, Any] = {
            "SPDXID": package_id,
            "name": wheel["metadata"]["name"],
            "versionInfo": wheel["metadata"]["version"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": wheel["license_declared"],
            "checksums": [
                {"algorithm": "SHA256", "checksumValue": wheel["sha256"]}
            ],
            "externalRefs": [],
        }
        if source.get("repository") and source.get("commit"):
            package["externalRefs"].append(
                {
                    "referenceCategory": "OTHER",
                    "referenceType": "vcs",
                    "referenceLocator": (
                        f"git+{source['repository']}@{source['commit']}"
                    ),
                }
            )
        document["packages"].append(package)
        document["documentDescribes"].append(package_id)
        for index, member in enumerate(wheel["members"]):
            file_id = (
                f"SPDXRef-File-{safe_id(wheel['key'])}-"
                f"{index}-{safe_id(member['path'])}"
            )
            document["files"].append(
                {
                    "SPDXID": file_id,
                    "fileName": f"./{wheel['filename']}/{member['path']}",
                    "checksums": [
                        {
                            "algorithm": "SHA256",
                            "checksumValue": member["sha256"],
                        }
                    ],
                    "licenseConcluded": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                }
            )
            document["relationships"].append(
                {
                    "spdxElementId": package_id,
                    "relationshipType": "CONTAINS",
                    "relatedSpdxElement": file_id,
                }
            )

        dependency_ids: set[str] = set()
        for requirement in wheel["metadata"]["requires_dist"]:
            dependency_name = re.split(r"[\s(<>=!~;]", requirement, maxsplit=1)[0]
            dependency_id = (
                f"SPDXRef-PythonDependency-{safe_id(wheel['key'])}-"
                f"{safe_id(dependency_name)}"
            )
            if dependency_id in dependency_ids:
                continue
            dependency_ids.add(dependency_id)
            document["packages"].append(
                {
                    "SPDXID": dependency_id,
                    "name": dependency_name,
                    "versionInfo": requirement,
                    "downloadLocation": (
                        f"https://pypi.org/project/{dependency_name}/"
                    ),
                    "filesAnalyzed": False,
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": "NOASSERTION",
                }
            )
            document["relationships"].append(
                {
                    "spdxElementId": package_id,
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": dependency_id,
                    "comment": "Declared by wheel METADATA Requires-Dist",
                }
            )

        runtime_dependencies: dict[str, dict[str, Any]] = {}
        for member in wheel["members"]:
            elf = member.get("elf")
            if not elf:
                continue
            resolution_by_name = {
                entry["name"]: entry
                for entry in elf["dependency_resolution"]["entries"]
            }
            for needed in elf["dynamic"]["needed"]:
                runtime_dependencies.setdefault(
                    needed, resolution_by_name.get(needed, {})
                )
        for needed, resolution in sorted(runtime_dependencies.items()):
            dependency_id = (
                f"SPDXRef-ELFDependency-{safe_id(wheel['key'])}-"
                f"{safe_id(needed)}"
            )
            document["packages"].append(
                {
                    "SPDXID": dependency_id,
                    "name": needed,
                    "versionInfo": resolution.get("dpkg_owner") or "NOASSERTION",
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": "NOASSERTION",
                    "comment": (
                        f"Resolved on audit host to {resolution.get('resolved')}"
                    ),
                }
            )
            document["relationships"].append(
                {
                    "spdxElementId": package_id,
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": dependency_id,
                    "comment": "ELF DT_NEEDED runtime dependency",
                }
            )

    for index, component in enumerate(source_dependencies):
        component_id = (
            f"SPDXRef-Dependency-{index}-{safe_id(component['name'])}"
        )
        document["packages"].append(
            {
                "SPDXID": component_id,
                "name": component["name"],
                "versionInfo": component.get("version", "NOASSERTION"),
                "downloadLocation": component.get("source", "NOASSERTION"),
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": component.get("license", "NOASSERTION"),
                "checksums": [
                    {
                        "algorithm": component["checksum_algorithm"],
                        "checksumValue": component["checksum"],
                    }
                ]
                if component.get("checksum")
                else [],
            }
        )
        owner = component["wheel"]
        document["relationships"].append(
            {
                "spdxElementId": package_ids[owner],
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": component_id,
                "comment": component.get("scope", ""),
            }
        )
    return document


def host_identity() -> dict[str, Any]:
    l4t = (
        Path("/etc/nv_tegra_release").read_text(encoding="utf-8").strip()
        if Path("/etc/nv_tegra_release").is_file()
        else None
    )
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "l4t": l4t,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--platform-id", required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dependencies", type=Path)
    parser.add_argument(
        "--created",
        help="SPDX timestamp (default: current UTC time)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    platform_spec = manifest["platforms"][args.platform_id]
    created = args.created or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "platform_id": args.platform_id,
        "created": created,
        "host": host_identity(),
        "wheels": [],
        "verdict": {},
    }
    violations: list[str] = []
    with tempfile.TemporaryDirectory(prefix="reachy-wheel-audit-") as temp:
        extraction_root = Path(temp)
        for key, spec in sorted(platform_spec["wheels"].items()):
            wheel_path = args.wheel_dir / spec["filename"]
            if not wheel_path.is_file():
                violations.append(f"{key}: wheel not found: {wheel_path}")
                continue
            wheel, wheel_violations = inspect_wheel(
                key, spec, wheel_path, extraction_root
            )
            audit["wheels"].append(wheel)
            violations.extend(wheel_violations)

    audit["verdict"] = {
        "passed": not violations,
        "violations": sorted(set(violations)),
        "scope": (
            "Archive members, wheel metadata, ELF headers/dependencies, "
            "host resolution, NVIDIA library-name/string references, required "
            "license hashes, and libculibos symbols"
        ),
    }
    source_dependencies = load_source_dependencies(args.source_dependencies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "wheel-inventory.json"
    inventory_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spdx = make_spdx(audit, source_dependencies, created)
    (args.output_dir / "sbom.spdx.json").write_text(
        json.dumps(spdx, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{wheel['sha256']}  {wheel['filename']}" for wheel in audit["wheels"]
    ]
    (args.output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(f"Wrote {inventory_path}")
    print(f"Verdict: {'PASS' if not violations else 'FAIL'}")
    for violation in sorted(set(violations)):
        print(f"- {violation}")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
