# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging" / "jetson-wheels.json"
RELEASE = (
    ROOT
    / "packaging"
    / "releases"
    / "native-jp72-cu132-py312-sm87-r1"
)
ORT_PATCH = ROOT / "patches" / "onnxruntime-v1.28.0-jp72.patch"


def test_jp72_manifest_is_fail_closed_and_complete():
    data = json.loads(MANIFEST.read_text())
    assert data["schema_version"] == 1

    assert list(data["platforms"]) == ["jp72-l4t39.2-cu13.2-py3.12-sm87"]
    platform = next(iter(data["platforms"].values()))
    assert platform["architecture"] == "aarch64"
    assert platform["l4t"] == "39.2.0"
    assert platform["cuda"] == "13.2"
    assert platform["python"] == "3.12"
    assert platform["gpu_family"] == "orin"
    assert platform["compute_capability"] == "8.7"
    assert platform["wheel_tag"] == "cp312-cp312-linux_aarch64"

    assert platform["release"]["tag"] == "native-jp72-cu132-py312-sm87-r1"
    assert platform["release"]["published"] is True
    for wheel in platform["wheels"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
        assert wheel["size"] > 0
        assert wheel["version"] in wheel["filename"]
        assert wheel["filename"].endswith("-cp312-cp312-linux_aarch64.whl")
        assert re.fullmatch(r"[0-9a-f]{40}", wheel["source"]["commit"])
        assert wheel["source"]["tag"].startswith("v")
        assert wheel["source"]["modified"] is True
        assert wheel["required_member_sha256"]
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in wheel["required_member_sha256"].values()
        )

    ort = platform["wheels"]["onnxruntime_gpu"]
    assert ort["source"]["patch"]["path"] == (
        "patches/onnxruntime-v1.28.0-jp72.patch"
    )
    assert ort["source"]["patch"]["sha256"] == _sha256(ORT_PATCH)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_release_evidence_preserves_licenses_notices_and_sbom():
    assert _sha256(
        RELEASE / "PATCHES/onnxruntime-v1.28.0-jp72.patch"
    ) == _sha256(ORT_PATCH)
    for cache_name in (
        "ctranslate2-CMakeCache.txt",
        "onnxruntime-CMakeCache.txt",
    ):
        assert (RELEASE / "build-config" / cache_name).stat().st_size > 1_000

    assert _sha256(RELEASE / "LICENSES/ctranslate2/CTranslate2-MIT.txt") == (
        "54aa79d9fe3c09e67a16dcd95b9e88676405a6ec174efda31036983cf7672ecb"
    )
    assert _sha256(RELEASE / "LICENSES/onnxruntime/LICENSE") == (
        "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"
    )
    notices = RELEASE / "LICENSES/onnxruntime/ThirdPartyNotices.txt"
    assert _sha256(notices) == (
        "0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2"
    )
    assert "MPL v2.0" in notices.read_text(encoding="utf-8")

    sbom = json.loads((RELEASE / "sbom.spdx.json").read_text())
    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert sbom["dataLicense"] == "CC0-1.0"
    assert len(sbom["files"]) == 369
    assert any(
        package["name"] == "Eigen"
        and package["licenseDeclared"] == "MPL-2.0"
        for package in sbom["packages"]
    )

    inventory = json.loads((RELEASE / "wheel-inventory.json").read_text())
    assert inventory["verdict"]["passed"] is True
    assert inventory["verdict"]["violations"] == []
    assert sum(wheel["member_count"] for wheel in inventory["wheels"]) == 369
    assert sum(
        wheel["native_member_count"] for wheel in inventory["wheels"]
    ) == 6
    for wheel in inventory["wheels"]:
        assert wheel["static_library_or_object_count"] == 0
        assert wheel["bundled_nvidia_library_count"] == 0
        for dependency in wheel["nvidia_runtime_dependencies"]:
            assert dependency["resolved"].startswith(("/usr/", "/lib/"))
            assert dependency["inside_wheel"] is False
            assert dependency["dpkg_owner"] in {
                "cuda-cudart-13-2",
                "libcublas-13-2",
                "libcudnn9-cuda-13",
            }

    scanner_status = (RELEASE / "SCANNER.md").read_text()
    assert "Status: **PENDING**." in scanner_status

    runtime = json.loads((RELEASE / "runtime-validation.json").read_text())
    assert runtime["passed"] is True
    assert runtime["ctranslate2_cuda_devices"] > 0
    assert "CUDAExecutionProvider" in runtime["onnxruntime_providers"]
    assert runtime["onnxruntime_output_device"] == "cuda"


def test_local_candidate_wheels_match_manifest_when_present():
    wheel_dir_value = os.environ.get("REACHY_JETSON_WHEEL_DIR")
    if not wheel_dir_value:
        pytest.skip("set REACHY_JETSON_WHEEL_DIR to validate candidate wheel files")

    wheel_dir = Path(wheel_dir_value)
    data = json.loads(MANIFEST.read_text())
    platform = next(iter(data["platforms"].values()))

    for key in ("ctranslate2", "onnxruntime_gpu"):
        wheel = platform["wheels"][key]
        path = wheel_dir / wheel["filename"]
        assert path.is_file(), f"missing candidate wheel: {path}"
        assert _sha256(path) == wheel["sha256"]
        assert path.stat().st_size == wheel["size"]
