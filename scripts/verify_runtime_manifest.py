#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the installed Thor runtime against the tested production manifest."""

import importlib.metadata
import json
import sys
from pathlib import Path

import ctranslate2
import onnxruntime as ort
import torch


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    manifest = json.loads((repo / "deploy" / "runtime-manifest.json").read_text())
    errors = []

    expected_python = manifest["platform"]["python"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        errors.append(f"Python {actual_python} != {expected_python}")

    for distribution, expected in manifest["python_packages"].items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package {distribution}")
            continue
        if actual != expected:
            errors.append(f"{distribution} {actual} != {expected}")

    if ctranslate2.__version__ != manifest["ctranslate2"]["version"]:
        errors.append("CTranslate2 version does not match manifest")
    if ctranslate2.get_cuda_device_count() < 1:
        errors.append("CTranslate2 CUDA device is unavailable")
    if not torch.cuda.is_available():
        errors.append("PyTorch CUDA is unavailable")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        errors.append("ONNX Runtime CUDAExecutionProvider is unavailable")

    if errors:
        for error in errors:
            print(f"RUNTIME_DRIFT: {error}", file=sys.stderr)
        return 1
    print("RUNTIME_MANIFEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
