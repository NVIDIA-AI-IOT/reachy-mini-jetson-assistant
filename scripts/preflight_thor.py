#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed production preflight for the Thor assistant."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import ctranslate2
import httpx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config


def main() -> int:
    config = Config.load(platform="thor")
    errors = config.validation_errors(require_web_auth=True)

    if not torch.cuda.is_available():
        errors.append("PyTorch cannot see the Thor GPU")
    if ctranslate2.get_cuda_device_count() < 1:
        errors.append("CTranslate2 cannot see a CUDA device")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        errors.append("ONNX Runtime CUDAExecutionProvider is unavailable")

    repo = Path(__file__).resolve().parent.parent
    free_bytes = shutil.disk_usage(repo).free
    if free_bytes < 10 * 1024**3:
        errors.append("less than 10 GiB disk space remains")

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{config.llm.base_url.rstrip('/')}/v1/models")
            response.raise_for_status()
            models = {item.get("id") for item in response.json().get("data", [])}
            if config.llm.model not in models:
                errors.append(
                    f"pinned model {config.llm.model!r} is not served by vLLM"
                )
    except Exception as exc:
        errors.append(f"vLLM readiness check failed: {type(exc).__name__}")

    env_file = Path(
        os.environ.get(
            "REACHY_ENV_FILE",
            str(Path.home() / ".config" / "reachy-assistant" / "env"),
        )
    )
    if config.runtime.production_mode:
        if not env_file.exists():
            errors.append(f"production environment file is missing: {env_file}")
        elif env_file.stat().st_mode & 0o077:
            errors.append(f"production environment file must have mode 600: {env_file}")

    if errors:
        for error in errors:
            print(f"PREFLIGHT_ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "PREFLIGHT_OK: configuration, CUDA runtimes, pinned VLM, disk, and secret permissions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
