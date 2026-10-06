#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${VENV:-$repo/venv-thor}"
onnx_url="https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/012/c10bef23a39f0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl"
onnx_sha="012c10bef23a39f074730d158b72f797a7314f2695c65835a0669b57282422f6"

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip==26.1.1 setuptools==81.0.0 wheel==0.47.0
"$venv/bin/python" -m pip install -r "$repo/requirements-thor.txt"
"$venv/bin/python" -m pip uninstall -y ctranslate2 onnxruntime onnxruntime-gpu || true
"$venv/bin/python" -m pip install --no-deps "$onnx_url#sha256=$onnx_sha"
VENV="$venv" "$repo/scripts/build_ctranslate2_thor.sh"
"$venv/bin/python" "$repo/scripts/verify_runtime_manifest.py"
