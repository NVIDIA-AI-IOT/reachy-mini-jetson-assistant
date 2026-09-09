#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "Usage: $0 <ctranslate2-wheel-path-or-URL> <onnxruntime-gpu-wheel-path-or-URL>" >&2
  echo "Activate the project Python 3.12 virtual environment first." >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: no active virtual environment" >&2
  usage
  exit 2
fi

python_bin="$VIRTUAL_ENV/bin/python"
ct2_wheel="$1"
ort_wheel="$2"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "error: JP7.2 wheels require an aarch64 Jetson" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"error: JP7.2 wheels require Python 3.12, found {sys.version.split()[0]}"
    )
PY

if [[ -r /etc/nv_tegra_release ]] && ! grep -q '^# R39 ' /etc/nv_tegra_release; then
  echo "error: these wheels target JetPack 7.2 / L4T r39" >&2
  exit 1
fi

"$python_bin" -m pip uninstall -y onnxruntime onnxruntime-gpu ctranslate2
"$python_bin" -m pip install --no-deps "$ct2_wheel" "$ort_wheel"

"$python_bin" - <<'PY'
import os

import ctranslate2
import onnxruntime as ort

cuda_devices = ctranslate2.get_cuda_device_count()
providers = ort.get_available_providers()
print(f"CTranslate2: {ctranslate2.__version__}; CUDA devices: {cuda_devices}")
print(f"ONNX Runtime: {ort.__version__}; providers: {providers}")
expected_ct2 = os.environ.get("REACHY_EXPECTED_CTRANSLATE2_VERSION")
expected_ort = os.environ.get("REACHY_EXPECTED_ONNXRUNTIME_VERSION")
if expected_ct2 and ctranslate2.__version__ != expected_ct2:
    raise SystemExit(
        f"error: CTranslate2 version {ctranslate2.__version__} != {expected_ct2}"
    )
if expected_ort and ort.__version__ != expected_ort:
    raise SystemExit(f"error: ONNX Runtime version {ort.__version__} != {expected_ort}")
if cuda_devices < 1:
    raise SystemExit("error: CTranslate2 cannot see a CUDA device")
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("error: ONNX Runtime has no CUDAExecutionProvider")
PY

echo "JP7.2 GPU runtime validation passed."
echo "Note: packages that declare 'onnxruntime' may make pip check report that"
echo "name as missing even though the onnxruntime-gpu module is installed."
