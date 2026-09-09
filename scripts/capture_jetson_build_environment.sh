#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR [BUILD_PYTHON]" >&2
  exit 2
fi

output_dir="$1"
build_python="${2:-python3}"
mkdir -p "$output_dir"

command -v "$build_python" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $build_python" >&2
  exit 1
}

{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -a)"
  echo "machine=$(uname -m)"
  echo "python=$("$build_python" --version 2>&1)"
  echo
  echo "[os-release]"
  sed -n '1,200p' /etc/os-release
  echo
  echo "[l4t]"
  if [[ -r /etc/nv_tegra_release ]]; then
    sed -n '1,40p' /etc/nv_tegra_release
  else
    echo "unavailable"
  fi
  echo
  echo "[device]"
  if [[ -r /proc/device-tree/model ]]; then
    tr '\0' '\n' </proc/device-tree/model
  else
    echo "unavailable"
  fi
  echo
  echo "[cuda]"
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version
  else
    echo "nvcc unavailable"
  fi
  echo
  echo "[compilers]"
  gcc --version | sed -n '1p'
  g++ --version | sed -n '1p'
  ld --version | sed -n '1p'
  cmake --version | sed -n '1p'
  ninja --version
  ldd --version | sed -n '1p'
  echo
  echo "[python-build-tools]"
  "$build_python" -m pip show pip setuptools wheel build pybind11 numpy 2>/dev/null |
    sed -n '/^Name:/p;/^Version:/p' || true
  echo
  echo "[jetpack-runtime-packages]"
  runtime_packages=(nvidia-l4t-core cuda-toolkit-13-2 cuda-cudart-13-2 libcublas-13-2 libcudnn9-cuda-13 libcudnn9-dev-cuda-13)
  dpkg-query -W -f='${binary:Package}\t${Version}\n' "${runtime_packages[@]}" 2>/dev/null |
    sort || true
  echo
  echo "[power-mode-capture]"
  if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -q 2>&1 || true
  else
    echo "nvpmodel unavailable"
  fi
  if command -v jetson_clocks >/dev/null 2>&1; then
    jetson_clocks --show 2>&1 || true
  else
    echo "jetson_clocks unavailable"
  fi
} >"$output_dir/build-environment.txt"

dpkg-query -W -f='${binary:Package}\t${Version}\n' |
  LC_ALL=C sort >"$output_dir/dpkg-packages.txt"
"$build_python" -m pip freeze --all |
  LC_ALL=C sort >"$output_dir/python-packages.txt"

(
  cd "$output_dir"
  sha256sum build-environment.txt dpkg-packages.txt python-packages.txt
) >"$output_dir/environment-SHA256SUMS"

echo "Build environment written to $output_dir"
