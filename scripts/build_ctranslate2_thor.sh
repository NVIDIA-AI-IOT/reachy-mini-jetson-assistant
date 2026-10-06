#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

commit="0d8bcd362ac75ef860ef161d6f0efad0ae439ff0"
source_dir="${CTRANSLATE2_SOURCE_DIR:-$HOME/src/ctranslate2-4.8.1-thor}"
prefix="${CTRANSLATE2_PREFIX:-$HOME/.local}"
venv="${VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/venv-thor}"

for command in git cmake nvcc python3; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done
[[ -x "$venv/bin/python" ]] || { echo "Missing virtual environment: $venv" >&2; exit 1; }

if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/OpenNMT/CTranslate2.git "$source_dir"
fi
git -C "$source_dir" fetch --tags origin
git -C "$source_dir" checkout --detach "$commit"

python3 - "$source_dir/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "  cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})"
replacement = """  # CMake 3.28's legacy FindCUDA helper only accepts one-digit major
  # compute capabilities. Jetson Thor is sm_110, so pass its gencode flag
  # directly until this project migrates away from FindCUDA.
  if(CUDA_ARCH_LIST STREQUAL \"11.0\")
    set(ARCH_FLAGS \"-gencode;arch=compute_110,code=sm_110\")
  else()
    cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
  endif()"""
if replacement not in text:
    if needle not in text:
        raise SystemExit("CTranslate2 CMake CUDA-architecture hook changed; refusing to patch")
    path.write_text(text.replace(needle, replacement, 1))
PY

cmake -S "$source_dir" -B "$source_dir/build-thor" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_SHARED_LIBS=ON \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DCUDA_ARCH_LIST=11.0 \
  -DOPENMP_RUNTIME=NONE
cmake --build "$source_dir/build-thor" --parallel "$(nproc)"
cmake --install "$source_dir/build-thor"

"$venv/bin/python" -m pip install --upgrade build pybind11
pushd "$source_dir/python" >/dev/null
rm -rf build dist-gpu
CTRANSLATE2_ROOT="$prefix" "$venv/bin/python" -m build --wheel --outdir dist-gpu
wheel="$(find dist-gpu -maxdepth 1 -name 'ctranslate2-4.8.1-*.whl' -print -quit)"
[[ -n "$wheel" ]] || { echo "CTranslate2 wheel was not produced" >&2; exit 1; }
"$venv/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
sha256sum "$wheel"
popd >/dev/null

LD_LIBRARY_PATH="$prefix/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/lib/aarch64-linux-gnu" \
  "$venv/bin/python" -c 'import ctranslate2; assert ctranslate2.get_cuda_device_count() > 0; print("CTranslate2 CUDA OK")'
