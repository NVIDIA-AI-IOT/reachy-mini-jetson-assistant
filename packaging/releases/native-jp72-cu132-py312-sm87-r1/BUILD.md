# Reproducing the JP7.2 wheel candidates

This record describes the candidate artifacts named in `packaging/jetson-wheels.json`. It is engineering evidence for OSRB review, not a legal conclusion or permission to redistribute NVIDIA software.

## Compatibility tuple

| Field | Value |
|---|---|
| Release tag | `native-jp72-cu132-py312-sm87-r1` |
| Architecture | `aarch64` |
| Jetson | Orin |
| L4T / JetPack | `39.2.0` / JetPack 7.2 |
| Ubuntu | 24.04 |
| CUDA | 13.2 |
| cuDNN | 9.20.0 |
| Python ABI | CPython 3.12, `cp312-cp312-linux_aarch64` |
| GPU target | Orin, SM 8.7 |

The exact installed APT packages, Python packages, OS, L4T release, compiler versions, and CUDA compiler output are archived in `build-environment.txt`, `dpkg-packages.txt`, and `python-packages.txt`. The configure state used by each build is preserved in `build-config/ctranslate2-CMakeCache.txt` and `build-config/onnxruntime-CMakeCache.txt`. Both candidate wheels record `Generator: setuptools (84.0.0)`.

## Source identity

| Project | Upstream tag | Tag object | Commit | Downstream status |
|---|---|---|---|---|
| CTranslate2 | `v4.8.1` | `399239a790ad0da4e4363e0dcbb83495b5abd742` | `0d8bcd362ac75ef860ef161d6f0efad0ae439ff0` | C++ source unmodified; wheel packaging and local version modified |
| ONNX Runtime | `v1.28.0` | lightweight tag | `da9b5e364c465de65c49d91e696cd6485270757f` | source and wheel packaging modified |

The exact ONNX Runtime source patch is `PATCHES/onnxruntime-v1.28.0-jp72.patch`, SHA-256 `035b1afa6e81e216351d1449186a2f2b9485ef5bed5cbfa1685dbb7119bf5786`. The local version `1.28.0+reachy.jp72.cu132.sm87`, wheel metadata, and bundled build notice all identify the build as modified.

## System prerequisites

```bash
sudo apt update
sudo apt install -y cuda-toolkit-13-2 libcudnn9-dev-cuda-13 build-essential cmake ninja-build git patchelf pkg-config python3-dev python3.12-venv
python3 -m venv /absolute/path/to/build-venv
source /absolute/path/to/build-venv/bin/activate
python -m pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0 build==1.5.0 pybind11==3.1.0 numpy==2.5.2
```

Use at least 8 GB of swap on an Orin Nano 8GB while building ONNX Runtime. The build captured here used GCC/G++ 13.3.0, GNU ld 2.42, CMake 3.28.3, Ninja 1.11.1, CUDA compiler 13.2.78, and glibc 2.39.

## CTranslate2

The following commands build the shared core for SM 8.7. Use new empty directories for every release candidate.

```bash
export REACHY_BUILD_ROOT=/absolute/path/to/reachy-wheel-build
export CT2_SOURCE="$REACHY_BUILD_ROOT/CTranslate2-v4.8.1"
export CT2_BUILD="$REACHY_BUILD_ROOT/ctranslate2-build"
export CT2_PREFIX="$REACHY_BUILD_ROOT/ctranslate2-prefix"
export CT2_STAGE="$REACHY_BUILD_ROOT/ctranslate2-wheel"
git clone --branch v4.8.1 --recurse-submodules https://github.com/OpenNMT/CTranslate2.git "$CT2_SOURCE"
test "$(git -C "$CT2_SOURCE" rev-parse HEAD)" = 0d8bcd362ac75ef860ef161d6f0efad0ae439ff0
test "$(git -C "$CT2_SOURCE" rev-parse refs/tags/v4.8.1)" = 399239a790ad0da4e4363e0dcbb83495b5abd742
test "$(git -C "$CT2_SOURCE" rev-parse refs/tags/v4.8.1^{})" = 0d8bcd362ac75ef860ef161d6f0efad0ae439ff0
cmake -S "$CT2_SOURCE" -B "$CT2_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_ARCH_LIST=8.7 -DWITH_MKL=OFF -DWITH_OPENBLAS=OFF -DWITH_RUY=OFF -DWITH_DNNL=OFF -DOPENMP_RUNTIME=NONE -DBUILD_CLI=OFF -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$CT2_PREFIX"
cmake --build "$CT2_BUILD" --parallel 2
cmake --install "$CT2_BUILD"
cp -a "$CT2_SOURCE/python" "$CT2_STAGE"
install -m 0755 "$CT2_PREFIX/lib/libctranslate2.so.4.8.1" "$CT2_STAGE/ctranslate2/libctranslate2.so.4"
```

Apply the exact downstream staging transformations. `RELEASE_EVIDENCE` must point to this release-evidence directory.

```bash
export RELEASE_EVIDENCE=/absolute/path/to/repository/packaging/releases/native-jp72-cu132-py312-sm87-r1
python - "$CT2_STAGE" "$RELEASE_EVIDENCE/LICENSES/ctranslate2" <<'PY'
from pathlib import Path
import shutil
import sys

stage = Path(sys.argv[1])
licenses = Path(sys.argv[2])
version_file = stage / "ctranslate2/version.py"
setup_file = stage / "setup.py"
version_text = version_file.read_text(encoding="utf-8")
old_version = '__version__ = "4.8.1"'
new_version = '__version__ = "4.8.1+reachy.jp72.cu132.sm87"'
assert version_text.count(old_version) == 1
version_file.write_text(version_text.replace(old_version, new_version), encoding="utf-8")
setup_text = setup_file.read_text(encoding="utf-8")
old_linux = '    ldflags.append("-Wl,-rpath,/usr/local/lib64:/usr/local/lib")'
new_linux = '    ldflags.append("-Wl,-rpath,$ORIGIN")\n    package_data["ctranslate2"] = ["libctranslate2.so.4"]'
assert setup_text.count(old_linux) == 1
setup_text = setup_text.replace(old_linux, new_linux)
old_license = '    license="MIT",'
new_license = '    license="MIT",\n    license_files=["LICENSES/*"],'
assert setup_text.count(old_license) == 1
setup_file.write_text(setup_text.replace(old_license, new_license), encoding="utf-8")
shutil.copytree(licenses, stage / "LICENSES")
PY
export CTRANSLATE2_ROOT="$CT2_PREFIX"
export CMAKE_BUILD_PARALLEL_LEVEL=2
cd "$CT2_STAGE"
python -m build --wheel --no-isolation
```

The resulting CTranslate2 wheel must contain the Python extension, the CTranslate2 core shared library, and the complete `LICENSES` directory. It must not contain CUDA, cuBLAS, or cuDNN shared/static libraries.

## ONNX Runtime

The ONNX Runtime source patch extends its CUDA 13.3 CCCL workaround to CUDA 13.2 and enables `ORT_QUICK_BUILD=1` only for the separately linked CUDA LLM/MoE object library. It does not modify Eigen or other third-party source.

```bash
export REACHY_REPO=/absolute/path/to/repository
export RELEASE_EVIDENCE="$REACHY_REPO/packaging/releases/native-jp72-cu132-py312-sm87-r1"
export ORT_SOURCE="$REACHY_BUILD_ROOT/onnxruntime-v1.28.0"
export ORT_BUILD="$ORT_SOURCE/build-jp72"
export CUDNN_DISCOVERY="$REACHY_BUILD_ROOT/cudnn-jp72"
git clone --branch v1.28.0 --recursive https://github.com/microsoft/onnxruntime.git "$ORT_SOURCE"
test "$(git -C "$ORT_SOURCE" rev-parse HEAD)" = da9b5e364c465de65c49d91e696cd6485270757f
test "$(sha256sum "$REACHY_REPO/patches/onnxruntime-v1.28.0-jp72.patch" | awk '{print $1}')" = 035b1afa6e81e216351d1449186a2f2b9485ef5bed5cbfa1685dbb7119bf5786
git -C "$ORT_SOURCE" apply "$REACHY_REPO/patches/onnxruntime-v1.28.0-jp72.patch"
mkdir -p "$CUDNN_DISCOVERY/include" "$CUDNN_DISCOVERY/lib"
ln -s /usr/include/aarch64-linux-gnu/cudnn.h /usr/include/aarch64-linux-gnu/cudnn_adv.h /usr/include/aarch64-linux-gnu/cudnn_adv_v9.h /usr/include/aarch64-linux-gnu/cudnn_backend.h /usr/include/aarch64-linux-gnu/cudnn_backend_v9.h /usr/include/aarch64-linux-gnu/cudnn_cnn.h /usr/include/aarch64-linux-gnu/cudnn_cnn_v9.h /usr/include/aarch64-linux-gnu/cudnn_graph.h /usr/include/aarch64-linux-gnu/cudnn_graph_v9.h /usr/include/aarch64-linux-gnu/cudnn_ops.h /usr/include/aarch64-linux-gnu/cudnn_ops_v9.h /usr/include/aarch64-linux-gnu/cudnn_version.h /usr/include/aarch64-linux-gnu/cudnn_version_v9.h "$CUDNN_DISCOVERY/include/"
ln -s /usr/lib/aarch64-linux-gnu/libcudnn.so /usr/lib/aarch64-linux-gnu/libcudnn.so.9 "$CUDNN_DISCOVERY/lib/"
"$ORT_SOURCE/build.sh" --build_dir "$ORT_BUILD" --config Release --update --build --build_wheel --skip_tests --skip_submodule_sync --parallel 4 --nvcc_threads 1 --compile_no_warning_as_error --use_cuda --disable_cuda_nhwc_ops --disable_ml_ops --disable_generation_ops --cuda_version 13.2 --cuda_home /usr/local/cuda --cudnn_home "$CUDNN_DISCOVERY" --cmake_generator Ninja --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF
```

The generated upstream wheel is then repacked to preserve the binaries while adding an explicit local version and release notices. The complete upstream `LICENSE` and `ThirdPartyNotices.txt` already present under `onnxruntime/` are copied byte-for-byte into `.dist-info/licenses/`.

```bash
export ORT_UPSTREAM_WHEEL="$ORT_BUILD/Release/dist/onnxruntime_gpu-1.28.0-cp312-cp312-linux_aarch64.whl"
export ORT_STAGE="$REACHY_BUILD_ROOT/onnxruntime-wheel"
export ORT_DIST="$REACHY_BUILD_ROOT/onnxruntime-dist"
mkdir -p "$ORT_STAGE" "$ORT_DIST"
python -m wheel unpack "$ORT_UPSTREAM_WHEEL" --dest "$ORT_STAGE"
python - "$ORT_STAGE" "$RELEASE_EVIDENCE/LICENSES/onnxruntime/BUILD-AND-LICENSE-NOTICE.txt" <<'PY'
from pathlib import Path
import shutil
import sys

stage_parent = Path(sys.argv[1])
notice = Path(sys.argv[2])
roots = list(stage_parent.glob("onnxruntime_gpu-1.28.0"))
assert len(roots) == 1
root = roots[0]
package = root / "onnxruntime"
dist_info = root / "onnxruntime_gpu-1.28.0.dist-info"
metadata = dist_info / "METADATA"
build_info = package / "capi/build_and_package_info.py"
local_version = "1.28.0+reachy.jp72.cu132.sm87"
metadata_text = metadata.read_text(encoding="utf-8")
assert metadata_text.count("Version: 1.28.0\n") == 1
metadata_text = metadata_text.replace("Version: 1.28.0\n", f"Version: {local_version}\n")
anchor = "Requires-Python: >=3.11\n"
license_fields = "License-File: LICENSE\nLicense-File: ThirdPartyNotices.txt\nLicense-File: BUILD-AND-LICENSE-NOTICE.txt\n"
assert metadata_text.count(anchor) == 1
metadata.write_text(metadata_text.replace(anchor, anchor + license_fields), encoding="utf-8")
build_info_text = build_info.read_text(encoding="utf-8")
assert build_info_text.count("__version__ = '1.28.0'") == 1
build_info.write_text(build_info_text.replace("__version__ = '1.28.0'", f"__version__ = '{local_version}'"), encoding="utf-8")
shutil.copy2(notice, package / "BUILD-AND-LICENSE-NOTICE.txt")
licenses = dist_info / "licenses"
licenses.mkdir()
shutil.copy2(package / "LICENSE", licenses / "LICENSE")
shutil.copy2(package / "ThirdPartyNotices.txt", licenses / "ThirdPartyNotices.txt")
shutil.copy2(notice, licenses / "BUILD-AND-LICENSE-NOTICE.txt")
new_dist_info = root / f"onnxruntime_gpu-{local_version}.dist-info"
dist_info.rename(new_dist_info)
new_root = stage_parent / f"onnxruntime_gpu-{local_version}"
root.rename(new_root)
print(new_root)
PY
python -m wheel pack "$ORT_STAGE/onnxruntime_gpu-1.28.0+reachy.jp72.cu132.sm87" --dest-dir "$ORT_DIST"
```

## Audit and runtime validation

Place both resulting wheels in one wheelhouse. Update the candidate filenames, sizes, and SHA-256 values in `packaging/jetson-wheels.json` before running the audit; a rebuild is a new artifact and must never reuse stale checksums.

```bash
python scripts/audit_jetson_wheels.py --manifest packaging/jetson-wheels.json --platform-id jp72-l4t39.2-cu13.2-py3.12-sm87 --wheel-dir /absolute/path/to/wheelhouse --output-dir packaging/releases/native-jp72-cu132-py312-sm87-r1 --source-dependencies packaging/releases/native-jp72-cu132-py312-sm87-r1/source-dependencies.json
python3 -m venv /absolute/path/to/validation-venv
source /absolute/path/to/validation-venv/bin/activate
python -m pip install numpy==2.5.2 onnx==1.22.0
python -m pip install --no-deps /absolute/path/to/wheelhouse/ctranslate2-*.whl /absolute/path/to/wheelhouse/onnxruntime_gpu-*.whl
python scripts/validate_jp72_gpu_wheels.py --output packaging/releases/native-jp72-cu132-py312-sm87-r1/runtime-validation.json
```

The release audit must report zero bundled NVIDIA library candidates, zero static libraries/object files in either wheel, no unresolved ELF dependencies, and JetPack package ownership for every CUDA/cuBLAS/cuDNN resolution. Compare the generated CMake caches with the copies in `build-config/` when reproducing the candidates. Run the organization-approved license and security scanner separately and attach its unmodified report before submission.
