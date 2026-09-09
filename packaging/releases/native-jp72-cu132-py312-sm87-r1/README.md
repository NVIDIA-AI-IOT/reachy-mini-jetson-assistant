# JP7.2 native wheel release evidence

Status: **PUBLISHED**. The wheel audit and Jetson Orin Nano runtime validation pass for the compatibility tuple below.

This directory is the evidence bundle for `native-jp72-cu132-py312-sm87-r1`. The wheel files remain external release assets and are identified by `SHA256SUMS` and `packaging/jetson-wheels.json`.

`EVIDENCE-SHA256SUMS` authenticates every file in this evidence directory except itself. The nested checksum files under `LICENSES/`, `PATCHES/`, and `build-config/` make those release-bundle sections independently verifiable.

## Validation status

| # | Validation item | Status | Evidence |
|---|---|---|---|
| 1 | Confirm the wheels do not bundle NVIDIA CUDA/cuDNN binaries | Pass | `wheel-inventory.json` inventories every member by content magic and filename; it reports zero bundled NVIDIA library candidates. |
| 2 | Inventory every shared object, static library, object, and dependency | Pass | All 369 non-directory members, six AArch64 ELF shared objects, zero static libraries/objects, Python `Requires-Dist` entries, ELF `DT_NEEDED` entries, fetched build dependencies, and runtime dependencies are in `wheel-inventory.json`, `source-dependencies.json`, and `sbom.spdx.json`. |
| 3 | Include upstream MIT licenses in each wheel and release bundle | Pass | CTranslate2’s MIT license is in its wheel license directory and `LICENSES/ctranslate2/`. ONNX Runtime’s MIT license is in the package, wheel metadata license directory, and `LICENSES/onnxruntime/`. |
| 4 | Preserve ONNX Runtime’s complete third-party notices | Pass | The wheel copy, upstream v1.28.0 copy, and release copy of `ThirdPartyNotices.txt` have SHA-256 `0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2`. The unchanged file includes Eigen’s MPL-2.0 text. |
| 5 | Provide the exact ONNX Runtime patch and mark the build modified | Pass | `PATCHES/onnxruntime-v1.28.0-jp72.patch`, SHA-256 `035b1afa6e81e216351d1449186a2f2b9485ef5bed5cbfa1685dbb7119bf5786`; the local version and build notice identify the downstream modification. |
| 6 | Record upstream tags and commit SHAs | Pass | `BUILD.md`, `packaging/jetson-wheels.json`, and `source-dependencies.json`. |
| 7 | Provide reproducible commands and complete build environment | Pass | `BUILD.md`, `build-environment.txt`, `dpkg-packages.txt`, `python-packages.txt`, `environment-SHA256SUMS`, and both exact CMake configure caches in `build-config/`. |
| 8 | Confirm dynamic use of JetPack CUDA/cuDNN | Pass | ELF resolution in `wheel-inventory.json` points outside the wheels to JetPack packages `cuda-cudart-13-2`, `libcublas-13-2`, and `libcudnn9-cuda-13`; no dependency is unresolved. `runtime-validation.json` records a CUDA-bound ONNX Runtime convolution and CTranslate2 CUDA device discovery. |

## Release artifacts

| Artifact | Size | SHA-256 |
|---|---:|---|
| `ctranslate2-4.8.1+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl` | 17,999,182 | `cd553ca1932127d16e4c2d2e38599b2e70a2a4ee8b7d06b1ec0eb37a1d38185a` |
| `onnxruntime_gpu-1.28.0+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl` | 55,961,323 | `a4dd9c97ba19eaaeb33a00a3ac8e7ada4da35cd935be15800dbbc56437126ba5` |

## Native artifact summary

CTranslate2 contains `ctranslate2/_ext.cpython-312-aarch64-linux-gnu.so` and `ctranslate2/libctranslate2.so.4`. ONNX Runtime contains `libonnxruntime.so.1.28.0`, `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so`, and `onnxruntime_pybind11_state.so`. All six are dynamically linked AArch64 ELF files, and neither wheel contains a `.a`, `.o`, `.obj`, NVIDIA runtime library, or NVIDIA driver library.

The ONNX Runtime build graph listed JetPack’s `libculibos.a` as a normal linker input. The finished, unstripped CUDA provider contains zero `culibos*` symbols and zero `culibos` strings, and the link used garbage collection without `--whole-archive`; this is evidence that no archive member from `libculibos.a` was incorporated. The wheel audit fails if a `culibos*` symbol is found.

## Recreate the evidence

```bash
./scripts/capture_jetson_build_environment.sh packaging/releases/native-jp72-cu132-py312-sm87-r1 /absolute/path/to/build-python
python scripts/audit_jetson_wheels.py --manifest packaging/jetson-wheels.json --platform-id jp72-l4t39.2-cu13.2-py3.12-sm87 --wheel-dir /absolute/path/to/wheelhouse --output-dir packaging/releases/native-jp72-cu132-py312-sm87-r1 --source-dependencies packaging/releases/native-jp72-cu132-py312-sm87-r1/source-dependencies.json
python scripts/validate_jp72_gpu_wheels.py --output packaging/releases/native-jp72-cu132-py312-sm87-r1/runtime-validation.json
```

`release.published: true` records that the immutable release assets are available to the exact-match installer.
