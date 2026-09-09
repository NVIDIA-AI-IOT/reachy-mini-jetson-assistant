# JetPack 7.2 native GPU wheels for Reachy Mini Jetson Assistant

This prerelease provides the exact native GPU runtime wheels used by `./scripts/setup_jetson.sh` on the validated JetPack 7.2 compatibility tuple.

## Supported tuple

- NVIDIA Jetson Orin family (`sm_87`)
- JetPack 7.2 / L4T 39.2.0
- CUDA 13.2 and cuDNN 9 supplied by JetPack
- Ubuntu 24.04 on `aarch64`
- CPython 3.12

These wheels are not compatible with JetPack 6, Thor (`sm_110`), another Python ABI, or a generic aarch64 machine. The installer fails closed when the host does not exactly match the manifest.

## Assets

- `ctranslate2-4.8.1+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl`
- `onnxruntime_gpu-1.28.0+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl`
- `reachy-mini-jetson-assistant-native-jp72-cu132-py312-sm87-r1-evidence.tar.gz`
- `reachy-mini-jetson-assistant-native-jp72-cu132-py312-sm87-r1-licenses.tar.gz`
- `native-jp72-cu132-py312-sm87-r1-SHA256SUMS`

The wheels dynamically use CUDA, cuBLAS, and cuDNN from JetPack; they do not bundle NVIDIA CUDA/cuDNN runtime libraries. The evidence archive contains the complete wheel member and ELF dependency inventory, SPDX 2.3 SBOM, upstream revisions, exact ONNX Runtime patch, build commands and environment, runtime validation, licenses, third-party notices, and nested checksums.

## Install

```bash
git clone --branch jetpack-7-2-port https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant.git
cd reachy-mini-jetson-assistant
./scripts/setup_jetson.sh --check-only
./scripts/setup_jetson.sh
```

## Release status

The local technical audit and Jetson Orin Nano runtime validation pass. The organization-approved NVIDIA license/security scanner report and final OSRB closure evidence are not included. This is therefore a prerelease and must not be described as a final/general-availability release.
