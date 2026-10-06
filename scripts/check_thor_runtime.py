#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report whether the Thor STT, TTS, CUDA, and Reachy runtime is importable."""

import ctranslate2
import faster_whisper
import kokoro_onnx  # noqa: F401
import onnxruntime
import reachy_mini  # noqa: F401
import torch


def main() -> None:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    ctranslate_cuda_devices = ctranslate2.get_cuda_device_count()
    onnx_providers = onnxruntime.get_available_providers()
    print(
        "CTranslate2",
        ctranslate2.__version__,
        "CUDA devices",
        ctranslate_cuda_devices,
    )
    print("Torch", torch.__version__, "CUDA", cuda_available, device_name)
    print(
        "ONNX Runtime",
        onnxruntime.__version__,
        onnx_providers,
    )
    print("faster-whisper", faster_whisper.__version__)
    print("Kokoro and Reachy SDK imports: OK")

    errors = []
    if not cuda_available:
        errors.append("PyTorch cannot see CUDA")
    if ctranslate_cuda_devices < 1:
        errors.append("CTranslate2 cannot see CUDA")
    if "CUDAExecutionProvider" not in onnx_providers:
        errors.append("ONNX Runtime CUDAExecutionProvider is unavailable")
    if errors:
        raise SystemExit("GPU_RUNTIME_CHECK_FAILED: " + "; ".join(errors))
    print("GPU_RUNTIME_CHECK_OK: all model runtimes have GPU execution")


if __name__ == "__main__":
    main()
