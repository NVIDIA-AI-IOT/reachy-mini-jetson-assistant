#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the installed JP7.2 candidate wheels with real CUDA execution."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import ctranslate2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("packaging/jetson-wheels.json"),
    )
    parser.add_argument(
        "--platform-id",
        default="jp72-l4t39.2-cu13.2-py3.12-sm87",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wheels = manifest["platforms"][args.platform_id]["wheels"]
    assert ctranslate2.__version__ == wheels["ctranslate2"]["version"]
    assert ort.__version__ == wheels["onnxruntime_gpu"]["version"]
    assert ctranslate2.get_cuda_device_count() > 0
    assert "CUDAExecutionProvider" in ort.get_available_providers()

    generator = np.random.default_rng(7)
    input_array = generator.standard_normal(
        (1, 3, 32, 32), dtype=np.float32
    )
    weights = generator.standard_normal((8, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["X", "W"], ["Y"], pads=[1, 1, 1, 1]
            )
        ],
        "jp72-cuda-cudnn-smoke",
        [
            helper.make_tensor_value_info(
                "X", TensorProto.FLOAT, input_array.shape
            )
        ],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, (1, 8, 32, 32)
            )
        ],
        [numpy_helper.from_array(weights, "W")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 10

    with tempfile.TemporaryDirectory(
        prefix="reachy-ort-cuda-smoke-"
    ) as directory:
        model_path = Path(directory) / "conv.onnx"
        onnx.save(model, model_path)
        session = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider"]
        )
        session.disable_fallback()
        input_gpu = ort.OrtValue.ortvalue_from_numpy(
            input_array, "cuda", 0
        )
        binding = session.io_binding()
        binding.bind_ortvalue_input("X", input_gpu)
        binding.bind_output("Y", "cuda", 0)
        session.run_with_iobinding(binding)
        output = binding.get_outputs()[0]
        assert output.device_name() == "cuda"
        assert output.shape() == [1, 8, 32, 32]

    result = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform_id": args.platform_id,
        "ctranslate2_version": ctranslate2.__version__,
        "ctranslate2_cuda_devices": ctranslate2.get_cuda_device_count(),
        "onnxruntime_version": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "onnxruntime_output_device": output.device_name(),
        "onnxruntime_output_shape": output.shape(),
        "passed": True,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
