# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import sys
from types import SimpleNamespace

import pytest

from app import tts_worker


CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


@pytest.fixture
def worker_runtime(monkeypatch, tmp_path):
    for filename in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        (tmp_path / filename).touch()
    providers = {"available": [CUDA, CPU], "session": [CUDA, CPU]}
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: providers["available"],
    ))

    def kokoro(model_path, voices_path):
        assert os.environ["ONNX_PROVIDER"] == CUDA
        return SimpleNamespace(sess=SimpleNamespace(
            get_providers=lambda: providers["session"],
        ))

    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=kokoro))
    monkeypatch.setenv("ONNX_PROVIDER", CPU)
    monkeypatch.setattr(sys, "argv", ["tts_worker.py", "--model-dir", str(tmp_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd": "shutdown"}\n'))
    return providers


def test_ready_protocol_reports_canonical_cuda_provider(worker_runtime, capsys):
    tts_worker.main()

    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.splitlines()]
    assert responses == [
        {"status": "ready", "provider": CUDA},
        {"status": "shutdown"},
    ]
    assert f"{CUDA} (required)" in captured.err


@pytest.mark.parametrize(
    "available,session,error",
    [
        ([CPU], [CPU], "GPU-required TTS needs"),
        ([CUDA, CPU], [CPU], "CUDA-primary session"),
        ([CUDA, CPU], [CPU, CUDA], "CUDA-primary session"),
        ([CUDA, CPU], [], "CUDA-primary session"),
    ],
)
def test_worker_rejects_missing_cuda_and_cpu_fallback(
    worker_runtime, capsys, available, session, error,
):
    worker_runtime.update(available=available, session=session)

    with pytest.raises(SystemExit) as stopped:
        tts_worker.main()

    assert stopped.value.code == 1
    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.splitlines()]
    assert len(responses) == 1
    assert responses[0]["status"] == "error"
    assert error in responses[0]["error"]
