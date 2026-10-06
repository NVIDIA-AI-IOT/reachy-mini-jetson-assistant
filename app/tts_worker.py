#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TTS subprocess worker — runs in a SEPARATE process to isolate GPL-licensed
# dependencies (kokoro-onnx -> phonemizer-fork GPL-3.0, espeak-ng GPL-3.0)
# from the main process which loads NVIDIA CUDA libraries.
#
# Protocol: JSON lines over stdin (requests) / stdout (responses).
# Log messages go to stderr so they appear in the parent's terminal.

import sys
import json
import base64
import argparse
import os
from pathlib import Path

import numpy as np


def _respond(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS subprocess worker")
    parser.add_argument("--model-dir", required=True, help="Directory containing model files")
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--lang", default="en-us")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"

    if not model_path.exists() or not voices_path.exists():
        _respond({"status": "error", "error": f"Model files not found in {model_dir}"})
        sys.exit(1)

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "GPU-required TTS needs ONNX Runtime CUDAExecutionProvider; "
                f"available providers: {available}"
            )

        # kokoro-onnx uses this as its requested primary provider. ORT still
        # registers CPU for a small number of shape-control nodes in this
        # exported graph, but CUDA must own the model compute or startup fails.
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"

        from kokoro_onnx import Kokoro
        kokoro = Kokoro(str(model_path), str(voices_path))
        providers = kokoro.sess.get_providers()
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "Kokoro did not create a CUDA-primary session; "
                f"session providers: {providers}"
            )
        provider = providers[0]

        _log(f"Kokoro TTS loaded — ONNX provider: {provider} (required)")
        _respond({"status": "ready", "provider": provider})

    except Exception as e:
        _respond({"status": "error", "error": str(e)})
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"error": "Invalid JSON"})
            continue

        cmd = req.get("cmd", "synthesize")

        if cmd == "synthesize":
            text = req.get("text", "")
            voice = req.get("voice", args.voice)
            speed = req.get("speed", args.speed)
            lang = req.get("lang", args.lang)

            if not text.strip():
                _respond({"error": "Empty text"})
                continue

            try:
                samples, sample_rate = kokoro.create(
                    text, voice=voice, speed=speed, lang=lang,
                )
                audio_int16 = (samples * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("ascii")
                _respond({"audio_b64": audio_b64, "sample_rate": sample_rate})
            except Exception as e:
                _respond({"error": str(e)})

        elif cmd == "health":
            _respond({"healthy": True})

        elif cmd == "shutdown":
            _respond({"status": "shutdown"})
            break


if __name__ == "__main__":
    main()
