#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load faster-whisper and Kokoro and exercise one inference on each."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import Config
from app.stt import STT
from app.tts import create_tts


def main() -> int:
    config = Config.load()
    stt = STT(
        model=config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    if not stt.load():
        print("SPEECH_CHECK_FAILED: faster-whisper did not load")
        return 1
    silence = np.zeros(16000, dtype=np.float32)
    stt_result = stt.transcribe(silence)
    if "error" in stt_result:
        print(f"SPEECH_CHECK_FAILED: STT inference: {stt_result['error']}")
        return 1
    print(f"STT_CHECK_OK: {stt.get_info()}")

    tts = create_tts(
        voice=config.tts.voice,
        speed=config.tts.speed,
        lang=config.tts.lang,
    )
    try:
        if not tts.load():
            print("SPEECH_CHECK_FAILED: Kokoro did not load")
            return 1
        result = tts.synthesize("Reachy speech pipeline check.")
        audio = result.get("audio")
        if audio is None or len(audio) == 0:
            print(f"SPEECH_CHECK_FAILED: TTS inference: {result.get('error', 'no audio')}")
            return 1
        print(
            f"TTS_CHECK_OK: backend={tts.backend_name} voice={tts.voice} "
            f"provider={tts.provider} samples={len(audio)} "
            f"rate={result.get('sample_rate')}"
        )
    finally:
        tts.unload()
        stt.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
