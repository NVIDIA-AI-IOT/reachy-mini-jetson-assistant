# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from app.stt import STT


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        segments = [
            SimpleNamespace(
                text=" What color is the bottle?",
                start=0.0,
                end=2.0,
                avg_logprob=-0.2,
                no_speech_prob=0.03,
            )
        ]
        return iter(segments), SimpleNamespace(language="en", duration=2.0)


def test_short_question_uses_quality_decoding_settings():
    stt = STT(model="large-v3", beam_size=5)
    stt._model = FakeModel()

    result = stt.transcribe(np.zeros(32000, dtype=np.float32))

    assert result["text"] == "What color is the bottle?"
    assert result["avg_logprob"] == -0.2
    assert result["no_speech_prob"] == 0.03
    assert stt._model.kwargs["beam_size"] == 5
    assert stt._model.kwargs["temperature"] == 0.0
    assert stt._model.kwargs["condition_on_previous_text"] is False
    assert stt._model.kwargs["no_speech_threshold"] == 0.6
    assert stt._model.kwargs["log_prob_threshold"] == -0.8
