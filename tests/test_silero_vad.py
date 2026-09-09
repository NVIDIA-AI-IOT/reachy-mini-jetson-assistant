# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from app.pipeline import SILERO_CHUNK_SAMPLES, SileroVAD


def test_silero_vad_streaming_state_and_reset():
    vad = SileroVAD()
    silence = np.zeros(SILERO_CHUNK_SAMPLES, dtype=np.int16).tobytes()

    probability = vad(silence)
    assert 0.0 <= probability <= 1.0
    assert np.any(vad._h) or np.any(vad._c)

    vad.reset()
    assert not np.any(vad._h)
    assert not np.any(vad._c)
    assert not np.any(vad._context)


def test_silero_vad_rejects_wrong_frame_size():
    vad = SileroVAD()
    short_frame = np.zeros(SILERO_CHUNK_SAMPLES - 1, dtype=np.int16).tobytes()

    with pytest.raises(ValueError, match="expects 512 samples"):
        vad(short_frame)
