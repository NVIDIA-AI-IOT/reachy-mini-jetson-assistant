# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from rich.console import Console

from app.pipeline import MicRecorder


def test_quiet_tail_resumes_after_consecutive_quiet_frames():
    mic = MicRecorder(Console(), chunk_ms=10)

    def publish_levels():
        for rms in (0.02, 0.01, 0.003, 0.002):
            time.sleep(0.01)
            with mic._capture_condition:
                mic._latest_capture_rms = rms
                mic._capture_sequence += 1
                mic._capture_condition.notify_all()

    producer = threading.Thread(target=publish_levels)
    producer.start()
    assert mic.wait_for_quiet_tail(quiet_ms=20, max_wait_ms=250, rms_threshold=0.004)
    producer.join()


def test_quiet_tail_timeout_is_bounded():
    mic = MicRecorder(Console(), chunk_ms=10)

    def publish_noise():
        for _ in range(6):
            time.sleep(0.01)
            with mic._capture_condition:
                mic._latest_capture_rms = 0.02
                mic._capture_sequence += 1
                mic._capture_condition.notify_all()

    producer = threading.Thread(target=publish_noise)
    producer.start()
    assert not mic.wait_for_quiet_tail(quiet_ms=20, max_wait_ms=80, rms_threshold=0.004)
    producer.join()
