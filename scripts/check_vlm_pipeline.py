#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture one Reachy camera frame and send it through the configured VLM."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.camera import Camera
from app.config import Config
from app.llm import LLM


def main() -> int:
    config = Config.load()
    camera = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if not camera.start():
        print("VLM_CHECK_FAILED: camera did not open")
        return 1

    try:
        frame = None
        deadline = time.monotonic() + 5.0
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.1)
            frame = camera.read_live()
        if frame is None:
            print("VLM_CHECK_FAILED: camera produced no frame")
            return 1

        llm = LLM(
            model=config.llm.model,
            base_url=config.llm.base_url,
            backend=config.llm.backend,
            max_tokens=32,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout,
            system_prompt=config.vision.system_prompt,
            guardrail_config=config.guardrails,
        )
        if not llm.load():
            print(f"VLM_CHECK_FAILED: server unavailable at {config.llm.base_url}")
            return 1

        response = "".join(
            content
            for content, _metadata in llm.generate_stream(
                prompt="Describe what you see in one short sentence.",
                images_b64=[frame],
            )
        ).strip()
        if not response:
            print("VLM_CHECK_FAILED: model returned no text")
            return 1
        print(f"VLM_CHECK_OK: {response}")
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
