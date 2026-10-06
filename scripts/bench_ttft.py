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

"""Benchmark TTFT across resolution, JPEG quality, and system prompt length.

Tests 8 combinations (2 resolutions x 2 JPEG qualities x 2 system prompts)
using streaming to measure true time-to-first-token.

Requires: VLM server on localhost:8001, USB camera available.
"""

import base64
import itertools
import os
import time

import cv2
import httpx

BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8001")
MODEL = os.environ.get(
    "VLM_MODEL", "google/gemma-4-E4B-it"
)
USER_PROMPT = "What do you see?"
MAX_TOKENS = 64
TEMPERATURE = 0.7
RUNS_PER_COMBO = 3

RESOLUTIONS = [
    ("640x480", 640, 480),
    ("320x240", 320, 240),
]

JPEG_QUALITIES = [
    ("q80", 80),
    ("q50", 50),
]

SYSTEM_PROMPTS = [
    ("full", (
        "You are Reachy Mini, a robot with a camera.\n"
        "An image is always attached. Only use the image when the user asks "
        "about what you see, colors, objects, or the scene.\n"
        "Otherwise ignore the image completely and respond naturally.\n"
        "Answer in one short sentence. No markdown or formatting."
    )),
    ("1-line", "You are a robot with a camera. Answer in one sentence."),
]


def capture_frame(cap, width, height, jpeg_quality):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        return None, 0
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None, 0
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
    return b64, len(jpg.tobytes())


def measure_ttft(system_prompt, img_b64):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
    })

    t0 = time.perf_counter()
    ttft = None
    full_resp = ""

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json={
            "model": MODEL, "messages": messages, "stream": True,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        }) as r:
            if r.status_code != 200:
                return None, None, f"[ERROR {r.status_code}]"
            for line in r.iter_lines():
                if not line or not line.strip().startswith("data:"):
                    continue
                line = line.strip()
                if line == "data: [DONE]":
                    break
                try:
                    import json
                    data = json.loads(line[5:])
                    content = ((data.get("choices") or [{}])[0]
                               .get("delta", {}).get("content", ""))
                    if content and ttft is None:
                        ttft = time.perf_counter() - t0
                    if content:
                        full_resp += content
                except Exception:
                    continue

    total = time.perf_counter() - t0
    return ttft, total, full_resp.strip()


def _open_camera():
    """Try Reachy SDK first, then iterate device indices."""
    try:
        from reachy_mini.media.camera_utils import find_camera
        cap, _ = find_camera()
        if cap and cap.isOpened():
            print("Camera opened via Reachy SDK")
            return cap
    except Exception:
        pass
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Camera opened at /dev/video{idx}")
            return cap
        cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()
    return None


def main():
    cap = _open_camera()
    if not cap:
        print("Cannot open camera! Is run_web_vision_chat.py holding it?")
        print("Stop the running app first: Ctrl-C in the terminal, then re-run this.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    combos = list(itertools.product(RESOLUTIONS, JPEG_QUALITIES, SYSTEM_PROMPTS))
    results = []

    print(f"Benchmarking {len(combos)} combinations x {RUNS_PER_COMBO} runs = {len(combos) * RUNS_PER_COMBO} total requests")
    print(f"User prompt: \"{USER_PROMPT}\"")
    print(f"Max tokens: {MAX_TOKENS}\n")

    for i, (res, qual, prompt) in enumerate(combos, 1):
        res_name, w, h = res
        qual_name, q = qual
        prompt_name, sp = prompt

        img_b64, jpg_bytes = capture_frame(cap, w, h, q)
        if not img_b64:
            print(f"  [{i}/{len(combos)}] SKIP — capture failed")
            continue

        cold_ttft = None
        warm_ttfts = []
        for run in range(RUNS_PER_COMBO):
            ttft, total, resp = measure_ttft(sp, img_b64)
            label = f"  [{i}/{len(combos)}] {res_name} {qual_name} {prompt_name} run{run+1}"
            if ttft:
                tag = "COLD" if run == 0 else "warm"
                print(f"{label}: TTFT={ttft:.2f}s total={total:.2f}s [{tag}] jpg={jpg_bytes/1024:.0f}KB")
                if run == 0:
                    cold_ttft = ttft
                else:
                    warm_ttfts.append(ttft)
            else:
                print(f"{label}: FAILED — {resp[:80]}")

        if cold_ttft is not None and warm_ttfts:
            avg_warm = sum(warm_ttfts) / len(warm_ttfts)
            results.append((res_name, qual_name, prompt_name, jpg_bytes, cold_ttft, avg_warm))

    cap.release()

    print(f"\n{'='*80}")
    print(f"  TTFT BENCHMARK RESULTS ({RUNS_PER_COMBO}-run averages)")
    print(f"{'='*80}")
    print(f"  {'Resolution':<12} {'JPEG':<6} {'Prompt':<8} {'JPG Size':<10} {'TTFT':<8} {'Total':<8} {'vs best'}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    if results:
        best_cold = min(r[4] for r in results)
        best_warm = min(r[5] for r in results)
        print(f"\n  {'Resolution':<12} {'JPEG':<6} {'Prompt':<8} {'JPG KB':<8} {'Cold TTFT':<10} {'Warm TTFT':<10} {'Speedup'}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for res, qual, prompt, jpg_bytes, cold, warm in sorted(results, key=lambda r: r[4]):
            speedup = f"{cold/warm:.1f}x" if warm > 0 else "--"
            print(f"  {res:<12} {qual:<6} {prompt:<8} {jpg_bytes/1024:>5.0f} KB  {cold:>7.2f}s   {warm:>7.2f}s    {speedup}")

    print()


if __name__ == "__main__":
    main()
