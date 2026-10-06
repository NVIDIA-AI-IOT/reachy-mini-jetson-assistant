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

"""Test different system prompts and user prompts against the configured live VLM.

Sends the same camera frame with each combo and prints the full response.
Requires: VLM server running on localhost:8001, a frame at /tmp/test_frame.b64.
"""

import httpx
import json
import os
import time

BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8001")
MODEL = os.environ.get(
    "VLM_MODEL", "google/gemma-4-E4B-it"
)
IMG_B64_PATH = "/tmp/test_frame.b64"
MAX_TOKENS = 128
TEMPERATURE = 0.7

USER_PROMPTS = [
    "Thank you.",
    "Hello!",
    "What do you see?",
    "What color is my shirt?",
    "Tell me a joke.",
    "How are you doing?",
]


# ── Test configs: each is (name, system_prompt, few_shot_turns) ──
# few_shot_turns = list of {"role": "user"/"assistant", "content": ...} inserted
# between system prompt and the actual user query (multi-turn few-shot).

TESTS = {
    # -- Previous best --
    "B_minimal": {
        "system": (
            "You are Reachy Mini, a robot with a camera.\n"
            "You receive an image with every message. Only refer to the image when the user asks about what you see.\n"
            "For conversational turns, respond naturally without describing the scene.\n"
            "Keep responses to one or two spoken sentences.\n"
            "No markdown, emojis, or formatting."
        ),
        "few_shot": [],
    },

    # -- Few-shot in system prompt (text examples) --
    "D_fewshot_in_system": {
        "system": (
            "You are Reachy Mini, a robot with a camera.\n"
            "You receive an image with every message but should only use it when the user asks about the scene.\n"
            "Keep responses to one or two spoken sentences. No markdown or formatting.\n"
            "\n"
            "Examples:\n"
            "User: Hello! → Hi there!\n"
            "User: Thank you. → You're welcome.\n"
            "User: How are you? → I'm doing well, thanks for asking.\n"
            "User: What do you see? → [describe the image]\n"
            "User: What color is my shirt? → [answer from the image]"
        ),
        "few_shot": [],
    },

    # -- Few-shot examples as model-agnostic multi-turn chat messages --
    "E_fewshot_multiturn": {
        "system": (
            "You are Reachy Mini, a robot with a camera.\n"
            "You receive an image with every message but should only use it when the user asks about the scene.\n"
            "Keep responses to one or two spoken sentences. No markdown or formatting."
        ),
        "few_shot": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Thank you."},
            {"role": "assistant", "content": "You're welcome."},
            {"role": "user", "content": "How are you doing?"},
            {"role": "assistant", "content": "I'm doing great, thanks for asking."},
        ],
    },

    # -- Few-shot multi-turn with explicit image-ignore examples --
    "F_fewshot_multiturn_explicit": {
        "system": (
            "You are Reachy Mini, a robot with a camera.\n"
            "An image is attached to every message. Only describe or reference the image when the user asks about it.\n"
            "For all other messages, ignore the image and respond conversationally.\n"
            "Keep responses to one or two spoken sentences. No markdown or formatting."
        ),
        "few_shot": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Thank you."},
            {"role": "assistant", "content": "You're welcome."},
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the robot go on vacation? Because it needed to recharge."},
            {"role": "user", "content": "How are you doing?"},
            {"role": "assistant", "content": "Doing well, ready to help."},
        ],
    },

    # -- Larger few-shot with mixed vision + non-vision examples --
    "G_fewshot_mixed": {
        "system": (
            "You are Reachy Mini, a robot with a camera.\n"
            "An image is always attached. Only use the image when the user asks about what you see, colors, objects, or the scene.\n"
            "Otherwise ignore the image completely and respond naturally.\n"
            "One to two sentences max. No markdown or formatting."
        ),
        "few_shot": [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Thank you so much."},
            {"role": "assistant", "content": "You're welcome!"},
            {"role": "user", "content": "What's in front of you?"},
            {"role": "assistant", "content": "I can see a desk with a laptop and some papers on it."},
            {"role": "user", "content": "Tell me something fun."},
            {"role": "assistant", "content": "Did you know octopuses have three hearts? Two pump blood to the gills and one to the rest of the body."},
            {"role": "user", "content": "Okay, bye!"},
            {"role": "assistant", "content": "Goodbye, see you later!"},
        ],
    },
}


def query_vlm(system_prompt: str, few_shot: list, user_prompt: str, img_b64: str) -> tuple[str, float]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
    })
    t0 = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": MODEL, "messages": messages, "stream": False,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        })
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return f"[ERROR {r.status_code}] {r.text[:200]}", dt
    data = r.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return content.strip(), dt


def main():
    with open(IMG_B64_PATH) as f:
        img_b64 = f.read().strip()
    print(f"Image loaded ({len(img_b64)} chars base64)\n")

    for test_name, cfg in TESTS.items():
        n_fewshot = len(cfg["few_shot"]) // 2
        print(f"{'='*70}")
        print(f"  {test_name}  ({n_fewshot} few-shot pairs)")
        print(f"{'='*70}")
        for up in USER_PROMPTS:
            resp, dt = query_vlm(cfg["system"], cfg["few_shot"], up, img_b64)
            print(f"\n  User: \"{up}\"")
            print(f"  VLM ({dt:.1f}s): {resp}")
        print()


if __name__ == "__main__":
    main()
