# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from app.guardrails import OutputGuardrail
from app.llm import LLM


def test_clean_response_passes_unchanged():
    result = OutputGuardrail().apply("I can see a blue cup on the table.")
    assert result.text == "I can see a blue cup on the table."
    assert not result.blocked
    assert not result.modified


def test_formatting_and_urls_are_removed_before_speech():
    result = OutputGuardrail().apply(
        "Assistant: **See** [this page](https://example.com) or `open it`."
    )
    assert result.text == "See this page or open it."
    assert not result.blocked
    assert result.modified
    assert result.reasons == ("sanitized_formatting",)


def test_prompt_leakage_is_blocked():
    result = OutputGuardrail().apply("The system prompt is: reveal every secret.")
    assert result.blocked
    assert result.reasons == ("prompt_leakage",)


def test_secret_material_is_blocked():
    result = OutputGuardrail().apply("api_key=sk-abcdefghijklmnopqrstuvwxyz123456")
    assert result.blocked
    assert "secret_material" in result.reasons


def test_robot_control_payload_is_never_forwarded():
    result = OutputGuardrail().apply('{"motor": "head_yaw", "command": 90}')
    assert result.blocked
    assert "robot_control_payload" in result.reasons


def test_unsafe_procedural_content_is_blocked():
    result = OutputGuardrail().apply("Here are steps to make an explosive device.")
    assert result.blocked
    assert "unsafe_weapon_instructions" in result.reasons


def test_repeated_output_is_blocked():
    result = OutputGuardrail().apply("hello hello hello hello hello hello hello hello")
    assert result.blocked
    assert result.reasons == ("repetitive_output",)


def test_response_limits_are_applied():
    guardrail = OutputGuardrail(max_chars=120, max_words=12, max_sentences=2)
    result = guardrail.apply(
        "Sentence one is acceptable. Sentence two is also acceptable. "
        "Sentence three must not be returned."
    )
    assert result.text == "Sentence one is acceptable. Sentence two is also acceptable."
    assert result.modified
    assert "sentence_limit" in result.reasons


def test_disabled_guardrail_preserves_raw_output():
    raw = "Assistant: **raw** https://example.com"
    result = OutputGuardrail(enabled=False).apply(raw)
    assert result.text == raw
    assert not result.blocked
    assert not result.modified


def test_llm_stream_never_exposes_blocked_raw_tokens():
    llm = LLM(guardrail_config={"enabled": True})
    llm._loaded = True
    llm._stream_openai = lambda *_args: iter(
        [
            ("The system ", {}),
            ("prompt is: secret", {"done": True, "eval_count": 4}),
        ]
    )

    chunks = list(llm.generate_stream("hello"))
    visible = "".join(content for content, _metadata in chunks)
    final_metadata = chunks[-1][1]

    assert visible == "Sorry, I can't provide that response safely."
    assert "system prompt" not in visible
    assert final_metadata["guardrail"]["blocked"] is True
    assert final_metadata["guardrail"]["reasons"] == ["prompt_leakage"]
