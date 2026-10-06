# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from app.conversation import ConversationHistory
from app.llm import LLM


def test_history_keeps_only_the_last_five_completed_turns():
    history = ConversationHistory(max_turns=5, max_message_chars=20)
    for index in range(6):
        history.add_turn(f" user {index} ", f" assistant {index} ")

    messages = history.messages()
    assert history.turn_count == 5
    assert len(messages) == 10
    assert messages[0] == {"role": "user", "content": "user 1"}
    assert messages[-1] == {"role": "assistant", "content": "assistant 5"}


def test_history_ignores_incomplete_turns_and_can_be_disabled():
    history = ConversationHistory(max_turns=5)
    history.add_turn("hello", "")
    assert history.messages() == []

    disabled = ConversationHistory(max_turns=0)
    disabled.add_turn("hello", "hi")
    assert disabled.messages() == []


def test_llm_orders_static_examples_then_history_then_current_image():
    llm = LLM(model="test")
    messages = llm._messages_multimodal(
        prompt="What color is it?",
        images_b64=["current-frame"],
        system_prompt="system",
        few_shot=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        history=[
            {"role": "user", "content": "What is on the desk?"},
            {"role": "assistant", "content": "A red cup."},
        ],
    )

    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "user", "assistant", "user"
    ]
    assert messages[3]["content"] == "What is on the desk?"
    current = messages[-1]["content"]
    assert current[0] == {"type": "text", "text": "What color is it?"}
    assert current[1]["image_url"]["url"].endswith("current-frame")


def test_history_configuration_is_bounded():
    from app.config import Config

    config = Config()
    config.vision.history_turns = 21
    assert "vision.history_turns must be between 0 and 20" in config.validation_errors()
