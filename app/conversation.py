# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, in-memory conversation history for a running assistant session."""

from collections import deque
from typing import Deque


class ConversationHistory:
    """Keep completed user/assistant turns without retaining camera frames."""

    def __init__(self, max_turns: int = 5, max_message_chars: int = 1000):
        self.max_turns = max(0, int(max_turns))
        self.max_message_chars = max(1, int(max_message_chars))
        self._turns: Deque[tuple[str, str]] = deque(maxlen=self.max_turns or None)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Store one completed turn; incomplete or disabled turns are ignored."""
        if self.max_turns == 0:
            return
        user = user_text.strip()[: self.max_message_chars]
        assistant = assistant_text.strip()[: self.max_message_chars]
        if not user or not assistant:
            return
        self._turns.append((user, assistant))

    def messages(self) -> list[dict[str, str]]:
        """Return fresh OpenAI-format text messages in chronological order."""
        messages: list[dict[str, str]] = []
        for user, assistant in self._turns:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def clear(self) -> None:
        self._turns.clear()
