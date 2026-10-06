# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic safety checks for text produced by the LLM or VLM."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator


_BLOCK_RULES = (
    (
        "prompt_leakage",
        re.compile(
            r"(?:\b(?:system|developer)\s+(?:prompt|message)\s*(?:is|:)|"
            r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b|"
            r"<\|(?:system|assistant|user|developer)[^>]*\|>)",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_material",
        re.compile(
            r"(?:-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----|"
            r"\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{20,}\b|"
            r"\b(?:api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+)",
            re.IGNORECASE,
        ),
    ),
    (
        "active_content",
        re.compile(
            r"(?:<\s*script\b|javascript\s*:|\bon(?:load|error|click)\s*=)",
            re.IGNORECASE,
        ),
    ),
    (
        "robot_control_payload",
        re.compile(
            r"(?:<\s*(?:tool_call|function_call)\b|"
            r"[\{,]\s*[\"'](?:action|command|motor|joint|tool_call|function_call)[\"']\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "unsafe_weapon_instructions",
        re.compile(
            r"\b(?:steps?|instructions?|recipe|how\s+to)\b.{0,100}"
            r"\b(?:make|build|assemble|synthesize)\b.{0,60}"
            r"\b(?:bomb|explosive|weapon|poison)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "unsafe_self_harm_instructions",
        re.compile(
            r"\b(?:steps?|instructions?|how\s+to)\b.{0,100}"
            r"\b(?:suicide|kill\s+(?:myself|yourself)|self[- ]harm)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "sexual_content_involving_minors",
        re.compile(
            r"\b(?:sexual|pornographic|explicit)\b.{0,50}"
            r"\b(?:child|minor|underage)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_ROLE_PREFIX = re.compile(
    r"^\s*(?:assistant|model|reachy(?:\s+mini)?|answer|response)\s*:\s*",
    re.IGNORECASE,
)
_REPEATED_WORD = re.compile(r"\b([A-Za-z][\w'-]*)\b(?:\s+\1\b){7,}", re.IGNORECASE)


@dataclass(frozen=True)
class GuardrailResult:
    text: str
    blocked: bool
    modified: bool
    reasons: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "modified": self.modified,
            "reasons": list(self.reasons),
        }


class OutputGuardrail:
    """Validate and normalize a complete model response before it is exposed."""

    def __init__(
        self,
        enabled: bool = True,
        max_chars: int = 500,
        max_words: int = 80,
        max_sentences: int = 3,
        fallback_response: str = "Sorry, I can't provide that response safely.",
        output_chunk_chars: int = 48,
    ):
        self.enabled = bool(enabled)
        self.max_chars = max(40, int(max_chars))
        self.max_words = max(8, int(max_words))
        self.max_sentences = max(1, int(max_sentences))
        self.fallback_response = fallback_response.strip()
        self.output_chunk_chars = max(16, int(output_chunk_chars))

    @classmethod
    def from_config(cls, config: Any = None) -> "OutputGuardrail":
        if config is None:
            return cls()
        if isinstance(config, dict):
            get = config.get
        else:
            get = lambda name, default: getattr(config, name, default)
        return cls(
            enabled=get("enabled", True),
            max_chars=get("max_chars", 500),
            max_words=get("max_words", 80),
            max_sentences=get("max_sentences", 3),
            fallback_response=get(
                "fallback_response", "Sorry, I can't provide that response safely."
            ),
            output_chunk_chars=get("output_chunk_chars", 48),
        )

    def apply(self, raw_text: str) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(raw_text, False, False, ())

        normalized = self._normalize(raw_text)
        reasons: list[str] = []

        for reason, pattern in _BLOCK_RULES:
            if pattern.search(normalized):
                reasons.append(reason)
        if _REPEATED_WORD.search(normalized):
            reasons.append("repetitive_output")

        if reasons:
            return GuardrailResult(
                self.fallback_response,
                True,
                True,
                tuple(dict.fromkeys(reasons)),
            )

        safe = self._sanitize(normalized)
        if safe != normalized:
            reasons.append("sanitized_formatting")

        safe, limit_reasons = self._limit(safe)
        reasons.extend(limit_reasons)

        if not safe:
            return GuardrailResult(
                self.fallback_response,
                True,
                True,
                ("empty_after_sanitization",),
            )

        return GuardrailResult(safe, False, bool(reasons), tuple(reasons))

    def chunks(self, text: str) -> Iterator[str]:
        for start in range(0, len(text), self.output_chunk_chars):
            yield text[start : start + self.output_chunk_chars]

    @staticmethod
    def _normalize(text: str) -> str:
        text = html.unescape(unicodedata.normalize("NFKC", text or ""))
        text = "".join(
            char
            for char in text
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _sanitize(text: str) -> str:
        text = _ROLE_PREFIX.sub("", text)
        text = _MARKDOWN_LINK.sub(lambda match: match.group(1), text)
        text = _URL.sub("", text)
        text = _HTML_TAG.sub("", text)
        text = re.sub(r"```(?:[A-Za-z0-9_+-]+)?", "", text)
        text = text.replace("```", "").replace("`", "")
        text = re.sub(r"[*~#]+", "", text)
        text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
        text = re.sub(r"([!?.,])\1{2,}", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    def _limit(self, text: str) -> tuple[str, list[str]]:
        reasons: list[str] = []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > self.max_sentences:
            text = " ".join(sentences[: self.max_sentences]).strip()
            reasons.append("sentence_limit")

        words = text.split()
        if len(words) > self.max_words:
            text = " ".join(words[: self.max_words]).strip()
            reasons.append("word_limit")

        if len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0].strip()
            reasons.append("character_limit")

        if reasons and text and text[-1] not in ".!?":
            text = text.rstrip(" ,;:-") + "."
        return text, reasons
