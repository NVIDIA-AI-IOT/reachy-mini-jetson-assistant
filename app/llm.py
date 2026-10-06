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

"""LLM — Ollama or OpenAI-compatible backend (llama.cpp)."""

import httpx
import json
from typing import Optional, Iterator, Dict, Any

from app.guardrails import OutputGuardrail


class LLM:
    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:8080",
        backend: str = "openai",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = "",
        timeout: float = 120.0,
        guardrail_config: Any = None,
        audit_logger: Any = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.backend = (backend or "openai").lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.guardrail = OutputGuardrail.from_config(guardrail_config)
        self.audit_logger = audit_logger
        self._loaded = False

    def load(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                if self.backend == "openai":
                    r = client.get(f"{self.base_url}/v1/models")
                    if r.status_code != 200:
                        return False
                    models = [m.get("id", "") for m in r.json().get("data", [])]
                    if not models:
                        return False
                    if not self.model or self.model not in models:
                        self.model = models[0]
                else:
                    r = client.get(f"{self.base_url}/api/tags")
                    if r.status_code != 200:
                        return False
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    base = self.model.split(":")[0]
                    if base not in [n.split(":")[0] for n in names] and self.model not in names:
                        print(f"Model '{self.model}' not found. Available: {', '.join(names)}")
                        return False
            self._loaded = True
            return True
        except Exception as e:
            print(f"LLM connection error: {e}")
            return False

    def _messages(
        self, prompt: str, system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
        history: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _messages_multimodal(
        self, prompt: str, images_b64: list[str],
        system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
        history: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        if history:
            msgs.extend(history)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        msgs.append({"role": "user", "content": content})
        return msgs

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        images_b64: Optional[list[str]] = None,
        few_shot: Optional[list[dict]] = None,
        history: Optional[list[dict]] = None,
    ) -> Iterator[tuple]:
        """Yield only output that passed the complete-response guardrail.

        Guarded mode intentionally buffers the backend stream so unsafe early
        tokens can never reach the display, TTS queue, or another consumer.
        Pass images_b64 for multimodal VLM requests.
        """
        if not self._loaded:
            yield ("", {})
            return
        mt = max_tokens or self.max_tokens
        t = temperature if temperature is not None else self.temperature
        if images_b64:
            msgs = self._messages_multimodal(
                prompt, images_b64, system_prompt, few_shot, history
            )
        else:
            msgs = self._messages(prompt, system_prompt, few_shot, history)

        if self.backend == "openai":
            raw_stream = self._stream_openai(msgs, mt, t)
        else:
            raw_stream = self._stream_ollama(msgs, mt, t)

        if not self.guardrail.enabled:
            yield from raw_stream
            return

        raw_parts: list[str] = []
        final_metadata: dict[str, Any] = {}
        for content, metadata in raw_stream:
            if content:
                raw_parts.append(content)
            if metadata:
                final_metadata.update(metadata)

        raw_text = "".join(raw_parts).strip()
        if not raw_text:
            yield ("", final_metadata)
            return

        result = self.guardrail.apply(raw_text)
        guardrail_metadata = result.metadata()
        if self.audit_logger is not None:
            self.audit_logger.write(
                "model_response_guardrail",
                model=self.model,
                blocked=result.blocked,
                modified=result.modified,
                reasons=result.reasons,
                raw_characters=len(raw_text),
                safe_characters=len(result.text),
            )
        if result.modified:
            status = "blocked" if result.blocked else "modified"
            reasons = ", ".join(result.reasons)
            print(f"\n  [Guardrail {status}: {reasons}]")

        for safe_chunk in self.guardrail.chunks(result.text):
            yield (safe_chunk, {})

        final_metadata["done"] = True
        final_metadata["guardrail"] = guardrail_metadata
        yield ("", final_metadata)

    def _stream_openai(self, messages, max_tokens, temperature) -> Iterator[tuple]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/v1/chat/completions", json={
                    "model": self.model, "messages": messages, "stream": True,
                    "max_tokens": max_tokens, "temperature": temperature,
                }) as r:
                    if r.status_code != 200:
                        err = r.read().decode(errors="replace")[:300]
                        print(f"\n  [LLM error {r.status_code}] {err}")
                        yield ("", {})
                        return
                    for line in r.iter_lines():
                        if not line or not line.strip().startswith("data:"):
                            continue
                        line = line.strip()
                        if line == "data: [DONE]":
                            yield ("", {"done": True})
                            return
                        try:
                            data = json.loads(line[5:])
                            usage = data.get("usage")
                            if usage:
                                yield ("", {"done": True, "eval_count": usage.get("completion_tokens", 0)})
                                return
                            content = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
                            if content:
                                yield (content, {})
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"LLM stream error: {e}")
            yield ("", {})

    def _stream_ollama(self, messages, max_tokens, temperature) -> Iterator[tuple]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json={
                    "model": self.model, "messages": messages, "stream": True,
                    "keep_alive": "1h",
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                }) as r:
                    if r.status_code != 200:
                        yield ("", {})
                        return
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            done = data.get("done", False)
                            meta = {}
                            if done:
                                meta = {"done": True, "eval_count": data.get("eval_count", 0)}
                            if content:
                                yield (content, meta)
                            elif done:
                                yield ("", meta)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"LLM stream error: {e}")
            yield ("", {})

    def health_check(self) -> bool:
        if not self._loaded:
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                url = f"{self.base_url}/v1/models" if self.backend == "openai" else f"{self.base_url}/api/tags"
                return client.get(url).status_code == 200
        except Exception:
            return False

    def unload(self):
        self._loaded = False
