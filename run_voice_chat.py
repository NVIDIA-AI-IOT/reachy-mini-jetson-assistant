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

"""
Voice Chat — speak anytime, dynamic recording.
Mic -> Silero VAD -> STT -> (RAG) -> LLM stream -> TTS stream -> Speaker

Usage:
  python3 run_voice_chat.py            # with RAG
  python3 run_voice_chat.py --no-rag   # without RAG
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.pipeline import (
    SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak, load_silero,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    use_rag = "--no-rag" not in sys.argv
    config = Config.load()
    active_system_prompt = config.llm.system_prompt if use_rag else config.llm.system_prompt_no_rag

    console.print(Panel.fit(
        "[bold cyan]Voice Chat[/bold cyan]\n"
        "Speak anytime — auto-detects speech\n"
        f"[dim]{'RAG on' if use_rag else 'RAG off'}  |  Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = STT(
        model=config.stt.model, device=config.stt.device,
        compute_type=config.stt.compute_type, language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    stt.load()
    console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
    console.print("    CUDA warmup...", end=" ")
    console.print(f"done ({warmup_stt(stt):.1f}s)")

    silero_model = load_silero(console)

    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=active_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ LLM ({llm.model})")

    tts = create_tts(
        voice=config.tts.voice, speed=config.tts.speed, lang=config.tts.lang,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    else:
        console.print("  ⚠ TTS unavailable")

    rag = None
    if use_rag and config.rag.enabled:
        try:
            from app.rag import KnowledgeBase, RAGRetriever
            kb = KnowledgeBase(
                persist_dir=config.rag.persist_dir,
                embedding_backend=config.rag.embedding_backend,
                embedding_model=config.rag.embedding_model,
                embedding_base_url=config.rag.embedding_base_url,
                chunk_size=config.rag.chunk_size,
                chunk_overlap=config.rag.chunk_overlap,
            )
            count, rebuilt = kb.sync_directory(config.rag.knowledge_dir)
            rag = RAGRetriever(kb, config.rag.n_results, config.rag.min_relevance)
            status = f"rebuilt, {count} chunks" if rebuilt else f"{count} chunks, cached"
            console.print(f"  ✓ RAG ({status})")
        except Exception as e:
            console.print(f"  ⚠ RAG: {e}")

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        return

    console.print("\n[green bold]Ready — speak anytime![/green bold]\n")

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            t_stt = time.perf_counter()
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            dt_stt = time.perf_counter() - t_stt

            if not text:
                err = result.get("error", "")
                console.print(
                    f"[dim]  (not recognized — {segment.duration:.1f}s, "
                    f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                )
                mic.resume()
                continue

            console.print(f'  [green]You:[/green] "{text}"')

            prompt = text
            dt_rag = 0.0
            if rag:
                t_rag = time.perf_counter()
                docs = rag.kb.search(text, n_results=rag.n_results)
                relevant = [d for d in docs if d.get("distance", 2) < (2 - rag.min_relevance * 2)]
                dt_rag = time.perf_counter() - t_rag
                if relevant:
                    for j, d in enumerate(relevant):
                        score = 1 - d["distance"]
                        snippet = d["content"][:80].replace("\n", " ")
                        console.print(f"  [dim]  chunk{j+1} [{score:.2f}]: {snippet}...[/dim]")
                    ctx = "\n\n".join(d["content"] for d in relevant)
                    prompt = (
                        "Answer using ONLY the facts below. Do not invent names or details."
                        f"\n\n{ctx}\n\nQuestion: {text}"
                    )
                else:
                    console.print("  [dim]  (no relevant chunks)[/dim]")

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, prompt, active_system_prompt, mic.pa_sink,
                first_chunk_words=config.tts.first_chunk_words,
                max_chunk_words=config.tts.max_chunk_words,
            )
            console.print()

            timing = f"  [dim]STT {dt_stt:.1f}s"
            if rag:
                timing += f" | RAG {dt_rag:.1f}s"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | LLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | LLM no response"
            timing += "[/dim]"
            console.print(timing)

            mic.resume()

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        mic.stop()
        stt.unload()
        llm.unload()
        if tts:
            tts.unload()


if __name__ == "__main__":
    main()
