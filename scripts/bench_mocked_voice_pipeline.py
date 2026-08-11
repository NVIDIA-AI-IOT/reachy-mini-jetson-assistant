#!/usr/bin/env python3
"""Measure the Reachy voice path with real AI stages and mocked robot I/O.

The robot microphone/camera/speaker/motion hardware is deliberately out of
scope.  The test generates a valid speech waveform with the same Kokoro TTS
client used by the app, runs it through the app STT class, calls the app's
streaming LLM client, and begins TTS on the first speakable response chunk.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.llm import LLM
from app.stt import STT
from app.tts import KokoroTTS


def ms(seconds):
    return round(seconds * 1000, 1)


def resample_linear(audio, src_rate, dst_rate=16000):
    n = round(len(audio) * dst_rate / src_rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt_text = "What is the capital of France?"
    system = "You are Reachy Mini. Answer in one short spoken sentence."
    tts = KokoroTTS()
    t0 = time.perf_counter()
    if not tts.load():
        raise SystemExit("Kokoro could not load")
    tts_load_s = time.perf_counter() - t0
    seed = tts.synthesize(prompt_text)
    if seed.get("audio") is None:
        raise SystemExit(f"Kokoro seed speech failed: {seed.get('error')}")
    utterance = resample_linear(seed["audio"].astype(np.float32) / 32768.0, seed["sample_rate"])

    stt = STT(model="small.en", device="cuda", compute_type="int8", language="en", beam_size=1)
    t0 = time.perf_counter()
    if not stt.load():
        raise SystemExit("STT could not load")
    stt_load_s = time.perf_counter() - t0

    llm = LLM(base_url=args.base_url, backend="openai", max_tokens=32, temperature=0.0, system_prompt=system)
    if not llm.load():
        raise SystemExit("LLM server is not ready")

    # Warm every executable stage, but do not include this in reported latency.
    stt.transcribe(utterance, 16000)
    list(llm.generate_stream(prompt=prompt_text))
    tts.synthesize("Paris is the capital of France.")

    results = []
    for i in range(args.runs):
        start = time.perf_counter()
        # Energy VAD stands in for the mic/VAD capture loop; its production
        # decision is dominated by audio cadence, not compute time.
        vad_start = time.perf_counter()
        rms = float(np.sqrt(np.mean(utterance ** 2)))
        vad_s = time.perf_counter() - vad_start

        stt_start = time.perf_counter()
        transcript = stt.transcribe(utterance, 16000)
        stt_s = time.perf_counter() - stt_start
        if not transcript.get("text"):
            raise SystemExit(f"STT failed: {transcript}")

        llm_start = time.perf_counter()
        ttft_s = None
        first_tts_done_s = None
        response = ""
        pending = ""
        tts_calls = 0
        tts_compute_s = 0.0
        for content, _meta in llm.generate_stream(prompt=transcript["text"]):
            if not content:
                continue
            if ttft_s is None:
                ttft_s = time.perf_counter() - llm_start
            response += content
            pending += content
            if len(pending.split()) >= 3 and first_tts_done_s is None:
                tts_start = time.perf_counter()
                audio = tts.synthesize(pending.strip())
                tts_compute_s += time.perf_counter() - tts_start
                if audio.get("audio") is None:
                    raise SystemExit(f"TTS failed: {audio.get('error')}")
                tts_calls += 1
                pending = ""
                first_tts_done_s = time.perf_counter() - start
        llm_s = time.perf_counter() - llm_start
        if pending.strip():
            tts_start = time.perf_counter()
            audio = tts.synthesize(pending.strip())
            tts_compute_s += time.perf_counter() - tts_start
            if audio.get("audio") is None:
                raise SystemExit(f"TTS failed: {audio.get('error')}")
            tts_calls += 1
        total_s = time.perf_counter() - start
        results.append({
            "run": i + 1, "energy_vad_ms": ms(vad_s), "rms": rms,
            "stt_ms": ms(stt_s), "llm_ttft_ms": ms(ttft_s or 0),
            "llm_stream_ms": ms(llm_s), "tts_compute_ms": ms(tts_compute_s),
            "first_audio_ready_ms": ms(first_tts_done_s or total_s),
            "pipeline_complete_ms": ms(total_s), "tts_calls": tts_calls,
            "transcript": transcript["text"], "response": response.strip(),
        })

    report = {
        "benchmark": "reachy-mini-jetson-assistant mocked voice pipeline",
        "hardware_mocked": ["Reachy microphone", "camera", "speaker playback", "USB motion controller"],
        "real_stages": ["energy VAD decision", "faster-whisper STT", "llama.cpp streaming LLM", "Kokoro TTS"],
        "stt_backend": stt.get_info(), "tts_provider": tts.provider,
        "cold_load_ms": {"stt": ms(stt_load_s), "tts": ms(tts_load_s)},
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    tts.unload()


if __name__ == "__main__":
    main()
