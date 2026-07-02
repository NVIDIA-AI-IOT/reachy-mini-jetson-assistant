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
Vision Chat — speak + see, the VLM describes what it sees.
Mic -> Silero VAD -> [camera capture] -> STT -> VLM (text + images) -> TTS -> Speaker

Usage:
  python3 run_vision_chat.py
"""

import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.llm import LLM
from app.tts import create_tts
from app.camera import Camera
from app.pipeline import (
    SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak, load_silero,
)
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from app.face_detector import FaceDetector
from app.face_tracker import FaceTracker
from app.movement_manager import MovementManager
from app.vision_capture import capture_frames_for_vlm
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        "[dim]Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Reachy Mini ──────────────────────────────────────────────
    reachy = connect_reachy(config, console)

    # ── Audio setup ──────────────────────────────────────────────
    result = find_alsa_device(name_hint=config.audio.input_device or "Reachy Mini Audio")
    if not result:
        console.print("[red]No mic found![/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Camera setup (background ring buffer) ────────────────────
    kill_stale_camera_holders(config.vision.camera_device, console)

    cam = Camera(
        device=config.vision.camera_device,
        width=config.vision.width,
        height=config.vision.height,
        jpeg_quality=config.vision.jpeg_quality,
        capture_fps=config.vision.capture_fps,
    )
    if cam.start():
        console.print(
            f"  ✓ Camera /dev/video{config.vision.camera_device} "
            f"({config.vision.width}x{config.vision.height}, "
            f"{config.vision.capture_fps} fps compressed ring buffer)"
        )
    else:
        console.print("[red]  ✗ Camera not found! Check USB webcam.[/red]")
        return

    # ── Pre-declare variables for cleanup closure ───────────────
    mic = None
    stt = None
    llm = None
    tts = None
    # ── Cleanup handler ──────────────────────────────────────────
    _cleanup_done = threading.Event()

    def _do_cleanup():
        if _cleanup_done.is_set():
            return
        _cleanup_done.set()
        console.print("\n[yellow]Shutting down...[/yellow]")
        if mic:
            try:
                mic.stop()
            except Exception:
                pass
        cam.close()
        if reachy and config.reachy.sleep_on_exit:
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except OSError:
                pass
            try:
                console.print("  Putting Reachy Mini to sleep...")
                reachy.goto_sleep()
                time.sleep(0.5)
                reachy.disable_motors()
                time.sleep(0.3)
            except Exception as e:
                console.print(f"  [dim]Sleep failed: {e}[/dim]")

    def _sig_cleanup(signum=None, frame=None):
        _do_cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sig_cleanup)
    signal.signal(signal.SIGTSTP, _sig_cleanup)
    signal.signal(signal.SIGTERM, _sig_cleanup)
    signal.signal(signal.SIGHUP, _sig_cleanup)

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

    vision_system_prompt = config.vision.system_prompt
    vision_few_shot = config.vision.few_shot or []
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=vision_system_prompt,
    )
    llm.load()
    console.print(f"  ✓ VLM ({llm.model})")

    tts = create_tts(
        voice=config.tts.voice, speed=config.tts.speed, lang=config.tts.lang,
    )
    tts = tts if tts.load() else None
    if tts:
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})")
    else:
        console.print("  ⚠ TTS unavailable")

    face_detector = None
    movement_manager = None
    face_tracker = None
    if reachy and config.reachy.face_tracking:
        face_detector = FaceDetector()
        if face_detector.load():
            console.print(f"  ✓ Face detection ({face_detector.backend})")
            movement_manager = MovementManager(
                reachy,
                pose_smoothing=config.reachy.tracking_pose_smoothing,
                pose_max_step_deg=config.reachy.tracking_pose_max_step_deg,
            )
            movement_manager.start()
            console.print("  ✓ Head controller (100 Hz)")

            face_tracker = FaceTracker(
                cam, face_detector, movement_manager, reachy,
                fps=config.reachy.tracking_fps,
                dead_zone=config.reachy.tracking_dead_zone,
                lock_zone=config.reachy.tracking_lock_zone,
                reacquire_zone=config.reachy.tracking_reacquire_zone,
                good_frame_zone=config.reachy.tracking_good_frame_zone,
                min_face_size=config.reachy.tracking_min_face_size,
                stable_frames=config.reachy.tracking_stable_frames,
                face_lost_delay=config.reachy.tracking_face_lost_delay,
                head_yaw_max_deg=config.reachy.tracking_head_yaw_max_deg,
                head_yaw_gain=config.reachy.tracking_head_yaw_gain,
                head_yaw_step=config.reachy.tracking_head_yaw_step,
                soft_center_head_yaw_max_deg=config.reachy.tracking_soft_center_head_yaw_max_deg,
                soft_center_head_yaw_step=config.reachy.tracking_soft_center_head_yaw_step,
                body_max_deg=config.reachy.tracking_body_max_deg,
                body_gain=config.reachy.tracking_body_gain,
                body_step=config.reachy.tracking_body_step,
                invert_body=config.reachy.tracking_invert_body,
                body_enabled=config.reachy.tracking_body_enabled,
                vertical=config.reachy.tracking_vertical,
                return_to_neutral=config.reachy.tracking_return_to_neutral,
                scan_enabled=config.reachy.tracking_scan_enabled,
                scan_body_range_deg=config.reachy.tracking_scan_body_range_deg,
                scan_speed_deg_per_sec=config.reachy.tracking_scan_speed_deg_per_sec,
            )
            face_tracker.start()
            console.print(f"  ✓ Face tracking ({config.reachy.tracking_fps:.0f} Hz)")
        else:
            console.print("  ⚠ Face detector unavailable")
            face_detector = None

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            t_cam = time.perf_counter()
            captured_frames, stable_at_capture = capture_frames_for_vlm(
                cam,
                face_tracker,
                n_frames,
                settle_secs=config.reachy.tracking_capture_settle_secs,
                acquire_timeout_secs=config.reachy.tracking_capture_acquire_timeout_secs,
            )
            dt_cam = time.perf_counter() - t_cam

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

            word_count = len(text.split())
            if word_count <= 2 and "?" not in text:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                mic.resume()
                continue

            n_imgs = len(captured_frames)
            console.print(
                f'  [green]You:[/green] "{text}" '
                f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
            )

            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, text, vision_system_prompt, mic.pa_sink,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
                first_chunk_words=config.tts.first_chunk_words,
                max_chunk_words=config.tts.max_chunk_words,
            )
            console.print()

            stability = "stable" if stable_at_capture else "latest"
            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} {stability} img)"
            if ttft is not None:
                toks = len(full_resp.split())
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += "[/dim]"
            console.print(timing)

            mic.resume()

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass

    _do_cleanup()
    if face_tracker:
        face_tracker.stop()
    if movement_manager:
        time.sleep(0.5)
        movement_manager.stop()
    try:
        if stt:
            stt.unload()
        if llm:
            llm.unload()
        if tts:
            tts.unload()
        if face_detector:
            face_detector.unload()
    except Exception:
        pass
    console.print("[yellow]Goodbye![/yellow]")
    os._exit(0)


if __name__ == "__main__":
    main()
