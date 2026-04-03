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
Web Vision Chat — browser UI + terminal output simultaneously.
Mic -> Silero VAD -> [camera] -> STT -> VLM -> TTS -> Speaker
               + WebSocket broadcast to connected browsers.

Usage:
  python3 run_web_vision_chat.py                 # default 0.0.0.0:8090
  python3 run_web_vision_chat.py --port 9000
  python3 run_web_vision_chat.py --host 127.0.0.1
"""

import argparse
import os
import queue
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
from app.monitor import get_system_stats, get_jetson_model
from app.pipeline import (
    SAMPLE_RATE, TTS_BREAKS, MicRecorder, warmup_stt, vad_loop,
    tts_player, load_silero,
)
from app.reachy import kill_stale_camera_holders, connect as connect_reachy
from app.emotion import Emotion, EmotionDetector, EmotionResult
from app.face_tracker import FaceTracker
from app.movement_manager import MovementManager
from app.movements import MovementController
from app.web import Broadcaster, start_web_server
from rich.console import Console
from rich.panel import Panel

console = Console()


# ── Background threads ───────────────────────────────────────────

def _frame_broadcast_thread(cam: Camera, broadcaster: Broadcaster, fps: float = 10.0):
    """Stream camera frames to browsers at UI fps via direct hardware reads.

    Uses cam.read_live() (bypasses the 3fps VLM ring buffer) so the browser
    gets a smooth video feed without affecting VLM frame selection.
    """
    interval = 1.0 / fps
    while cam.health_check():
        if broadcaster.client_count > 0:
            b64 = cam.read_live()
            if b64:
                broadcaster.send({"type": "frame", "data": b64})
        time.sleep(interval)


def _stats_broadcast_thread(
    broadcaster: Broadcaster,
    models: dict,
    reachy,
    tracker=None,
    interval: float = 2.0,
):
    """Periodically send system stats + robot status to all WebSocket clients."""
    while True:
        try:
            s = get_system_stats()
            msg = {
                "type": "stats",
                "cpu": round(s.cpu_percent, 1),
                "ram_used": round(s.ram_used_mb / 1024, 1),
                "ram_total": round(s.ram_total_mb / 1024, 1),
                "models": models,
                "clients": broadcaster.client_count,
            }
            if s.gpu_percent is not None:
                msg["gpu"] = round(s.gpu_percent, 1)
            broadcaster.send(msg)

            robot_msg = {
                "type": "robot",
                "connected": reachy is not None,
                "motors": True if reachy else False,
                "head": "Up" if reachy else "N/A",
            }
            if tracker is not None:
                robot_msg["tracking"] = tracker.is_tracking
            broadcaster.send(robot_msg)
        except Exception:
            pass
        time.sleep(interval)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision Chat with Web UI")
    parser.add_argument("--host", default=None, help="Web server bind address")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    args = parser.parse_args()

    config = Config.load()
    web_host = args.host or config.web.host
    web_port = args.port or config.web.port
    broadcaster = Broadcaster()

    console.print(Panel.fit(
        "[bold cyan]Web Vision Chat[/bold cyan]\n"
        "Speak anytime — camera captures when you speak\n"
        f"[dim]Web UI: http://{{host}}:{web_port}  |  Ctrl-C to quit[/dim]",
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

    # ── Camera setup ─────────────────────────────────────────────
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
            f"{config.vision.capture_fps} fps ring buffer)"
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

    emotion_detector = None
    mover = None
    movement_manager = None
    face_tracker = None
    if config.emotion.enabled:
        emotion_detector = EmotionDetector()
        if emotion_detector.load():
            prov = emotion_detector.provider.replace("ExecutionProvider", "")
            console.print(f"  ✓ Emotion (YuNet + FER+, {prov})")
            if reachy:
                movement_manager = MovementManager(reachy)
                movement_manager.start()
                console.print("  ✓ Movement manager (100 Hz control loop)")

                mover = MovementController(movement_manager, config.reachy.antenna_rest_position)
                console.print("  ✓ Emotion movements enabled")

                if config.reachy.face_tracking:
                    face_tracker = FaceTracker(
                        cam, emotion_detector, movement_manager,
                        fps=config.reachy.tracking_fps,
                    )
                    face_tracker.start()
                    console.print(f"  ✓ Face tracking ({config.reachy.tracking_fps:.0f} Hz)")
        else:
            console.print("  ⚠ Emotion detector unavailable")
            emotion_detector = None

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, config.audio.input_device or "Reachy Mini Audio"):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        cam.close()
        return

    # ── Start web server + background threads ────────────────────
    web_thread = start_web_server(broadcaster, host=web_host, port=web_port)
    time.sleep(0.5)
    console.print(f"  ✓ Web UI  →  [bold]http://{web_host}:{web_port}[/bold]")

    threading.Thread(
        target=_frame_broadcast_thread,
        args=(cam, broadcaster, config.web.ui_fps),
        daemon=True, name="frame-broadcaster",
    ).start()

    model_info = {
        "stt": f"faster-whisper ({config.stt.model})",
        "vlm": llm.model,
        "tts": f"{tts.backend_name} ({tts.voice})" if tts else "unavailable",
        "vad": "Silero",
    }

    threading.Thread(
        target=_stats_broadcast_thread,
        args=(broadcaster, model_info, reachy, face_tracker),
        daemon=True, name="stats-broadcaster",
    ).start()

    platform_name = get_jetson_model()
    config_info = {
        "max_tokens": config.llm.max_tokens,
        "temperature": config.llm.temperature,
        "vision_frames": config.vision.frames,
        "capture_fps": config.vision.capture_fps,
        "ui_fps": config.web.ui_fps,
        "jpeg_quality": config.vision.jpeg_quality,
        "resolution": f"{config.vision.width}x{config.vision.height}",
        "silero_threshold": config.vad.silero_threshold,
        "beam_size": config.stt.beam_size,
    }
    broadcaster.send({
        "type": "info",
        "models": model_info,
        "platform": platform_name,
        "config": config_info,
    })

    n_frames = config.vision.frames
    n_fewshot = len(vision_few_shot) // 2
    first_chunk_words = config.tts.first_chunk_words
    max_chunk_words = config.tts.max_chunk_words

    console.print(
        f"\n[green bold]Ready — speak anytime! "
        f"({config.vision.capture_fps} fps, {n_frames} frame{'s' if n_frames > 1 else ''} "
        f"per query{f', {n_fewshot} few-shot pairs' if n_fewshot else ''})[/green bold]\n"
    )

    if broadcaster.ptt_active:
        broadcaster.send({"type": "status", "stage": "listening"})
    else:
        broadcaster.send({"type": "status", "stage": "muted"})

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            if not broadcaster.ptt_active:
                broadcaster.send({"type": "status", "stage": "muted"})
                mic.resume()
                continue

            broadcaster.send({"type": "status", "stage": "transcribing"})

            t_cam = time.perf_counter()
            captured_frames = cam.get_speech_frames(
                speech_start=segment.start_time,
                speech_end=segment.end_time,
                max_frames=n_frames,
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
                broadcaster.send({"type": "status", "stage": "listening"})
                mic.resume()
                continue

            word_count = len(text.split())
            if word_count <= 2 and "?" not in text:
                console.print(f"[dim]  (skipped filler: \"{text}\")[/dim]")
                broadcaster.send({"type": "status", "stage": "listening"})
                mic.resume()
                continue

            emotion_tag = ""
            emo = None
            moved = False
            if emotion_detector:
                frame_b64 = captured_frames[0] if captured_frames else None
                emo = emotion_detector.detect(frame_b64) if frame_b64 else EmotionResult(Emotion.NEUTRAL, 0.0, 0.0)
                if emo.emotion == Emotion.NEUTRAL:
                    text_emo = emotion_detector.detect_text(text)
                    if text_emo is not None:
                        emo = EmotionResult(text_emo, 0.95, 0.0)
                moved = mover.react(emo.emotion, emo.confidence) if mover else False
                emotion_tag = (
                    f" | {emo.emotion.value} ({emo.confidence:.0%}, {emo.inference_ms:.0f}ms)"
                    f"{'*' if moved else ''}"
                )

            n_imgs = len(captured_frames)
            emo_label = emo.emotion.value if emo else None
            if emo and emo.emotion != Emotion.NEUTRAL:
                console.print(
                    f'  [green]You:[/green] "{text}" '
                    f'[dim]({n_imgs} img)[/dim] '
                    f'[bold yellow][{emo_label} {emo.confidence:.0%}][/bold yellow]'
                    f'{"[bold cyan] ↺[/bold cyan]" if moved else ""}'
                )
            else:
                console.print(
                    f'  [green]You:[/green] "{text}" '
                    f'[dim]({n_imgs} frame{"s" if n_imgs != 1 else ""} captured)[/dim]'
                )

            broadcaster.send({
                "type": "transcript",
                "text": text,
                "stt_time": round(dt_stt, 2),
                "duration": round(segment.duration, 1),
                "emotion": emo_label,
                "emotion_confidence": round(emo.confidence, 2) if emo else None,
                "emotion_ms": round(emo.inference_ms, 1) if emo else None,
                "face_detected": emo.face_detected if emo else False,
                "face_box": list(emo.face_box) if emo and emo.face_box else None,
                "moved": moved,
            })

            # ── VLM streaming with TTS + WebSocket broadcast ─────
            broadcaster.send({"type": "status", "stage": "thinking"})
            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            tts_q = None
            tts_thread = None
            if tts:
                tts_q = queue.Queue()
                tts_thread = threading.Thread(
                    target=tts_player, args=(tts, tts_q, mic.pa_sink), daemon=True,
                )
                tts_thread.start()

            full_resp = ""
            tts_buf = ""
            first_tts_sent = False
            t_llm = time.perf_counter()
            ttft = None

            for chunk_data in llm.generate_stream(
                prompt=text, system_prompt=vision_system_prompt,
                images_b64=captured_frames if captured_frames else None,
                few_shot=vision_few_shot if vision_few_shot else None,
            ):
                content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_llm
                        broadcaster.send({"type": "status", "stage": "speaking"})
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_resp += content

                    broadcaster.send({"type": "token", "text": content})

                    if tts_q is not None:
                        tts_buf += content
                        words = len(tts_buf.split())
                        limit = first_chunk_words if not first_tts_sent else max_chunk_words
                        hit_break = any(c in content for c in TTS_BREAKS) and words >= 2
                        if hit_break or words >= limit:
                            tts_q.put(tts_buf.strip())
                            tts_buf = ""
                            first_tts_sent = True

            dt_llm = time.perf_counter() - t_llm

            if tts_q is not None:
                if tts_buf.strip():
                    tts_q.put(tts_buf.strip())
                tts_q.put(None)
                tts_thread.join()

            console.print()

            toks = len(full_resp.split())
            timing = f"  [dim]STT {dt_stt:.1f}s | CAM {dt_cam*1000:.0f}ms ({n_imgs} img from buf)"
            if ttft is not None:
                timing += f" | TTFT {ttft:.1f}s | VLM {dt_llm:.1f}s ~{toks/(dt_llm or 1):.0f}w/s"
            else:
                timing += " | VLM no response"
            timing += emotion_tag
            timing += "[/dim]"
            console.print(timing)

            broadcaster.send({
                "type": "done",
                "ttft": round(ttft, 2) if ttft else None,
                "vlm_time": round(dt_llm, 2),
                "tokens": toks,
            })
            broadcaster.send({"type": "status", "stage": "listening"})

            mic.resume()

    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass

    _do_cleanup()
    if face_tracker:
        face_tracker.stop()
    if mover:
        mover.reset()
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
        if emotion_detector:
            emotion_detector.unload()
    except Exception:
        pass
    console.print("[yellow]Goodbye![/yellow]")
    os._exit(0)


if __name__ == "__main__":
    main()
