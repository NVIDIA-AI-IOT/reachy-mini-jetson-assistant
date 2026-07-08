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

"""Pipeline — shared audio I/O, VAD, TTS streaming, and mic recording.

Extracts the common infrastructure used by both run_voice_chat.py and
run_vision_chat.py so each entry point only contains its unique logic.
"""

import sys
import time
import wave
import subprocess
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Iterator, Union

import numpy as np
from pathlib import Path
from rich.console import Console

from app.audio import kill_pulseaudio
from app.config import VADConfig

# Suppress noisy ALSA error messages (underrun warnings etc.)
# The callback reference must be kept alive to avoid segfault from GC.
_ALSA_ERR_T = None
_alsa_handler = None
try:
    import ctypes
    _ALSA_ERR_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


# ── Audio constants (fixed by hardware, not user-tunable) ────────

SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples (32ms) at 16kHz
CHANNELS = 1

TTS_BREAKS = frozenset('.,;:!?\n')
AEC_SOURCE_NAME = "reachy_aec_source"
AEC_SINK_NAME = "reachy_aec_sink"


# ── Audio helpers ─────────────────────────────────────────────────

def chunk_rms(raw: bytes) -> float:
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def save_wav(chunks: list[bytes], path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(chunks))


def warmup_stt(stt_obj) -> float:
    """Run a dummy transcription to warm up CUDA. Returns elapsed seconds."""
    path = "/tmp/_warmup.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
    t0 = time.perf_counter()
    stt_obj.transcribe(path, sample_rate=SAMPLE_RATE)
    Path(path).unlink(missing_ok=True)
    return time.perf_counter() - t0


def _pa_match(needle: str, haystack: str) -> bool:
    """Match a name hint against a PulseAudio device name, ignoring space/underscore differences."""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """Find a PulseAudio input source matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sources"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]) and "monitor" not in parts[1].lower():
                return parts[1]
    except Exception:
        pass
    return None


def find_pa_sink(name_hint: str) -> Optional[str]:
    """Find a PulseAudio output sink matching name_hint."""
    for sink in list_pa_sinks():
        if _pa_match(name_hint, sink["id"]):
            return sink["id"]
    return None


def _pa_sink_label(sink_name: str) -> str:
    """Return a concise, user-facing label for a PulseAudio sink."""
    normalized = sink_name.lower()
    if "pollen_robotics_reachy_mini_audio" in normalized:
        return "Reachy Mini"
    if "anker_powerconf" in normalized:
        return "Anker PowerConf"
    if "platform-sound" in normalized:
        return "Jetson Audio"
    return sink_name


def list_pa_sinks() -> list[dict[str, str]]:
    """List available PulseAudio output sinks for the web speaker selector."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sinks"],
                           capture_output=True, text=True, timeout=5)
        sinks = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and not parts[1].startswith("reachy_aec_"):
                sinks.append({"id": parts[1], "label": _pa_sink_label(parts[1])})
        return sinks
    except Exception:
        return []


def get_default_pa_sink() -> Optional[str]:
    """Return PulseAudio's current default sink, if available."""
    try:
        r = subprocess.run(["pactl", "get-default-sink"],
                           capture_output=True, text=True, timeout=5)
        sink = r.stdout.strip()
        return sink or None
    except Exception:
        return None


class SpeakerSelector:
    """Thread-safe, live-selectable PulseAudio output routing."""

    def __init__(self, preferred_hint: Optional[str], fallback_hint: Optional[str]):
        self._lock = threading.Lock()
        self._preferred_hint = preferred_hint
        self._fallback_hint = fallback_hint
        self._sink: Optional[str] = None
        self._speakers: list[dict[str, str]] = []
        self.refresh()

    @staticmethod
    def _matching_sink(speakers: list[dict[str, str]], hint: Optional[str]) -> Optional[str]:
        if not hint:
            return None
        for speaker in speakers:
            if _pa_match(hint, speaker["id"]) or _pa_match(hint, speaker["label"]):
                return speaker["id"]
        return None

    def refresh(self) -> dict:
        speakers = list_pa_sinks()
        available = {speaker["id"] for speaker in speakers}
        default_sink = get_default_pa_sink()
        with self._lock:
            selected = self._sink if self._sink in available else None
            selected = selected or self._matching_sink(speakers, self._preferred_hint)
            selected = selected or self._matching_sink(speakers, self._fallback_hint)
            selected = selected or (default_sink if default_sink in available else None)
            selected = selected or (speakers[0]["id"] if speakers else None)
            self._speakers = speakers
            self._sink = selected
            return self._state_unlocked()

    def _state_unlocked(self) -> dict:
        return {
            "speakers": [dict(speaker) for speaker in self._speakers],
            "selected": self._sink,
        }

    def state(self) -> dict:
        return self.refresh()

    def get_sink(self) -> Optional[str]:
        with self._lock:
            return self._sink

    def select(self, sink_id: str) -> dict:
        speakers = list_pa_sinks()
        available = {speaker["id"] for speaker in speakers}
        if sink_id not in available:
            state = self.refresh()
            state["error"] = "Selected speaker is no longer available"
            return state

        try:
            result = subprocess.run(
                ["pactl", "set-default-sink", sink_id],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                state = self.refresh()
                state["error"] = result.stderr.strip() or "Could not switch speaker"
                return state
        except Exception as exc:
            state = self.refresh()
            state["error"] = str(exc)
            return state

        with self._lock:
            self._speakers = speakers
            self._sink = sink_id
            return self._state_unlocked()


class PulseAudioAEC:
    """Managed WebRTC echo-cancellation route for one mic and selected speaker."""

    def __init__(
        self,
        source_master: str,
        source_name: str = AEC_SOURCE_NAME,
        sink_name: str = AEC_SINK_NAME,
    ):
        self.source_master = source_master
        self.source_name = source_name
        self.sink_name = sink_name
        self.sink_master: Optional[str] = None
        self.module_id: Optional[int] = None
        self.last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.module_id is not None

    def _managed_module_ids(self) -> list[int]:
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "modules"],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return []

        module_ids = []
        for line in result.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3 or parts[1] != "module-echo-cancel":
                continue
            args = parts[2]
            if f"source_name={self.source_name}" in args or f"sink_name={self.sink_name}" in args:
                try:
                    module_ids.append(int(parts[0]))
                except ValueError:
                    pass
        return module_ids

    def cleanup_stale(self):
        for module_id in self._managed_module_ids():
            try:
                subprocess.run(
                    ["pactl", "unload-module", str(module_id)],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:
                pass

    def start(self, sink_master: str) -> bool:
        self.stop()
        self.cleanup_stale()
        self.last_error = None
        args = [
            "pactl", "load-module", "module-echo-cancel",
            "aec_method=webrtc",
            f"source_master={self.source_master}",
            f"sink_master={sink_master}",
            f"source_name={self.source_name}",
            f"sink_name={self.sink_name}",
            "source_properties=device.description=Reachy_AEC_Microphone",
            "sink_properties=device.description=Reachy_AEC_Speaker",
        ]
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                self.last_error = result.stderr.strip() or "Could not load WebRTC AEC"
                return False
            self.module_id = int(result.stdout.strip())
            self.sink_master = sink_master
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.module_id = None
            self.sink_master = None
            return False

    def stop(self):
        module_id = self.module_id
        self.module_id = None
        self.sink_master = None
        if module_id is None:
            return
        try:
            subprocess.run(
                ["pactl", "unload-module", str(module_id)],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            pass


def play_audio(audio: np.ndarray, sample_rate: int, sink: Optional[str] = None):
    """Play int16 audio and do not return before its real-time duration."""
    raw = audio.astype(np.int16).tobytes()
    expected_duration = audio.size / sample_rate if sample_rate > 0 else 0.0
    started_at = time.monotonic()
    submitted = False
    try:
        if sink:
            cmd = ["paplay", f"--device={sink}", "--format=s16le",
                   f"--rate={sample_rate}", "--channels=1", "--raw"]
        else:
            cmd = ["aplay", "-f", "S16_LE", "-r", str(sample_rate),
                   "-c", "1", "-t", "raw", "-q"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.stdin.write(raw)
        p.stdin.close()
        submitted = True
        p.wait(timeout=30)
    except Exception:
        pass
    finally:
        if submitted:
            remaining = expected_duration - (time.monotonic() - started_at)
            if remaining > 0:
                time.sleep(remaining)


def tts_player(
    tts_obj,
    tts_q: queue.Queue,
    sink: Optional[Union[str, Callable[[], Optional[str]]]] = None,
    on_audio_start: Optional[Callable[[], None]] = None,
    on_audio_end: Optional[Callable[[], None]] = None,
):
    """Synthesize queued text and expose the true first/last audio events."""
    audio_started = False
    try:
        while True:
            text = tts_q.get()
            if text is None:
                return
            r = tts_obj.synthesize(text)
            if r.get("audio") is not None:
                if not audio_started:
                    audio_started = True
                    if on_audio_start:
                        try:
                            on_audio_start()
                        except Exception as exc:
                            print(f"TTS start callback failed: {exc}")
                current_sink = sink() if callable(sink) else sink
                play_audio(r["audio"], r["sample_rate"], sink=current_sink)
    finally:
        if audio_started and on_audio_end:
            try:
                on_audio_end()
            except Exception as exc:
                print(f"TTS end callback failed: {exc}")


# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    """Thin wrapper around the Silero VAD ONNX model."""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self._model = load_silero_vad(onnx=True)
        self._torch = torch

    def __call__(self, raw_audio: bytes) -> float:
        """Return speech probability for raw int16 PCM audio at 16 kHz."""
        pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(pcm)
        return self._model(tensor, SAMPLE_RATE).item()

    def reset(self):
        self._model.reset_states()


def load_silero(console: Optional[Console] = None) -> SileroVAD:
    """Load Silero VAD. Raises if unavailable (silero-vad is required)."""
    t0 = time.perf_counter()
    vad = SileroVAD()
    dt = time.perf_counter() - t0
    if console:
        console.print(f"  ✓ Silero VAD (ONNX, loaded in {dt:.1f}s)")
    return vad


# ── Speech segment ────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """A completed speech utterance from the VAD."""
    audio: np.ndarray
    raw_chunks: list
    duration: float
    rms: float
    start_time: float
    end_time: float


# ── Mic recorder ──────────────────────────────────────────────────

class MicRecorder:
    """Manages mic recording via parecord/arecord with a background reader thread."""

    def __init__(self, console: Console, chunk_ms: int = 30):
        self.console = console
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * CHANNELS * 2
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._proc: Optional[subprocess.Popen] = None
        self._route_lock = threading.RLock()
        self._hw = ""
        self.pa_source: Optional[str] = None
        self.pa_sink: Optional[str] = None
        self.speaker_selector: Optional[SpeakerSelector] = None
        self.aec: Optional[PulseAudioAEC] = None

    def start(
        self,
        hw: str,
        mic_hint: str,
        speaker_hint: Optional[str] = None,
        echo_cancellation: bool = True,
    ) -> bool:
        """Start recording. Returns True on success."""
        subprocess.run(["pkill", "-9", "parecord"], capture_output=True)
        subprocess.run(["pkill", "-9", "arecord"], capture_output=True)
        time.sleep(0.3)

        self._hw = hw
        self.pa_source = find_pa_source(mic_hint)
        self.speaker_selector = SpeakerSelector(
            preferred_hint=speaker_hint,
            fallback_hint=mic_hint,
        )
        self.pa_sink = self.speaker_selector.get_sink()

        if self.pa_source:
            if echo_cancellation and self.pa_sink:
                self.aec = PulseAudioAEC(self.pa_source)
                if self.aec.start(self.pa_sink):
                    self.console.print(
                        f"  AEC: [green]WebRTC active[/green] "
                        f"(Reachy mic → {_pa_sink_label(self.pa_sink)})"
                    )
                    self.pa_sink = self.aec.sink_name
                else:
                    self.console.print(
                        f"  [yellow]AEC unavailable: {self.aec.last_error}; "
                        "using direct audio[/yellow]"
                    )
            self.console.print(f"  PA source: {self.pa_source.split('.')[-2]}")
        else:
            self.console.print("  [yellow]PA source not found, using ALSA direct[/yellow]")
            kill_pulseaudio()
            time.sleep(0.5)

        if not self._start_capture():
            if self.aec and self.aec.active:
                self.console.print("  [yellow]AEC capture failed; using direct audio[/yellow]")
                self.aec.stop()
                self.pa_sink = self.speaker_selector.get_sink()
                if not self._start_capture():
                    return False
            else:
                return False

        self._check_capture()
        return True

    def _record_command(self) -> list[str]:
        if self.pa_source:
            source = self.aec.source_name if self.aec and self.aec.active else self.pa_source
            return [
                "parecord", "-d", source, "--format=s16le",
                f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}", "--raw",
            ]
        plughw = self._hw.replace("hw:", "plughw:")
        return [
            "arecord", "-D", plughw, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS), "-t", "raw",
        ]

    def _start_capture(self) -> bool:
        rec_cmd = self._record_command()
        for attempt in range(3):
            proc = subprocess.Popen(rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(0.5)
            if proc.poll() is None:
                self._proc = proc
                threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
                return True
            err = proc.stderr.read().decode(errors="replace").strip()
            self.console.print(f"  [red]Mic attempt {attempt+1} failed: {err}[/red]")
            time.sleep(1)
        self._proc = None
        return False

    def _check_capture(self):
        time.sleep(0.5)
        test_chunks = []
        for _ in range(10):
            try:
                test_chunks.append(self.audio_q.get(timeout=0.5))
            except queue.Empty:
                break
        if test_chunks:
            rms = chunk_rms(b"".join(test_chunks))
            if rms > 0.003:
                self.console.print("  Mic: [green]✓ live[/green]")
            else:
                self.console.print("  Mic: [yellow]quiet, waiting for speech[/yellow]")
        else:
            running = self._proc is not None and self._proc.poll() is None
            self.console.print(f"  [yellow]Mic: no startup audio; capture running: {running}[/yellow]")

    def _stop_capture(self):
        proc = self._proc
        self._proc = None
        if not proc or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _rebuild_aec_route(self) -> bool:
        with self._route_lock:
            if not self.speaker_selector:
                return False
            physical_sink = self.speaker_selector.get_sink()
            was_listening = self.listening.is_set()
            self.pause()
            self._stop_capture()

            aec_ok = False
            if self.aec and self.pa_source and physical_sink:
                aec_ok = self.aec.start(physical_sink)
            self.pa_sink = self.aec.sink_name if aec_ok else physical_sink

            capture_ok = self._start_capture()
            if not capture_ok and aec_ok:
                self.aec.stop()
                self.pa_sink = physical_sink
                capture_ok = self._start_capture()

            if was_listening:
                self.resume()
            else:
                self.flush()
            return capture_ok

    def get_pa_sink(self) -> Optional[str]:
        with self._route_lock:
            return self.pa_sink

    def _add_aec_state(self, state: dict) -> dict:
        state["aec_enabled"] = bool(self.aec and self.aec.active)
        if self.aec and self.aec.last_error and not self.aec.active:
            state["aec_error"] = self.aec.last_error
        return state

    def speaker_state(self) -> dict:
        if not self.speaker_selector:
            return {"speakers": [], "selected": None, "aec_enabled": False}
        previous_sink = self.speaker_selector.get_sink()
        state = self.speaker_selector.state()
        selected_sink = self.speaker_selector.get_sink()
        if (
            selected_sink
            and previous_sink != selected_sink
            and self.aec
            and self.aec.sink_master != selected_sink
        ):
            self._rebuild_aec_route()
        return self._add_aec_state(state)

    def select_speaker(self, sink_id: str) -> dict:
        if not self.speaker_selector:
            return {
                "speakers": [],
                "selected": None,
                "aec_enabled": False,
                "error": "Speaker routing is not initialized",
            }
        state = self.speaker_selector.select(sink_id)
        if state.get("error"):
            return self._add_aec_state(state)
        if not self._rebuild_aec_route():
            state["error"] = "Could not restart microphone after speaker switch"
        return self._add_aec_state(state)

    def _reader(self, proc: subprocess.Popen):
        while self.alive and proc.poll() is None:
            raw = proc.stdout.read(self.chunk_bytes)
            if not raw:
                if proc.poll() is not None and proc is self._proc:
                    err = proc.stderr.read().decode(errors="replace").strip()
                    if err:
                        self.console.print(f"\n  [red]audio capture died: {err}[/red]")
                break
            if proc is self._proc and self.listening.is_set():
                self.audio_q.put(raw)

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """Stop queuing audio and drain the buffer."""
        self.listening.clear()
        self.flush()

    def resume(self):
        """Drain any stale audio and resume queuing."""
        self.flush()
        self.listening.set()

    def stop(self):
        self.alive = False
        with self._route_lock:
            self._stop_capture()
            if self.aec:
                self.aec.stop()


# ── VAD loop ──────────────────────────────────────────────────────

def vad_loop(
    mic: MicRecorder,
    console: Console,
    vad_cfg: Optional[VADConfig] = None,
    silero: Optional[SileroVAD] = None,
) -> Iterator[SpeechSegment]:
    """Yields SpeechSegment each time a complete utterance is detected.

    Uses Silero VAD for speech detection with RMS as a cheap pre-filter
    to skip dead silence without invoking the model.

    The caller is responsible for calling mic.resume() after processing
    each segment (so audio stays paused during STT/LLM/TTS).
    """
    if silero is None:
        raise RuntimeError("Silero VAD is required (pip install silero-vad)")

    cfg = vad_cfg or VADConfig()
    chunk_ms = cfg.chunk_ms
    silence_chunks = int(cfg.silence_duration_ms / chunk_ms)
    lookback_chunks = int(cfg.lookback_ms / chunk_ms)
    max_chunks = int(cfg.max_speech_secs * 1000 / chunk_ms)

    silero_thresh = cfg.silero_threshold
    rms_silence_floor = min(0.002, cfg.min_utterance_rms)

    lookback: deque[bytes] = deque(maxlen=lookback_chunks)
    speech_raw: list[bytes] = []
    is_speaking = False
    silence_count = 0
    speech_start_t: float = 0.0

    while mic.alive:
        try:
            raw = mic.audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        rms = chunk_rms(raw)

        if rms < rms_silence_floor:
            is_speech = False
        else:
            is_speech = silero(raw) > silero_thresh

        if is_speech:
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                speech_start_t = time.monotonic()
                speech_raw = list(lookback)
                sys.stdout.write("  🎤 Listening...\r")
                sys.stdout.flush()
            speech_raw.append(raw)
            if len(speech_raw) < max_chunks:
                continue
        else:
            if is_speaking:
                speech_raw.append(raw)
                silence_count += 1
                if silence_count < silence_chunks:
                    continue
            else:
                lookback.append(raw)
                continue

        is_speaking = False
        captured = speech_raw
        speech_raw = []
        silence_count = 0
        lookback.clear()
        speech_end_t = time.monotonic()

        silero.reset()

        dur_s = len(captured) * chunk_ms / 1000
        cap_rms = chunk_rms(b"".join(captured))

        sys.stdout.write("                              \r")
        sys.stdout.flush()
        mic.pause()

        if dur_s < cfg.min_utterance_secs or cap_rms < cfg.min_utterance_rms:
            console.print(f"[dim]  (noise: {dur_s:.1f}s, rms={cap_rms:.4f})[/dim]")
            mic.resume()
            continue

        raw_audio = b"".join(captured)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        yield SpeechSegment(
            audio=audio_np,
            raw_chunks=captured,
            duration=dur_s,
            rms=cap_rms,
            start_time=speech_start_t,
            end_time=speech_end_t,
        )


# ── LLM streaming with TTS ───────────────────────────────────────

def stream_and_speak(
    llm,
    tts_obj,
    prompt: str,
    system_prompt: str,
    pa_sink: Optional[str] = None,
    images_b64: Optional[list[str]] = None,
    few_shot: Optional[list[dict]] = None,
    first_chunk_words: int = 3,
    max_chunk_words: int = 8,
) -> tuple[str, float, Optional[float]]:
    """Stream LLM response while chunking text to TTS for real-time playback.

    Returns (full_response, elapsed_seconds, time_to_first_token).
    """
    tts_q = None
    tts_thread = None
    if tts_obj:
        tts_q = queue.Queue()
        tts_thread = threading.Thread(
            target=tts_player, args=(tts_obj, tts_q, pa_sink), daemon=True,
        )
        tts_thread.start()

    full_resp = ""
    tts_buf = ""
    first_tts_sent = False
    t_llm = time.perf_counter()
    ttft = None

    for chunk_data in llm.generate_stream(
        prompt=prompt, system_prompt=system_prompt,
        images_b64=images_b64, few_shot=few_shot,
    ):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if content:
            if ttft is None:
                ttft = time.perf_counter() - t_llm
            sys.stdout.write(content)
            sys.stdout.flush()
            full_resp += content

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

    return full_resp, dt_llm, ttft
