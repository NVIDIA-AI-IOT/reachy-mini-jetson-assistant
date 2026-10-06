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

"""Configuration — loads settings.yaml into typed dataclasses."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from urllib.parse import urlparse
import yaml


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"
PROFILE_ALIASES = {
    "orin": "orin_nano",
    "orin-nano": "orin_nano",
    "orin_nano": "orin_nano",
    "agx": "agx_orin",
    "agx-orin": "agx_orin",
    "agx_orin": "agx_orin",
    "thor": "thor",
    "jetson-thor": "thor",
}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Configuration file does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class PlatformConfig:
    profile: str = "orin_nano"
    display_name: str = "Jetson Orin Nano"
    supported: bool = True
    implementation_status: str = "validated"
    jetpack: str = "7.2"
    model_server: str = "llama_cpp"
    notes: str = ""


@dataclass
class LLMConfig:
    model: str = ""
    base_url: str = "http://localhost:8080"
    backend: str = "openai"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 120.0
    system_prompt: str = "You are a helpful AI assistant."
    system_prompt_no_rag: str = "You are a helpful AI assistant. Answer from your own knowledge."


@dataclass
class GuardrailConfig:
    enabled: bool = True
    max_chars: int = 500
    max_words: int = 80
    max_sentences: int = 2
    output_chunk_chars: int = 48
    fallback_response: str = "Sorry, I can't provide that response safely."


@dataclass
class RuntimeConfig:
    production_mode: bool = False
    audit_log_path: str = "./data/logs/audit.jsonl"
    audit_max_bytes: int = 10_485_760
    audit_backup_count: int = 5


@dataclass
class STTConfig:
    model: str = "base.en"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 1


@dataclass
class TTSConfig:
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"
    first_chunk_words: int = 3
    max_chunk_words: int = 8


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 2
    input_device: Optional[str] = "Reachy Mini Audio"
    output_device: Optional[str] = None
    echo_cancellation: bool = False
    playback_tail_quiet_ms: int = 192
    playback_tail_max_wait_ms: int = 1200
    playback_tail_rms_threshold: float = 0.004


@dataclass
class VADConfig:
    speech_threshold: float = 0.008
    silence_duration_ms: int = 500
    lookback_ms: int = 250
    max_speech_secs: int = 15
    chunk_ms: int = 30
    min_utterance_secs: float = 0.3
    min_utterance_rms: float = 0.001
    silero_threshold: float = 0.5


@dataclass
class VisionConfig:
    camera_device: int = 1
    width: int = 640
    height: int = 480
    jpeg_quality: int = 80
    frames: int = 3
    capture_fps: float = 10.0
    system_prompt: str = (
        "You are a vision assistant on an NVIDIA Jetson device with a live camera. "
        "Answer in one to two sentences. Be direct and concise."
    )
    few_shot: List[Dict[str, str]] = field(default_factory=list)
    history_turns: int = 5


@dataclass
class ReachyConfig:
    enabled: bool = True
    spawn_daemon: bool = True
    timeout: float = 30.0
    media_backend: str = "no_media"
    automatic_body_yaw: bool = False
    wake_on_start: bool = True
    sleep_on_exit: bool = False
    antenna_rest_position: List[float] = field(
        default_factory=lambda: [-0.1745, 0.1745]
    )
    daemon_retry_attempts: int = 3
    daemon_startup_wait: float = 15.0
    face_tracking: bool = True
    tracking_motion_enabled: bool = True
    tracking_fps: float = 15.0
    tracking_dead_zone: float = 0.12
    tracking_lock_zone: float = 0.18
    tracking_reacquire_zone: float = 0.45
    tracking_good_frame_zone: float = 0.18
    tracking_min_face_size: float = 0.06
    tracking_stable_frames: int = 2
    tracking_face_lost_delay: float = 3.0
    tracking_head_yaw_max_deg: float = 20.0
    tracking_head_yaw_gain: float = 18.0
    tracking_head_yaw_step: float = 1.4
    tracking_soft_center_head_yaw_max_deg: float = 12.0
    tracking_soft_center_head_yaw_step: float = 0.75
    tracking_pose_smoothing: float = 0.18
    tracking_pose_max_step_deg: float = 6.0
    tracking_body_max_deg: float = 30.0
    tracking_body_gain: float = 18.0
    tracking_body_step: float = 0.7
    tracking_invert_body: bool = True
    tracking_body_enabled: bool = True
    tracking_vertical: bool = True
    tracking_return_to_neutral: bool = False
    tracking_scan_enabled: bool = True
    tracking_scan_body_range_deg: float = 20.0
    tracking_scan_speed_deg_per_sec: float = 5.0
    tracking_capture_settle_secs: float = 0.35
    tracking_capture_acquire_timeout_secs: float = 0.4
    speaking_movements_enabled: bool = True
    speaking_movement_excitement_probability: float = 0.4


@dataclass
class RAGConfig:
    enabled: bool = True
    knowledge_dir: str = "./knowledge_base"
    persist_dir: str = "./data/chromadb"
    embedding_backend: str = "llamacpp"
    embedding_model: str = "bge-small-en-v1.5"
    embedding_base_url: str = "http://localhost:8081"
    n_results: int = 3
    min_relevance: float = 0.5
    chunk_size: int = 200
    chunk_overlap: int = 20


@dataclass
class WebConfig:
    ui_fps: float = 10.0
    host: str = "0.0.0.0"
    port: int = 8090
    require_auth: bool = False
    api_token: str = ""
    allowed_hosts: List[str] = field(
        default_factory=lambda: ["*"]
    )
    max_clients: int = 4


_SECTIONS = [
    ("platform", "platform", PlatformConfig),
    ("llm", "llm", LLMConfig),
    ("guardrails", "guardrails", GuardrailConfig),
    ("runtime", "runtime", RuntimeConfig),
    ("stt", "stt", STTConfig),
    ("tts", "tts", TTSConfig),
    ("audio", "audio", AudioConfig),
    ("vad", "vad", VADConfig),
    ("vision", "vision", VisionConfig),
    ("reachy", "reachy", ReachyConfig),
    ("rag", "rag", RAGConfig),
    ("web", "web", WebConfig),
]


@dataclass
class Config:
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    reachy: ReachyConfig = field(default_factory=ReachyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> "Config":
        """Load common settings, a platform profile, then custom overrides.

        Selection priority is the explicit argument, REACHY_PLATFORM, a custom
        config's platform.profile, then the common Orin Nano default.
        Custom overrides come from config_path or REACHY_ASSISTANT_CONFIG.
        """
        common_path = CONFIG_ROOT / "settings.yaml"
        common_data = _read_yaml(common_path)

        custom_data: dict = {}
        override_path = config_path or os.environ.get("REACHY_ASSISTANT_CONFIG")
        if override_path:
            custom_path = Path(override_path).expanduser().resolve()
            if custom_path != common_path.resolve():
                custom_data = _read_yaml(custom_path)

        requested = (
            platform
            or os.environ.get("REACHY_PLATFORM")
            or custom_data.get("platform", {}).get("profile")
            or common_data.get("platform", {}).get("profile")
            or "orin_nano"
        )
        canonical = PROFILE_ALIASES.get(str(requested).strip().lower())
        if canonical is None:
            available = ", ".join(cls.available_platforms())
            raise ValueError(
                f"Unknown Jetson platform profile {requested!r}; available: {available}"
            )

        profile_data = _read_yaml(
            CONFIG_ROOT / "platforms" / f"{canonical}.yaml"
        )
        data = _deep_merge(common_data, profile_data)
        data = _deep_merge(data, custom_data)

        config = cls()
        for yaml_key, attr_name, _ in _SECTIONS:
            section_obj = getattr(config, attr_name)
            section_data = data.get(yaml_key, {})
            if not isinstance(section_data, dict):
                raise ValueError(
                    f"Configuration section {yaml_key!r} must be a mapping"
                )
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
        config.platform.profile = canonical

        token = os.environ.get("REACHY_WEB_API_TOKEN")
        if token:
            config.web.api_token = token
        production = os.environ.get("REACHY_PRODUCTION_MODE")
        if production is not None:
            config.runtime.production_mode = production.lower() in {
                "1", "true", "yes", "on"
            }
        return config

    @staticmethod
    def available_platforms() -> List[str]:
        profiles = CONFIG_ROOT / "platforms"
        return sorted(path.stem for path in profiles.glob("*.yaml"))

    def validation_errors(self, require_web_auth: bool = True) -> List[str]:
        """Return configuration errors, including fail-closed production rules."""
        errors: List[str] = []
        if not self.platform.supported:
            errors.append(
                f"platform profile {self.platform.profile!r} is reserved but not implemented: "
                f"{self.platform.notes or self.platform.implementation_status}"
            )
        if not (1 <= self.web.port <= 65535):
            errors.append("web.port must be between 1 and 65535")
        if self.web.port == 8000:
            errors.append("web.port 8000 is reserved for the Reachy daemon")
        if not (1 <= self.web.max_clients <= 16):
            errors.append("web.max_clients must be between 1 and 16")
        if self.llm.max_tokens < 1 or self.llm.max_tokens > 512:
            errors.append("llm.max_tokens must be between 1 and 512")
        if self.guardrails.max_sentences < 1 or self.guardrails.max_sentences > 5:
            errors.append("guardrails.max_sentences must be between 1 and 5")
        if self.guardrails.max_words < 8 or self.guardrails.max_words > 200:
            errors.append("guardrails.max_words must be between 8 and 200")
        if self.vision.history_turns < 0 or self.vision.history_turns > 20:
            errors.append("vision.history_turns must be between 0 and 20")
        if not (0 <= self.audio.playback_tail_quiet_ms <= 1000):
            errors.append("audio.playback_tail_quiet_ms must be between 0 and 1000")
        if not (0 <= self.audio.playback_tail_max_wait_ms <= 3000):
            errors.append("audio.playback_tail_max_wait_ms must be between 0 and 3000")
        if self.audio.playback_tail_max_wait_ms < self.audio.playback_tail_quiet_ms:
            errors.append("audio.playback_tail_max_wait_ms must cover the quiet window")
        if not (0.0001 <= self.audio.playback_tail_rms_threshold <= 0.1):
            errors.append("audio.playback_tail_rms_threshold must be between 0.0001 and 0.1")

        if not self.runtime.production_mode:
            return errors

        if not self.llm.model.strip():
            errors.append("llm.model must be pinned explicitly in production mode")
        if not self.guardrails.enabled:
            errors.append("guardrails must be enabled in production mode")
        if self.stt.device != "cuda":
            errors.append("stt.device must be 'cuda' in production mode")
        if require_web_auth:
            if not self.web.require_auth:
                errors.append("web.require_auth must be true in production mode")
            if self.web.require_auth and len(self.web.api_token) < 32:
                errors.append(
                    "REACHY_WEB_API_TOKEN must contain at least 32 characters"
                )
            if not self.web.allowed_hosts or "*" in self.web.allowed_hosts:
                errors.append("web.allowed_hosts must be explicit in production mode")
            if self.web.host not in {"localhost", "127.0.0.1", "::1"}:
                errors.append(
                    "web.host must use loopback in production; expose it through TLS or an SSH tunnel"
                )
        llm_url = urlparse(self.llm.base_url)
        if llm_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            errors.append("llm.base_url must use a loopback host in production mode")
        if self.reachy.tracking_head_yaw_max_deg > 25:
            errors.append("tracking_head_yaw_max_deg exceeds production limit 25")
        if self.reachy.tracking_body_max_deg > 35:
            errors.append("tracking_body_max_deg exceeds production limit 35")
        if self.reachy.tracking_scan_body_range_deg > 25:
            errors.append("tracking_scan_body_range_deg exceeds production limit 25")
        return errors

    def require_valid(self, require_web_auth: bool = True) -> None:
        errors = self.validation_errors(require_web_auth=require_web_auth)
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))
