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
import yaml


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
class STTConfig:
    model: str = "base.en"
    device: str = "cuda"
    compute_type: str = "int8"
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


@dataclass
class VADConfig:
    speech_threshold: float = 0.008
    silence_duration_ms: int = 500
    lookback_ms: int = 250
    max_speech_secs: int = 15
    chunk_ms: int = 30
    min_utterance_secs: float = 0.3
    min_utterance_rms: float = 0.005
    silero_threshold: float = 0.5


@dataclass
class VisionConfig:
    camera_device: int = 0
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


@dataclass
class ReachyConfig:
    enabled: bool = True
    spawn_daemon: bool = True
    timeout: float = 30.0
    media_backend: str = "no_media"
    automatic_body_yaw: bool = False
    wake_on_start: bool = True
    sleep_on_exit: bool = False
    antenna_rest_position: List[float] = field(default_factory=lambda: [0.0, 0.0])
    daemon_retry_attempts: int = 3
    daemon_startup_wait: float = 15.0
    face_tracking: bool = True
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
    tracking_vertical: bool = False
    tracking_return_to_neutral: bool = False
    tracking_scan_enabled: bool = True
    tracking_scan_body_range_deg: float = 20.0
    tracking_scan_speed_deg_per_sec: float = 5.0
    tracking_capture_settle_secs: float = 0.35
    tracking_capture_acquire_timeout_secs: float = 0.4


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


_SECTIONS = [
    ("llm", "llm", LLMConfig),
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    reachy: ReachyConfig = field(default_factory=ReachyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        config = cls()
        if not os.path.exists(config_path):
            return config
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            for yaml_key, attr_name, _ in _SECTIONS:
                section_obj = getattr(config, attr_name)
                for k, v in data.get(yaml_key, {}).items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        except Exception as e:
            print(f"Error loading config: {e}")
        return config
