# Jetson Platform Architecture

The application uses layered configuration so robot behavior, guardrails, and
UI code stay shared while model servers, CUDA runtimes, memory choices, camera
nodes, and deployment policy remain platform-specific.

Configuration is merged in this order:

1. `config/settings.yaml` — shared behavior and Orin Nano JetPack 7.2 defaults.
2. `config/platforms/<profile>.yaml` — hardware and model-server overrides.
3. An optional custom configuration file (`--config` or `REACHY_ASSISTANT_CONFIG`) — deployment-specific overrides.
4. Secret/environment overrides such as `REACHY_WEB_API_TOKEN`.

Selection priority is `--platform`, `REACHY_PLATFORM`, a custom file's
`platform.profile`, then the default `orin_nano` profile.

## Orin Nano

- Profile: `orin_nano`
- Status: default platform; JP7.2 runtime with a retained JP6 overlay
- JetPack: 7.2 / L4T r39 / Python 3.12 / CUDA 13.2 (`sm_87`)
- Model server: llama.cpp on port 8080
- Default VLM: Gemma 4 E2B GGUF
- Speech: faster-whisper `small.en` CUDA FP16 and Kokoro ONNX CUDA
- Camera default: `/dev/video1`
- Launcher: `./run_reachy_orin_nano.sh`
- Setup: [JetPack 7.2](docs/JETPACK_7_2_SETUP.md)

Existing JetPack 6 / L4T r36 / Python 3.10 installations use
`REACHY_ASSISTANT_CONFIG=config/settings.jp6.yaml` to restore CUDA INT8 STT and
the legacy speaker preference. They must also select the r36/CUDA 12.6
llama.cpp image documented in [SETUP.md](SETUP.md). The Orin launcher prefers
`.venv`, falls back to `venv`, and accepts `VENV` to select an environment
explicitly.

Existing direct commands continue to select Orin Nano when neither
`--platform` nor `REACHY_PLATFORM` is provided.

## Jetson AGX Thor

- Profile: `thor`
- Status: runtime profile implemented
- JetPack: 7.1 / Python 3.12 / CUDA 13 (`sm_110`)
- Model server: pinned vLLM/Gemma on loopback port 8001
- Speech: custom CTranslate2 `sm_110` FP16 and ONNX Runtime GPU
- Camera default: automatic discovery of the Reachy camera's `video-index0`
- UI: LAN-accessible on port `8090`, matching the Orin Nano workflow
- Launcher: `./run_reachy_thor.sh`
- Setup: `SETUP_THOR.md`; optional hardening: `PRODUCTION_THOR.md`

Thor uses its own pinned installer and runtime manifest. The published Orin
JP7.2 `sm_87` wheel bundle is not compatible with this profile.

## Jetson AGX Orin

- Profile: `agx_orin`
- Status: architecture reserved; implementation intentionally skipped
- Launcher: `./run_reachy_agx_orin.sh` exits with a clear unsupported message

The profile already fits the shared schema and can later define its model
server, JetPack/CUDA runtime, STT compute type, memory limits, camera node,
installer, manifest, and production validation. Until those are implemented
and tested, `Config.require_valid()` rejects the profile before touching the
robot, microphone, or camera.

## Custom overrides

Examples:

```bash
REACHY_PLATFORM=orin_nano python3 run_web_vision_chat.py
python3 run_web_vision_chat.py --platform thor
```

A custom YAML file can contain only the fields being changed and specify its
base platform:

```yaml
platform:
  profile: thor
llm:
  temperature: 0.1
```

Platform profiles must never silently fall back to another device. Unknown or
unimplemented profiles fail validation explicitly.
