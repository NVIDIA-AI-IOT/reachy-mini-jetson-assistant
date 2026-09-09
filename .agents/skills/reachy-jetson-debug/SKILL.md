---
name: reachy-jetson-debug
description: Diagnose Reachy Mini Jetson Assistant failures with read-only checks before repairs. Use for missing CUDA acceleration, STT or TTS fallback, VLM disconnects, camera or audio failures, Reachy daemon and motor connection problems, dashboard or WebSocket issues, high memory, crashes, and slow end-to-end latency.
---

# Reachy Jetson Debug

Collect evidence without stopping a working pipeline. Do not reinstall packages, restart containers, change audio routing, or move Reachy until the failing layer is identified or the user requested a repair.

## Snapshot

Run these checks from the repository root:

```bash
git status --short
./scripts/setup_jetson.sh --check-only
pgrep -af 'run_web_vision_chat.py|reachy-mini-daemon'
curl -fsS http://127.0.0.1:8080/health
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/
free -h
pactl list short sources
pactl list short sinks
ls -l /dev/ttyACM* /dev/video* 2>/dev/null
```

Also verify CTranslate2's CUDA device count and ONNX Runtime's available providers from `.venv`. Read model-container and assistant logs without exposing base64 camera frames or secrets.

## Route the failure

- **STT CPU/failure:** inspect the CTranslate2 version, CUDA device count, `sm_87` match, and `float16` configuration.
- **TTS CPU/failure:** inspect the ONNX Runtime version/provider, Kokoro worker stderr, model files, and CUDA/cuDNN shared-library resolution.
- **VLM disconnect/latency:** inspect container health, OOM/restart state, context/batch settings, image prefill, and a fresh-connection streaming test.
- **Mic/speaker/AEC:** inspect PulseAudio devices, selected sink, capture process, and AEC state before changing routes.
- **Camera/tracking:** check device ownership and frame delivery before changing motor-control settings.
- **Dashboard:** distinguish HTTP, WebSocket, broadcaster, and browser state.

## Repair discipline

Make the smallest reversible fix, rerun the failed layer test, then run the full test suite. Preserve a working Reachy daemon. Never replace a manifest-selected GPU wheel with a generic `aarch64` package. Do not commit or push unless asked.
