---
name: reachy-jetson-deploy
description: Launch, verify, and stop the Reachy Mini Jetson Assistant on connected hardware or in no-Reachy mode. Use for starting Gemma or another llama.cpp VLM, running the web vision pipeline, checking the live dashboard and WebSocket, selecting safe configuration overlays, and performing orderly shutdown.
---

# Reachy Jetson Deploy

Deploy from the repository root and reuse healthy services instead of starting duplicate model, assistant, or Reachy daemon processes.

## Preflight

1. Run `./scripts/setup_jetson.sh --check-only` and verify `.venv` exists.
2. Inspect `pgrep -af` for assistant and `reachy-mini-daemon` processes. Check `http://127.0.0.1:8080/health` and `http://127.0.0.1:8090/` before starting replacements.
3. For connected hardware, verify the serial controller, camera, PulseAudio source, and sink. Use `config/settings.jp72-reachy.yaml` for conservative initial motion. Use `config/settings.jp72-no-reachy.yaml` only when hardware I/O must be disabled.

## Launch

Start the JP7.2 VLM with bounded Orin Nano memory settings:

```bash
NP=1 ./run_llama_cpp.sh unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

The launcher downloads and reuses a model-family-named projector. If a generic projector was downloaded manually, select it explicitly and keep it beside the model: `MMPROJ=./models/mmproj-F16.gguf ./run_llama_cpp.sh ./models/model.gguf`. Never attach a generic projector to a model by filename guesswork.

Then activate the project environment and start the web pipeline in a managed foreground session:

```bash
source .venv/bin/activate
python run_web_vision_chat.py
```

Use `REACHY_ASSISTANT_CONFIG` only when an overlay is intended. Report the exact active overlay.

## Verify and stop

- Require startup evidence for CUDA STT, ONNX CUDA TTS, VAD, VLM, camera, mic, speaker/AEC, tracking, and `Ready`.
- Verify dashboard HTTP 200 and receive `ptt_state` plus `speaker_state` from `ws://127.0.0.1:8090/ws`.
- Leave services running when the user wants to interact with the dashboard.
- Stop the assistant with Ctrl-C so Reachy sleeps cleanly. Stop the model container separately. Do not kill the daemon or motors unless requested or graceful shutdown has failed.
