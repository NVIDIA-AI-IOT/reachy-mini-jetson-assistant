# Jetson AGX Thor Setup

This guide selects the `thor` profile. Do not use the Orin Nano llama.cpp
instructions for this platform.

The pinned Thor runtime targets JetPack 7.1, Python 3.12, CUDA 13, and GPU
architecture `sm_110`. The default Orin Nano JetPack 7.2 setup script and
published `sm_87` wheel bundle target a different compatibility tuple; use the
Thor installer and manifest described here.

The Thor profile uses faster-whisper `large-v3` with CUDA FP16. Its default UI
is reachable directly on the trusted LAN, matching the Orin Nano workflow.
For an optional authenticated, loopback-only deployment, see
`PRODUCTION_THOR.md`.

This setup keeps the existing faster-whisper STT and Kokoro ONNX TTS pipeline,
and serves `google/gemma-4-E4B-it` through the Jetson AI Lab Gemma 4 vLLM image.

## Ports

- `8000`: Reachy Mini daemon
- `8001`: vLLM OpenAI-compatible API
- `8090`: Reachy assistant web UI

## Python environment

```bash
python3 -m venv venv-thor
source venv-thor/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-thor.txt
```

The Thor build is GPU-only. Build CTranslate2 4.8.1 with CUDA 13, cuDNN 9,
and `sm_110`, then install its Python wheel. Install the Jetson AI Lab CUDA
ONNX Runtime wheel instead of the generic CPU wheel:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install --no-deps \
  --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/ \
  onnxruntime-gpu==1.24.0
```

The application deliberately fails during startup if faster-whisper cannot
use CTranslate2 CUDA or if Kokoro cannot create a CUDA-primary ONNX session.
Kokoro's neural model compute runs on CUDA; ONNX Runtime still assigns a few
shape-control nodes in the exported graph to CPU and cannot initialize this
model when CPU execution is disabled entirely.

## Start the VLM

The default vLLM memory fraction is `0.50`. On Jetson unified-memory systems,
drop reclaimable filesystem caches before starting vLLM so the reservation is
available while still leaving roughly half of memory for faster-whisper,
Kokoro, camera processing, robot control, and the OS.

```bash
sync
sudo sysctl -w vm.drop_caches=3
./run_vllm_thor.sh
curl http://127.0.0.1:8001/v1/models
```

To run it in the background:

```bash
sync
sudo sysctl -w vm.drop_caches=3
DETACH=1 ./run_vllm_thor.sh
docker logs -f reachy-vllm
```

## Verify the Python runtime without Reachy

These checks inspect imports and CUDA availability without connecting to a
robot or starting the assistant:

```bash
source venv-thor/bin/activate
LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}" \
  python scripts/check_thor_runtime.py
python -c "from reachy_mini import ReachyMini; print('Reachy Mini SDK: OK')"
python -c "import faster_whisper; print('faster-whisper: OK')"
python -c "import kokoro_onnx; print('Kokoro: OK')"
REACHY_PLATFORM=thor python main.py info
```

Passing these checks does not validate microphone capture, camera frames,
speaker playback, or robot motion. Connect Reachy before running the assistant
and complete those checks separately.

## Run the assistant

```bash
./run_reachy_thor.sh
```

Open `http://<thor-ip>:8090` from a browser on the same network. The default
Thor profile matches the Orin Nano UI workflow and does not require an SSH
tunnel or browser token. The app starts the local Reachy daemon automatically
and refuses to enter the listening loop if STT, VLM, or TTS failed to
initialize.

## Model-output guardrails

Model output is buffered and validated before it reaches the terminal, web UI,
TTS queue, or speaking-movement callback. The deterministic guardrail blocks
prompt leakage, secret material, active markup, robot-control payloads, unsafe
procedural content, and repetitive output. It also removes URLs and formatting
and enforces the configured character, word, and sentence limits.

Guardrail settings live under `guardrails` in `config/settings.yaml`. When a
response is blocked, only the configured safe fallback sentence is displayed
and spoken. Setting `enabled: false` restores raw streaming and should be used
only for debugging.
