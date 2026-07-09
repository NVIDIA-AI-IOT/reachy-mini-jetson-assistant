# Setup Guide

Full installation instructions for the Reachy Mini Jetson Assistant.

## Prerequisites

### Hardware

- **NVIDIA Jetson Orin Nano** (8GB) — other Jetson modules may work but are untested
- **[Reachy Mini Lite](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started)** — the developer version, USB connection to your computer. Provides camera, microphone, speaker, and 9-DOF motor control in one cable. [Buy Reachy Mini](https://www.hf.co/reachy-mini/)
- **NVMe SSD** recommended — for swap space and model storage

If you're new to Reachy Mini, start with the [official getting started guide](https://huggingface.co/docs/reachy_mini/index) and the [Reachy Mini Lite setup](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started). The [Python SDK documentation](https://huggingface.co/docs/reachy_mini/SDK/readme) covers movement, camera, audio, and AI integrations.

### Software

- **JetPack 6.x** (L4T r36.x, Ubuntu 22.04, CUDA 12.6)
- **Python 3.10** (ships with JetPack 6 Ubuntu 22.04)
- **Docker** with NVIDIA runtime (`nvidia-container-toolkit`)
- **PulseAudio** (for mic/speaker multiplexing)

> **Important:** This project requires **Python 3.10** specifically. The Jetson ONNX Runtime GPU wheels, CTranslate2 builds, and Reachy Mini SDK are all built against Python 3.10 on JetPack 6. Using a different Python version will cause compatibility issues.

### Minimal L4T Installations

A Jetson flashed with only the minimal L4T packages may not include CUDA, cuDNN, development tools, Docker, or the NVIDIA Container Runtime required by this project.

The recommended installation path is the complete JetPack metapackage:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-jetpack
sudo reboot
```

If installing the complete JetPack metapackage is not appropriate for your deployment, install the equivalent CUDA, cuDNN, and NVIDIA Container Runtime packages individually and validate them with the checks below. That configuration is not currently tested by this project.

Do not add CUDA paths to `.bashrc` automatically. First verify whether CUDA is already available. If required, configure it for the current shell:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

### Platform Preflight Checks

Run these checks before installing the application:

```bash
# Jetson Linux / L4T (tested with r36.x)
cat /etc/nv_tegra_release

# Full JetPack installation (recommended for minimal-L4T systems)
dpkg-query -W nvidia-jetpack

# Python (must be 3.10.x)
python3.10 --version

# CUDA toolkit
/usr/local/cuda/bin/nvcc --version

# cuDNN packages
dpkg -l | grep -E 'libcudnn[0-9]'

# Docker and NVIDIA runtime
docker --version
docker info --format '{{json .Runtimes}}' | grep nvidia
```

If any required component is missing on a minimal-L4T system, install `nvidia-jetpack` before continuing.

## Hardware Setup

### Reachy Mini Lite

1. Connect Reachy Mini Lite to your Jetson via USB. The robot provides camera, microphone, speaker, and motor control over a single USB connection.

2. Add udev rules so the SDK can access the robot's serial ports without root:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="000a", MODE="0666", SYMLINK+="reachy_mini"' \
  | sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

3. Add your user to the `dialout` group and reboot:

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

4. Verify the device is visible:

```bash
ls -la /dev/ttyACM*
# Should show /dev/ttyACM0, /dev/ttyACM1, etc.
```

### NVMe Swap (Required for 8GB Jetson)

Running STT + VLM + TTS simultaneously exceeds 8GB RAM. Setting up swap on NVMe prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Installation

### Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  python3-dev \
  python3.10-venv \
  libcairo2-dev \
  libgirepository1.0-dev \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12
```

The Cairo and GObject development packages are required when PyGObject is built as part of the Reachy Mini media dependencies. Verify that `pkg-config` can find them:

```bash
pkg-config --modversion cairo
pkg-config --modversion gobject-introspection-1.0
```

### Step 2: Clone and Create Virtual Environment

```bash
git clone https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant
cd reachy-mini-jetson-assistant
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Packages

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

### Step 4: Install ONNX Runtime GPU (Jetson-Specific)

The default `onnxruntime` from pip is CPU-only. For GPU inference (Kokoro TTS, Silero VAD) on Jetson:

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed after install, uninstall the CPU version first: `pip uninstall onnxruntime && pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Step 5: Install Reachy Mini SDK

The project is currently runtime-tested with Reachy Mini SDK 1.3.1:

```bash
pip install "reachy-mini==1.3.1"
```

Verify the installed version:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('reachy-mini'))"
```

> **Known dependency limitation:** Reachy Mini 1.3.1 declares `numpy>=2.2.5`, while this project currently uses NumPy 1.26.4 for its tested Jetson ONNX Runtime stack. The combination runs on the development system, but `pip check` reports the version mismatch. A clean dependency combination still needs validation before the related installation issue is closed.

### Step 6: Pin NumPy (Compatibility Fix)

The Jetson `onnxruntime-gpu` wheel requires NumPy 1.x:

```bash
pip install "numpy==1.26.4"
```

Verify the runtime version:

```bash
python -c "import numpy; print(numpy.__version__)"
```

### Step 7: Build CTranslate2 with CUDA (GPU-Accelerated STT)

The pip `ctranslate2` package is CPU-only. For GPU-accelerated speech-to-text on Jetson, build from source:

```bash
pip install pybind11

cd ~
git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git submodule update --init --recursive

mkdir build && cd build
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_LIST="8.7" -DOPENMP_RUNTIME=NONE -DWITH_MKL=OFF

make -j$(nproc)
cmake --install . --prefix ~/.local

export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
cd ../python
pip install .
```

Persist the library path in your venv activation script:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/reachy-mini-jetson-assistant/venv/bin/activate
```

### Verify Installation

```bash
source venv/bin/activate
python3 -c "
import importlib.metadata; print('Reachy Mini SDK:', importlib.metadata.version('reachy-mini'))
import numpy; print('NumPy:', numpy.__version__)
import ctranslate2; print('CTranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())
import onnxruntime; print('ONNX providers:', onnxruntime.get_available_providers())
from reachy_mini import ReachyMini; print('Reachy Mini SDK: OK')
import faster_whisper; print('faster-whisper: OK')
import kokoro_onnx; print('kokoro-onnx: OK')
"
```

Expected output:
```
Reachy Mini SDK: 1.3.1
NumPy: 1.26.4
CTranslate2 CUDA devices: 1
ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Reachy Mini SDK: OK
faster-whisper: OK
kokoro-onnx: OK
```

## Models

### LLM / VLM (served via llama.cpp Docker)

Models download automatically from HuggingFace on first launch. No manual download needed.

| Model | Use | Launch Command |
|-------|-----|----------------|
| Cosmos-Reason2-2B (Q4_K_M) | Vision VLM | `NP=1 ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M` |
| Gemma 3 1B (Q8) | Text LLM | `./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0` |
| bge-small-en-v1.5 (Q8) | RAG embeddings | `./run_llama_embedding.sh ggml-org/bge-small-en-v1.5-Q8_0-GGUF:Q8_0` |

Models are cached in `~/.cache/huggingface` and reused across runs.

### TTS Voices

**Kokoro TTS** (default) downloads automatically on first run (~340 MB). No manual step needed.

To pre-download for offline use:

```bash
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Configure voice in `config/settings.yaml`:

```yaml
tts:
  voice: "af_sarah"    # kokoro voices: af_sarah, af_bella, am_adam, bf_emma, bm_george
```

### Emotion Model

The emotion classifier (DistilBERT SST-2, ~268 MB) downloads automatically on first run. No manual step needed.

To pre-download for offline use:

```bash
mkdir -p models/emotion
wget -O models/emotion/model.onnx \
  "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model.onnx"
wget -O models/emotion/tokenizer.json \
  "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json"
```

## Troubleshooting

**PyGObject fails to build while installing Reachy Mini:**
Install the native Cairo and GObject development packages, then retry the SDK installation:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  python3-dev \
  libcairo2-dev \
  libgirepository1.0-dev

source venv/bin/activate
pip install "reachy-mini==1.3.1"
```

**CUDA or cuDNN is missing on a minimal-L4T installation:**
Install the complete JetPack stack and reboot:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-jetpack
sudo reboot
```

**CUDA is installed but `nvcc` is not on `PATH`:**
Configure only the current shell until the installation path is verified:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

**`CUDAExecutionProvider` not available:**
Uninstall CPU onnxruntime and reinstall the GPU version:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**CTranslate2 not finding CUDA:**
Make sure the library path is set: `export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH`

**VLM server not responding:**
Check the Docker container is running: `docker ps`. View logs: `docker logs assistant-llm`

**Process won't exit / robot stays awake after Ctrl+C:**
The app handles Ctrl+C cleanly — the robot should go to sleep. If the process is stuck, run `pkill -9 -f run_web_vision_chat` and `pkill -f reachy-mini-daemon`.

**Port 8090 already in use:**
A previous instance is still running. Kill it: `lsof -ti :8090 | xargs kill -9`

**Camera not found:**
Check the device is available: `ls /dev/video*`. If another process holds it: `fuser -k /dev/video0`

## Installation-Issue Validation

Before closing the PyGObject or minimal-L4T installation issues, repeat the complete setup on:

1. A clean standard JetPack 6 image.
2. A clean minimal-L4T image followed by `nvidia-jetpack` installation.
3. A new Python 3.10 virtual environment.

Record the following output with the test results:

```bash
cat /etc/nv_tegra_release
python3.10 --version
/usr/local/cuda/bin/nvcc --version
docker info --format '{{json .Runtimes}}'
python -m pip check
```

Keep the PyGObject issue open until the SDK installs successfully on a clean image. Keep the minimal-L4T issue open until CUDA, cuDNN, the NVIDIA Container Runtime, and the application have all been validated on that installation path.
