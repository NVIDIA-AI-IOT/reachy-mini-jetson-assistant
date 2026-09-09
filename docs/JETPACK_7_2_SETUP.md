# JetPack 7.2 Setup (Default)

This is the default installation path for the Reachy Mini Jetson Assistant. The validated tuple is JetPack 7.2 / L4T 39.2.0, Ubuntu 24.04, CUDA 13.2, cuDNN 9, Python 3.12, `aarch64`, and an Orin GPU (`sm_87`). It was tested on an Orin Nano 8GB.

> **JetPack 6 users:** keep using the complete [legacy JetPack 6 setup guide](../SETUP.md). Its Python 3.10, CUDA 12.6, and r36 container steps remain supported and are intentionally separate from this path.

## Before you start

- Flash JetPack 7.2 and confirm the host reports L4T 39.2.0.
- Install CUDA 13.2 and cuDNN 9; the compatibility check requires CUDA to be present before it changes the Python environment.
- Install Docker with the NVIDIA container runtime for the llama.cpp VLM.
- Ensure the account has `sudo` access and internet access for initial package and model downloads.
- Use an NVMe SSD for model storage and build swap when possible.
- Connecting a Reachy Mini Lite is optional for installation and software-only benchmarking, but required for hardware validation.

The setup script installs application dependencies; it does not flash JetPack or configure Docker and the NVIDIA container runtime.

## Recommended installation

Run all commands from a terminal on the Jetson.

### 1. Clone the repository

```bash
git clone https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant.git
cd reachy-mini-jetson-assistant
```

### 2. Check compatibility

Run the non-mutating check before installing anything:

```bash
./scripts/setup_jetson.sh --check-only
```

The check detects architecture, L4T, CUDA, Python, GPU family, and compute capability. It proceeds only when exactly one entry in `packaging/jetson-wheels.json` matches all six values.

### 3. Install the application

Install the exact published prerelease wheel set with:

```bash
./scripts/setup_jetson.sh
```

The current JP7.2 wheel set is published as a GitHub prerelease. Its technical audit passes, but the organization-approved scanner report and final OSRB closure evidence are not included; publication must not be described as final OSRB approval. Maintainers testing replacement candidates can use a local wheel directory:

```bash
./scripts/setup_jetson.sh \
  --wheel-dir /absolute/path/to/wheelhouse
```

Add `--venv` to validate a candidate without changing the project environment:

```bash
./scripts/setup_jetson.sh \
  --wheel-dir /absolute/path/to/wheelhouse \
  --venv /tmp/reachy-jp72-test
```

The installer:

- installs missing Ubuntu audio, Cairo, GObject, and Python environment packages;
- creates or reuses `.venv`;
- installs the application requirements and Reachy Mini SDK;
- downloads only the exact CTranslate2 and ONNX Runtime wheel filenames selected by the manifest when the release is published;
- verifies each wheel's size, SHA-256, distribution, version, Python ABI, and `linux_aarch64` platform tag before installing it;
- replaces any portable CPU runtime pulled in during Python dependency resolution;
- validates a CUDA device for CTranslate2 and `CUDAExecutionProvider` for ONNX Runtime; and
- runs the production Silero VAD smoke test.

It stops on an unsupported tuple, missing artifact, checksum mismatch, incorrect wheel metadata, unavailable GPU backend, or an active assistant using the target virtual environment. It never accepts a generic ARM CPU wheel as the final STT or TTS runtime.

### 4. Confirm success

A successful run ends with output equivalent to:

```text
Detected: architecture=aarch64 L4T=39.2.0 CUDA=13.2 Python=3.12 GPU=orin sm_8.7
CTranslate2: 4.8.1+reachy.jp72.cu132.sm87; CUDA devices: 1
ONNX Runtime: 1.28.0+reachy.jp72.cu132.sm87; providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
JP7.2 GPU runtime validation passed.
Silero VAD smoke test: <probability>
Setup complete for jp72-l4t39.2-cu13.2-py3.12-sm87.
```

Whisper, Kokoro, and Gemma model assets download on their first corresponding launch and are reused for offline inference afterward.

### Supported runtime policy

The application defaults to native CTranslate2 STT and ONNX Runtime Kokoro TTS. It does not silently replace either GPU runtime with a generic ARM CPU package. Speaches can be evaluated as a separately managed compatibility service, but it is not installed, started, or selected automatically by this setup path.

## Maintainer installation and source-build fallback

The following build instructions are retained for maintainers adding a new compatibility tuple or diagnosing a native runtime. Normal installations should use `scripts/setup_jetson.sh`.

### Native development packages

Install the narrower development set below, or use `nvidia-jetpack-dev` in place of the CUDA/cuDNN packages if the complete JetPack development stack is desired.

```bash
sudo apt update
sudo apt install -y \
  cuda-toolkit-13-2 libcudnn9-dev-cuda-13 \
  build-essential cmake ninja-build git patchelf pkg-config \
  python3-dev python3.12-venv \
  portaudio19-dev libasound2-dev pulseaudio-utils acl \
  libcairo2-dev libgirepository1.0-dev libssl-dev
```

Confirm the platform before compiling:

```bash
/usr/local/cuda/bin/nvcc --version
test "$(uname -m)" = aarch64
```

### Build memory on an 8 GB Orin Nano

ONNX Runtime's largest translation units can exhaust an 8 GB Orin Nano. Make at least 8 GB of swap available on fast local storage before building. This example creates a dedicated, non-persistent swap file; add a matching `/etc/fstab` entry only if it should survive reboots.

```bash
sudo fallocate -l 8G /absolute/path/to/reachy-jp72.swap
sudo chmod 600 /absolute/path/to/reachy-jp72.swap
sudo mkswap /absolute/path/to/reachy-jp72.swap
sudo swapon /absolute/path/to/reachy-jp72.swap
free -h
```

The ONNX Runtime command below uses four parallel jobs with that swap enabled. Use `--parallel 2` when no additional swap is available.

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "reachy-mini==1.3.1"
```

Keep this environment active when launching the assistant so the Reachy SDK can find the `reachy-mini-daemon` executable installed in `.venv/bin`.

Do not build separate `faster-whisper` or `kokoro-onnx` wheels. They are Python packages. Their GPU execution comes from the two native runtimes below: CUDA-enabled CTranslate2 for faster-whisper, and CUDA-enabled ONNX Runtime for Kokoro.

Silero VAD also runs directly through ONNX Runtime using the model bundled with `faster-whisper==1.2.1`; a separate `silero-vad`, PyTorch, or torchaudio install is not needed.

### Install candidate GPU wheels manually

For JetPack 7.2 on Orin, the lower-level installer accepts two already-selected local wheel paths or GitHub Release asset URLs. It does not select or checksum assets itself; prefer `setup_jetson.sh`, which performs those checks first:

```bash
source .venv/bin/activate
./scripts/install_jp72_gpu_wheels.sh \
  /path/to/ctranslate2-4.8.1+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl \
  /path/to/onnxruntime_gpu-1.28.0+reachy.jp72.cu132.sm87-cp312-cp312-linux_aarch64.whl
```

The lower-level installer checks aarch64, Python 3.12, and L4T r39; replaces the portable CPU runtimes that pip may have resolved; then asserts that CTranslate2 sees a CUDA device and ONNX Runtime exposes `CUDAExecutionProvider`. Release assets must be published only after the repository's OSRB/license review and checksum process is complete.

If the release wheels are unavailable for a newer JetPack, use the source-build fallback below and publish a new platform-tagged wheel set only after testing.

### Adding a new JetPack/runtime tuple

1. Record the exact architecture, L4T, CUDA, Python ABI, GPU family, and compute capability; never broaden an existing tuple.
2. Build CTranslate2 and ONNX Runtime from pinned upstream revisions.
3. Validate CUDA providers, the software pipeline benchmark, and connected Reachy hardware on the target image.
4. Run `scripts/audit_jetson_wheels.py` to inventory every archive member and ELF dependency, verify the in-wheel licenses/notices, prove NVIDIA runtime libraries resolve outside the wheels, and generate an SPDX 2.3 SBOM.
5. Complete OSRB review and archive the upstream licenses, complete third-party notices, exact patches, tags and commit SHAs, build commands, complete build environment, SBOM, and the unmodified report from NVIDIA’s organization-approved license/security scanner.
6. Create an immutable GitHub Release tag and attach both wheels plus provenance material; do not overwrite assets under an existing tag.
7. Add a new `published: false` manifest entry with exact filenames, versions, sizes, wheel tag, and SHA-256 values.
8. Run the installer in a clean temporary venv using `--wheel-dir`, then run the repository tests with `REACHY_JETSON_WHEEL_DIR` set to that wheelhouse.
9. Publish the assets, verify the release URL from a clean device, and only then change that tuple to `published: true`. Keep older validated tuples intact.

The current prerelease’s [release evidence bundle](../packaging/releases/native-jp72-cu132-py312-sm87-r1/README.md) records the exact audit scope and remaining scanner/OSRB gate for promotion to a final release.

### Source-build fallback: CTranslate2 for Orin (`sm_87`)

The commands below use CTranslate2 4.8.1 and install it under the virtual environment instead of modifying `/usr/local`.

```bash
git clone --branch v4.8.1 --depth 1 --recurse-submodules \
  https://github.com/OpenNMT/CTranslate2.git ../CTranslate2-v4.8.1
cmake -S ../CTranslate2-v4.8.1 -B ../CTranslate2-v4.8.1/build-jp72 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_ARCH_LIST=8.7 \
  -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DBUILD_CLI=OFF -DBUILD_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV/ctranslate2"
cmake --build ../CTranslate2-v4.8.1/build-jp72 --parallel 2
cmake --install ../CTranslate2-v4.8.1/build-jp72

export CTRANSLATE2_ROOT="$VIRTUAL_ENV/ctranslate2"
export CMAKE_BUILD_PARALLEL_LEVEL=2
python -m pip install --force-reinstall --no-deps ../CTranslate2-v4.8.1/python
CT2_EXT="$(find "$VIRTUAL_ENV"/lib/python*/site-packages/ctranslate2 \
  -name '_ext*.so' -print -quit)"
test -n "$CT2_EXT"
patchelf --set-rpath "$CTRANSLATE2_ROOT/lib" "$CT2_EXT"
```

Verify that faster-whisper can see the Orin GPU:

```bash
python - <<'PY'
import ctranslate2
print("CTranslate2:", ctranslate2.__version__)
print("CUDA devices:", ctranslate2.get_cuda_device_count())
print("CUDA types:", ctranslate2.get_supported_compute_types("cuda"))
PY
```

Use `float16` for faster-whisper on this CUDA 13.2/SM87 stack. CTranslate2 reports plain `int8` as supported, but transcription fails at runtime with `CUBLAS_STATUS_NOT_SUPPORTED`; `float16`, `int8_float16`, and `int8_float32` all execute on CUDA, with `float16` fastest in the mocked-utterance check.

### Source-build fallback: ONNX Runtime with CUDA

JetPack 7.2/CUDA 13.2 does not use the repository's old JP6/cu126 wheel URL. When a matching wheel is unavailable, build ONNX Runtime 1.28.0 locally. The small cuDNN discovery tree handles the multiarch header/library locations used by Ubuntu on Jetson.

```bash
git clone --branch v1.28.0 --depth 1 --recursive \
  https://github.com/microsoft/onnxruntime.git ../onnxruntime-v1.28.0

# Extend ONNX Runtime's CCCL workaround to CUDA 13.2 and limit only its
# optional LLM/MoE object library to the built-in quick-build kernel set.
git -C ../onnxruntime-v1.28.0 apply \
  "$PWD/patches/onnxruntime-v1.28.0-jp72.patch"

mkdir -p ../cudnn-jp72/include ../cudnn-jp72/lib
ln -s /usr/include/aarch64-linux-gnu/cudnn*.h ../cudnn-jp72/include/
ln -s /usr/lib/aarch64-linux-gnu/libcudnn*.so* ../cudnn-jp72/lib/

../onnxruntime-v1.28.0/build.sh \
  --build_dir ../onnxruntime-v1.28.0/build-jp72 \
  --config Release --update --build --build_wheel \
  --skip_tests --skip_submodule_sync --parallel 4 --nvcc_threads 1 \
  --compile_no_warning_as_error --use_cuda --disable_cuda_nhwc_ops \
  --disable_ml_ops --disable_generation_ops \
  --cuda_version 13.2 --cuda_home /usr/local/cuda \
  --cudnn_home ../cudnn-jp72 --cmake_generator Ninja \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF

python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install --no-deps \
  ../onnxruntime-v1.28.0/build-jp72/Release/dist/onnxruntime_gpu-*.whl
```

Those disabled domains reduce compile time and memory. They are valid for the bundled Kokoro and Silero models, which use standard `ai.onnx` operators only. Contrib must remain enabled because ONNX Runtime 1.28's core CUDA LLM sources reference a contrib CUDA constant during compilation. Omit the corresponding disable flags if another ONNX model needs ML, generation, or CUDA NHWC operators.

The JP7.2 patch leaves the normal CUDA provider intact. It extends ONNX Runtime's build-local CCCL parser workaround to CUDA 13.2 and enables `ORT_QUICK_BUILD` only for the separately linked LLM/MoE CUDA object library. Kokoro and Silero do not contain LLM/MoE operators. Remove that final patch hunk when the wheel must provide the exhaustive LLM/MoE CUTLASS kernel set.

Confirm the CUDA provider before launching Kokoro:

```bash
python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
assert "CUDAExecutionProvider" in providers
PY
```

## llama.cpp container on JetPack 7.2

`run_llama_cpp.sh` defaults to `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`. Override `LLAMA_CPP_IMAGE` to pin a known-compatible tag or digest. The previous `r36.4`/CUDA 12.6 image is a JetPack 6 image and should not be used on r39.

```bash
NP=1 ./run_llama_cpp.sh unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

On first use, the launcher downloads both the selected quantization and its `mmproj-F16.gguf` vision projector, then reuses the local files offline.

For a manually downloaded generic projector, select it explicitly and keep it beside the model so Docker can mount both from the same directory:

```bash
MMPROJ=./models/mmproj-F16.gguf NP=1 ./run_llama_cpp.sh ./models/gemma-4-E2B-it-Q4_K_M.gguf
```

For multimodal models on Orin Nano 8GB, the launcher defaults to `CTX=1024`, `BATCH=128`, and `UBATCH=128`. Gemma 4 E2B Q4_K_M was validated for ten consecutive live-camera requests with the rest of the GPU assistant pipeline resident. Keep `BATCH` and `UBATCH` equal when lowering them: setting a smaller micro-batch than the image decode batch causes current llama.cpp multimodal builds to abort. Text-only and embedding launches retain the larger 4096/2048/ 512 defaults. All values can be overridden explicitly:

```bash
CTX=2048 BATCH=256 UBATCH=256 NP=1 ./run_llama_cpp.sh /path/to/model.gguf
```

The launcher also defaults to `REASONING=off` so Gemma returns spoken-answer tokens in the OpenAI `content` field instead of spending the 128-token reply budget on `reasoning_content`. `CACHE_RAM=0` disables llama.cpp's server-wide prompt cache, whose multi-gigabyte default is unsafe when VLM, CUDA STT, and CUDA TTS share an 8GB Orin Nano. Both settings remain explicitly overridable for other deployments, for example `REASONING=auto CACHE_RAM=256`.

## Connected Reachy Mini

The base configuration is the normal JP7.2 connected-hardware profile, so no overlay is required. `config/settings.jp72-reachy.yaml` is an optional, conservative bring-up profile that also disables body assist and scanning.

### Serial permissions and device preflight

The user running the assistant needs access to the USB motor controller. For persistent access, add the account to `dialout`, then log out and back in (or reboot) before continuing:

```bash
sudo usermod -aG dialout "$USER"
```

For a single plugged-in session, use a narrower temporary ACL instead:

```bash
sudo setfacl -m u:"$USER":rw /dev/ttyACM0
```

The ACL disappears when the device node is removed or recreated. Confirm the controller, camera, microphone, and speaker before starting the application:

```bash
ls -l /dev/ttyACM0 /dev/video0 /dev/video1
lsusb
pactl list short sources
pactl list short sinks
```

A Reachy Mini Lite should expose a serial controller, a Reachy Mini camera, and a Reachy Mini audio input/output device. Adjust the device indices in `config/settings.yaml` only if enumeration differs.

### Launch the connected assistant

Keep the project virtual environment active. The Reachy SDK spawns `reachy-mini-daemon` by executable name, so launching with an unactivated environment can make the daemon appear missing even when it is installed.

Terminal 1:

```bash
source .venv/bin/activate
NP=1 ./run_llama_cpp.sh unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

Terminal 2:

```bash
source .venv/bin/activate
unset REACHY_ASSISTANT_CONFIG
python run_web_vision_chat.py
```

For conservative first motion, export `REACHY_ASSISTANT_CONFIG=config/settings.jp72-reachy.yaml` before the Python command. A healthy startup reports a connected/awake Reachy, live camera and microphone, CUDA faster-whisper, the VLM model, and Kokoro using `CUDAExecutionProvider`.

Stop the application with Ctrl-C so its cleanup handler returns Reachy to sleep and disables its motors. Stop the model container separately:

```bash
docker stop assistant-llm
```

The controlled JP7.2 hardware pass included a human-spoken turn and validated the microphone, Silero VAD, GPU STT/VLM/TTS, camera, Reachy speaker playback, USB motor controller, and an official speaking gesture.

## No-Reachy software benchmark

The profile disables robot connection, motion, hardware audio selection, and RAG. All ordinary runners pick it up through the common config loader:

```bash
export REACHY_ASSISTANT_CONFIG=config/settings.jp72-no-reachy.yaml
```

Start a llama.cpp model on port 18080, then run the benchmark from the project root. The test uses real Silero VAD, faster-whisper, llama.cpp streaming, and Kokoro while mocking Reachy hardware I/O:

```bash
PORT=18080 NP=1 NAME=reachy-bench-llm \
  ./run_llama_cpp.sh /absolute/path/to/model.gguf

python scripts/bench_mocked_voice_pipeline.py \
  --base-url http://127.0.0.1:18080 \
  --runs 3 --out /tmp/reachy-jp72-gpu-voice-pipeline.json

docker stop reachy-bench-llm
```

The report must show a CUDA STT backend and `CUDAExecutionProvider` for TTS. Silero VAD remains on `CPUExecutionProvider` by design. These results characterize the software pipeline only. Run `unset REACHY_ASSISTANT_CONFIG` to return to the default connected-Reachy configuration when the robot is attached.
