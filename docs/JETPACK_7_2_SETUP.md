# JetPack 7.2 setup (Jetson Orin)

This is the default setup path. It targets JetPack 7.2 / L4T r39, CUDA 13.2,
cuDNN 9, Ubuntu 24.04, Python 3.12, and an Orin GPU (`sm_87`). The base
`config/settings.yaml` and default llama.cpp image target this environment.

Two validation paths are provided: a software-only benchmark that mocks
Reachy hardware I/O, and the normal connected-Reachy application. The
software benchmark does not validate USB, motors, camera, or audio hardware.
See [SETUP.md](../SETUP.md) for the retained JetPack 6 legacy path.

## Native development packages

Install the narrower development set below, or use `nvidia-jetpack-dev` in
place of the CUDA/cuDNN packages if the complete JetPack development stack is
desired.

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

## Build memory on an 8 GB Orin Nano

ONNX Runtime's largest translation units can exhaust an 8 GB Orin Nano. Make
at least 8 GB of swap available on fast local storage before building. This
example creates a dedicated, non-persistent swap file; add a matching `/etc/fstab`
entry only if it should survive reboots.

```bash
sudo fallocate -l 8G /absolute/path/to/reachy-jp72.swap
sudo chmod 600 /absolute/path/to/reachy-jp72.swap
sudo mkswap /absolute/path/to/reachy-jp72.swap
sudo swapon /absolute/path/to/reachy-jp72.swap
free -h
```

The ONNX Runtime command below uses four parallel jobs with that swap enabled.
Use `--parallel 2` when no additional swap is available.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install "reachy-mini==1.3.1"
```

Keep this environment active when launching the assistant so the Reachy SDK
can find the `reachy-mini-daemon` executable installed in `.venv/bin`.

Do not build separate `faster-whisper` or `kokoro-onnx` wheels. They are Python
packages. Their GPU execution comes from the two native runtimes below:
CUDA-enabled CTranslate2 for faster-whisper, and CUDA-enabled ONNX Runtime for
Kokoro.

## Build CTranslate2 for Orin (`sm_87`)

The commands below use CTranslate2 4.8.1 and install it under the virtual
environment instead of modifying `/usr/local`.

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

Use `float16` for faster-whisper on this CUDA 13.2/SM87 stack. CTranslate2
reports plain `int8` as supported, but transcription fails at runtime with
`CUBLAS_STATUS_NOT_SUPPORTED`; `float16`, `int8_float16`, and `int8_float32`
all execute on CUDA, with `float16` fastest in the mocked-utterance check.

## Build ONNX Runtime with CUDA

JetPack 7.2/CUDA 13.2 does not use the repository's old JP6/cu126 wheel URL.
Build ONNX Runtime 1.28.0 locally. The small cuDNN discovery tree handles the
multiarch header/library locations used by Ubuntu on Jetson.

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

Those disabled domains reduce compile time and memory. They are valid for the
bundled Kokoro and Silero models, which use standard `ai.onnx` operators only.
Contrib must remain enabled because ONNX Runtime 1.28's core CUDA LLM sources
reference a contrib CUDA constant during compilation. Omit the corresponding
disable flags if another ONNX model needs ML, generation, or CUDA NHWC
operators.

The JP7.2 patch leaves the normal CUDA provider intact. It extends ONNX
Runtime's build-local CCCL parser workaround to CUDA 13.2 and enables
`ORT_QUICK_BUILD` only for the separately linked LLM/MoE CUDA object library.
Kokoro and Silero do not contain LLM/MoE operators. Remove that final patch
hunk when the wheel must provide the exhaustive LLM/MoE CUTLASS kernel set.

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

`run_llama_cpp.sh` defaults to
`ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`. Override
`LLAMA_CPP_IMAGE` to pin a known-compatible tag or digest. The previous
`r36.4`/CUDA 12.6 image is a JetPack 6 image and should not be used on r39.

```bash
NP=1 ./run_llama_cpp.sh /absolute/path/to/model.gguf
```

## Connected Reachy Mini

The base configuration is the normal JP7.2 connected-hardware profile, so no
overlay is required. `config/settings.jp72-reachy.yaml` is an optional,
conservative bring-up profile that also disables body assist and scanning.

### Serial permissions and device preflight

The user running the assistant needs access to the USB motor controller. For
persistent access, add the account to `dialout`, then log out and back in
(or reboot) before continuing:

```bash
sudo usermod -aG dialout "$USER"
```

For a single plugged-in session, use a narrower temporary ACL instead:

```bash
sudo setfacl -m u:"$USER":rw /dev/ttyACM0
```

The ACL disappears when the device node is removed or recreated. Confirm the
controller, camera, microphone, and speaker before starting the application:

```bash
ls -l /dev/ttyACM0 /dev/video0 /dev/video1
lsusb
pactl list short sources
pactl list short sinks
```

A Reachy Mini Lite should expose a serial controller, a Reachy Mini camera,
and a Reachy Mini audio input/output device. Adjust the device indices in
`config/settings.yaml` only if enumeration differs.

### Launch the connected assistant

Keep the project virtual environment active. The Reachy SDK spawns
`reachy-mini-daemon` by executable name, so launching with an unactivated
environment can make the daemon appear missing even when it is installed.

Terminal 1:

```bash
source .venv/bin/activate
NP=1 ./run_llama_cpp.sh /absolute/path/to/multimodal-model.gguf
```

Terminal 2:

```bash
source .venv/bin/activate
unset REACHY_ASSISTANT_CONFIG
python run_web_vision_chat.py
```

For conservative first motion, export
`REACHY_ASSISTANT_CONFIG=config/settings.jp72-reachy.yaml` before the Python
command. A healthy startup reports a connected/awake Reachy, live camera and
microphone, CUDA faster-whisper, the VLM model, and Kokoro using
`CUDAExecutionProvider`.

Stop the application with Ctrl-C so its cleanup handler returns Reachy to
sleep and disables its motors. Stop the model container separately:

```bash
docker stop assistant-llm
```

The controlled JP7.2 hardware pass validated the motor controller, camera,
GPU STT/VLM/TTS, Reachy speaker playback, and a speaking gesture. Its query
was injected at the STT boundary; complete a human-spoken microphone-to-Silero
VAD turn when validating a new installation.

## No-Reachy software benchmark

The profile disables robot connection, motion, hardware audio selection, and
RAG. All ordinary runners pick it up through the common config loader:

```bash
export REACHY_ASSISTANT_CONFIG=config/settings.jp72-no-reachy.yaml
```

Start a llama.cpp model on port 18080, then run the benchmark from the project
root. The test uses real Silero VAD, faster-whisper, llama.cpp streaming, and
Kokoro while mocking Reachy hardware I/O:

```bash
PORT=18080 NP=1 NAME=reachy-bench-llm \
  ./run_llama_cpp.sh /absolute/path/to/model.gguf

python scripts/bench_mocked_voice_pipeline.py \
  --base-url http://127.0.0.1:18080 \
  --runs 3 --out benchmarks/jp72_gpu_voice_pipeline.json

docker stop reachy-bench-llm
```

The report must show a CUDA STT backend and `CUDAExecutionProvider` for TTS.
Silero VAD remains on `CPUExecutionProvider` by design. These results
characterize the software pipeline only. Run `unset REACHY_ASSISTANT_CONFIG`
to return to the default connected-Reachy configuration when the robot is
attached.
