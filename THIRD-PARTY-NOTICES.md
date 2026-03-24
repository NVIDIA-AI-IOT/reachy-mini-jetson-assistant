# Third-Party Software Notices

This project uses the following third-party open source software.

## Direct Dependencies

| Package | License | URL |
|---------|---------|-----|
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| Rich | MIT | https://github.com/Textualize/rich |
| Typer | MIT | https://github.com/fastapi/typer |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| sounddevice | MIT | https://github.com/spatialaudio/python-sounddevice |
| Silero VAD | MIT | https://github.com/snakers4/silero-vad |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| kokoro-onnx | MIT | https://github.com/thewh1teagle/kokoro-onnx |
| opencv-python-headless | Apache-2.0 | https://github.com/opencv/opencv-python |
| ChromaDB | Apache-2.0 | https://github.com/chroma-core/chroma |
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |

## Separately Installed Dependencies

| Package | License | URL |
|---------|---------|-----|
| onnxruntime-gpu | MIT | https://github.com/microsoft/onnxruntime |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| reachy-mini | Apache-2.0 | https://github.com/pollen-robotics/reachy_mini |

## Key Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchaudio | BSD-3-Clause | https://github.com/pytorch/audio |
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| sentence-transformers | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| Pydantic | MIT | https://github.com/pydantic/pydantic |
| espeakng-loader | MIT | https://github.com/thewh1teagle/espeakng-loader |

## GPL-Licensed Transitive Dependencies (Subprocess-Isolated)

The following GPL-licensed packages are transitive dependencies of `kokoro-onnx`
(MIT). They run in a **separate subprocess** (`app/tts_worker.py`) that does not
load any NVIDIA proprietary libraries. The main application process never imports
these packages. Communication between processes uses JSON over stdin/stdout pipes
(standard IPC), which does not constitute linking under GPL.

| Package | License | URL |
|---------|---------|-----|
| phonemizer-fork | GPL-3.0 | https://github.com/thewh1teagle/phonemizer |
| espeak-ng | GPL-3.0 | https://github.com/espeak-ng/espeak-ng |

## External Services (Process-Isolated)

The following run as separate Docker containers and communicate via HTTP API:

| Software | License | URL |
|----------|---------|-----|
| llama.cpp | MIT | https://github.com/ggerganov/llama.cpp |

## Model Licenses

| Model | License | URL |
|-------|---------|-----|
| Cosmos-Reason2-2B | Apache-2.0 | https://huggingface.co/nvidia/Cosmos-Reason2-2B |
| faster-whisper (small.en) | MIT | https://huggingface.co/Systran/faster-whisper-small.en |
| Kokoro v1.0 | Apache-2.0 | https://huggingface.co/hexgrad/Kokoro-82M |
| YuNet face detection | MIT | https://huggingface.co/opencv/face_detection_yunet |
| FER+ int8 emotion | MIT | https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8 |
| reachy-mini-emotions-library | Apache-2.0 | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |
| bge-small-en-v1.5 | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 |
