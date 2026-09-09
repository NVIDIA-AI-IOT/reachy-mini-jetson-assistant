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
| Silero VAD model (bundled by faster-whisper) | MIT | https://github.com/snakers4/silero-vad |
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
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| reachy-mini | Apache-2.0 | https://github.com/pollen-robotics/reachy_mini |

## Key Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| sentence-transformers | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| Pydantic | MIT | https://github.com/pydantic/pydantic |
| espeakng-loader | MIT | https://github.com/thewh1teagle/espeakng-loader |

## Development and Release Validation Dependencies

| Package | License | URL |
|---------|---------|-----|
| ONNX | Apache-2.0 | https://github.com/onnx/onnx |
| pytest | MIT | https://github.com/pytest-dev/pytest |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy |
| build | MIT | https://github.com/pypa/build |
| wheel | MIT | https://github.com/pypa/wheel |
| setuptools | MIT | https://github.com/pypa/setuptools |
| pybind11 | BSD-3-Clause | https://github.com/pybind/pybind11 |

## GPL-Licensed TTS Dependencies

The following GPL-licensed packages are transitive dependencies of `kokoro-onnx` (MIT). They run in the separate `app/tts_worker.py` process and communicate with the main application over JSON stdin/stdout pipes. Distributions that include them must provide the required license notices and corresponding source material.

| Package | License | URL |
|---------|---------|-----|
| phonemizer-fork | GPL-3.0 | https://github.com/thewh1teagle/phonemizer |
| espeak-ng | GPL-3.0 | https://github.com/espeak-ng/espeak-ng |

## JetPack 7.2 Native Wheel Distribution

The optional JetPack 7.2 wheels are local builds of unmodified CTranslate2 4.8.1 and patched ONNX Runtime 1.28.0 source. The wheels do not bundle the CUDA or cuDNN system libraries. Any published release must preserve the upstream license and third-party-notice files embedded in the wheels.

ONNX Runtime's third-party notices include components under several permissive licenses and Eigen under MPL-2.0. The release evidence archives the exact source tags, the repository's ONNX Runtime patch, build commands, final wheel hashes, licenses, and third-party notices.

## External Services (Process-Isolated)

The following run as separate Docker containers and communicate via HTTP API:

| Software | License | URL |
|----------|---------|-----|
| llama.cpp | MIT | https://github.com/ggerganov/llama.cpp |

## Model Licenses

| Model | License | URL |
|-------|---------|-----|
| Cosmos-Reason2-2B | Apache-2.0 | https://huggingface.co/nvidia/Cosmos-Reason2-2B |
| Gemma 4 E2B (Unsloth GGUF) | Apache-2.0 | https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF |
| faster-whisper (small.en) | MIT | https://huggingface.co/Systran/faster-whisper-small.en |
| Kokoro v1.0 | Apache-2.0 | https://huggingface.co/hexgrad/Kokoro-82M |
| YuNet face detection | MIT | https://huggingface.co/opencv/face_detection_yunet |
| FER+ int8 emotion | MIT | https://huggingface.co/onnxmodelzoo/emotion-ferplus-12-int8 |
| reachy-mini-emotions-library | Apache-2.0 | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |
| bge-small-en-v1.5 | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 |

The current Gemma 4 distributor metadata labels the GGUF repository Apache-2.0 and also links Google's Gemma 4 terms at https://ai.google.dev/gemma/docs/gemma_4_license. Capture and review both the model-card metadata and linked terms before redistributing model artifacts.

### Kokoro model data attributions

The Kokoro v1.0 model card identifies the following Creative Commons Attribution audio in its training data. Preserve these credits when distributing the model or a bundle containing it.

| Audio data | License | Source |
|---|---|---|
| Koniwa `tnc` | CC-BY-3.0 | https://github.com/koniwa/koniwa |
| SIWIS | CC-BY-4.0 | https://datashare.ed.ac.uk/items/1de74991-eede-4b48-8fbe-6c2abaed88d8 |

See the upstream Kokoro model card for its complete acknowledgements and training-data statement: https://huggingface.co/hexgrad/Kokoro-82M.

## Dependency-resolution requirement

The tables above are a human-readable notice index. Keep an exact dependency lock for every supported installation profile and archive the applicable upstream license and notice texts with each release.
