#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/packaging/jetson-wheels.json"
VENV_DIR="$REPO_ROOT/.venv"
WHEEL_DIR=""
WHEEL_BASE_URL="${REACHY_WHEEL_BASE_URL:-}"
PYTHON_BIN="${REACHY_PYTHON:-python3}"
SKIP_SYSTEM_PACKAGES=false
CHECK_ONLY=false

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_jetson.sh [options]

Detect the Jetson platform, select only an exact wheel manifest match, create
the Python environment, install dependencies, and verify GPU STT/TTS runtimes.

Options:
  --wheel-dir DIR          Use already-downloaded release wheels from DIR
  --wheel-base-url URL     Download wheels from this immutable release URL
  --venv DIR               Virtual environment path (default: .venv)
  --manifest FILE          Compatibility manifest path
  --python EXECUTABLE      Python used to create the venv (default: python3)
  --skip-system-packages   Do not install missing Ubuntu runtime/build packages
  --check-only             Detect compatibility and optionally verify wheel files
  -h, --help               Show this help

Until the candidate wheels are published, use --wheel-dir. The default release
URL is enabled only after packaging/jetson-wheels.json marks it published.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel-dir)
      [[ $# -ge 2 ]] || die "--wheel-dir requires a value"
      WHEEL_DIR="$2"
      shift 2
      ;;
    --wheel-base-url)
      [[ $# -ge 2 ]] || die "--wheel-base-url requires a value"
      WHEEL_BASE_URL="${2%/}"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || die "--venv requires a value"
      VENV_DIR="$2"
      shift 2
      ;;
    --manifest)
      [[ $# -ge 2 ]] || die "--manifest requires a value"
      MANIFEST="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-system-packages)
      SKIP_SYSTEM_PACKAGES=true
      shift
      ;;
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
[[ -r "$MANIFEST" ]] || die "wheel manifest not found: $MANIFEST"

architecture="$(uname -m)"
[[ -r /etc/nv_tegra_release ]] || die "not a supported Jetson: /etc/nv_tegra_release is missing"
l4t_line="$(head -n 1 /etc/nv_tegra_release)"
if [[ "$l4t_line" =~ \#\ R([0-9]+).*REVISION:\ ([0-9.]+) ]]; then
  l4t_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
else
  die "could not parse L4T version from /etc/nv_tegra_release"
fi

python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

cuda_version=""
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  cuda_version="$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
fi
if [[ -z "$cuda_version" && -r /usr/local/cuda/version.json ]]; then
  cuda_version="$($PYTHON_BIN - /usr/local/cuda/version.json <<'PY'
import json
import re
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
match = re.search(r"[0-9]+\.[0-9]+", json.dumps(data))
print(match.group(0) if match else "")
PY
)"
fi
[[ -n "$cuda_version" ]] || die "could not detect CUDA; expected /usr/local/cuda"

compatible=""
if [[ -r /proc/device-tree/compatible ]]; then
  compatible="$(tr '\0' '\n' </proc/device-tree/compatible)"
fi
if grep -q 'tegra234' <<<"$compatible"; then
  gpu_family="orin"
  compute_capability="8.7"
elif grep -q 'tegra264' <<<"$compatible"; then
  gpu_family="thor"
  compute_capability="11.0"
else
  die "unsupported or unknown Jetson SoC; refusing to guess a CUDA SM target"
fi

manifest_row="$($PYTHON_BIN - "$MANIFEST" "$architecture" "$l4t_version" "$cuda_version" "$python_version" "$gpu_family" "$compute_capability" <<'PY'
import json
import sys

path, architecture, l4t, cuda, python, family, compute = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    manifest = json.load(stream)

matches = []
for platform_id, platform in manifest.get("platforms", {}).items():
    if (
        platform.get("architecture") == architecture
        and platform.get("l4t") == l4t
        and platform.get("cuda") == cuda
        and platform.get("python") == python
        and platform.get("gpu_family") == family
        and platform.get("compute_capability") == compute
    ):
        matches.append((platform_id, platform))

if len(matches) != 1:
    raise SystemExit(
        "no unique compatible wheel set: "
        f"arch={architecture} l4t={l4t} cuda={cuda} python={python} "
        f"family={family} sm={compute}; matches={len(matches)}"
    )

platform_id, platform = matches[0]
release = platform.get("release", {})
ct2 = platform["wheels"]["ctranslate2"]
ort = platform["wheels"]["onnxruntime_gpu"]
values = [
    platform_id,
    platform["wheel_tag"],
    str(release.get("published", False)).lower(),
    release.get("base_url", ""),
    ct2["filename"], ct2["sha256"], str(ct2["size"]),
    ct2["distribution"], ct2["version"],
    ort["filename"], ort["sha256"], str(ort["size"]),
    ort["distribution"], ort["version"],
]
print("\t".join(values))
PY
)" || die "this Jetson has no exact entry in $MANIFEST"

IFS=$'\t' read -r \
  PLATFORM_ID WHEEL_TAG RELEASE_PUBLISHED MANIFEST_BASE_URL \
  CT2_FILENAME CT2_SHA256 CT2_SIZE CT2_DISTRIBUTION CT2_VERSION \
  ORT_FILENAME ORT_SHA256 ORT_SIZE ORT_DISTRIBUTION ORT_VERSION \
  <<<"$manifest_row"

echo "Detected: architecture=$architecture L4T=$l4t_version CUDA=$cuda_version Python=$python_version GPU=$gpu_family sm_$compute_capability"
echo "Selected: $PLATFORM_ID"
echo "CTranslate2: $CT2_FILENAME"
echo "ONNX Runtime: $ORT_FILENAME"

verify_wheel() {
  local path="$1"
  local expected_sha="$2"
  local expected_size="$3"
  local expected_distribution="$4"
  local expected_version="$5"
  local expected_tag="$6"
  [[ -f "$path" ]] || die "wheel not found: $path"

  local actual_sha actual_size
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  actual_size="$(stat -Lc '%s' "$path")"
  [[ "$actual_sha" == "$expected_sha" ]] || die "SHA-256 mismatch: $path"
  [[ "$actual_size" == "$expected_size" ]] || die "size mismatch: $path"

  "$PYTHON_BIN" - "$path" "$expected_distribution" "$expected_version" "$expected_tag" <<'PY'
import email.parser
import re
import sys
import zipfile

wheel_path, expected_name, expected_version, expected_tag = sys.argv[1:]
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
with zipfile.ZipFile(wheel_path) as archive:
    metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    wheel_files = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
    if len(metadata_files) != 1 or len(wheel_files) != 1:
        raise SystemExit(f"invalid wheel metadata layout: {wheel_path}")
    metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_files[0]))
    wheel_metadata = archive.read(wheel_files[0]).decode("utf-8", errors="replace")
if normalize(metadata["Name"]) != normalize(expected_name):
    raise SystemExit(f"wheel distribution mismatch: {metadata['Name']} != {expected_name}")
if metadata["Version"] != expected_version:
    raise SystemExit(f"wheel version mismatch: {metadata['Version']} != {expected_version}")
if f"Tag: {expected_tag}" not in wheel_metadata:
    raise SystemExit(f"wheel does not have the required {expected_tag} tag")
PY
  echo "Verified: $(basename "$path")"
}

if [[ "$CHECK_ONLY" == true ]]; then
  if [[ -n "$WHEEL_DIR" ]]; then
    verify_wheel "$WHEEL_DIR/$CT2_FILENAME" "$CT2_SHA256" "$CT2_SIZE" "$CT2_DISTRIBUTION" "$CT2_VERSION" "$WHEEL_TAG"
    verify_wheel "$WHEEL_DIR/$ORT_FILENAME" "$ORT_SHA256" "$ORT_SIZE" "$ORT_DISTRIBUTION" "$ORT_VERSION" "$WHEEL_TAG"
  elif [[ "$RELEASE_PUBLISHED" != true ]]; then
    echo "Candidate wheel set is not published; use --wheel-dir for local verification."
  else
    echo "Published release: ${WHEEL_BASE_URL:-$MANIFEST_BASE_URL}"
  fi
  echo "Compatibility check passed."
  exit 0
fi

if [[ "$SKIP_SYSTEM_PACKAGES" != true ]]; then
  system_packages=(
    ca-certificates curl build-essential pkg-config "python${python_version}-venv"
    portaudio19-dev libasound2-dev pulseaudio-utils acl
    libcairo2-dev libgirepository1.0-dev libssl-dev
  )
  missing_packages=()
  for package in "${system_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
      missing_packages+=("$package")
    fi
  done
  if [[ ${#missing_packages[@]} -gt 0 ]]; then
    echo "Installing missing system packages: ${missing_packages[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
  else
    echo "System packages already installed."
  fi
else
  echo "Skipping system package installation."
fi

if [[ -n "$WHEEL_DIR" ]]; then
  CT2_WHEEL="$WHEEL_DIR/$CT2_FILENAME"
  ORT_WHEEL="$WHEEL_DIR/$ORT_FILENAME"
else
  if [[ -z "$WHEEL_BASE_URL" ]]; then
    [[ "$RELEASE_PUBLISHED" == true ]] || die "candidate wheels are not published; provide --wheel-dir or --wheel-base-url"
    WHEEL_BASE_URL="$MANIFEST_BASE_URL"
  fi
  cache_dir="$REPO_ROOT/.cache/jetson-wheels/$PLATFORM_ID"
  mkdir -p "$cache_dir"
  CT2_WHEEL="$cache_dir/$CT2_FILENAME"
  ORT_WHEEL="$cache_dir/$ORT_FILENAME"
  for item in "$CT2_FILENAME:$CT2_WHEEL" "$ORT_FILENAME:$ORT_WHEEL"; do
    filename="${item%%:*}"
    destination="${item#*:}"
    if [[ ! -f "$destination" ]]; then
      echo "Downloading $filename"
      curl --fail --location --retry 3 --output "$destination.part" "$WHEEL_BASE_URL/$filename"
      mv "$destination.part" "$destination"
    fi
  done
fi

verify_wheel "$CT2_WHEEL" "$CT2_SHA256" "$CT2_SIZE" "$CT2_DISTRIBUTION" "$CT2_VERSION" "$WHEEL_TAG"
verify_wheel "$ORT_WHEEL" "$ORT_SHA256" "$ORT_SIZE" "$ORT_DISTRIBUTION" "$ORT_VERSION" "$WHEEL_TAG"

if [[ -d "$VENV_DIR" ]]; then
  while IFS= read -r pid; do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    if [[ "$command_line" == "$VENV_DIR/bin/python "* ]]; then
      die "an assistant process is using $VENV_DIR; stop it or select a different --venv"
    fi
  done < <(pgrep -f 'run_(web_vision|vision|voice)_chat.py' || true)
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt" "reachy-mini==1.3.1"

VIRTUAL_ENV="$VENV_DIR" \
REACHY_EXPECTED_CTRANSLATE2_VERSION="$CT2_VERSION" \
REACHY_EXPECTED_ONNXRUNTIME_VERSION="$ORT_VERSION" \
PATH="$VENV_DIR/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
  "$SCRIPT_DIR/install_jp72_gpu_wheels.sh" "$CT2_WHEEL" "$ORT_WHEEL"

PYTHONPATH="$REPO_ROOT" "$VENV_DIR/bin/python" - <<'PY'
import importlib.util
import numpy as np

from app.pipeline import SILERO_CHUNK_SAMPLES, SileroVAD

vad = SileroVAD()
probability = vad(np.zeros(SILERO_CHUNK_SAMPLES, dtype=np.int16).tobytes())
if not 0.0 <= probability <= 1.0:
    raise SystemExit("Silero VAD smoke test returned an invalid probability")
print(f"Silero VAD smoke test: {probability:.6f}")
if importlib.util.find_spec("silero_vad") is not None:
    print("Note: optional silero-vad package is installed but is not used by this app.")
PY

echo
echo "Setup complete for $PLATFORM_ID."
echo "Activate: source $VENV_DIR/bin/activate"
echo "Model:    NP=1 ./run_llama_cpp.sh unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
echo "App:      python run_web_vision_chat.py"
