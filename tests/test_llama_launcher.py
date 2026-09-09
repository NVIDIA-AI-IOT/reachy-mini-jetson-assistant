# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _launcher_fixture(tmp_path):
    project = tmp_path / "project"
    models = project / "models"
    fake_bin = tmp_path / "bin"
    models.mkdir(parents=True)
    fake_bin.mkdir()
    launcher = project / "run_llama_cpp.sh"
    shutil.copy2(ROOT / "run_llama_cpp.sh", launcher)

    docker = fake_bin / "docker"
    docker.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "ps" ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$DOCKER_ARGS_LOG"\n'
    )
    docker.chmod(0o755)
    wget = fake_bin / "wget"
    wget.write_text('#!/bin/sh\nexit 1\n')
    wget.chmod(0o755)
    return launcher, models, fake_bin


def _run(launcher, fake_bin, model, **overrides):
    env = os.environ.copy()
    env.update(overrides)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["DOCKER_ARGS_LOG"] = str(launcher.parent / "docker-args.log")
    return subprocess.run(
        [str(launcher), str(model)],
        cwd=launcher.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_huggingface_spec_never_reuses_an_unrelated_quantized_model(tmp_path):
    launcher, models, fake_bin = _launcher_fixture(tmp_path)
    (models / "cosmos-reason2-Q4_K_M.gguf").touch()

    result = _run(
        launcher,
        fake_bin,
        "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
    )

    assert result.returncode != 0
    assert "Model not found" in result.stdout
    assert not (launcher.parent / "docker-args.log").exists()


def test_generic_projector_requires_explicit_selection(tmp_path):
    launcher, models, fake_bin = _launcher_fixture(tmp_path)
    model = models / "gemma-4-E2B-it-Q4_K_M.gguf"
    projector = models / "mmproj-F16.gguf"
    model.touch()
    projector.touch()

    result = _run(launcher, fake_bin, model)
    assert result.returncode == 0, result.stderr


def test_vlm_defaults_disable_reasoning_and_prompt_cache(tmp_path):
    launcher, models, fake_bin = _launcher_fixture(tmp_path)
    model = models / "gemma-4-E2B-it-Q4_K_M.gguf"
    projector = models / "mmproj-F16.gguf"
    model.touch()
    projector.touch()

    result = _run(
        launcher,
        fake_bin,
        model,
        MMPROJ=str(projector),
    )

    assert result.returncode == 0, result.stderr
    docker_args = (launcher.parent / "docker-args.log").read_text().splitlines()
    reasoning_index = docker_args.index("--reasoning")
    cache_ram_index = docker_args.index("--cache-ram")
    assert docker_args[reasoning_index + 1] == "off"
    assert docker_args[cache_ram_index + 1] == "0"
