# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA "
    "CORPORATION & AFFILIATES. All rights reserved."
)
LICENSE = "SPDX-License-Identifier: Apache-2.0"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".css",
    ".h",
    ".hpp",
    ".html",
    ".js",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def repository_files():
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [ROOT / name for name in result.stdout.splitlines()]


def test_nvidia_source_files_have_spdx_headers():
    missing = []
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if "packaging" in relative.parts:
            continue
        first_lines = "\n".join(
            path.read_text(encoding="utf-8").splitlines()[:20]
        )
        if COPYRIGHT not in first_lines or LICENSE not in first_lines:
            missing.append(str(relative))
    assert missing == [], f"source files missing NVIDIA SPDX headers: {missing}"
