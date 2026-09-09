# OSRB source-release checklist

Status: **PUBLIC PRERELEASE; FINAL CLOSURE PENDING**. The repository-side controls are in place and the release owner authorized prerelease publication, but the exact internal NVIDIA header template, complete resolved application dependency license bundle, approved scanner report, immutable GitHub links, and final OSRB closure evidence are still pending. Do not represent the prerelease as final/general availability.

This checklist covers the NVIDIA-authored application source release. The separately built JetPack wheels have their own evidence bundle under `packaging/releases/native-jp72-cu132-py312-sm87-r1/`.

## Requirement status

| Requirement | Status | Repository evidence |
|---|---|---|
| Release NVIDIA-authored application code under Apache 2.0 | Pass | `LICENSE` contains the full Apache License 2.0 text; `README.md` identifies Apache 2.0 as the project license. |
| Carry NVIDIA copyright and Apache-2.0 identifiers in source files | Pass against the repository's current template; internal template confirmation pending | Source and configuration files use the exact two-line SPDX header documented in `CONTRIBUTING.md`. `tests/test_source_license_headers.py` prevents missing headers. |
| Preserve project attribution | Pass | `NOTICE` carries the project and NVIDIA attribution and points to the third-party notice record. |
| Document third-party software | Partial | `THIRD-PARTY-NOTICES.md` inventories direct, separately installed, important transitive, GPL, service, and model dependencies. Full license texts are preserved for the candidate native wheels. A complete resolved application dependency license bundle still requires an exact dependency lock and approved scan. |
| Use NVIDIA's ongoing IP review process | Pass as repository policy | `CONTRIBUTING.md` requires NVIDIA maintainers to complete the internal IP review before merging. |
| Require DCO for external contributions | Pass | `CONTRIBUTING.md` includes the verbatim DCO 1.1 and requires `Signed-off-by` on every commit. |
| Provide immutable links for OSRB closure | Blocked until commit and push | Use commit-SHA GitHub blob links after the approved change is committed and pushed; do not submit links to uncommitted local files or a mutable branch head. |

## Header standard used by this repository

```text
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

The SPDX lines are placed in comments appropriate to each source language and near the top of the file. The internal Confluence page supplied by OSRB is not accessible from the build environment, so an NVIDIA reviewer must compare this text and its year range with the internal canonical header before closure. If the internal text differs, update `CONTRIBUTING.md`, the source files, and `tests/test_source_license_headers.py` together.

## Links to provide after the approved commit is pushed

Use immutable URLs of the form `https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant/blob/<commit-sha>/<path>` for:

- `LICENSE`
- `NOTICE`
- `THIRD-PARTY-NOTICES.md`
- `CONTRIBUTING.md`
- `tests/test_source_license_headers.py`
- representative application and release-tool source files carrying the header
- `packaging/releases/native-jp72-cu132-py312-sm87-r1/README.md`

## Remaining closure work

1. Have an NVIDIA reviewer confirm the copyright header exactly against the internal Apache 2.0 guidance.
2. Resolve and lock the complete application dependency set, export every applicable third-party license/notice, and reconcile it with `THIRD-PARTY-NOTICES.md`.
3. Run NVIDIA's approved license/security scanner over the source, dependency bundle, both wheels, unpacked wheels, and SPDX SBOM; archive the unmodified report.
4. Obtain OSRB approval, then commit with DCO sign-off, push the review branch, and provide immutable commit-SHA links. Do not merge as part of the evidence-submission step.
