---
name: reachy-jetson-setup
description: Install and validate the Reachy Mini Jetson Assistant and its GPU-native dependencies. Use for first-time setup, JetPack compatibility checks, CTranslate2 or ONNX Runtime wheel installation, clean virtual environments, and upgrades to a new JetPack, CUDA, Python, Orin, or Thor target.
---

# Reachy Jetson Setup

Set up only an exact platform match from `packaging/jetson-wheels.json`. Never substitute a generic ARM wheel for a Jetson GPU wheel.

## Workflow

1. Resolve the repository root with `git rev-parse --show-toplevel` and inspect `git status --short`. Preserve unrelated or uncommitted work.
2. Run `./scripts/setup_jetson.sh --check-only`. Stop if architecture, L4T, CUDA, Python, GPU family, or compute capability has no unique manifest entry.
3. For a published wheel set, run `./scripts/setup_jetson.sh`. For an unpublished candidate supplied by the developer, run:

   ```bash
   ./scripts/setup_jetson.sh --wheel-dir /absolute/path/to/wheelhouse
   ```

4. Use `--venv /tmp/name` for isolated validation. Use the default `.venv` only when no assistant process is using it.
5. Require the final output to report a CUDA device for CTranslate2, `CUDAExecutionProvider` for ONNX Runtime, and a valid Silero VAD probability.
6. Run the repository tests after changing requirements, manifests, installers, or runtime code.
7. Set `REACHY_JETSON_WHEEL_DIR` to a candidate wheelhouse and run `pytest tests/test_jetson_wheel_manifest.py` to validate both files.
8. Before publication, inspect the tuple’s `packaging/releases/<tag>/README.md` and run `scripts/audit_jetson_wheels.py`. Require the organization-approved NVIDIA scanner report plus OSRB sign-off for a final release. If the release owner explicitly authorizes an earlier prerelease, retain every pending gate in the evidence and never describe it as final or submission-ready.

## Release guardrails

- Treat the manifest filename, version, size, tag, and SHA-256 as one immutable compatibility record.
- Keep a candidate `published: false` until its assets are actually available. A published candidate with pending organizational gates must remain a GitHub prerelease and preserve those gates in its evidence.
- Before publication, verify the no-argument installer refuses the candidate. After publication, mark the release true only when its immutable URL works in a clean environment, then rerun the exact-wheel and GPU checks.
- Add a new manifest entry for a new JetPack, CUDA, Python ABI, or SM target; never widen an old match speculatively.
- Preserve the JetPack 6 instructions while JetPack 7.2 is the default.
- Do not commit, publish a release, or push unless the user explicitly asks.
