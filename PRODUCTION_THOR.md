# Reachy Mini on Jetson Thor: Optional Hardened Operations

The default `thor` profile intentionally exposes an unauthenticated UI on the
trusted LAN, as documented in `SETUP_THOR.md`. The systemd example uses that
same mode. This guide describes an optional loopback-only, token-authenticated
deployment for environments that require stronger network isolation.

Before enabling hardened mode, change the `web` section in
`config/platforms/thor.yaml` to:

```yaml
web:
  host: 127.0.0.1
  port: 8090
  require_auth: true
  allowed_hosts:
    - 127.0.0.1
    - localhost
```

These settings are required by the production configuration validator. Keep
the normal LAN settings if direct access at `http://<thor-ip>:8090` is desired.

## Provision once

1. Install the JetPack 7.1 build prerequisites documented in `SETUP_THOR.md`.
2. Run `scripts/install_thor_python.sh` to create the pinned Python runtime.
3. Verify drift with `venv-thor/bin/python scripts/verify_runtime_manifest.py`.
4. Create the production secret file only after applying the loopback settings:

```bash
install -d -m 700 ~/.config/reachy-assistant
token="$(openssl rand -hex 32)"
printf 'REACHY_PRODUCTION_MODE=true\nREACHY_PLATFORM=thor\nREACHY_WEB_API_TOKEN=%s\n' "$token" \
  > ~/.config/reachy-assistant/env
chmod 600 ~/.config/reachy-assistant/env
```

Never add this file or token to Git, shell history, screenshots, or logs.

## Validate before installation

Start the pinned vLLM launcher temporarily, then run:

```bash
./run_vllm_thor.sh
source ~/.config/reachy-assistant/env
venv-thor/bin/python scripts/preflight_thor.py
venv-thor/bin/python -m pytest -q tests
```

`preflight_thor.py` verifies production configuration, secret permissions,
PyTorch CUDA, CTranslate2 CUDA, ONNX CUDA, available disk, and the exact model
served by vLLM.

## Install supervised services

```bash
./deploy/install_systemd.sh
sudo systemctl enable --now reachy-assistant.service
```

The assistant unit requires the vLLM unit, waits for its model endpoint,
restarts after failures, propagates SIGINT for robot cleanup, and applies
systemd hardening. The vLLM unit reclaims filesystem caches before reserving
unified memory. The install script deliberately does not enable or start
anything automatically.

## Secure UI access

Create an SSH tunnel from the operator workstation:

```bash
ssh -L 8090:127.0.0.1:8090 jetson@10.111.65.10
```

Then open `http://127.0.0.1:8090/?token=<REACHY_WEB_API_TOKEN>`. The token is
removed from the browser address bar and retained only in session storage.
For multi-user deployment, terminate TLS in a reviewed reverse proxy and pass
the token through an authorization header instead of a query parameter.

## Health and diagnostics

```bash
curl --fail http://127.0.0.1:8090/health/live
curl --fail http://127.0.0.1:8090/health/ready
docker inspect --format '{{json .State.Health}}' reachy-vllm
journalctl -u reachy-assistant.service -u reachy-vllm.service
```

Readiness reports component booleans but no prompts, images, transcripts, or
secrets. Audit records are JSONL in `data/logs/audit.jsonl`, mode 600, rotated
at ten MiB with five backups. Audit logging rejects sensitive field names.

## Upgrade and rollback

1. Stop the units.
2. Back up the current repository and environment manifest.
3. Review changes and update pinned digests/revisions only after validation.
4. Run runtime verification, unit tests, camera/robot checks, and a supervised
   spoken end-to-end test.
5. Start the units and monitor readiness, audit events, temperature, memory,
   and restart counts.
6. Roll back to the previous reviewed tree and manifest on any failure.

Never use a mutable container tag, unpinned model revision, placeholder token,
or `--trust-remote-code` with an unreviewed model revision in production.

## Remaining assurance work

This repository provides production engineering controls, not a safety
certification. Release approval still requires hardware-specific endurance and
thermal testing, microphone/speaker end-to-end testing, fault injection,
security review, guardrail red-teaming, privacy review, and an operator-defined
emergency-stop procedure. Kokoro uses CUDA for neural compute but ONNX Runtime
assigns a few shape-control operations to CPU. YuNet and Silero remain auxiliary
CPU workloads.
