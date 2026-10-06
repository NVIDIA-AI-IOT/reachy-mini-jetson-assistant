# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured, privacy-preserving audit events for production operation."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class AuditLogger:
    """Write bounded JSONL audit records without prompts, images, or responses."""

    def __init__(
        self,
        path: str,
        max_bytes: int = 10_485_760,
        backup_count: int = 5,
    ):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"reachy.audit.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        handler = RotatingFileHandler(
            self.path,
            maxBytes=max(1_048_576, int(max_bytes)),
            backupCount=max(1, int(backup_count)),
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        sensitive_names = {
            "prompt", "response", "text", "image", "audio", "token",
            "secret", "password", "api_key", "authorization",
        }
        for key, value in fields.items():
            normalized_key = key.lower()
            if any(name in normalized_key for name in sensitive_names):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
            elif isinstance(value, (list, tuple)):
                record[key] = [str(item)[:128] for item in value[:32]]
            else:
                record[key] = str(value)[:256]
        try:
            line = json.dumps(record, separators=(",", ":"), sort_keys=True)
            with self._lock:
                self._logger.info(line)
        except Exception:
            # Audit failure must not expose model data or crash robot cleanup.
            pass
