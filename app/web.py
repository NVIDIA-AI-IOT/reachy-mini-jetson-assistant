# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web UI — lightweight FastAPI server with WebSocket broadcasting.

Provides a Broadcaster class for thread-safe message delivery from the
pipeline thread to all connected browser clients, and a FastAPI app
that serves the single-page frontend and a WebSocket endpoint.
"""

import asyncio
import base64
import hashlib
import json
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

STATIC_DIR = Path(__file__).parent.parent / "static"


def _inline_script_hashes() -> str:
    path = STATIC_DIR / "index.html"
    if not path.exists():
        return "'none'"
    html_text = path.read_text(encoding="utf-8")
    hashes = []
    for script in re.findall(r"<script[^>]*>(.*?)</script>", html_text, re.DOTALL):
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        hashes.append(f"'sha256-{digest}'")
    return " ".join(hashes) or "'none'"


class Broadcaster:
    """Thread-safe fan-out from the synchronous pipeline to async WebSocket clients.

    Each connected client gets its own asyncio.Queue.  The pipeline calls
    send() from any thread; messages are routed into the event loop via
    call_soon_threadsafe.

    Also holds shared push-to-talk (PTT) state: set = unmuted / listening
    when active, cleared = muted by default. Any client can toggle via WebSocket.
    """

    def __init__(self):
        self._clients: dict[asyncio.Queue, None] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ptt = threading.Event()
        self._speaker_getter: Optional[Callable[[], dict]] = None
        self._speaker_setter: Optional[Callable[[str], dict]] = None
        self._speaker_volume_setter: Optional[Callable[[int], dict]] = None

        self._ready = False
        self._components: dict[str, bool] = {}
        self._started_at = time.monotonic()
        self._last_activity = time.monotonic()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def register(self, q: asyncio.Queue):
        with self._lock:
            self._clients[q] = None

    def unregister(self, q: asyncio.Queue):
        with self._lock:
            self._clients.pop(q, None)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def ptt_active(self) -> bool:
        return self._ptt.is_set()

    def set_ptt(self, active: bool):
        if active:
            self._ptt.set()
        else:
            self._ptt.clear()
        self.send({"type": "ptt_state", "active": active})

    def configure_speakers(
        self,
        getter: Callable[[], dict],
        setter: Callable[[str], dict],
        volume_setter: Optional[Callable[[int], dict]] = None,
    ):
        """Attach live speaker state, selection, and volume callbacks."""
        self._speaker_getter = getter
        self._speaker_setter = setter
        self._speaker_volume_setter = volume_setter

    def get_speaker_state(self) -> dict:
        if not self._speaker_getter:
            return {"type": "speaker_state", "speakers": [], "selected": None}
        try:
            return {"type": "speaker_state", **self._speaker_getter()}
        except Exception as exc:
            return {
                "type": "speaker_state",
                "speakers": [],
                "selected": None,
                "error": str(exc),
            }

    def select_speaker(self, sink_id: str):
        if not self._speaker_setter:
            state = {
                "speakers": [],
                "selected": None,
                "error": "Speaker routing is not initialized",
            }
        else:
            try:
                state = self._speaker_setter(sink_id)
            except Exception as exc:
                state = {**self.get_speaker_state(), "error": str(exc)}
                state.pop("type", None)
        self.send({"type": "speaker_state", **state})

    def set_speaker_volume(self, volume: int):
        if not self._speaker_volume_setter:
            state = {
                **self.get_speaker_state(),
                "error": "Speaker volume control is not initialized",
            }
            state.pop("type", None)
        else:
            try:
                state = self._speaker_volume_setter(volume)
            except Exception as exc:
                state = {**self.get_speaker_state(), "error": str(exc)}
                state.pop("type", None)
        self.send({"type": "speaker_state", **state})

    def set_ready(self, ready: bool, components: Optional[dict[str, bool]] = None):
        with self._lock:
            self._ready = bool(ready)
            if components is not None:
                self._components = dict(components)
            self._last_activity = time.monotonic()

    def health_status(self) -> dict:
        with self._lock:
            return {
                "ready": self._ready,
                "components": dict(self._components),
                "clients": len(self._clients),
                "uptime_seconds": round(time.monotonic() - self._started_at, 1),
                "last_activity_seconds": round(
                    time.monotonic() - self._last_activity, 1
                ),
            }

    @staticmethod
    def _enqueue_latest(q: asyncio.Queue, msg: dict):
        """Keep a slow client's queue current without leaking payloads to logs."""
        try:
            q.put_nowait(msg)
            return
        except asyncio.QueueFull:
            pass

        # Camera frames are large and quickly become stale. Drop one queued
        # message and retry instead of letting QueueFull escape from the event
        # loop callback, where asyncio would log the complete base64 payload.
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def send(self, msg: dict):
        """Enqueue *msg* for every connected client (thread-safe)."""
        loop = self._loop
        if not loop:
            return
        with self._lock:
            self._last_activity = time.monotonic()
            for q in self._clients:
                try:
                    loop.call_soon_threadsafe(self._enqueue_latest, q, msg)
                except Exception:
                    pass


def _token_from_authorization(value: str) -> str:
    scheme, _, token = (value or "").partition(" ")
    return token if scheme.lower() == "bearer" else ""


def _authorized(expected: str, candidate: str) -> bool:
    return bool(expected and candidate) and secrets.compare_digest(expected, candidate)


def create_app(
    broadcaster: Broadcaster,
    api_token: str = "",
    require_auth: bool = False,
    allowed_hosts: Optional[list[str]] = None,
    max_clients: int = 4,
    audit_logger=None,
) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        broadcaster.set_loop(asyncio.get_running_loop())
        yield
        broadcaster.set_ready(False)

    app = FastAPI(
        title="Reachy Mini Vision Chat",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    hosts = allowed_hosts or ["localhost", "127.0.0.1"]
    script_hashes = _inline_script_hashes()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    def _audit(event: str, **fields):
        if audit_logger is not None:
            audit_logger.write(event, **fields)

    @app.middleware("http")
    async def _security(request: Request, call_next):
        public_health = request.url.path in {"/health/live", "/health/ready"}
        if require_auth and not public_health:
            candidate = (
                _token_from_authorization(request.headers.get("authorization", ""))
                or request.headers.get("x-api-key", "")
                or request.query_params.get("token", "")
            )
            if not _authorized(api_token, candidate):
                _audit("web_auth_rejected", path=request.url.path)
                response = JSONResponse({"detail": "unauthorized"}, status_code=401)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            f"style-src 'self' 'unsafe-inline'; script-src 'self' {script_hashes}; "
            "connect-src 'self' ws: wss:; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.get("/health/live")
    async def _live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def _ready():
        status = broadcaster.health_status()
        return JSONResponse(
            {"status": "ready" if status["ready"] else "not_ready", **status},
            status_code=200 if status["ready"] else 503,
        )

    @app.get("/")
    async def _index():
        path = STATIC_DIR / "index.html"
        if path.exists():
            return FileResponse(
                path, media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)

    @app.websocket("/ws")
    async def _ws(ws: WebSocket):
        candidate = (
            _token_from_authorization(ws.headers.get("authorization", ""))
            or ws.query_params.get("token", "")
        )
        origin = ws.headers.get("origin")
        origin_host = urlparse(origin).hostname if origin else None
        if require_auth and not _authorized(api_token, candidate):
            _audit("websocket_auth_rejected")
            await ws.close(code=1008)
            return
        if origin_host and "*" not in hosts and origin_host not in hosts:
            _audit("websocket_origin_rejected", origin_host=origin_host)
            await ws.close(code=1008)
            return
        if broadcaster.client_count >= max(1, int(max_clients)):
            _audit("websocket_capacity_rejected")
            await ws.close(code=1013)
            return

        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        broadcaster.register(q)
        _audit("websocket_connected", clients=broadcaster.client_count)

        q.put_nowait({"type": "ptt_state", "active": broadcaster.ptt_active})
        q.put_nowait(broadcaster.get_speaker_state())

        async def _sender():
            try:
                while True:
                    msg = await q.get()
                    await ws.send_text(json.dumps(msg))
            except (WebSocketDisconnect, Exception):
                pass

        async def _receiver():
            received_at: list[float] = []
            try:
                while True:
                    data = await ws.receive_text()
                    if len(data) > 2048:
                        await ws.close(code=1009)
                        return
                    now = time.monotonic()
                    received_at[:] = [item for item in received_at if now - item < 1.0]
                    received_at.append(now)
                    if len(received_at) > 10:
                        _audit("websocket_rate_limited")
                        await ws.close(code=1008)
                        return
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ptt":
                            broadcaster.set_ptt(bool(msg.get("active", False)))
                        elif msg.get("type") == "set_speaker":
                            sink_id = msg.get("sink")
                            if isinstance(sink_id, str) and sink_id:
                                await asyncio.to_thread(
                                    broadcaster.select_speaker, sink_id,
                                )
                        elif msg.get("type") == "set_speaker_volume":
                            volume = msg.get("volume")
                            if (
                                isinstance(volume, (int, float))
                                and not isinstance(volume, bool)
                            ):
                                await asyncio.to_thread(
                                    broadcaster.set_speaker_volume, int(volume),
                                )
                    except (ValueError, KeyError):
                        pass
            except (WebSocketDisconnect, Exception):
                pass

        send_task = asyncio.ensure_future(_sender())
        recv_task = asyncio.ensure_future(_receiver())
        try:
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception:
            send_task.cancel()
            recv_task.cancel()
        finally:
            broadcaster.unregister(q)
            _audit("websocket_disconnected", clients=broadcaster.client_count)

    return app


def start_web_server(
    broadcaster: Broadcaster,
    host: str = "0.0.0.0",
    port: int = 8090,
    api_token: str = "",
    require_auth: bool = False,
    allowed_hosts: Optional[list[str]] = None,
    max_clients: int = 4,
    audit_logger=None,
) -> threading.Thread:
    """Start uvicorn in a daemon thread.  Returns immediately."""
    import uvicorn

    app = create_app(
        broadcaster,
        api_token=api_token,
        require_auth=require_auth,
        allowed_hosts=allowed_hosts,
        max_clients=max_clients,
        audit_logger=audit_logger,
    )

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="web-server")
    t.start()
    return t
