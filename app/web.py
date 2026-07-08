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
import json
import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

STATIC_DIR = Path(__file__).parent.parent / "static"


class Broadcaster:
    """Thread-safe fan-out from the synchronous pipeline to async WebSocket clients.

    Each connected client gets its own asyncio.Queue.  The pipeline calls
    send() from any thread; messages are routed into the event loop via
    call_soon_threadsafe.

    Also holds shared push-to-talk (PTT) state: set = unmuted / listening
    by default, cleared = muted. Any client can toggle via WebSocket.
    """

    def __init__(self):
        self._clients: dict[asyncio.Queue, None] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ptt = threading.Event()
        self._ptt.set()
        self._speaker_getter: Optional[Callable[[], dict]] = None
        self._speaker_setter: Optional[Callable[[str], dict]] = None

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
    ):
        """Attach live speaker state and selection callbacks."""
        self._speaker_getter = getter
        self._speaker_setter = setter

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
            for q in self._clients:
                try:
                    loop.call_soon_threadsafe(self._enqueue_latest, q, msg)
                except Exception:
                    pass


def create_app(broadcaster: Broadcaster) -> FastAPI:
    app = FastAPI(title="Reachy Mini Vision Chat")

    @app.on_event("startup")
    async def _startup():
        broadcaster.set_loop(asyncio.get_event_loop())

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
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        broadcaster.register(q)

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
            try:
                while True:
                    data = await ws.receive_text()
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

    return app


def start_web_server(
    broadcaster: Broadcaster,
    host: str = "0.0.0.0",
    port: int = 8090,
) -> threading.Thread:
    """Start uvicorn in a daemon thread.  Returns immediately."""
    import uvicorn

    app = create_app(broadcaster)

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="web-server")
    t.start()
    return t
