import logging

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

_feed_clients: list[WebSocket] = []
_event_clients: list[WebSocket] = []
_latest_frame: dict | None = None  # {"frame": np.ndarray, "servo_pan": float, "servo_tilt": float}


def store_latest_frame(frame, servo_pan: float, servo_tilt: float):
    """Called from vision loop to cache the latest raw frame for depth queries."""
    global _latest_frame
    _latest_frame = {"frame": frame, "servo_pan": servo_pan, "servo_tilt": servo_tilt}


def get_latest_frame() -> dict | None:
    return _latest_frame


def has_feed_clients() -> bool:
    return len(_feed_clients) > 0


async def broadcast_to_clients(data: dict):
    """Broadcast frame data to feed WebSocket clients only."""
    if not _feed_clients:
        return
    disconnected = []
    for ws in _feed_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _feed_clients:
            _feed_clients.remove(ws)


async def broadcast_event(data: dict):
    """Broadcast event data to event WebSocket clients only."""
    disconnected = []
    for ws in _event_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _event_clients:
            _event_clients.remove(ws)


class MJPEGStreamReader:
    """Reads MJPEG frames from the ESP32-CAM HTTP stream."""

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self._client: httpx.AsyncClient | None = None
        self._response = None

    async def connect(self):
        self._client = httpx.AsyncClient(timeout=None)
        self._response = await self._client.stream("GET", self.stream_url).__aenter__()

    async def read_frame(self) -> np.ndarray | None:
        if not self._response:
            return None

        buffer = b""
        async for chunk in self._response.aiter_bytes(chunk_size=4096):
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg_bytes = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                return frame
        return None

    async def close(self):
        if self._response:
            await self._response.aclose()
        if self._client:
            await self._client.aclose()


@router.websocket("/ws/feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    _feed_clients.append(websocket)
    logger.info("WebSocket feed client connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _feed_clients:
            _feed_clients.remove(websocket)
        logger.info("WebSocket feed client disconnected")


@router.websocket("/ws/events")
async def event_feed(websocket: WebSocket):
    await websocket.accept()
    _event_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _event_clients:
            _event_clients.remove(websocket)
