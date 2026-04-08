import logging

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

_ws_clients: list[WebSocket] = []


async def broadcast_to_clients(data: dict):
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


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
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected")


@router.websocket("/ws/events")
async def event_feed(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
