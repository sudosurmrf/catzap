import asyncio
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
            await asyncio.wait_for(ws.send_json(data), timeout=0.5)
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
    """Reads MJPEG frames from the ESP32-CAM HTTP stream with auto-reconnect."""

    def __init__(self, stream_url: str, max_retries: int = 0):
        self.stream_url = stream_url
        self._client: httpx.AsyncClient | None = None
        self._response = None
        self._stream_ctx = None
        self._buffer = b""
        self._byte_iter = None
        self._connected = False

    async def connect(self):
        await self._open_stream()

    async def _open_stream(self):
        """Open (or reopen) the MJPEG stream connection."""
        # Clean up any existing connection first
        await self._close_stream()
        self._buffer = b""
        self._client = httpx.AsyncClient(timeout=None)
        self._stream_ctx = self._client.stream("GET", self.stream_url)
        self._response = await self._stream_ctx.__aenter__()
        self._byte_iter = self._response.aiter_bytes(chunk_size=4096).__aiter__()
        self._connected = True
        logger.info("ESP32-CAM stream connected")

    async def _close_stream(self):
        """Close current stream connection silently."""
        self._connected = False
        self._byte_iter = None
        try:
            if self._stream_ctx:
                await self._stream_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        self._stream_ctx = None
        self._response = None
        try:
            if self._client:
                await self._client.aclose()
        except Exception:
            pass
        self._client = None

    async def _reconnect(self):
        """Attempt to reconnect to the stream."""
        logger.info("Reconnecting to ESP32-CAM stream...")
        try:
            await self._open_stream()
            return True
        except Exception as e:
            logger.warning(f"Reconnect failed: {e}")
            return False

    async def read_frame(self) -> np.ndarray | None:
        if not self._connected:
            # Try to reconnect
            await asyncio.sleep(1)
            if not await self._reconnect():
                return None

        max_buffer = 2 * 1024 * 1024  # 2MB safety limit
        while True:
            try:
                chunk = await asyncio.wait_for(self._byte_iter.__anext__(), timeout=5.0)
            except (StopAsyncIteration, asyncio.TimeoutError, httpx.StreamClosed, httpx.StreamConsumed, Exception) as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                log_msg = "timed out" if isinstance(e, asyncio.TimeoutError) else str(type(e).__name__)
                logger.warning(f"ESP32-CAM stream {log_msg}, will reconnect")
                self._connected = False
                return None

            self._buffer += chunk
            # Prevent unbounded buffer growth from corrupted streams
            if len(self._buffer) > max_buffer:
                start = self._buffer.find(b"\xff\xd8", -max_buffer // 2)
                self._buffer = self._buffer[start:] if start != -1 else b""
            start = self._buffer.find(b"\xff\xd8")
            end = self._buffer.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg_bytes = self._buffer[start : end + 2]
                self._buffer = self._buffer[end + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                return frame

    async def close(self):
        await self._close_stream()


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
