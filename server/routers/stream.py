import asyncio
import logging
import socket
import threading
from urllib.parse import urlparse

import cv2
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
    """Reads MJPEG frames from the ESP32-CAM HTTP stream using raw sockets."""

    def __init__(self, stream_url: str, max_retries: int = 0):
        self.stream_url = stream_url
        parsed = urlparse(stream_url)
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._path = parsed.path or "/"
        self._sock: socket.socket | None = None
        self._connected = False
        self._stop = threading.Event()
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self):
        """Open the MJPEG stream using a raw socket (bypasses urllib/httpx)."""
        # Clean up any previous connection
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False

        self._loop = asyncio.get_event_loop()
        logger.info(f"Connecting to ESP32-CAM at {self._host}:{self._port}{self._path}")

        self._sock = socket.create_connection((self._host, self._port), timeout=10)
        self._sock.settimeout(10)

        # Send minimal HTTP GET
        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            f"Connection: keep-alive\r\n"
            f"\r\n"
        )
        self._sock.sendall(request.encode())

        # Read response headers
        header_buf = b""
        while b"\r\n\r\n" not in header_buf:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise ConnectionError("Connection closed while reading headers")
            header_buf += chunk

        header_text = header_buf.split(b"\r\n\r\n")[0].decode("utf-8", errors="replace")
        logger.info(f"ESP32-CAM response headers:\n{header_text}")

        # Everything after headers is the start of the body
        body_start = header_buf.split(b"\r\n\r\n", 1)[1]

        self._sock.settimeout(5)  # read timeout for stream chunks
        self._connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, args=(body_start,), daemon=True)
        self._thread.start()
        logger.info("ESP32-CAM stream connected and reading")

    def _read_loop(self, initial_data: bytes):
        """Background thread: reads from socket and extracts JPEG frames."""
        buf = initial_data
        max_buffer = 2 * 1024 * 1024
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
            except socket.timeout:
                continue  # no data yet, try again
            except Exception as e:
                logger.warning(f"ESP32-CAM stream read error: {e}")
                break

            buf += chunk
            if len(buf) > max_buffer:
                start = buf.find(b"\xff\xd8", -max_buffer // 2)
                buf = buf[start:] if start != -1 else b""

            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2 if start != -1 else 0)
                if start == -1 or end == -1 or end <= start:
                    break
                jpg_bytes = buf[start : end + 2]
                buf = buf[end + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self._loop.call_soon_threadsafe(self._frame_queue.put_nowait, frame)

        self._connected = False
        logger.info("ESP32-CAM stream read loop ended")

    async def read_frame(self) -> np.ndarray | None:
        if not self._connected:
            await asyncio.sleep(1)
            try:
                await self.connect()
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")
                return None

        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("ESP32-CAM stream timed out waiting for frame, will reconnect")
            self._connected = False
            return None

    async def close(self):
        self._stop.set()
        self._connected = False
        if self._thread:
            self._thread.join(timeout=3)
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None


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
