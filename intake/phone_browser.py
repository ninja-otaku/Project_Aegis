import asyncio
from datetime import datetime, timezone

import cv2
import numpy as np

from .base import BaseVideoIntake


class PhoneBrowserIntake(BaseVideoIntake):
    """Receives JPEG frames over WebSocket from a phone browser.

    Frame flow:
      Phone browser  --[JPEG binary]-->  /ws/intake  -->  process_frame()

    Preprocessing pipeline applied to each incoming frame:
      1. Decode JPEG bytes into a BGR ndarray.
      2. Convert to grayscale (reduces downstream processing cost ~3×).
      3. Apply CLAHE for local contrast enhancement (handles HDR scenes,
         dark UI elements, bloom from monitors).

    Only the single latest frame is retained — no queue, no disk writes.
    An asyncio.Lock guards the shared frame slot so the WebSocket handler
    and any async reader never race.

    Target: 10 FPS (phone sends a frame every 100 ms).
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._timestamp: datetime | None = None
        self._lock = asyncio.Lock()
        # CLAHE: clip limit 2.0 suppresses noise amplification; 8×8 tile
        # grid balances local vs global contrast at typical 720p resolution.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # BaseVideoIntake interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        # No hardware to initialise; the WebSocket route in main.py drives
        # ingest by calling process_frame() directly.
        async with self._lock:
            self._frame = None
            self._timestamp = None

    async def stop(self) -> None:
        async with self._lock:
            self._frame = None
            self._timestamp = None

    def get_latest_frame(self) -> np.ndarray | None:
        # Plain attribute read — safe in CPython because assignment is atomic
        # (pointer swap protected by the GIL).  We return a copy so the
        # caller can't corrupt the stored frame.
        frame = self._frame
        return frame.copy() if frame is not None else None

    def get_frame_timestamp(self) -> datetime | None:
        return self._timestamp

    # ------------------------------------------------------------------
    # WebSocket handler entry-point
    # ------------------------------------------------------------------

    async def process_frame(self, data: bytes) -> None:
        """Decode and preprocess a raw JPEG payload received from the phone.

        Called from the FastAPI WebSocket handler for every incoming message.
        Silently drops malformed payloads rather than raising so a single
        bad frame never kills the WebSocket connection.
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return  # malformed JPEG — drop silently

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)  # returns a new ndarray

        async with self._lock:
            self._frame = enhanced
            self._timestamp = datetime.now(tz=timezone.utc)
