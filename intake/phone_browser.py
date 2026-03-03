import asyncio
from datetime import datetime, timezone

import cv2
import numpy as np

from config import settings
from .base import BaseVideoIntake


class PhoneBrowserIntake(BaseVideoIntake):
    """Receives JPEG frames over WebSocket from a phone browser.

    Frame flow:
      Phone browser  --[JPEG binary]-->  /ws/intake  -->  process_frame()

    Preprocessing pipeline applied to each incoming frame:
      1. Decode JPEG bytes into a BGR ndarray.
      2. Resize to PHONE_FRAME_WIDTH (aspect-ratio-preserving, INTER_AREA).
         Happens before CLAHE so the enhancement operates at the target
         resolution rather than the raw camera resolution.
      3. Convert to grayscale.
      4. Apply CLAHE for local contrast enhancement.
      5. Re-encode to JPEG at PHONE_COMPRESS_QUALITY and store as bytes.
         Storing bytes instead of the ndarray reduces per-frame memory by
         ~10–20× at typical resolutions.

    get_latest_frame() decodes the stored JPEG back to ndarray on each read —
    cheap because frames are only read once per analysis interval.

    The asyncio.Lock wraps only the buffer write so the read path is lockless
    (bytes reference assignment is atomic in CPython under the GIL).

    Target: 10 FPS (phone sends a frame every 100 ms).
    """

    def __init__(self) -> None:
        self._frame_bytes: bytes | None = None
        self._timestamp: datetime | None = None
        self._lock = asyncio.Lock()
        # CLAHE: clip limit 2.0 suppresses noise amplification; 8×8 tile
        # grid balances local vs global contrast at typical 720p resolution.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # BaseVideoIntake interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        async with self._lock:
            self._frame_bytes = None
            self._timestamp = None

    async def stop(self) -> None:
        async with self._lock:
            self._frame_bytes = None
            self._timestamp = None

    def get_latest_frame(self) -> np.ndarray | None:
        # Lockless read — bytes reference is assigned atomically in CPython.
        frame_bytes = self._frame_bytes
        if frame_bytes is None:
            return None
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        # Decode as grayscale; the stored bytes are already a grayscale JPEG.
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    def get_frame_timestamp(self) -> datetime | None:
        return self._timestamp

    # ------------------------------------------------------------------
    # WebSocket handler entry-point
    # ------------------------------------------------------------------

    async def process_frame(self, data: bytes) -> None:
        """Decode, resize, preprocess, compress and store one phone frame.

        Called from the FastAPI WebSocket handler for every incoming message.
        Silently drops malformed payloads so a single bad frame never kills
        the WebSocket connection.
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return  # malformed JPEG — drop silently

        # ── 1. Resize to target width (before CLAHE) ──────────────────
        target_w = settings.PHONE_FRAME_WIDTH
        h, w = bgr.shape[:2]
        if w != target_w:
            scale = target_w / w
            new_h = int(h * scale)
            # INTER_AREA is best for downscaling — avoids moiré artifacts
            bgr = cv2.resize(bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)

        # ── 2. Grayscale + CLAHE ──────────────────────────────────────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)

        # ── 3. Re-encode at compressed quality for storage ────────────
        ok, buf = cv2.imencode(
            ".jpg", enhanced,
            [cv2.IMWRITE_JPEG_QUALITY, settings.PHONE_COMPRESS_QUALITY]
        )
        if not ok:
            return  # encoding failure — drop silently

        # Lock only wraps the buffer write (not decode or encode above)
        async with self._lock:
            self._frame_bytes = buf.tobytes()
            self._timestamp = datetime.now(tz=timezone.utc)
