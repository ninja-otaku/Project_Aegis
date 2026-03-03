import threading
from datetime import datetime, timezone

import cv2
import numpy as np

from .base import BaseVideoIntake

# OpenCV's grab-decode loop is ~33 ms per frame at 30 FPS.
# The background thread loops tight; cv2.VideoCapture.read() blocks
# naturally, so no explicit sleep is needed — the camera's hardware
# frame rate acts as the rate limiter.
_TARGET_FPS = 30


class CaptureCardIntake(BaseVideoIntake):
    """Reads frames from a capture card (or any V4L2 / DirectShow device).

    A daemon thread runs the cv2.VideoCapture read loop continuously.
    No preprocessing is applied — the raw BGR frame is stored as-is so
    downstream providers receive full-colour, full-resolution data.

    Only the single latest frame is retained. A threading.Lock guards
    the shared frame slot between the background capture thread and any
    calling thread.

    Target: 30 FPS (limited by the capture card hardware).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._frame: np.ndarray | None = None
        self._timestamp: datetime | None = None
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # BaseVideoIntake interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the capture device and start the background read thread."""
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"CaptureCardIntake: failed to open device {self._device_index!r}. "
                "Check CAPTURE_DEVICE_INDEX in .env and that the device is connected."
            )
        # Request the target frame rate; the driver may ignore this.
        cap.set(cv2.CAP_PROP_FPS, _TARGET_FPS)

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="capture-card-reader",
            daemon=True,  # exits automatically when the main process dies
        )
        self._thread.start()

    async def stop(self) -> None:
        """Signal the read loop to stop and release the capture device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
            self._timestamp = None

    def get_latest_frame(self) -> np.ndarray | None:
        """Return a copy of the latest frame, or None if none captured yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_timestamp(self) -> datetime | None:
        with self._lock:
            return self._timestamp

    # ------------------------------------------------------------------
    # Background capture loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Continuously read frames from the capture card.

        cv2.VideoCapture.read() blocks until the next hardware frame
        arrives, providing natural rate-limiting without a sleep().
        On read failure (device disconnect, etc.) the loop exits cleanly.
        """
        assert self._cap is not None  # guaranteed by start()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # Device disconnected or stream ended — stop gracefully.
                self._running = False
                break
            with self._lock:
                self._frame = frame          # store raw BGR ndarray
                self._timestamp = datetime.now(tz=timezone.utc)
