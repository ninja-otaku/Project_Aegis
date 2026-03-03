"""Smoke tests for the intake subsystem.

No real camera or network is required — tests use a synthesised JPEG and mocks.
Run with:  pytest tests/
"""

import asyncio

import cv2
import numpy as np
import pytest

from intake.base import BaseVideoIntake
from intake.capture_card import CaptureCardIntake
from intake.phone_browser import PhoneBrowserIntake


# ---------------------------------------------------------------------------
# Interface tests — both classes must implement the full ABC
# ---------------------------------------------------------------------------

def test_phone_browser_intake_implements_base():
    """PhoneBrowserIntake is a concrete subclass of BaseVideoIntake."""
    assert issubclass(PhoneBrowserIntake, BaseVideoIntake)


def test_capture_card_intake_implements_base():
    """CaptureCardIntake is a concrete subclass of BaseVideoIntake."""
    assert issubclass(CaptureCardIntake, BaseVideoIntake)


def test_phone_browser_intake_has_required_methods():
    """All abstract methods from BaseVideoIntake are present."""
    for method in ("start", "stop", "get_latest_frame", "get_frame_timestamp"):
        assert callable(getattr(PhoneBrowserIntake, method, None)), (
            f"PhoneBrowserIntake is missing {method}()"
        )


def test_capture_card_intake_has_required_methods():
    """All abstract methods from BaseVideoIntake are present."""
    for method in ("start", "stop", "get_latest_frame", "get_frame_timestamp"):
        assert callable(getattr(CaptureCardIntake, method, None)), (
            f"CaptureCardIntake is missing {method}()"
        )


# ---------------------------------------------------------------------------
# PhoneBrowserIntake — state before any frame is received
# ---------------------------------------------------------------------------

def test_phone_browser_returns_none_initially():
    """get_latest_frame() must return None before any frame is processed."""
    intake = PhoneBrowserIntake()
    assert intake.get_latest_frame() is None


def test_phone_browser_timestamp_is_none_initially():
    """get_frame_timestamp() must return None before any frame is processed."""
    intake = PhoneBrowserIntake()
    assert intake.get_frame_timestamp() is None


# ---------------------------------------------------------------------------
# PhoneBrowserIntake — frame round-trip with a synthetic JPEG
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 64, height: int = 48) -> bytes:
    """Create a minimal valid JPEG (solid grey rectangle)."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok, "cv2.imencode failed in test helper"
    return buf.tobytes()


def test_phone_browser_returns_frame_after_process():
    """After process_frame() with a valid JPEG, get_latest_frame() returns an ndarray."""
    intake = PhoneBrowserIntake()
    jpeg = _make_jpeg()
    # process_frame is async; use asyncio.run to drive it.
    asyncio.run(intake.process_frame(jpeg))

    frame = intake.get_latest_frame()
    assert frame is not None, "get_latest_frame() returned None after processing a valid JPEG"
    assert isinstance(frame, np.ndarray)
    # PhoneBrowserIntake stores grayscale (2D) after CLAHE preprocessing.
    assert frame.ndim in (2, 3), f"Unexpected ndim: {frame.ndim}"


def test_phone_browser_frame_dimensions_bounded():
    """Stored frame width must not exceed PHONE_FRAME_WIDTH."""
    from config import settings

    intake = PhoneBrowserIntake()
    # Use a wide image to exercise the resize path.
    jpeg = _make_jpeg(width=1280, height=720)
    asyncio.run(intake.process_frame(jpeg))

    frame = intake.get_latest_frame()
    assert frame is not None
    frame_w = frame.shape[1]
    assert frame_w <= settings.PHONE_FRAME_WIDTH, (
        f"Frame width {frame_w} exceeds PHONE_FRAME_WIDTH {settings.PHONE_FRAME_WIDTH}"
    )


def test_phone_browser_drops_malformed_jpeg():
    """process_frame() with garbage bytes must not raise and must leave frame as-is."""
    intake = PhoneBrowserIntake()
    asyncio.run(intake.process_frame(b"this is not a jpeg"))
    # No exception raised, and frame is still None (nothing was stored).
    assert intake.get_latest_frame() is None


# ---------------------------------------------------------------------------
# CaptureCardIntake — state before start() is called
# ---------------------------------------------------------------------------

def test_capture_card_returns_none_initially():
    """get_latest_frame() must return None before any frame is captured."""
    # Use device_index=99 — an invalid index that won't be opened (start() not called).
    intake = CaptureCardIntake(device_index=99)
    assert intake.get_latest_frame() is None


def test_capture_card_timestamp_is_none_initially():
    """get_frame_timestamp() must return None before any frame is captured."""
    intake = CaptureCardIntake(device_index=99)
    assert intake.get_frame_timestamp() is None
