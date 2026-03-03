import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import anthropic

from intake.base import BaseVideoIntake
from providers.base import BaseAIProvider

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Drives the intake → AI provider analysis loop.

    A single asyncio background task wakes every ``interval_ms`` milliseconds,
    grabs the latest frame from the intake source, and dispatches it to the AI
    provider.  The result is stored in a single-slot cache (``_latest``) and
    pushed to all active WebSocket subscribers via per-subscriber asyncio
    queues.

    Subscriber model:
      - ``subscribe()``   — returns a ``Queue[dict]`` that receives results.
      - ``unsubscribe()`` — removes the queue; call in a ``finally`` block.
      - Queues have ``maxsize=1``.  If a slow subscriber hasn't consumed the
        previous result, the new one is silently dropped — analysis results
        are always fresh, never backlogged.

    Error handling:
      - ``anthropic.AuthenticationError`` → logged once, processor stops.
      - All other provider exceptions    → logged, loop continues.
    """

    def __init__(
        self,
        intake: BaseVideoIntake,
        provider: BaseAIProvider,
        interval_ms: int = 2000,
    ) -> None:
        self._intake = intake
        self._provider = provider
        self._interval = interval_ms / 1000.0

        self._latest: dict[str, Any] | None = None
        self._result_lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self._loop(), name="frame-processor"
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Public read / subscribe API
    # ------------------------------------------------------------------

    def get_latest(self) -> dict | None:
        """Return the most-recent analysis result dict, or None."""
        return self._latest

    def subscribe(self) -> "asyncio.Queue[dict]":
        """Register a new subscriber queue and return it."""
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[dict]") -> None:
        self._subscribers.discard(q)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        logger.info(
            "FrameProcessor started (interval=%.1fs)", self._interval
        )
        while True:
            await asyncio.sleep(self._interval)

            frame = self._intake.get_latest_frame()
            if frame is None:
                logger.debug("No frame available yet — skipping analysis.")
                continue

            try:
                analysis = await self._provider.analyze_frame(frame)
            except anthropic.AuthenticationError:
                logger.error(
                    "ANTHROPIC_API_KEY is missing or invalid. "
                    "Set it in .env and restart.  Processor stopping."
                )
                return  # Non-recoverable — don't retry endlessly.
            except Exception:
                logger.exception("Provider analysis failed — will retry.")
                continue

            result: dict[str, Any] = {
                "analysis": analysis,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

            async with self._result_lock:
                self._latest = result

            self._notify_subscribers(result)

    def _notify_subscribers(self, result: dict) -> None:
        """Push result to all subscriber queues, dropping if full."""
        for q in list(self._subscribers):
            if not q.full():
                try:
                    q.put_nowait(result)
                except asyncio.QueueFull:
                    pass  # subscriber is behind — newest frame wins
