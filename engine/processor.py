import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import cv2
import numpy as np

from config import settings
from intake.base import BaseVideoIntake
from providers.base import BaseAIProvider

if TYPE_CHECKING:
    from engine.tts import TTSEngine

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Drives the intake → AI provider analysis loop.

    A single asyncio background task wakes every ``interval_ms`` milliseconds,
    grabs the latest frame from the intake source, applies ROI cropping from
    the active game profile, and dispatches it to the AI provider.

    Frame diff:
      Before each API call the current frame is compared to the last one that
      was actually sent to the provider.  The per-profile ``frame_diff_threshold``
      takes precedence over the global FRAME_DIFF_THRESHOLD setting.

    Error surfacing:
      Provider exceptions are caught, classified, and broadcast as structured
      error payloads over the analysis WebSocket so the frontend can show a
      toast rather than going silently stale.  Rate-limit errors trigger a
      single retry after RATE_LIMIT_RETRY_SECONDS.

    Result format (flat dict, JSON-safe):
      game_state     (str)       — current game situation
      threats        (list[str]) — immediate threats / opportunities
      recommendation (str)       — single recommended action
      confidence     (str)       — "low" | "medium" | "high"
      timestamp      (str)       — UTC ISO-8601

    Error payload format:
      status         (str) — always "error"
      error_type     (str) — "rate_limit" | "provider_error" | "timeout" | "unknown"
      message        (str) — human-readable description
      provider       (str) — active AI_PROVIDER name
      timestamp      (str) — UTC ISO-8601

    Subscriber model:
      - ``subscribe()``   — returns a ``Queue[dict]`` that receives results.
      - ``unsubscribe()`` — removes the queue; call in a ``finally`` block.
      - Queues have ``maxsize=1``.  Slow subscribers drop stale results.
    """

    def __init__(
        self,
        intake: BaseVideoIntake,
        provider: BaseAIProvider,
        interval_ms: int = 2000,
        history_max: int = 50,
        tts: "TTSEngine | None" = None,
    ) -> None:
        self._intake = intake
        self._provider = provider
        self._interval = interval_ms / 1000.0
        self._tts = tts

        self._latest: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=history_max)
        self._result_lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

        # Stores the last frame that was actually sent to the AI provider,
        # used for frame-diff comparison on the next tick.
        self._last_analyzed_frame: np.ndarray | None = None

        # Load the active game profile (system prompt, ROI crops, thresholds).
        self._load_profile()

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
        return self._latest

    def get_history(self) -> list[dict]:
        """Return history list, newest entry first."""
        return list(reversed(self._history))

    def subscribe(self) -> "asyncio.Queue[dict]":
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[dict]") -> None:
        self._subscribers.discard(q)

    def get_active_profile(self) -> str:
        """Return the name of the currently loaded profile (JSON stem)."""
        return self._active_profile_name

    def reload_profile(self, name: str) -> str:
        """Hot-reload a named profile without restarting the processor.

        Returns the name of the profile that was actually loaded (may differ
        from *name* if the file was not found and default.json was used).
        """
        self._load_profile_by_name(name)
        return self._active_profile_name

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    def _load_profile(self) -> None:
        """Load the profile specified by settings.ACTIVE_PROFILE."""
        self._load_profile_by_name(settings.ACTIVE_PROFILE)

    def _load_profile_by_name(self, name: str) -> None:
        """Load a named profile from PROFILES_DIR, falling back to default.json."""
        profile_path = Path(settings.PROFILES_DIR) / f"{name}.json"
        if not profile_path.exists():
            logger.warning(
                "Profile '%s' not found at %s — falling back to default.json",
                name, profile_path,
            )
            profile_path = Path(settings.PROFILES_DIR) / "default.json"

        try:
            with open(profile_path, encoding="utf-8") as fh:
                profile = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to load profile from %s: %s — using built-in defaults.",
                profile_path, exc,
            )
            profile = {}

        # Store all profile-derived attributes used by the analysis loop.
        self._active_profile_name: str = profile_path.stem
        self._profile_system_prompt: str = profile.get(
            "system_prompt", settings.ANALYSIS_SYSTEM_PROMPT
        )
        self._roi_strategy: str = profile.get("roi_strategy", "full_frame")
        self._roi_crops: dict = profile.get("roi_crops", {})
        # Per-profile threshold overrides the global FRAME_DIFF_THRESHOLD when present.
        self._profile_frame_diff_threshold: float = float(
            profile.get("frame_diff_threshold", settings.FRAME_DIFF_THRESHOLD)
        )
        logger.info(
            "Loaded profile '%s' (strategy=%s, diff_threshold=%.3f)",
            self._active_profile_name, self._roi_strategy,
            self._profile_frame_diff_threshold,
        )

    # ------------------------------------------------------------------
    # ROI cropping
    # ------------------------------------------------------------------

    def _apply_roi_cropping(self, frame: np.ndarray) -> np.ndarray:
        """Crop and composite the frame according to the active profile's ROI strategy.

        Coordinates in roi_crops are percentages [0.0–1.0] of frame
        width/height, making them resolution-independent.

        Strategies:
          full_frame       — return frame unchanged (no cropping)
          horizontal_stack — hstack regions, all resized to the same height
          vertical_stack   — vstack regions, all resized to the same width
          grid             — 2×2 grid; pads with blank tiles if fewer than 4 regions
        """
        # Skip cropping if strategy is full_frame or no crops are defined.
        if self._roi_strategy == "full_frame" or not self._roi_crops:
            return frame

        h, w = frame.shape[:2]
        regions: list[np.ndarray] = []
        for crop in self._roi_crops.values():
            # Convert percentage coordinates to pixel values.
            x  = int(crop["x"] * w)
            y  = int(crop["y"] * h)
            cw = int(crop["w"] * w)
            ch = int(crop["h"] * h)
            # Clamp to frame bounds to avoid index errors on edge-case crops.
            x  = max(0, min(x,  w - 1))
            y  = max(0, min(y,  h - 1))
            cw = max(1, min(cw, w - x))
            ch = max(1, min(ch, h - y))
            regions.append(frame[y:y + ch, x:x + cw])

        if not regions:
            return frame

        if self._roi_strategy == "horizontal_stack":
            # Resize all regions to the tallest crop's height, then hstack.
            target_h = max(r.shape[0] for r in regions)
            resized = [
                cv2.resize(
                    r,
                    (max(1, int(r.shape[1] * target_h / r.shape[0])), target_h),
                    interpolation=cv2.INTER_AREA,
                ) if r.shape[0] != target_h else r
                for r in regions
            ]
            return np.hstack(resized)

        if self._roi_strategy == "vertical_stack":
            # Resize all regions to the widest crop's width, then vstack.
            target_w = max(r.shape[1] for r in regions)
            resized = [
                cv2.resize(
                    r,
                    (target_w, max(1, int(r.shape[0] * target_w / r.shape[1]))),
                    interpolation=cv2.INTER_AREA,
                ) if r.shape[1] != target_w else r
                for r in regions
            ]
            return np.vstack(resized)

        if self._roi_strategy == "grid":
            # Arrange in a 2×2 grid; pad with blank tiles if fewer than 4 crops.
            blank = np.zeros_like(regions[0])
            while len(regions) < 4:
                regions.append(blank)
            target_h = max(r.shape[0] for r in regions[:4])
            target_w = max(r.shape[1] for r in regions[:4])
            tiles = [
                cv2.resize(r, (target_w, target_h), interpolation=cv2.INTER_AREA)
                for r in regions[:4]
            ]
            return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

        # Unknown strategy — pass frame through unchanged.
        return frame

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        logger.info("FrameProcessor started (interval=%.1fs)", self._interval)
        while True:
            await asyncio.sleep(self._interval)

            frame = self._intake.get_latest_frame()
            if frame is None:
                logger.debug("No frame available yet — skipping analysis.")
                continue

            # ── Frame diff check (uses per-profile threshold) ──────────────
            # Skip the API call when the scene hasn't changed enough.
            if settings.FRAME_DIFF_ENABLED and self._last_analyzed_frame is not None:
                diff = _compute_diff(frame, self._last_analyzed_frame)
                if diff < self._profile_frame_diff_threshold:
                    logger.debug(
                        "Frame diff %.4f < threshold %.4f — skipping API call.",
                        diff, self._profile_frame_diff_threshold,
                    )
                    continue

            # ── Apply ROI cropping from the active profile ─────────────────
            cropped = self._apply_roi_cropping(frame)

            # ── AI provider call with error surfacing ──────────────────────
            try:
                # Pass the profile system prompt so game-specific instructions
                # are used instead of the global ANALYSIS_SYSTEM_PROMPT.
                analysis = await self._provider.analyze_frame(
                    cropped, system_prompt=self._profile_system_prompt
                )
            except anthropic.AuthenticationError:
                # Broadcast an error toast so the frontend shows a red message
                # instead of going silently stale, then stop the loop.
                msg = (
                    "ANTHROPIC_API_KEY is missing or invalid — "
                    "set it in .env and restart."
                )
                logger.error(msg)
                self._notify_subscribers(
                    self._make_error_payload("provider_error", msg)
                )
                return
            except asyncio.TimeoutError:
                # Broadcast timeout so the frontend can show a toast.
                logger.warning("Provider call timed out.")
                self._notify_subscribers(
                    self._make_error_payload("timeout", "Analysis request timed out.")
                )
                continue
            except Exception as exc:
                error_type = _classify_exception(exc)
                if error_type == "rate_limit":
                    # One retry after RATE_LIMIT_RETRY_SECONDS backoff.
                    wait = settings.RATE_LIMIT_RETRY_SECONDS
                    logger.warning("Rate limited — retrying in %ds.", wait)
                    # Notify subscribers immediately so the UI can show a toast.
                    self._notify_subscribers(
                        self._make_error_payload(
                            "rate_limit",
                            f"Rate limited — retrying in {wait}s",
                        )
                    )
                    await asyncio.sleep(wait)
                    try:
                        analysis = await self._provider.analyze_frame(
                            cropped, system_prompt=self._profile_system_prompt
                        )
                    except Exception as retry_exc:
                        # Retry also failed — broadcast the error and move on.
                        retry_type = _classify_exception(retry_exc)
                        logger.error(
                            "Retry also failed (%s): %s", retry_type, retry_exc
                        )
                        self._notify_subscribers(
                            self._make_error_payload(retry_type, str(retry_exc))
                        )
                        continue
                else:
                    # Non-rate-limit error — broadcast and continue the loop.
                    logger.exception("Provider call failed (%s).", error_type)
                    self._notify_subscribers(
                        self._make_error_payload(error_type, str(exc))
                    )
                    continue

            # ── Success path ───────────────────────────────────────────────
            # Record the original (uncropped) frame for next-tick diff comparison.
            self._last_analyzed_frame = frame

            result: dict[str, Any] = {
                **analysis,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

            async with self._result_lock:
                self._latest = result
                self._history.append(result)

            if self._tts and result.get("recommendation"):
                self._tts.speak(result["recommendation"])

            self._notify_subscribers(result)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _make_error_payload(self, error_type: str, message: str) -> dict:
        """Build the standardised error dict broadcast to analysis WebSocket subscribers."""
        return {
            "status": "error",
            "error_type": error_type,
            "message": message,
            "provider": settings.AI_PROVIDER,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Subscriber notify
    # ------------------------------------------------------------------

    def _notify_subscribers(self, result: dict) -> None:
        for q in list(self._subscribers):
            if not q.full():
                try:
                    q.put_nowait(result)
                except asyncio.QueueFull:
                    pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _classify_exception(exc: Exception) -> str:
    """Map a provider SDK exception to a canonical error_type string.

    Checks in priority order:
      1. asyncio.TimeoutError                                    → "timeout"
      2. HTTP 429 via status_code / code attribute              → "rate_limit"
      3. Class name pattern (RateLimit, ResourceExhausted, …)   → "rate_limit"
      4. Known SDK module (anthropic, openai, google, ollama …) → "provider_error"
      5. Anything else                                           → "unknown"
    """
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"

    # HTTP 429 — works for anthropic, openai, and any httpx-based SDK.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return "rate_limit"

    # Class-name pattern matching for SDKs that encode rate-limit differently.
    name = type(exc).__name__
    if any(k in name for k in ("RateLimit", "ResourceExhausted", "TooManyRequests")):
        return "rate_limit"

    # Any exception from a known AI SDK module → provider_error.
    module = type(exc).__module__ or ""
    if any(m in module for m in ("anthropic", "openai", "google", "ollama", "httpx", "grpc")):
        return "provider_error"

    return "unknown"


def _compute_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two frames, normalised to [0, 1].

    Both frames are converted to grayscale before comparison so the metric
    is independent of whether the intake source delivers colour or greyscale.
    If the shapes differ (e.g. after a resolution change) b is resized to
    match a before comparison.
    """
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) / 255.0
