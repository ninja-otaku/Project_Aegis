"""Google Gemini vision provider using the google-generativeai SDK.

Default model: gemini-1.5-flash — fast, cost-effective, vision-capable.
Set GEMINI_MODEL in .env to switch (e.g. gemini-1.5-pro for higher quality).

Requires: pip install google-generativeai
"""

import json
import logging

import cv2
import numpy as np

from .base import BaseAIProvider

logger = logging.getLogger(__name__)

_FALLBACK = {
    "game_state": "Unable to parse structured response.",
    "threats": [],
    "recommendation": "",
    "confidence": "low",
}


class GeminiVisionProvider(BaseAIProvider):
    """Sends game frames to the Gemini API for structured vision analysis."""

    def __init__(self) -> None:
        # Lazy import so the package is only required when this provider is used.
        import google.generativeai as genai
        from config import settings

        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to .env and restart."
            )

        genai.configure(api_key=settings.GEMINI_API_KEY)

        # system_instruction is supported on gemini-1.5-flash and later.
        self._model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=settings.ANALYSIS_SYSTEM_PROMPT,
        )

    async def analyze_frame(self, frame: np.ndarray) -> dict:
        jpeg_bytes = _encode_frame_bytes(frame)

        # Pass image as inline data + a short user prompt.
        # The system_instruction already carries the full analysis brief.
        image_part = {"mime_type": "image/jpeg", "data": jpeg_bytes}
        response = await self._model.generate_content_async(
            [image_part, "Analyse this game frame. Return valid JSON only."]
        )

        raw = (response.text or "").strip()
        return _parse_json(raw, provider="Gemini")


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _encode_frame_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy frame (grayscale or BGR) to JPEG bytes."""
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _parse_json(raw: str, provider: str) -> dict:
    """Strip markdown fences and parse JSON; fall back gracefully."""
    # Strip ```json ... ``` fences if the model wraps its output
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
        for key in ("game_state", "threats", "recommendation", "confidence"):
            if key not in data:
                raise ValueError(f"Missing key: {key}")
        return data
    except (json.JSONDecodeError, ValueError):
        logger.warning("%s: structured parse failed. Raw: %.200s", provider, raw)
        return {**_FALLBACK, "game_state": text or "No response."}
