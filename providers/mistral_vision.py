"""Mistral vision provider using the mistralai SDK.

Default model: pixtral-12b-2409 — Mistral's multimodal vision model.
Set MISTRAL_MODEL in .env to switch.

Requires: pip install mistralai
"""

import base64
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


class MistralVisionProvider(BaseAIProvider):
    """Sends game frames to the Mistral API for structured vision analysis."""

    def __init__(self) -> None:
        from mistralai import Mistral
        from config import settings

        if not settings.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Add it to .env and restart."
            )

        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self._model = settings.MISTRAL_MODEL
        self._system_prompt = settings.ANALYSIS_SYSTEM_PROMPT

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str | None = None,
    ) -> dict:
        b64_str = _encode_frame_b64(frame)
        data_url = f"data:image/jpeg;base64,{b64_str}"
        effective_prompt = system_prompt if system_prompt is not None else self._system_prompt

        response = await self._client.chat.complete_async(
            model=self._model,
            messages=[
                {"role": "system", "content": effective_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse this game frame. Return valid JSON only."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        return _parse_json(raw)


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _encode_frame_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a numpy frame (grayscale or BGR) to a base64 JPEG string."""
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON; fall back gracefully."""
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
        logger.warning("Mistral: structured parse failed. Raw: %.200s", raw)
        return {**_FALLBACK, "game_state": text or "No response."}
