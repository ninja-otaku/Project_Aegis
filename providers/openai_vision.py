"""OpenAI vision provider using the openai SDK.

Default model: gpt-4o-mini — cheapest vision-capable model, widely available.
Set OPENAI_MODEL in .env to switch (e.g. gpt-4o for higher quality).

Requires: pip install openai>=1.0.0
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


class OpenAIVisionProvider(BaseAIProvider):
    """Sends game frames to the OpenAI API for structured vision analysis."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        from config import settings

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env and restart."
            )

        # AsyncOpenAI reads OPENAI_API_KEY from env if api_key not passed,
        # but we pass it explicitly for clarity.
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_MODEL
        self._system_prompt = settings.ANALYSIS_SYSTEM_PROMPT

    async def analyze_frame(self, frame: np.ndarray) -> dict:
        b64_str = _encode_frame_b64(frame)
        # OpenAI vision requires a data URL for base64 images
        data_url = f"data:image/jpeg;base64,{b64_str}"

        response = await self._client.chat.completions.create(
            model=self._model,
            # json_object mode guarantees parseable JSON without markdown fences
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "Analyse this game frame."},
                    ],
                },
            ],
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(raw)
            for key in ("game_state", "threats", "recommendation", "confidence"):
                if key not in data:
                    raise ValueError(f"Missing key: {key}")
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("OpenAI: structured parse failed. Raw: %.200s", raw)
            return {**_FALLBACK, "game_state": raw or "No response."}


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
