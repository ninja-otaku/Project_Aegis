"""Groq vision provider using the groq SDK (OpenAI-compatible API).

Default model: llama-3.2-11b-vision-preview — Groq's fastest vision model.
Set GROQ_MODEL in .env to switch (e.g. llama-3.2-90b-vision-preview).

Requires: pip install groq
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


class GroqVisionProvider(BaseAIProvider):
    """Sends game frames to the Groq API for structured vision analysis.

    Groq provides extremely fast inference for open-source models.
    No self-hosting required — just an API key from console.groq.com.
    """

    def __init__(self) -> None:
        from groq import AsyncGroq
        from config import settings

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env and restart."
            )

        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL
        self._system_prompt = settings.ANALYSIS_SYSTEM_PROMPT

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str | None = None,
    ) -> dict:
        b64_str = _encode_frame_b64(frame)
        data_url = f"data:image/jpeg;base64,{b64_str}"
        effective_prompt = system_prompt if system_prompt is not None else self._system_prompt

        response = await self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": effective_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse this game frame."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=1024,
        )

        raw = (response.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
            for key in ("game_state", "threats", "recommendation", "confidence"):
                if key not in data:
                    raise ValueError(f"Missing key: {key}")
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("Groq: structured parse failed. Raw: %.200s", raw)
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
