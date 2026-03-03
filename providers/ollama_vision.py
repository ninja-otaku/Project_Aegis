"""Ollama vision provider using the ollama Python SDK.

Runs fully offline — no API key required.
Default model: llava (most widely used local vision model).

Requirements:
    1. Install Ollama: https://ollama.com/download
    2. Pull the model:  ollama pull llava
    3. Set AI_PROVIDER=ollama in .env (OLLAMA_HOST defaults to localhost)

Requires: pip install ollama
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


class OllamaVisionProvider(BaseAIProvider):
    """Sends game frames to a local Ollama instance for vision analysis.

    Requires no API key — Ollama runs entirely on your machine.
    Latency depends on your GPU/CPU; a dedicated GPU is recommended for
    analysis intervals below 5 seconds.
    """

    def __init__(self) -> None:
        import ollama
        from config import settings

        # AsyncClient is non-blocking — safe to call from the asyncio loop.
        self._client = ollama.AsyncClient(host=settings.OLLAMA_HOST)
        self._model = settings.OLLAMA_MODEL
        self._system_prompt = settings.ANALYSIS_SYSTEM_PROMPT

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str | None = None,
    ) -> dict:
        b64_str = _encode_frame_b64(frame)
        # Use profile-supplied system prompt when available, else fall back to default.
        effective_prompt = system_prompt if system_prompt is not None else self._system_prompt

        # Ollama's chat API accepts base64 image strings in the images field.
        # The system role sets analysis context; user role carries the image.
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": effective_prompt},
                {
                    "role": "user",
                    "content": "Analyse this game frame. Return valid JSON only.",
                    "images": [b64_str],
                },
            ],
        )

        raw = (response.message.content or "").strip()
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
        logger.warning("Ollama: structured parse failed. Raw: %.200s", raw)
        return {**_FALLBACK, "game_state": text or "No response."}
