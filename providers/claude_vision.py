import base64
import json
import logging

import anthropic
import cv2
import numpy as np

from .base import BaseAIProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema for structured output
# ---------------------------------------------------------------------------

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "game_state": {
            "type": "string",
            "description": "Brief description of the current game situation (1-2 sentences).",
        },
        "threats": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of immediate threats or opportunities visible in the frame.",
        },
        "recommendation": {
            "type": "string",
            "description": "Single most important action the player should take right now.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence in the analysis given image clarity and game context.",
        },
    },
    "required": ["game_state", "threats", "recommendation", "confidence"],
    "additionalProperties": False,
}

_FALLBACK_RESULT = {
    "game_state": "Unable to parse structured response.",
    "threats": [],
    "recommendation": "",
    "confidence": "low",
}


class ClaudeVisionProvider(BaseAIProvider):
    """Sends game frames to Claude's vision API for structured analysis.

    Frame encoding:
      - Grayscale (H, W)     — from PhoneBrowserIntake (CLAHE-enhanced)
      - BGR colour (H, W, 3) — from CaptureCardIntake (raw capture)

    Both are encoded to JPEG before transmission.  Grayscale is promoted
    to BGR so cv2.imencode produces a well-formed JPEG.

    Structured output (output_config JSON schema) guarantees the response
    matches the GameAnalysis schema.  A JSON parse fallback is provided in
    case the schema enforcement is unavailable on the configured model.

    Adaptive thinking is enabled so Claude scales reasoning depth to the
    complexity of the scene.  Streaming prevents HTTP timeouts.
    """

    def __init__(self, model: str, system_prompt: str) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._system_prompt = system_prompt

    async def analyze_frame(
        self,
        frame: np.ndarray,
        system_prompt: str | None = None,
    ) -> dict:
        """Encode frame, call Claude vision, return structured analysis dict."""
        image_b64 = _encode_frame(frame)
        # Use profile-supplied system prompt when available, else fall back to default.
        effective_prompt = system_prompt if system_prompt is not None else self._system_prompt

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=effective_prompt,
            output_config={"format": {"type": "json_schema", "json_schema": {"name": "game_analysis", "schema": _ANALYSIS_SCHEMA}}},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyse this game frame.",
                    },
                ],
            }],
        ) as stream:
            response = await stream.get_final_message()

        # Extract text blocks (skip thinking blocks).
        text_parts = [b.text for b in response.content if b.type == "text"]
        raw = "\n".join(text_parts).strip()

        try:
            data = json.loads(raw)
            # Ensure required keys are present.
            for key in ("game_state", "threats", "recommendation", "confidence"):
                if key not in data:
                    raise ValueError(f"Missing key: {key}")
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Structured output parse failed — returning raw text as game_state.\n"
                "Raw response: %s", raw[:200]
            )
            return {**_FALLBACK_RESULT, "game_state": raw or "No response."}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray, jpeg_quality: int = 85) -> str:
    """Return a base64-encoded JPEG string from a numpy frame."""
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame

    ok, buf = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed — frame may be corrupt")

    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")
