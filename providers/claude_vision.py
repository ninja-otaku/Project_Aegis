import base64
import logging

import anthropic
import cv2
import numpy as np

from .base import BaseAIProvider

logger = logging.getLogger(__name__)


class ClaudeVisionProvider(BaseAIProvider):
    """Sends game frames to Claude's vision API for real-time analysis.

    Frame encoding:
      - Grayscale (H, W)     — from PhoneBrowserIntake (CLAHE-enhanced)
      - BGR colour (H, W, 3) — from CaptureCardIntake (raw capture)

    Both are encoded to JPEG before transmission.  Grayscale is first
    promoted to BGR so cv2.imencode produces a well-formed JPEG; Claude
    handles monochrome images correctly regardless.

    Streaming is used unconditionally: it prevents HTTP timeouts on
    slow connections and gives us .get_final_message() for clean result
    extraction.  Adaptive thinking is enabled so Claude scales reasoning
    depth to the complexity of the scene.
    """

    def __init__(self, model: str, system_prompt: str) -> None:
        # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment.
        # Construction succeeds even if the key is absent; the first API
        # call will raise anthropic.AuthenticationError in that case.
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._system_prompt = system_prompt

    async def analyze_frame(self, frame: np.ndarray) -> str:
        """Encode frame, call Claude vision, return analysis text."""
        image_b64 = _encode_frame(frame)

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=self._system_prompt,
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
                        "text": "Analyse this game frame and provide your assessment.",
                    },
                ],
            }],
        ) as stream:
            response = await stream.get_final_message()

        # Collect only text blocks — skip thinking blocks (internal reasoning).
        parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray, jpeg_quality: int = 85) -> str:
    """Return a base64-encoded JPEG string from a numpy frame."""
    if frame.ndim == 2:
        # Grayscale → BGR so imencode produces standard 3-channel JPEG.
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame  # already BGR from CaptureCardIntake

    ok, buf = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed — frame may be corrupt")

    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")
