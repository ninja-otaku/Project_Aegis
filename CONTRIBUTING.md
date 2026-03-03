# Contributing to Project Aegis

Welcome! Aegis is built around two clean abstractions — `BaseVideoIntake` and `BaseAIProvider` — that make it straightforward to add new screen-capture sources and AI backends without touching any existing code. If you want to support a new provider, game genre, or intake method, you're in the right place.

---

## Adding a new intake source

Subclass `BaseVideoIntake` in `intake/` and implement three methods:

```python
from intake.base import BaseVideoIntake
import numpy as np

class MyIntake(BaseVideoIntake):

    async def start(self) -> None:
        # Open hardware, start threads, etc.
        ...

    async def stop(self) -> None:
        # Release hardware, stop threads, clear frame buffer.
        ...

    def get_latest_frame(self) -> np.ndarray | None:
        # Return a copy of the latest frame, or None if none yet.
        # Must be thread-safe.
        ...
```

Then add a branch in `main.py`'s `_build_intake()` factory and a value to the `IntakeMode` enum in `config.py`.

---

## Adding a new AI provider

Subclass `BaseAIProvider` in `providers/` and implement one async method:

```python
from providers.base import BaseAIProvider
import numpy as np

class MyProvider(BaseAIProvider):

    async def analyze_frame(self, frame: np.ndarray) -> dict:
        # frame is either grayscale (H, W) or BGR (H, W, 3).
        # Call your API, parse the response, return a dict with these keys:
        return {
            "game_state":     "What is currently happening in the game.",
            "threats":        ["Threat or opportunity 1", "Threat 2"],
            "recommendation": "The single best action to take right now.",
            "confidence":     "high",  # "low" | "medium" | "high"
        }
```

Then add a branch in `main.py`'s `_build_provider()` factory and a value to `AI_PROVIDER` in `.env.example`.

**Interface contract:**
- `frame` may be grayscale `(H, W)` or BGR `(H, W, 3)` — handle both.
- Always return a dict with the four required keys above.
- Never raise on a malformed API response — log and return a fallback dict.
- Use `async`/`await` throughout; never block the event loop.

---

## Good first issues

- **Frame diff tuning presets** — per-game-genre threshold presets (e.g. FPS=0.05, RTS=0.01, RPG=0.02) configurable in `.env`
- **New intake sources** — NDI stream, WebRTC, RTSP/RTMP from a capture device
- **New AI providers** — Mistral Vision, Groq (LLaVA), local Gemma via Ollama
- **Per-game ROI cropping** — crop to minimap / health bar before sending to Claude to reduce token cost and improve accuracy
- **Structured prompt library** — curated system prompts tuned for specific popular games

---

## Dev setup

```bash
git clone https://github.com/ninja-otaku/Project_Aegis.git
cd Project_Aegis
pip install -r requirements.txt
cp .env.example .env
# Add your API key to .env
python main.py
```

---

## Pull request checklist

- [ ] No new dependencies added without a comment explaining why they're necessary
- [ ] `BaseVideoIntake` and `BaseAIProvider` interfaces are unchanged (no new abstract methods without prior discussion)
- [ ] New provider returns the correct four-key dict with graceful JSON parse fallback
- [ ] Tested with at least one intake mode (`phone` or `capture`)
- [ ] No secrets, `.env` files, or generated certs committed
