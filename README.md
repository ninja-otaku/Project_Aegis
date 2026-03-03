![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

# Project Aegis

An air-gapped AI gaming companion that observes your game from a physically separate device — no code injection, no kernel hooks, no process memory access. Works with Claude, Gemini, GPT-4o, or fully offline with Ollama.

## How it works

```
Game monitor
     │
     ▼  (phone camera or HDMI capture card)
[Separate device running Aegis]
     │
     ├── Receives frames via WebSocket (phone) or cv2.VideoCapture (capture card)
     ├── Preprocesses frames (resize + grayscale + CLAHE contrast enhancement)
     ├── Skips API call if frame hasn't changed enough (frame diff)
     ├── Sends frames to the configured AI provider every N seconds
     └── Streams structured analysis back to your browser in real-time
```

Because Aegis runs on a **separate machine** and only passively observes the screen (like a human watching a monitor), it is invisible to kernel-level anti-cheat software on the gaming PC.

---

## Supported AI Providers

| Provider | Default Model | Requires | Notes |
|---|---|---|---|
| `claude` | `claude-opus-4-5` | Anthropic API key | Default, best quality |
| `gemini` | `gemini-1.5-flash` | Google API key | Fast, cost-effective |
| `openai` | `gpt-4o-mini` | OpenAI API key | Widely available |
| `ollama` | `llava` | Ollama running locally | Free, fully offline |

Set `AI_PROVIDER` in `.env` to switch. Only install the SDK for the provider you use.

---

## Architecture

```
Project_Aegis/
├── main.py                  # FastAPI app, lifespan, all routes & WebSockets
├── config.py                # ENV-based config (pydantic-settings)
├── intake/
│   ├── base.py              # BaseVideoIntake ABC
│   ├── phone_browser.py     # WebSocket JPEG intake, resize + CLAHE, compressed storage
│   └── capture_card.py      # cv2.VideoCapture in background thread
├── engine/
│   ├── processor.py         # Frame → AI loop, frame diff, history, TTS, pub/sub
│   └── tts.py               # pyttsx3 TTS engine (daemon thread, queue-based)
├── providers/
│   ├── base.py              # BaseAIProvider ABC  →  analyze_frame() -> dict
│   ├── claude_vision.py     # Claude Opus 4.5 (Anthropic)
│   ├── gemini_vision.py     # Gemini 1.5 Flash (Google)
│   ├── openai_vision.py     # GPT-4o Mini (OpenAI)
│   └── ollama_vision.py     # LLaVA / any local Ollama model
├── scripts/
│   └── generate_cert.py     # Self-signed TLS cert generator (trustme)
├── static/
│   └── index.html           # Phone frontend — camera stream + analysis UI
├── .env.example
└── requirements.txt
```

---

## Structured analysis output

Every frame analysis returns a structured JSON object:

```json
{
  "game_state":     "Player is low health, two enemies flanking from the right.",
  "threats":        ["Enemy sniper on rooftop", "Low ammo"],
  "recommendation": "Take cover behind the wall to the left immediately.",
  "confidence":     "high",
  "timestamp":      "2026-03-03T14:23:01.123Z"
}
```

---

## Intake modes

### Phone browser (default)
Open `http://<server-ip>:<port>` on your phone — it streams the rear camera at 10 FPS over WebSocket. No app required.

### Capture card
Connect an HDMI capture card to the Aegis machine and set `INTAKE_MODE=capture`. Reads directly from the device at 30 FPS.

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/ninja-otaku/Project_Aegis.git
cd Project_Aegis
pip install -r requirements.txt
```

**2. Configure**

```bash
cp .env.example .env
# Edit .env: set AI_PROVIDER and the matching API key
```

**3. Run**

```bash
python main.py
```

Open `http://<server-ip>:8765` on your phone or any browser on the same network.

---

### Phone camera not working?

Most mobile browsers block camera access (`getUserMedia`) over plain HTTP on non-localhost connections. You need HTTPS.

```bash
pip install trustme
python scripts/generate_cert.py
# Set TLS_ENABLED=true in .env
python main.py
# Open https://<server-ip>:8765 on your phone
# Accept the self-signed cert warning
```

To remove the browser warning entirely, install `certs/ca.pem` as a trusted CA on your phone (Settings → Security → Install certificate).

---

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Phone browser frontend |
| `GET /health` | Service status, provider info, frame/analysis timestamps |
| `GET /analysis` | Latest AI analysis (polling) |
| `GET /history` | All retained analyses, newest first |
| `WS /ws/intake` | Binary JPEG frame ingestion (phone mode) |
| `WS /ws/analysis` | Real-time structured analysis stream |
| `GET /profiles` | List available profiles and active profile |
| `POST /profiles/activate` | Hot-reload a profile `{"profile": "valorant"}` |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `INTAKE_MODE` | `phone` | `phone` or `capture` |
| `CAPTURE_DEVICE_INDEX` | `0` | cv2 device index (capture mode) |
| `PHONE_WS_PORT` | `8765` | Server listen port |
| `PHONE_FRAME_WIDTH` | `640` | Resize phone frames to this width (px) |
| `PHONE_COMPRESS_QUALITY` | `70` | JPEG quality for compressed frame storage |
| `AI_PROVIDER` | `claude` | `claude` \| `gemini` \| `openai` \| `ollama` |
| `ANTHROPIC_API_KEY` | — | Required for `claude` provider |
| `CLAUDE_MODEL` | `claude-opus-4-5` | Claude model string |
| `GEMINI_API_KEY` | — | Required for `gemini` provider |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model string |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model string |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llava` | Ollama model name |
| `ANALYSIS_INTERVAL_MS` | `2000` | AI call frequency (ms) |
| `ANALYSIS_SYSTEM_PROMPT` | *(see .env.example)* | System prompt (game-specific tuning) |
| `FRAME_DIFF_ENABLED` | `true` | Skip API call when scene unchanged |
| `FRAME_DIFF_THRESHOLD` | `0.02` | Min pixel change to trigger analysis (~2%) |
| `TLS_ENABLED` | `false` | Enable HTTPS (needed for phone camera) |
| `TLS_CERT_PATH` | `certs/cert.pem` | Path to TLS certificate |
| `TLS_KEY_PATH` | `certs/key.pem` | Path to TLS private key |
| `TTS_ENABLED` | `false` | Speak recommendations aloud (pyttsx3) |
| `HISTORY_MAX_ENTRIES` | `50` | Number of past analyses to retain |
| `RATE_LIMIT_RETRY_SECONDS` | `5` | Wait time before retrying a rate-limited call |
| `ACTIVE_PROFILE` | `default` | Game profile name (without `.json`) |
| `PROFILES_DIR` | `profiles` | Directory containing game profile JSON files |

---

## Game Profiles

Drop a `.json` file into `profiles/` to teach Aegis how to analyse a specific game — no code changes required.

**Schema**

```json
{
  "game_name": "My Game",
  "system_prompt": "You are a <game> coach. Output JSON: game_state, threats, recommendation, confidence.",
  "frame_diff_threshold": 0.02,
  "roi_strategy": "full_frame",
  "roi_crops": {
    "minimap": { "x": 0.82, "y": 0.75, "w": 0.18, "h": 0.25, "label": "Minimap" }
  }
}
```

- **Coordinates** are percentages `[0.0–1.0]` of frame width/height — fully resolution-independent.
- **`roi_strategy`** — `full_frame` (no crop) · `horizontal_stack` · `vertical_stack` · `grid` (2×2)
- **`frame_diff_threshold`** — overrides the global `.env` value for this game
- If `roi_crops` is empty or strategy is `full_frame`, the whole frame is sent to the AI

**Built-in profiles**

| Profile | Strategy | Crops |
|---|---|---|
| `default` | `full_frame` | None — full screen |
| `league_of_legends` | `horizontal_stack` | Minimap + Health/Mana bar |
| `valorant` | `horizontal_stack` | Minimap + Health/Shield + Abilities |

**Activate a profile**

```bash
# Via .env (requires restart)
ACTIVE_PROFILE=valorant

# Or hot-reload at runtime (no restart)
curl -X POST http://localhost:8765/profiles/activate \
  -H "Content-Type: application/json" \
  -d '{"profile": "valorant"}'
```

> **Tip:** To measure ROI coordinates, take a screenshot of your game at your native resolution, open it in browser DevTools or any image editor, and read the pixel positions of the UI element you want to crop. Divide by frame width/height to get percentages.

---

## Docker Setup

```bash
# Pull the LLaVA model first (one-time, ~4 GB) — only needed for AI_PROVIDER=ollama
docker-compose run --rm ollama ollama pull llava

# Start Aegis + Ollama
docker-compose up -d

# Open http://<server-ip>:8765 on your phone
```

Game profiles are mounted as a read-only volume — add or edit `.json` files in `profiles/` and hit `POST /profiles/activate` without rebuilding.

**TLS with Docker:** Generate certs first (`python scripts/generate_cert.py`), then uncomment the `certs` volume mount in `docker-compose.yml` and set `TLS_ENABLED=true` in `.env`.

---

## Running Tests

```bash
pip install pytest
pytest tests/
```

Tests use mocks — no API keys or hardware required.

---

## Extending

**New intake source** — subclass `BaseVideoIntake` in `intake/`, implement `start()`, `stop()`, `get_latest_frame()`.

**New AI provider** — subclass `BaseAIProvider` in `providers/`, implement `async analyze_frame(frame, system_prompt=None) -> dict`. The dict must contain `game_state`, `threats`, `recommendation`, `confidence`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for code templates and a full checklist.

---

## Roadmap

- Additional providers: Mistral Vision, Groq (LLaVA), Gemma via Ollama
- Aegis II: mobile-native companion app (React Native) replacing the browser frontend

---

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new providers and intake sources.

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Project Aegis Contributors
