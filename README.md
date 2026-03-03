# Project Aegis

An air-gapped AI gaming companion that observes your game from a physically separate device — no code injection, no kernel hooks, no process memory access.

## How it works

```
Game monitor
     │
     ▼  (phone camera or HDMI capture card)
[Separate device running Aegis]
     │
     ├── Receives frames via WebSocket (phone) or cv2.VideoCapture (capture card)
     ├── Preprocesses frames (grayscale + CLAHE contrast enhancement)
     ├── Sends frames to Claude Opus 4.6 vision every N seconds
     └── Streams structured AI analysis back to your browser in real-time
```

Because Aegis runs on a **separate machine** and only passively observes the screen (like a human watching a monitor), it is invisible to kernel-level anti-cheat software on the gaming PC.

---

## Architecture

```
Project_Aegis/
├── main.py                  # FastAPI app, lifespan, all routes & WebSockets
├── config.py                # ENV-based config (pydantic-settings)
├── intake/
│   ├── base.py              # BaseVideoIntake ABC
│   ├── phone_browser.py     # WebSocket JPEG intake + CLAHE preprocessing
│   └── capture_card.py      # cv2.VideoCapture in background thread
├── engine/
│   ├── processor.py         # Frame → AI loop, history deque, TTS, pub/sub
│   └── tts.py               # pyttsx3 TTS engine (daemon thread, queue-based)
├── providers/
│   ├── base.py              # BaseAIProvider ABC  →  analyze_frame() -> dict
│   └── claude_vision.py     # Claude Opus 4.6 vision, structured JSON output
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
  "timestamp":      "2026-03-02T14:23:01.123Z"
}
```

This is displayed in the browser UI and optionally spoken aloud via TTS.

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
# Edit .env and add your ANTHROPIC_API_KEY
```

**3. Run**

```bash
python main.py
```

Open `http://<server-ip>:8765` on your phone or any browser on the same network.

---

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Phone browser frontend |
| `GET /health` | Service status + frame/analysis timestamps |
| `GET /analysis` | Latest AI analysis (polling) |
| `GET /history` | All retained analyses, newest first |
| `WS /ws/intake` | Binary JPEG frame ingestion (phone mode) |
| `WS /ws/analysis` | Real-time structured analysis stream |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `INTAKE_MODE` | `phone` | `phone` or `capture` |
| `CAPTURE_DEVICE_INDEX` | `0` | cv2 device index (capture mode) |
| `PHONE_WS_PORT` | `8765` | Server listen port |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (**required**) |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Model for vision analysis |
| `ANALYSIS_INTERVAL_MS` | `2000` | AI call frequency (ms) |
| `ANALYSIS_SYSTEM_PROMPT` | *(see .env.example)* | System prompt (game-specific tuning) |
| `TTS_ENABLED` | `false` | Speak recommendations aloud via pyttsx3 |
| `HISTORY_MAX_ENTRIES` | `50` | Number of past analyses to retain |

---

## Extending

**New intake source** — subclass `BaseVideoIntake` in `intake/`, implement `start()`, `stop()`, `get_latest_frame()`.

**New AI provider** — subclass `BaseAIProvider` in `providers/`, implement `async analyze_frame(frame) -> dict`. The dict must contain `game_state`, `threats`, `recommendation`, `confidence`.

Swap providers in `main.py`'s `_build_processor()` factory with no other changes required.

---

## Requirements

- Python 3.11+
- A second device with a camera (phone) **or** an HDMI capture card
- Anthropic API key
- `pyttsx3` only required if `TTS_ENABLED=true`
