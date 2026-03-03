# Project Aegis

An air-gapped AI gaming companion that observes your game from a physically separate device — no code injection, no kernel hooks, no process memory access.

## How it works

```
Game monitor
     │
     ▼  (camera or HDMI capture card)
[Separate device running Aegis]
     │
     ├── Receives frames via WebSocket (phone) or cv2.VideoCapture (capture card)
     ├── Preprocesses frames (grayscale + CLAHE contrast enhancement)
     ├── Sends frames to Claude vision API every N seconds
     └── Streams AI analysis back to your browser in real-time
```

Because Aegis runs on a **separate machine** and only passively observes the screen (like a human watching a monitor), it is invisible to kernel-level anti-cheat software running on the gaming PC.

---

## Architecture

```
Project_Aegis/
├── main.py                  # FastAPI app, lifespan, all routes
├── config.py                # ENV-based config (pydantic-settings)
├── intake/
│   ├── base.py              # BaseVideoIntake ABC
│   ├── phone_browser.py     # WebSocket JPEG intake + CLAHE preprocessing
│   └── capture_card.py      # cv2.VideoCapture in background thread
├── engine/
│   └── processor.py         # Frame → AI provider loop, pub/sub to WS clients
├── providers/
│   ├── base.py              # BaseAIProvider ABC
│   └── claude_vision.py     # Claude Opus 4.6 vision provider
├── static/
│   └── index.html           # Phone browser frontend (camera + analysis display)
├── .env.example
└── requirements.txt
```

---

## Intake modes

### Phone browser (default)
Your phone opens `http://<server-ip>:<port>` and streams its rear camera at 10 FPS over WebSocket. No app required — pure browser.

### Capture card
Connect an HDMI capture card to the Aegis machine and set `INTAKE_MODE=capture`. Reads directly from the device at 30 FPS.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/Project_Aegis.git
cd Project_Aegis
pip install -r requirements.txt
```

**2. Configure**

```bash
cp .env.example .env
```

Edit `.env`:

```env
INTAKE_MODE=phone           # phone | capture
PHONE_WS_PORT=8765
ANTHROPIC_API_KEY=sk-ant-...   # your Anthropic API key
CLAUDE_MODEL=claude-opus-4-6
ANALYSIS_INTERVAL_MS=2000   # how often to call the AI (ms)
```

**3. Run**

```bash
python main.py
```

Then open `http://<server-ip>:8765` on your phone (or any browser on the same network).

---

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Phone browser frontend |
| `GET /health` | Service status + frame/analysis timestamps |
| `GET /analysis` | Latest AI analysis (polling) |
| `WS /ws/intake` | Binary JPEG frame ingestion (phone mode) |
| `WS /ws/analysis` | Real-time AI analysis stream |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `INTAKE_MODE` | `phone` | `phone` or `capture` |
| `CAPTURE_DEVICE_INDEX` | `0` | cv2 device index (capture mode) |
| `PHONE_WS_PORT` | `8765` | Server port |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required) |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Model used for vision analysis |
| `ANALYSIS_INTERVAL_MS` | `2000` | AI analysis frequency in ms |
| `ANALYSIS_SYSTEM_PROMPT` | *(see .env.example)* | System prompt for the AI |

---

## Extending

- **New intake source** — subclass `BaseVideoIntake` in `intake/`, implement `start()`, `stop()`, `get_latest_frame()`
- **New AI provider** — subclass `BaseAIProvider` in `providers/`, implement `analyze_frame(frame) -> str`
- Swap providers in `main.py`'s `_build_processor()` factory

---

## Requirements

- Python 3.11+
- A second device with a camera (phone) **or** an HDMI capture card
- Anthropic API key
