from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class IntakeMode(str, Enum):
    PHONE = "phone"
    CAPTURE = "capture"


class Settings(BaseSettings):
    # ── Intake ────────────────────────────────────────────────────────────────
    INTAKE_MODE: IntakeMode = IntakeMode.PHONE
    CAPTURE_DEVICE_INDEX: int = 0
    PHONE_WS_PORT: int = 8765

    # ── Phone frame compression ───────────────────────────────────────────────
    # Resize incoming phone frames to this width (aspect ratio preserved) before
    # CLAHE and storage.  Lower = cheaper pipeline; 640 is a good default.
    PHONE_FRAME_WIDTH: int = 640
    # JPEG quality used when re-encoding the preprocessed frame for storage.
    # Lower = smaller memory footprint; 70 balances quality and size.
    PHONE_COMPRESS_QUALITY: int = 70

    # ── AI provider ───────────────────────────────────────────────────────────
    # Which provider to use: claude | gemini | openai | ollama
    AI_PROVIDER: str = "claude"

    # Claude (Anthropic) — ANTHROPIC_API_KEY is read by the SDK from env
    CLAUDE_MODEL: str = "claude-opus-4-5"

    # Gemini (Google)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama (local, no API key required — run: ollama pull llava)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llava"

    # How often the frame processor calls the AI provider (milliseconds).
    # Lower = more responsive, higher = cheaper.
    ANALYSIS_INTERVAL_MS: int = 2000

    # System prompt sent with every vision request.
    ANALYSIS_SYSTEM_PROMPT: str = (
        "You are an expert AI gaming companion. Analyse the game screenshot "
        "and return a JSON object with: game_state (brief current situation), "
        "threats (list of immediate threats or opportunities), recommendation "
        "(the single most important action to take right now), and confidence "
        "(low/medium/high). Be direct, tactical, and concise."
    )

    # ── Frame diff (API cost optimization) ───────────────────────────────────
    # Skip the AI call when the scene hasn't changed enough between frames.
    # 0.02 = ~2% pixel change — tune up for fast-paced games (lots of motion),
    # down for slow / turn-based games where small changes matter.
    FRAME_DIFF_ENABLED: bool = True
    FRAME_DIFF_THRESHOLD: float = 0.02

    # ── TLS ───────────────────────────────────────────────────────────────────
    # Enable HTTPS (required for camera access on most phone browsers).
    # Generate certs first: python scripts/generate_cert.py
    TLS_ENABLED: bool = False
    TLS_CERT_PATH: str = "certs/cert.pem"
    TLS_KEY_PATH: str = "certs/key.pem"

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_ENABLED: bool = False

    # ── History ───────────────────────────────────────────────────────────────
    HISTORY_MAX_ENTRIES: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # allow ANTHROPIC_API_KEY etc. in .env without declaring them
    )


settings = Settings()
