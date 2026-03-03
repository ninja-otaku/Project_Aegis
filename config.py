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

    # ── AI provider ───────────────────────────────────────────────────────────
    # ANTHROPIC_API_KEY is read directly by the anthropic SDK from the
    # environment; no need to declare it here.
    CLAUDE_MODEL: str = "claude-opus-4-6"

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

    # ── TTS ───────────────────────────────────────────────────────────────────
    # Speak the recommendation aloud via the system TTS engine (pyttsx3).
    TTS_ENABLED: bool = False

    # ── History ───────────────────────────────────────────────────────────────
    # Number of past analyses to retain in memory (accessible via GET /history).
    HISTORY_MAX_ENTRIES: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # allow ANTHROPIC_API_KEY etc. in .env without declaring them
    )


settings = Settings()
