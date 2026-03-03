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
        "You are an expert AI gaming companion with real-time game-state "
        "awareness. Analyse the provided screenshot and concisely describe: "
        "(1) current game state, (2) immediate threats or opportunities, "
        "(3) a recommended action. Be direct and tactical."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
