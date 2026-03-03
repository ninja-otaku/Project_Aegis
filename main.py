import asyncio
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import IntakeMode, settings
from engine import FrameProcessor, TTSEngine
from intake import BaseVideoIntake, CaptureCardIntake, PhoneBrowserIntake
from providers.claude_vision import ClaudeVisionProvider

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _build_intake() -> BaseVideoIntake:
    if settings.INTAKE_MODE == IntakeMode.PHONE:
        return PhoneBrowserIntake()
    return CaptureCardIntake(device_index=settings.CAPTURE_DEVICE_INDEX)


def _build_tts() -> TTSEngine | None:
    if not settings.TTS_ENABLED:
        return None
    tts = TTSEngine()
    tts.start()
    return tts


def _build_processor(intake: BaseVideoIntake, tts: TTSEngine | None) -> FrameProcessor:
    provider = ClaudeVisionProvider(
        model=settings.CLAUDE_MODEL,
        system_prompt=settings.ANALYSIS_SYSTEM_PROMPT,
    )
    return FrameProcessor(
        intake=intake,
        provider=provider,
        interval_ms=settings.ANALYSIS_INTERVAL_MS,
        history_max=settings.HISTORY_MAX_ENTRIES,
        tts=tts,
    )


intake: BaseVideoIntake = _build_intake()
tts: TTSEngine | None = _build_tts()
processor: FrameProcessor = _build_processor(intake, tts)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await intake.start()
    await processor.start()
    yield
    await processor.stop()
    await intake.stop()
    if tts:
        tts.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Project Aegis", version="0.3.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes — static + REST
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    frame_ts: datetime | None = intake.get_frame_timestamp()
    latest = processor.get_latest()
    return JSONResponse({
        "status": "ok",
        "intake_mode": settings.INTAKE_MODE.value,
        "tts_enabled": settings.TTS_ENABLED,
        "latest_frame_timestamp": frame_ts.isoformat() if frame_ts else None,
        "latest_analysis_timestamp": latest.get("timestamp") if latest else None,
        "history_count": len(processor.get_history()),
        "server_time": datetime.now(tz=timezone.utc).isoformat(),
    })


@app.get("/analysis")
async def analysis() -> JSONResponse:
    """Latest AI analysis result (polling-friendly)."""
    latest = processor.get_latest()
    if latest is None:
        return JSONResponse({"game_state": None, "threats": [], "recommendation": None, "confidence": None, "timestamp": None})
    return JSONResponse(latest)


@app.get("/history")
async def history() -> JSONResponse:
    """All retained analysis results, newest first."""
    return JSONResponse({"history": processor.get_history()})


# ---------------------------------------------------------------------------
# WebSocket — intake (phone mode only)
# ---------------------------------------------------------------------------

if settings.INTAKE_MODE == IntakeMode.PHONE:
    assert isinstance(intake, PhoneBrowserIntake)

    @app.websocket("/ws/intake")
    async def ws_intake(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                await intake.process_frame(data)
        except WebSocketDisconnect:
            pass


# ---------------------------------------------------------------------------
# WebSocket — analysis results (all modes)
# ---------------------------------------------------------------------------

@app.websocket("/ws/analysis")
async def ws_analysis(websocket: WebSocket) -> None:
    """Push structured AI analysis results to the client as they arrive.

    On connect: immediately sends the latest cached result (if any).
    Ongoing:    pushed whenever the processor completes a new analysis.
    Keepalive:  {"type": "keepalive"} every 30 s to prevent proxy timeouts.
    """
    await websocket.accept()
    q = processor.subscribe()
    try:
        latest = processor.get_latest()
        if latest:
            await websocket.send_json(latest)

        while True:
            try:
                result = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_json(result)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        pass
    finally:
        processor.unsubscribe(q)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PHONE_WS_PORT,
        reload=False,
    )
