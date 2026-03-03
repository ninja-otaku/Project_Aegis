import logging
import queue
import threading

logger = logging.getLogger(__name__)

_WORDS_PER_MINUTE = 175


class TTSEngine:
    """Speaks analysis recommendations aloud using the system TTS engine.

    Uses pyttsx3 (SAPI5 on Windows, espeak on Linux) in a dedicated daemon
    thread so speech never blocks the asyncio event loop.

    Queue design:
      - maxsize=1 — only the latest recommendation is ever queued.
      - If a new recommendation arrives while the previous is still queued
        (not yet spoken), the old one is replaced.  If speech is already in
        progress it finishes before the next item is spoken.

    Graceful degradation:
      - If pyttsx3 is not installed, all calls are silent no-ops.
      - If the TTS engine fails at runtime, the error is logged and the
        thread exits cleanly without crashing the main process.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._running = False
        self._available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        try:
            import pyttsx3  # noqa: F401 — availability check only
            self._available = True
        except ImportError:
            logger.warning(
                "pyttsx3 not installed — TTS disabled. "
                "Run: pip install pyttsx3"
            )
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="tts-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info("TTSEngine started.")

    def stop(self) -> None:
        self._running = False
        # Unblock the thread if it's waiting on the queue.
        try:
            self._queue.put_nowait("\x00")  # sentinel
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Queue text for speech, replacing any unspoken pending item."""
        if not self._available or not self._running:
            return
        # Drain the queue so the latest text always wins.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", _WORDS_PER_MINUTE)
        except Exception:
            logger.exception("TTSEngine: failed to initialise pyttsx3.")
            return

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text == "\x00":  # sentinel — exit
                break

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logger.exception("TTSEngine: speech error.")
