"""Smoke tests for AI provider implementations.

All SDK calls are mocked — no real API keys or network connections are required.
Run with:  pytest tests/
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from providers.base import BaseAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 8, w: int = 8) -> np.ndarray:
    """Return a small solid-colour BGR frame for provider tests."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Interface tests — every provider must be a concrete BaseAIProvider subclass
# ---------------------------------------------------------------------------

def test_claude_provider_is_subclass():
    from providers.claude_vision import ClaudeVisionProvider
    assert issubclass(ClaudeVisionProvider, BaseAIProvider)


def test_gemini_provider_is_subclass():
    from providers.gemini_vision import GeminiVisionProvider
    assert issubclass(GeminiVisionProvider, BaseAIProvider)


def test_openai_provider_is_subclass():
    from providers.openai_vision import OpenAIVisionProvider
    assert issubclass(OpenAIVisionProvider, BaseAIProvider)


def test_ollama_provider_is_subclass():
    from providers.ollama_vision import OllamaVisionProvider
    assert issubclass(OllamaVisionProvider, BaseAIProvider)


def test_mistral_provider_is_subclass():
    from providers.mistral_vision import MistralVisionProvider
    assert issubclass(MistralVisionProvider, BaseAIProvider)


def test_groq_provider_is_subclass():
    from providers.groq_vision import GroqVisionProvider
    assert issubclass(GroqVisionProvider, BaseAIProvider)


def test_all_providers_have_analyze_frame():
    """Each provider must expose the analyze_frame coroutine method."""
    from providers.claude_vision import ClaudeVisionProvider
    from providers.gemini_vision import GeminiVisionProvider
    from providers.openai_vision import OpenAIVisionProvider
    from providers.ollama_vision import OllamaVisionProvider
    from providers.mistral_vision import MistralVisionProvider
    from providers.groq_vision import GroqVisionProvider

    for cls in (ClaudeVisionProvider, GeminiVisionProvider,
                OpenAIVisionProvider, OllamaVisionProvider,
                MistralVisionProvider, GroqVisionProvider):
        assert callable(getattr(cls, "analyze_frame", None)), (
            f"{cls.__name__} is missing analyze_frame()"
        )


# ---------------------------------------------------------------------------
# Claude — exception propagation
# ---------------------------------------------------------------------------

def test_claude_provider_propagates_sdk_exceptions():
    """ClaudeVisionProvider must not wrap SDK exceptions in a bare Exception."""

    async def run():
        # Patch anthropic.AsyncAnthropic so __init__ succeeds without a real key.
        with patch("providers.claude_vision.anthropic.AsyncAnthropic"):
            from providers.claude_vision import ClaudeVisionProvider
            provider = ClaudeVisionProvider(model="test", system_prompt="test")

            # Replace the client with one whose stream context manager raises.
            original_exc = RuntimeError("sdk failure")
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(side_effect=original_exc)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            provider._client = MagicMock()
            provider._client.messages.stream.return_value = mock_ctx

            caught = None
            try:
                await provider.analyze_frame(_blank_frame())
            except Exception as exc:
                caught = exc

            assert caught is original_exc, (
                "Expected the original exception to propagate unchanged"
            )
            assert type(caught) is not Exception, (
                "Exception was wrapped in a bare Exception — should propagate typed"
            )

    asyncio.run(run())


def test_claude_provider_accepts_system_prompt_override():
    """analyze_frame must accept the optional system_prompt keyword argument."""

    async def run():
        with patch("providers.claude_vision.anthropic.AsyncAnthropic"):
            from providers.claude_vision import ClaudeVisionProvider
            provider = ClaudeVisionProvider(model="test", system_prompt="default")

            # Mock the stream to return a minimal valid response.
            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text='{"game_state":"ok","threats":[],"recommendation":"hold","confidence":"high"}')]
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.get_final_message = AsyncMock(return_value=mock_response)
            provider._client = MagicMock()
            provider._client.messages.stream.return_value = mock_ctx

            # Should not raise when system_prompt kwarg is passed.
            result = await provider.analyze_frame(
                _blank_frame(), system_prompt="override prompt"
            )
            assert isinstance(result, dict)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Ollama — exception propagation (no API key needed)
# ---------------------------------------------------------------------------

def test_ollama_provider_propagates_sdk_exceptions():
    """OllamaVisionProvider must propagate SDK exceptions without wrapping."""

    async def run():
        from providers.ollama_vision import OllamaVisionProvider
        provider = OllamaVisionProvider()  # ollama client requires no API key

        original_exc = ConnectionRefusedError("ollama not running")
        provider._client = MagicMock()
        provider._client.chat = AsyncMock(side_effect=original_exc)

        caught = None
        try:
            await provider.analyze_frame(_blank_frame())
        except Exception as exc:
            caught = exc

        assert caught is original_exc
        assert type(caught) is not Exception

    asyncio.run(run())


def test_ollama_provider_accepts_system_prompt_override():
    """analyze_frame must accept the optional system_prompt keyword argument."""

    async def run():
        from providers.ollama_vision import OllamaVisionProvider
        provider = OllamaVisionProvider()

        mock_response = MagicMock()
        mock_response.message.content = (
            '{"game_state":"ok","threats":[],"recommendation":"push","confidence":"medium"}'
        )
        provider._client = MagicMock()
        provider._client.chat = AsyncMock(return_value=mock_response)

        result = await provider.analyze_frame(
            _blank_frame(), system_prompt="custom prompt"
        )
        assert isinstance(result, dict)
        # Verify the custom prompt was passed as the system message.
        call_args = provider._client.chat.call_args
        # Ollama chat is called with keyword arguments only — messages lives in .kwargs.
        messages = call_args.kwargs.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert any(m["content"] == "custom prompt" for m in system_messages), (
            "system_prompt override was not passed to the Ollama chat call"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# OpenAI — exception propagation (patched settings + SDK)
# ---------------------------------------------------------------------------

def test_openai_provider_propagates_sdk_exceptions():
    """OpenAIVisionProvider must propagate SDK exceptions without wrapping."""

    async def run():
        import config

        # Provide a fake API key so the provider doesn't raise ValueError.
        fake_settings = config.Settings(OPENAI_API_KEY="fake-key")

        # Patch the openai module as it's imported locally inside __init__.
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}), \
             patch("config.settings", fake_settings):
            # Re-import with patched sys.modules so the local import picks up the mock.
            if "providers.openai_vision" in sys.modules:
                del sys.modules["providers.openai_vision"]
            from providers.openai_vision import OpenAIVisionProvider
            provider = OpenAIVisionProvider()

            original_exc = RuntimeError("openai sdk error")
            provider._client.chat.completions.create = AsyncMock(side_effect=original_exc)

            caught = None
            try:
                await provider.analyze_frame(_blank_frame())
            except Exception as exc:
                caught = exc

            assert caught is original_exc
            assert type(caught) is not Exception

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Gemini — exception propagation (patched settings + SDK)
# ---------------------------------------------------------------------------


def test_gemini_provider_propagates_sdk_exceptions():
    """GeminiVisionProvider must propagate SDK exceptions without wrapping."""

    async def run():
        import config

        fake_settings = config.Settings(GEMINI_API_KEY="fake-key")

        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        # google.generativeai is imported as `genai` inside __init__,
        # so we patch the entire google.generativeai entry in sys.modules.
        mock_google = MagicMock()
        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.generativeai": mock_genai,
        }), patch("config.settings", fake_settings):
            if "providers.gemini_vision" in sys.modules:
                del sys.modules["providers.gemini_vision"]
            from providers.gemini_vision import GeminiVisionProvider
            provider = GeminiVisionProvider()

            original_exc = RuntimeError("gemini api error")
            provider._model.generate_content_async = AsyncMock(side_effect=original_exc)

            caught = None
            try:
                await provider.analyze_frame(_blank_frame())
            except Exception as exc:
                caught = exc

            assert caught is original_exc
            assert type(caught) is not Exception

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Mistral — exception propagation (patched settings + SDK)
# ---------------------------------------------------------------------------

def test_mistral_provider_propagates_sdk_exceptions():
    """MistralVisionProvider must propagate SDK exceptions without wrapping."""

    async def run():
        import config

        fake_settings = config.Settings(MISTRAL_API_KEY="fake-key")

        mock_mistral = MagicMock()
        mock_client = MagicMock()
        mock_mistral.Mistral.return_value = mock_client

        with patch.dict(sys.modules, {"mistralai": mock_mistral}), \
             patch("config.settings", fake_settings):
            if "providers.mistral_vision" in sys.modules:
                del sys.modules["providers.mistral_vision"]
            from providers.mistral_vision import MistralVisionProvider
            provider = MistralVisionProvider()

            original_exc = RuntimeError("mistral api error")
            provider._client.chat.complete_async = AsyncMock(side_effect=original_exc)

            caught = None
            try:
                await provider.analyze_frame(_blank_frame())
            except Exception as exc:
                caught = exc

            assert caught is original_exc
            assert type(caught) is not Exception

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Groq — exception propagation (patched settings + SDK)
# ---------------------------------------------------------------------------

def test_groq_provider_propagates_sdk_exceptions():
    """GroqVisionProvider must propagate SDK exceptions without wrapping."""

    async def run():
        import config

        fake_settings = config.Settings(GROQ_API_KEY="fake-key")

        mock_groq = MagicMock()
        mock_client = MagicMock()
        mock_groq.AsyncGroq.return_value = mock_client

        with patch.dict(sys.modules, {"groq": mock_groq}), \
             patch("config.settings", fake_settings):
            if "providers.groq_vision" in sys.modules:
                del sys.modules["providers.groq_vision"]
            from providers.groq_vision import GroqVisionProvider
            provider = GroqVisionProvider()

            original_exc = RuntimeError("groq api error")
            provider._client.chat.completions.create = AsyncMock(side_effect=original_exc)

            caught = None
            try:
                await provider.analyze_frame(_blank_frame())
            except Exception as exc:
                caught = exc

            assert caught is original_exc
            assert type(caught) is not Exception

    asyncio.run(run())
