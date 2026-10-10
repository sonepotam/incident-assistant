"""Thin provider adapters. The agent depends only on the `LLMClient` protocol."""

from __future__ import annotations

import logging
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}


class LLMError(Exception):
    """Transport/provider failure (after the SDK's own retries)."""


class LLMClient(Protocol):
    name: str

    def complete(self, stage: str, system: str, messages: list[Message]) -> str:
        """Return the raw text answer. `stage` is used for logging and by test doubles."""
        ...


class AnthropicClient:
    def __init__(self, settings: Settings):
        import anthropic

        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s, max_retries=2
        )
        self._settings = settings
        self.name = f"anthropic:{settings.anthropic_model}"

    def complete(self, stage: str, system: str, messages: list[Message]) -> str:
        log.debug("anthropic call stage=%s messages=%d", stage, len(messages))
        try:
            resp = self._client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=self._settings.max_tokens,
                # anthropic SDK 1.x Messages.create has no `temperature` argument
                system=system,
                messages=messages,
            )
        except Exception as exc:  # SDK raises several exception types; normalise them
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient:
    """Also works with any OpenAI-compatible endpoint (Ollama, vLLM, ...) via OPENAI_BASE_URL."""

    def __init__(self, settings: Settings):
        import openai

        if not settings.openai_api_key and not settings.openai_base_url:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_s,
            max_retries=2,
        )
        self._settings = settings
        self.name = f"openai:{settings.openai_model}"

    def complete(self, stage: str, system: str, messages: list[Message]) -> str:
        log.debug("openai call stage=%s messages=%d", stage, len(messages))
        try:
            resp = self._client.chat.completions.create(
                model=self._settings.openai_model,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
                messages=[{"role": "system", "content": system}, *messages],
            )
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc
        return resp.choices[0].message.content or ""


def build_client(settings: Settings) -> LLMClient:
    provider = settings.resolved_provider()
    if provider == "anthropic":
        return AnthropicClient(settings)
    if provider == "openai":
        return OpenAIClient(settings)
    if provider == "mock":
        from .mock_llm import MockLLM

        return MockLLM()
    raise LLMError(f"Unknown LLM_PROVIDER {provider!r} (expected auto|anthropic|openai|mock)")
