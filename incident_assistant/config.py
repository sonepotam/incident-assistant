"""Configuration loaded from environment variables (optionally from a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent / "data"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    provider: str  # auto | anthropic | openai | mock
    anthropic_api_key: str | None
    anthropic_model: str
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None
    temperature: float
    max_tokens: int
    request_timeout_s: float
    max_attempts: int
    knowledge_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=_env("OPENAI_BASE_URL"),
            temperature=float(_env("LLM_TEMPERATURE", "0")),
            max_tokens=int(_env("LLM_MAX_TOKENS", "1500")),
            request_timeout_s=float(_env("LLM_TIMEOUT_S", "60")),
            max_attempts=int(_env("AGENT_MAX_ATTEMPTS", "3")),
            knowledge_dir=Path(_env("KNOWLEDGE_DIR", str(DEFAULT_KNOWLEDGE_DIR))),
        )

    def resolved_provider(self) -> str:
        """'auto' picks the first provider that has an API key, otherwise the offline mock."""
        if self.provider != "auto":
            return self.provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "mock"
