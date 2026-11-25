"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# OpenAI-compatible endpoints for the supported providers.
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("DATAPILOT_PROVIDER", "groq").lower())
    executor: str = field(default_factory=lambda: os.getenv("DATAPILOT_EXECUTOR", "local").lower())
    exec_timeout: int = field(default_factory=lambda: _int("DATAPILOT_EXEC_TIMEOUT", 60))
    max_messages: int = field(default_factory=lambda: _int("DATAPILOT_MAX_MESSAGES", 12))
    max_tokens_per_run: int = field(default_factory=lambda: _int("DATAPILOT_MAX_TOKENS_PER_RUN", 200_000))
    max_upload_mb: int = field(default_factory=lambda: _int("DATAPILOT_MAX_UPLOAD_MB", 50))
    team: str = field(default_factory=lambda: os.getenv("DATAPILOT_TEAM", "round_robin").lower())
    runs_dir: Path = field(default_factory=lambda: Path(os.getenv("DATAPILOT_RUNS_DIR") or PROJECT_ROOT / "runs"))
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def provider_cfg(self) -> dict[str, str]:
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Choose one of: {', '.join(PROVIDERS)}")
        return PROVIDERS[self.provider]

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.provider_cfg["key_env"]) or None

    @property
    def model(self) -> str:
        cfg = self.provider_cfg
        return os.getenv(cfg["model_env"]) or cfg["default_model"]


def get_settings() -> Settings:
    return Settings()
