"""Model client factory for Groq / DeepSeek via their OpenAI-compatible APIs."""

from __future__ import annotations

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


def _model_info(model: str) -> ModelInfo:
    reasoning = "reasoner" in model or "r1" in model.lower()
    return ModelInfo(
        vision=False,
        function_calling=not reasoning,
        json_output=not reasoning,
        family=ModelFamily.R1 if reasoning else ModelFamily.UNKNOWN,
        structured_output=False,
        multiple_system_messages=False,
    )


def make_model_client(settings: Settings | None = None, *, temperature: float = 0.0) -> OpenAIChatCompletionClient:
    """Build an AutoGen chat client for the configured provider.

    temperature=0 keeps Planner/Coder/Reviewer output as deterministic as the provider allows.
    """
    s = settings or get_settings()
    if not s.api_key:
        raise MissingAPIKeyError(
            f"{s.provider_cfg['key_env']} is not set. Add it to .env (see .env.example)."
        )
    return OpenAIChatCompletionClient(
        model=s.model,
        base_url=s.provider_cfg["base_url"],
        api_key=s.api_key,
        model_info=_model_info(s.model),
        temperature=temperature,
    )
