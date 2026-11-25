"""Smoke-test the configured LLM provider: one tiny request through the AutoGen client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autogen_core.models import UserMessage  # noqa: E402

from datapilot.config import get_settings  # noqa: E402
from datapilot.llm import MissingAPIKeyError, make_model_client  # noqa: E402


async def main() -> int:
    s = get_settings()
    print(f"Provider: {s.provider}  Model: {s.model}  Endpoint: {s.provider_cfg['base_url']}")
    try:
        client = make_model_client(s)
    except MissingAPIKeyError as e:
        print(f"SKIP: {e}")
        return 2
    result = await client.create([UserMessage(content="Reply with exactly: PONG", source="user")])
    print(f"Response: {result.content!r}")
    print(f"Usage: prompt={result.usage.prompt_tokens} completion={result.usage.completion_tokens}")
    await client.close()
    return 0 if "PONG" in str(result.content).upper() else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
