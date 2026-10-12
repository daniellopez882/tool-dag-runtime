"""Streaming client for OpenAI-compatible ``/chat/completions`` endpoints.

Works with any server that speaks that API (OpenAI, Ollama, vLLM, LM Studio, llama.cpp
server). The model name, base URL and key come from the caller; nothing is defaulted.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import aiohttp

from tooldag.providers.base import ProviderError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tooldag.providers.base import ChatMessage

ENV_BASE_URL = "TOOLDAG_LLM_BASE_URL"
ENV_MODEL = "TOOLDAG_LLM_MODEL"
ENV_API_KEY = "TOOLDAG_LLM_API_KEY"


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.name = f"openai-compatible ({self.model} at {self.base_url})"

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        base_url = os.environ.get(ENV_BASE_URL)
        model = os.environ.get(ENV_MODEL)
        if not base_url or not model:
            raise ProviderError(
                f"set {ENV_BASE_URL} (e.g. http://localhost:11434/v1) and {ENV_MODEL}"
            )
        return cls(base_url, model, os.environ.get(ENV_API_KEY))

    async def stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        async with (
            aiohttp.ClientSession(timeout=self.timeout) as session,
            session.post(url, json=payload, headers=headers) as response,
        ):
            if response.status != 200:
                body = (await response.text())[:300]
                raise ProviderError(f"{url} returned HTTP {response.status}: {body}")
            async for raw in response.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"malformed stream chunk: {data[:120]}") from exc
                if "error" in chunk:
                    raise ProviderError(f"provider error: {chunk['error']}")
                for choice in chunk.get("choices", ()):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta
