"""OpenAI-compatible adapter against a local stand-in server (no network)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tooldag.providers import OpenAICompatibleProvider, ProviderError, complete


def _chunk(text: str) -> bytes:
    body = {"choices": [{"index": 0, "delta": {"content": text}}]}
    return f"data: {json.dumps(body)}\n\n".encode()


def make_app(seen: list[dict[str, Any]], mode: str) -> web.Application:
    async def chat(request: web.Request) -> web.StreamResponse:
        seen.append({"body": await request.json(), "auth": request.headers.get("Authorization")})
        if mode == "http_error":
            return web.json_response({"error": "bad model"}, status=404)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b": comment lines are ignored\n\n")
        await response.write(_chunk("1. resolve"))
        await response.write(b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n')
        if mode == "stream_error":
            await response.write(b'data: {"error": {"message": "overloaded"}}\n\n')
        await response.write(_chunk('_title("Heat")\n'))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app


def call(mode: str, api_key: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    async def go() -> str:
        async with TestServer(make_app(seen, mode)) as server:
            provider = OpenAICompatibleProvider(
                str(server.make_url("/v1")), "local-model", api_key
            )
            return await complete(provider, [{"role": "user", "content": "hi"}], max_tokens=50)

    return asyncio.run(go()), seen


def test_streams_deltas_and_sends_the_request_shape() -> None:
    text, seen = call("ok", api_key="secret")
    assert text == '1. resolve_title("Heat")\n'
    body = seen[0]["body"]
    assert body["stream"] is True
    assert body["model"] == "local-model"
    assert body["max_tokens"] == 50
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert seen[0]["auth"] == "Bearer secret"


def test_no_key_means_no_authorization_header() -> None:
    _, seen = call("ok")
    assert seen[0]["auth"] is None


@pytest.mark.parametrize(
    ("mode", "message"), [("http_error", "HTTP 404"), ("stream_error", "overloaded")]
)
def test_errors_become_provider_errors(mode: str, message: str) -> None:
    with pytest.raises(ProviderError, match=message):
        call(mode)


def test_from_env_requires_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLDAG_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TOOLDAG_LLM_MODEL", raising=False)
    with pytest.raises(ProviderError, match="TOOLDAG_LLM_BASE_URL"):
        OpenAICompatibleProvider.from_env()
    monkeypatch.setenv("TOOLDAG_LLM_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("TOOLDAG_LLM_MODEL", "some-model")
    provider = OpenAICompatibleProvider.from_env()
    assert provider.base_url == "http://localhost:11434/v1"
