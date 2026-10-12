"""The chat-model interface the LLM planner and joiner use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


class ChatMessage(TypedDict):
    role: str
    content: str


class ProviderError(RuntimeError):
    pass


class ChatProvider(Protocol):
    name: str

    def stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Yield the assistant reply as text deltas."""
        ...


async def complete(
    provider: ChatProvider, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
) -> str:
    parts = [delta async for delta in provider.stream(messages, max_tokens=max_tokens)]
    return "".join(parts)
