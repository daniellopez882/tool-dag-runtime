"""Deterministic provider for tests and offline runs: replays scripted replies."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tooldag.providers.base import ProviderError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tooldag.providers.base import ChatMessage


class FakeProvider:
    """Returns ``replies`` in order, each streamed in ``chunk_chars``-sized deltas."""

    name = "fake (scripted replies)"

    def __init__(self, replies: Sequence[str], *, chunk_chars: int = 5) -> None:
        self._replies = list(replies)
        self.chunk_chars = chunk_chars
        self.calls: list[list[ChatMessage]] = []

    async def stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        if not self._replies:
            raise ProviderError("fake provider has no scripted reply left")
        reply = self._replies.pop(0)
        for start in range(0, len(reply), self.chunk_chars):
            await asyncio.sleep(0)
            yield reply[start : start + self.chunk_chars]
