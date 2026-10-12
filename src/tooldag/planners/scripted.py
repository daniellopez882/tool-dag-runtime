"""Planner that replays fixed plan texts, one per round, for fault-injection runs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tooldag.rounds import PlanContext, PlannerError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


class ScriptedPlanner:
    """Round ``k`` streams ``scripts[k-1]`` in ``chunk_chars``-sized pieces.

    Scripts can be malformed, cyclic or reference missing tasks on purpose; that is
    how the replan loop is exercised without a model.
    """

    name = "scripted"

    def __init__(self, scripts: Sequence[str], *, chunk_chars: int = 7) -> None:
        if chunk_chars < 1:
            raise ValueError("chunk_chars must be positive")
        self.scripts = list(scripts)
        self.chunk_chars = chunk_chars
        self.contexts: list[PlanContext] = []

    async def stream_plan(self, ctx: PlanContext) -> AsyncIterator[str]:
        self.contexts.append(ctx)
        if ctx.round > len(self.scripts):
            raise PlannerError(f"no scripted plan for round {ctx.round}")
        text = self.scripts[ctx.round - 1]
        for start in range(0, len(text), self.chunk_chars):
            await asyncio.sleep(0)
            yield text[start : start + self.chunk_chars]
