"""Append-only, replayable event log for one run.

Every lifecycle transition (plan line parsed, task started, retried, finished, joiner
decision) becomes an ``Event`` with a sequence number and a run-relative timestamp.
The SSE endpoint replays the log from any ``Last-Event-ID`` and then follows it live.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

PREVIEW_CHARS = 80
PREVIEW_ITEMS = 6


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    t_ms: float
    type: str
    round: int | None = None
    data: dict[str, Any] = {}


class EventLog:
    def __init__(self, run_id: str = "", clock: Callable[[], float] = time.perf_counter) -> None:
        self.run_id = run_id
        self._clock = clock
        self._t0 = clock()
        self._events: list[Event] = []
        self._waiters: set[asyncio.Future[None]] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def elapsed_ms(self) -> float:
        return (self._clock() - self._t0) * 1000.0

    def emit(self, type_: str, round_: int | None = None, **data: Any) -> Event:
        if self._closed:
            raise RuntimeError("event log is closed")
        event = Event(
            seq=len(self._events) + 1,
            t_ms=round(self.elapsed_ms(), 3),
            type=type_,
            round=round_,
            data=data,
        )
        self._events.append(event)
        self._wake()
        return event

    def close(self) -> None:
        self._closed = True
        self._wake()

    def _wake(self) -> None:
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    async def follow(self, after_seq: int = 0) -> AsyncIterator[Event]:
        """Yield events with ``seq > after_seq``, then live ones until the log closes."""
        position = max(after_seq, 0)
        while True:
            while position < len(self._events):
                yield self._events[position]
                position += 1
            if self._closed:
                return
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.discard(waiter)


def preview(value: Any, depth: int = 0) -> Any:
    """Compact JSON-safe summary of a tool result or argument for an event payload."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) <= PREVIEW_CHARS:
            return value
        return f"{value[:PREVIEW_CHARS]}... ({len(value)} chars)"
    custom = getattr(value, "preview", None)
    if callable(custom):
        return custom()
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list | tuple):
        items = [preview(v, depth + 1) for v in value[:PREVIEW_ITEMS]]
        if len(value) > PREVIEW_ITEMS:
            items.append(f"... (+{len(value) - PREVIEW_ITEMS})")
        return items
    if isinstance(value, dict):
        keys = list(value)[:PREVIEW_ITEMS]
        return {str(k): preview(value[k], depth + 1) for k in keys}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: preview(getattr(value, f.name), depth + 1) for f in dataclasses.fields(value)
        }
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": list(shape), "dtype": str(getattr(value, "dtype", ""))}
    text = repr(value)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."
