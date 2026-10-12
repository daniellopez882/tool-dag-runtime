"""Scheduler and parser overhead with no-op tools.

The tools return immediately, so the measured time is the runtime's own bookkeeping:
graph validation, readiness tracking, queueing under limits, the asyncio task per
attempt, argument resolution and binding, and the lifecycle events.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import TYPE_CHECKING, Any

from tooldag.bench.common import log
from tooldag.bench.replan import fixture_question
from tooldag.events import EventLog
from tooldag.plan.ast import Call, ListArg, Ref
from tooldag.plan.parser import StreamingPlanParser
from tooldag.plan.render import render_plan
from tooldag.planners import TemplatePlanner
from tooldag.rounds import PlanContext
from tooldag.scheduler import Limits, SequentialExecutor, TaskFetchingUnit
from tooldag.tools.base import Param, Tool, ToolRegistry
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder
from tooldag.tools.movies import MovieTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from tooldag.scheduler import RoundExecutor


def _noop(**_: Any) -> int:
    return 0


def _noop_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool("noop", "returns 0", (Param("inputs", list[Any]),), _noop, "int"))
    return registry


def _noop_copy(registry: ToolRegistry) -> ToolRegistry:
    """Same tool names and parameters as ``registry``, every function a no-op."""
    out = ToolRegistry()
    for tool in registry:
        params = tuple(Param(p.name, object, default=p.default) for p in tool.params)
        out.register(Tool(tool.name, tool.description, params, _noop, "int"))
    return out


def _independent(n: int) -> list[Call]:
    return [Call(i, "noop", (ListArg(()),)) for i in range(1, n + 1)]


def _chain(n: int) -> list[Call]:
    calls = [Call(1, "noop", (ListArg(()),))]
    calls += [Call(i, "noop", (ListArg((Ref(i - 1),)),)) for i in range(2, n + 1)]
    return calls


def _template_shape() -> tuple[list[Call], ToolRegistry]:
    """The real 34-task plan for a fixture question, with every tool replaced by a no-op."""
    q = fixture_question(0)
    tools = MovieTools(Catalog.fixture(), HashingEmbedder()).registry()
    ctx = PlanContext(q["question"], tuple(q["options"]), tools, 1)
    calls = [i for i in TemplatePlanner().build(ctx) if isinstance(i, Call)]
    return calls, _noop_copy(tools)


async def _drain_once(
    factory: Callable[[EventLog], RoundExecutor], calls: list[Call]
) -> tuple[float, int]:
    log_ = EventLog()
    unit = factory(log_)
    start = time.perf_counter()
    for call in calls:
        unit.submit(call)
    unit.end_of_plan()
    records = await unit.drain()
    elapsed = time.perf_counter() - start
    if not all(r.result == 0 for r in records.values()):
        raise RuntimeError("a no-op task did not succeed")
    return elapsed, len(log_)


def _time_executor(
    executor: str, calls: list[Call], registry: ToolRegistry, repeats: int
) -> dict[str, Any]:
    limits = Limits(max_concurrency=6, timeout_s=None)

    def factory(log_: EventLog) -> RoundExecutor:
        cls = TaskFetchingUnit if executor == "dag" else SequentialExecutor
        return cls(registry, log_, limits=limits)

    async def go() -> list[tuple[float, int]]:
        await _drain_once(factory, calls)  # warm-up
        return [await _drain_once(factory, calls) for _ in range(repeats)]

    timings = asyncio.run(go())
    per_task = [seconds / len(calls) * 1e6 for seconds, _ in timings]
    return {
        "us_per_task_median": round(statistics.median(per_task), 2),
        "us_per_task_min": round(min(per_task), 2),
        "events_per_task": round(timings[-1][1] / len(calls), 2),
        "repeats": repeats,
    }


def _parser_cost(text: str, repeats: int, chunk_chars: int = 7) -> dict[str, Any]:
    chunks = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    lines = text.count("\n")
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        parser = StreamingPlanParser()
        for chunk in chunks:
            parser.feed(chunk)
        parser.close()
        timings.append(time.perf_counter() - start)
    per_line = [t / lines * 1e6 for t in timings]
    return {
        "lines": lines,
        "chunk_chars": chunk_chars,
        "us_per_line_median": round(statistics.median(per_line), 2),
        "repeats": repeats,
    }


def bench_overhead(*, smoke: bool) -> dict[str, Any]:
    n = 200 if smoke else 2000
    repeats = 3 if smoke else 15
    template_calls, template_registry = _template_shape()
    noop = _noop_registry()
    shapes = [
        (f"{n} independent tasks", _independent(n), noop),
        (f"chain of {n} tasks", _chain(n), noop),
        (f"template plan shape ({len(template_calls)} tasks)", template_calls, template_registry),
    ]
    rows = []
    for name, calls, registry in shapes:
        for executor in ("dag", "sequential"):
            log(f"overhead: {name}, {executor}")
            timing = _time_executor(executor, calls, registry, repeats)
            rows.append({"shape": name, "executor": executor, "tasks": len(calls), **timing})
    plan_text = render_plan(template_calls)
    return {
        "benchmark": "overhead",
        "tools": "no-op functions run inline on the event loop",
        "max_concurrency": 6,
        "executors": rows,
        "parser": _parser_cost(plan_text, 50 if smoke else 500),
    }
