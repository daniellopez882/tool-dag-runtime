"""Fault scenarios for the joiner/replan loop, run offline on the bundled fixture.

Each scenario feeds the runtime a plan with one specific defect (a malformed line, a
cycle, a dangling reference, an unknown tool, a flaky tool, a title the catalogue does
not know) and records how many rounds, executed tasks, reused tasks and retries it took
to reach an answer. Model calls are counted structurally; no model is run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import TYPE_CHECKING, Any

from tooldag.events import EventLog
from tooldag.joiner import RuleJoiner
from tooldag.plan.ast import Call, ListArg, Ref
from tooldag.plan.render import render_item, render_plan
from tooldag.planners import ScriptedPlanner, TemplatePlanner
from tooldag.rounds import PlanContext
from tooldag.runtime import Runtime, RuntimeConfig
from tooldag.scheduler import Limits, RetryPolicy
from tooldag.tools.base import ToolRegistry, TransientToolError
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder
from tooldag.tools.movies import MovieTools

if TYPE_CHECKING:
    from collections.abc import Callable

    from tooldag.plan.ast import PlanItem
    from tooldag.rounds import Planner

FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.005)


def fixture_question(n: int = 0) -> dict[str, Any]:
    raw = resources.files("tooldag.fixtures").joinpath("mini_catalog.json").read_text("utf-8")
    return dict(json.loads(raw)["questions"][n])


def _clean_items(tools: ToolRegistry, question: str, options: tuple[str, ...]) -> list[PlanItem]:
    ctx = PlanContext(question, options, tools, 1)
    return TemplatePlanner().build(ctx)


def _first(items: list[PlanItem], tool: str) -> Call:
    return next(i for i in items if isinstance(i, Call) and i.tool == tool)


def _replace_call(items: list[PlanItem], new: Call) -> list[PlanItem]:
    return [new if isinstance(i, Call) and i.idx == new.idx else i for i in items]


def _malformed(items: list[PlanItem]) -> str:
    target = _first(items, "cooccurrence")
    lines = []
    for item in items:
        text = render_item(item)
        lines.append(text[:-1] if item is target else text)  # drop the closing parenthesis
    return "\n".join(lines) + "\n"


def _cyclic(items: list[PlanItem]) -> str:
    co = _first(items, "cooccurrence")
    gj = _first(items, "genre_jaccard")
    # cooccurrence now waits on genre_jaccard and vice versa.
    items = _replace_call(items, dataclasses.replace(co, args=(co.args[0], Ref(gj.idx))))
    items = _replace_call(items, dataclasses.replace(gj, args=(gj.args[0], Ref(co.idx))))
    return render_plan(items)


def _dangling(items: list[PlanItem]) -> str:
    rank = _first(items, "rank")
    name, value = rank.kwargs[0]
    assert isinstance(value, ListArg)
    broken = ListArg((*value.items[:-1], Ref(99)))
    kwargs = ((name, broken), *rank.kwargs[1:])
    return render_plan(_replace_call(items, dataclasses.replace(rank, kwargs=kwargs)))


def _unknown_tool(items: list[PlanItem]) -> str:
    gj = _first(items, "genre_jaccard")
    return render_plan(_replace_call(items, Call(gj.idx, "imdb_rating", (gj.args[1],))))


def _flaky(registry: ToolRegistry, tool_name: str, failures: int) -> ToolRegistry:
    """Copy of ``registry`` whose ``tool_name`` raises a transient error ``failures`` times."""
    remaining = {"n": failures}
    out = ToolRegistry()
    for tool in registry:
        if tool.name != tool_name:
            out.register(tool)
            continue
        inner = tool.fn

        def flaky(*args: Any, _inner: Callable[..., Any] = inner, **kwargs: Any) -> Any:
            if remaining["n"] > 0:
                remaining["n"] -= 1
                raise TransientToolError("injected transient failure")
            return _inner(*args, **kwargs)

        out.register(dataclasses.replace(tool, fn=flaky))
    return out


async def _run(
    tools: ToolRegistry,
    planner: Planner,
    question: str,
    options: tuple[str, ...],
    pool: ThreadPoolExecutor,
    max_rounds: int = 3,
) -> dict[str, Any]:
    runtime = Runtime(
        tools,
        planner,
        RuleJoiner(),
        RuntimeConfig(limits=Limits(max_concurrency=4), retry=FAST_RETRY, max_rounds=max_rounds),
        executor=pool,
    )
    log = EventLog("scenario")
    result = await runtime.run(question, options, log=log)
    counts: dict[str, int] = {}
    for event in log.events:
        counts[event.type] = counts.get(event.type, 0) + 1
    return {
        "answer": result.answer,
        "rounds": len(result.rounds),
        "tasks_executed": result.tasks_executed,
        "tasks_reused": result.tasks_reused,
        "retries": counts.get("task_retry", 0),
        "parse_issues": counts.get("plan_issue", 0),
        "rejected": counts.get("task_rejected", 0),
        "cancelled": counts.get("task_cancelled", 0),
        "model_calls": result.model_calls(),
        "final_reason": result.decision.reason if result.decision else None,
    }


async def _scenarios() -> list[dict[str, Any]]:
    catalog = Catalog.fixture()
    movie_tools = MovieTools(catalog, HashingEmbedder())
    tools = movie_tools.registry()
    q = fixture_question(0)
    question, options, expected = q["question"], tuple(q["options"]), q["answer"]
    clean = _clean_items(tools, question, options)
    clean_text = render_plan(clean)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        cases: list[tuple[str, str, ToolRegistry, Planner, str, int, str | None]] = [
            ("clean template plan", "-", tools, TemplatePlanner(), question, 3, expected),
            (
                "malformed line",
                "round 1 has a call without its closing parenthesis",
                tools,
                ScriptedPlanner([_malformed(clean), clean_text]),
                question,
                3,
                expected,
            ),
            (
                "dependency cycle",
                "two signal tasks wait on each other",
                tools,
                ScriptedPlanner([_cyclic(clean), clean_text]),
                question,
                3,
                expected,
            ),
            (
                "dangling reference",
                "rank() reads $99, which is never defined",
                tools,
                ScriptedPlanner([_dangling(clean), clean_text]),
                question,
                3,
                expected,
            ),
            (
                "unknown tool",
                "one call names a tool that does not exist",
                tools,
                ScriptedPlanner([_unknown_tool(clean), clean_text]),
                question,
                3,
                expected,
            ),
            (
                "transient tool failure",
                "embed() fails twice with a retryable error",
                _flaky(tools, "embed", 2),
                TemplatePlanner(),
                question,
                3,
                expected,
            ),
            (
                "unknown title",
                "the question names a film the catalogue lacks",
                tools,
                TemplatePlanner(),
                question.replace(
                    "Find a movie similar to ", "Find a movie similar to Nowhere Man, "
                ),
                3,
                expected,
            ),
            (
                "never recovers",
                "every round's plan is malformed; the joiner gives up",
                tools,
                ScriptedPlanner([_malformed(clean)] * 3),
                question,
                3,
                None,
            ),
        ]
        for name, defect, registry, planner, text, max_rounds, want in cases:
            row = await _run(registry, planner, text, options, pool, max_rounds)
            rows.append({"scenario": name, "defect": defect, "expected_answer": want, **row})
    return rows


def run_scenarios() -> list[dict[str, Any]]:
    return asyncio.run(_scenarios())


def bench_replan() -> dict[str, Any]:
    rows = run_scenarios()
    return {
        "benchmark": "replan",
        "catalog": "fixture (12 invented films, hashing embedder)",
        "planner": "template and scripted (no model run)",
        "joiner": "rule",
        "scenarios": rows,
        "all_expected": all(r["answer"] == r["expected_answer"] for r in rows),
    }
