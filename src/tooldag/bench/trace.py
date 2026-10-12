"""Capture one real SSE event stream from the HTTP service, for the README walkthrough.

The service is started on a loopback port, one question is posted to ``/v1/ask`` and
the raw ``text/event-stream`` body is stored as received, next to a per-task timing
table derived from the same events.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from tooldag.bench.common import log
from tooldag.bigbench import load_questions, split_fit_eval
from tooldag.fetch import asset_path
from tooldag.server import ServiceState, create_app
from tooldag.service import make_joiner, make_planner

if TYPE_CHECKING:
    from pathlib import Path

    from tooldag.service import Domain


def parse_sse(body: str) -> list[dict[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


def task_table(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per task: when its line was parsed, when it started and when it finished (ms)."""
    rows: dict[int, dict[str, Any]] = {}
    for event in events:
        data = event.get("data", {})
        idx = data.get("idx")
        if idx is None or event["type"] == "plan_join":
            continue
        row = rows.setdefault(idx, {"idx": idx, "tool": data.get("tool")})
        match event["type"]:
            case "task_received":
                row["received_ms"] = event["t_ms"]
            case "task_started":
                row.setdefault("started_ms", event["t_ms"])
            case "task_succeeded" | "task_failed" | "task_cancelled" | "task_reused":
                row["finished_ms"] = event["t_ms"]
                row["state"] = event["type"].removeprefix("task_")
    return [rows[i] for i in sorted(rows)]


async def _capture(domain: Domain, question: str, options: list[str]) -> str:
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="tool") as pool:
        state = ServiceState(
            tools=domain.tools,
            planner_factory=lambda kind: make_planner(kind, domain),
            joiner_factory=make_joiner,
            executor=pool,
            info={"catalog": domain.catalog.source, "embedder": domain.embedder.name},
        )
        runner = web.AppRunner(create_app(state))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/ask",
                    json={"question": question, "options": options, "max_concurrency": 6},
                ) as response,
            ):
                response.raise_for_status()
                return await response.text()
        finally:
            await runner.cleanup()


def bench_trace(root: Path | None, domain: Domain) -> dict[str, Any]:
    if root is not None and domain.catalog.source != "fixture":
        questions = load_questions(asset_path(root, "bigbench-movie-recommendation"))
        q = split_fit_eval(questions)[1][0]
        question, options = q.text, list(q.options)
    else:
        from tooldag.bench.replan import fixture_question

        fq = fixture_question(0)
        question, options = fq["question"], list(fq["options"])
    log(f"trace: {question}")
    body = asyncio.run(_capture(domain, question, options))
    events = parse_sse(body)
    finished = next((e for e in events if e["type"] == "run_finished"), None)
    return {
        "benchmark": "trace",
        "endpoint": "POST /v1/ask (SSE)",
        "catalog": domain.catalog.source,
        "embedder": domain.embedder.name,
        "scorer": domain.scorer.origin,
        "question": question,
        "options": options,
        "answer": finished["data"]["answer"] if finished else None,
        "makespan_ms": finished["data"]["makespan_ms"] if finished else None,
        "event_count": len(events),
        "tasks": task_table(events),
        "sse": body,
    }
