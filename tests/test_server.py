"""SSE contract of the HTTP service, exercised with aiohttp's test client."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tooldag.bench.trace import parse_sse, task_table
from tooldag.joiner import RuleJoiner
from tooldag.planners import TemplatePlanner
from tooldag.server import ServiceState, create_app
from tooldag.tools.base import Param, Tool, ToolRegistry
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder
from tooldag.tools.movies import MovieTools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

QUESTION = "Find a movie similar to Orbit of Ash, Starfall Protocol, The Last Lighthouse on Vega"
OPTIONS = ["Comet Wardens", "The Wedding Season", "Silent Orchard", "Tidewater"]


def _state(tools: ToolRegistry, planner_factory: Callable[[str], Any]) -> ServiceState:
    return ServiceState(
        tools=tools,
        planner_factory=planner_factory,
        joiner_factory=lambda kind: RuleJoiner(),
        executor=ThreadPoolExecutor(max_workers=2),
        info={"catalog": "fixture"},
        keepalive_s=0.05,
    )


def fixture_state() -> ServiceState:
    tools = MovieTools(Catalog.fixture(), HashingEmbedder()).registry()

    def planner(kind: str) -> TemplatePlanner:
        if kind != "template":
            raise ValueError("set TOOLDAG_LLM_BASE_URL and TOOLDAG_LLM_MODEL")
        return TemplatePlanner()

    return _state(tools, planner)


def with_client(state: ServiceState, body: Callable[[TestClient], Awaitable[None]]) -> None:
    async def go() -> None:
        async with TestClient(TestServer(create_app(state))) as client:
            await body(client)

    asyncio.run(go())


def test_health_and_tools() -> None:
    async def body(client: TestClient) -> None:
        health = await (await client.get("/healthz")).json()
        assert health["status"] == "ok"
        assert health["catalog"] == "fixture"
        tools = await (await client.get("/v1/tools")).json()
        assert {t["name"] for t in tools["tools"]} >= {"resolve_title", "rank"}

    with_client(fixture_state(), body)


def test_ask_streams_a_complete_well_formed_event_stream() -> None:
    async def body(client: TestClient) -> None:
        response = await client.post("/v1/ask", json={"question": QUESTION, "options": OPTIONS})
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")
        text = await response.text()
        blocks = [b for b in text.split("\n\n") if b.strip()]
        ids = []
        for block in blocks:
            fields = dict(line.split(": ", 1) for line in block.splitlines())
            assert set(fields) == {"id", "event", "data"}
            payload = json.loads(fields["data"])
            assert payload["type"] == fields["event"]
            assert payload["seq"] == int(fields["id"])
            ids.append(int(fields["id"]))
        assert ids == list(range(1, len(ids) + 1))
        events = parse_sse(text)
        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_finished"
        assert events[-1]["data"]["answer"] == "Comet Wardens"
        times = [e["t_ms"] for e in events]
        assert times == sorted(times)
        table = task_table(events)
        assert all(r["state"] == "succeeded" for r in table)
        for row in table:
            assert row["received_ms"] <= row["started_ms"] <= row["finished_ms"]

    with_client(fixture_state(), body)


def test_run_resource_status_and_replay_from_last_event_id() -> None:
    async def body(client: TestClient) -> None:
        created = await client.post("/v1/runs", json={"question": QUESTION, "options": OPTIONS})
        assert created.status == 201
        info = await created.json()
        full = parse_sse(await (await client.get(info["events_url"])).text())
        status = await (await client.get(info["status_url"])).json()
        assert status["state"] == "finished"
        assert status["answer"] == "Comet Wardens"
        assert status["events"] == len(full)
        replay = await client.get(info["events_url"], headers={"Last-Event-ID": "10"})
        tail = parse_sse(await replay.text())
        assert [e["seq"] for e in tail] == [e["seq"] for e in full[10:]]
        by_query = parse_sse(await (await client.get(info["events_url"] + "?after=12")).text())
        assert by_query[0]["seq"] == 13
        bad = await client.get(info["events_url"], headers={"Last-Event-ID": "x"})
        assert bad.status == 400

    with_client(fixture_state(), body)


def test_validation_errors_and_unknown_runs() -> None:
    async def body(client: TestClient) -> None:
        assert (await client.post("/v1/runs", data=b"not json")).status == 400
        invalid = await client.post("/v1/runs", json={"question": "", "max_concurrency": 0})
        assert invalid.status == 422
        assert (await invalid.json())["error"] == "invalid run request"
        llm = await client.post("/v1/runs", json={"question": QUESTION, "planner": "llm"})
        assert llm.status == 400
        assert "TOOLDAG_LLM_BASE_URL" in (await llm.json())["error"]
        assert (await client.get("/v1/runs/nope")).status == 404
        assert (await client.delete("/v1/runs/nope")).status == 404

    with_client(fixture_state(), body)


def _blocking_state(gate: asyncio.Event, cancelled: list[str]) -> ServiceState:
    async def wait_forever(x: str) -> str:
        try:
            gate.set()
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(x)
            raise
        return x

    tools = ToolRegistry()
    tools.register(Tool("wait", "sleeps", (Param("x", str),), wait_forever))
    tools.register(Tool("rank", "never reached", (Param("x", object),), lambda x: x))

    class Plan:
        name = "fixed"

        async def stream_plan(self, ctx: Any):
            yield '1. wait("a")\n2. wait("b")\n3. rank([$1, $2])\n4. join()\n'

    return _state(tools, lambda kind: Plan())


def test_delete_cancels_in_flight_tasks() -> None:
    gate = asyncio.Event()
    cancelled: list[str] = []

    async def body(client: TestClient) -> None:
        info = await (await client.post("/v1/runs", json={"question": "q"})).json()
        await asyncio.wait_for(gate.wait(), 5)
        await asyncio.sleep(0.05)
        response = await client.delete(info["status_url"])
        assert response.status == 202
        events = parse_sse(await (await client.get(info["events_url"])).text())
        types = [e["type"] for e in events]
        assert types[-1] == "run_cancelled"
        assert types.count("task_cancelled") == 3
        status = await (await client.get(info["status_url"])).json()
        assert status["state"] == "cancelled"
        again = await client.delete(info["status_url"])
        assert (await again.json())["state"] == "cancelled"

    with_client(_blocking_state(gate, cancelled), body)
    assert sorted(cancelled) == ["a", "b"]


def test_keepalive_comments_are_sent_while_idle() -> None:
    gate = asyncio.Event()

    async def body(client: TestClient) -> None:
        info = await (await client.post("/v1/runs", json={"question": "q"})).json()
        await asyncio.wait_for(gate.wait(), 5)
        response = await client.get(info["events_url"])
        seen = b""
        while b": keep-alive" not in seen:
            seen += await asyncio.wait_for(response.content.readany(), 5)
        await client.delete(info["status_url"])
        await response.read()

    with_client(_blocking_state(gate, []), body)


@pytest.mark.parametrize("path", ["/v1/ask"])
def test_client_disconnect_cancels_the_run(path: str) -> None:
    gate = asyncio.Event()
    cancelled: list[str] = []
    state = _blocking_state(gate, cancelled)

    async def body(client: TestClient) -> None:
        response = await client.post(path, json={"question": "q"})
        await asyncio.wait_for(gate.wait(), 5)
        response.close()
        for _ in range(200):
            if all(h.task.done() for h in state.runs.values()):
                break
            await asyncio.sleep(0.02)
        assert all(h.task.cancelled() for h in state.runs.values())

    with_client(state, body)
    assert sorted(cancelled) == ["a", "b"]
