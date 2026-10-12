"""aiohttp service that runs questions and streams task-lifecycle events over SSE.

Routes::

    GET    /healthz                 liveness plus which catalogue and embedder are loaded
    GET    /v1/tools                tool signatures the planner can use
    POST   /v1/runs                 start a run, returns its id and event URL (201)
    GET    /v1/runs/{id}            status and answer
    GET    /v1/runs/{id}/events     SSE: replay from Last-Event-ID (or ?after=N), then live
    DELETE /v1/runs/{id}            cancel a run; in-flight tasks receive CancelledError
    POST   /v1/ask                  start a run and stream its events in the same response;
                                    if the client disconnects, the run is cancelled
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError

from tooldag.events import Event, EventLog
from tooldag.runtime import RunResult, Runtime, RuntimeConfig
from tooldag.scheduler import Limits

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor

    from tooldag.rounds import Joiner, Planner
    from tooldag.tools.base import ToolRegistry

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=20)
    planner: Literal["template", "llm"] = "template"
    joiner: Literal["rule", "llm"] = "rule"
    mode: Literal["dag", "sequential"] = "dag"
    max_concurrency: int = Field(default=6, ge=1, le=64)
    max_rounds: int = Field(default=3, ge=1, le=5)


@dataclass(slots=True)
class RunHandle:
    run_id: str
    log: EventLog
    task: asyncio.Task[RunResult]
    created: float

    @property
    def state(self) -> str:
        if not self.task.done():
            return "running"
        if self.task.cancelled():
            return "cancelled"
        return "failed" if self.task.exception() is not None else "finished"


@dataclass(slots=True)
class ServiceState:
    tools: ToolRegistry
    planner_factory: Callable[[str], Planner]
    joiner_factory: Callable[[str], Joiner]
    executor: Executor | None
    info: dict[str, Any]
    max_runs: int = 64
    keepalive_s: float = 15.0
    runs: OrderedDict[str, RunHandle] = field(default_factory=OrderedDict)

    def start(self, request: RunRequest) -> RunHandle:
        runtime = Runtime(
            self.tools,
            self.planner_factory(request.planner),
            self.joiner_factory(request.joiner),
            RuntimeConfig(
                limits=Limits(max_concurrency=request.max_concurrency),
                max_rounds=request.max_rounds,
                mode=request.mode,
            ),
            executor=self.executor,
        )
        run_id = uuid.uuid4().hex[:12]
        log = EventLog(run_id)
        task = asyncio.create_task(runtime.run(request.question, request.options, log=log))
        # Retrieve the exception so a failed run does not log "never retrieved".
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        handle = RunHandle(run_id, log, task, time.time())
        self.runs[run_id] = handle
        self._evict()
        return handle

    def _evict(self) -> None:
        finished = [rid for rid, h in self.runs.items() if h.task.done()]
        while len(self.runs) > self.max_runs and finished:
            self.runs.pop(finished.pop(0), None)


STATE_KEY = web.AppKey("tooldag_state", ServiceState)


def encode_sse(run_id: str, event: Event) -> bytes:
    payload = {"run_id": run_id, **event.model_dump()}
    body = json.dumps(payload, separators=(",", ":"), default=str)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {body}\n\n".encode()


async def _write_events(
    request: web.Request, log: EventLog, after: int, keepalive_s: float
) -> web.StreamResponse:
    response = web.StreamResponse(headers=SSE_HEADERS)
    await response.prepare(request)
    async with contextlib.aclosing(log.follow(after, heartbeat_s=keepalive_s)) as stream:
        async for event in stream:
            if event is None:
                # The comment line keeps proxies from closing an idle stream and makes a
                # write fail promptly once the client has gone away.
                await response.write(b": keep-alive\n\n")
            else:
                await response.write(encode_sse(log.run_id, event))
    with contextlib.suppress(ConnectionError):
        await response.write_eof()
    return response


def _error(status: int, message: str, **extra: Any) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


def _last_event_id(request: web.Request) -> int:
    raw = request.headers.get("Last-Event-ID") or request.query.get("after") or "0"
    try:
        return max(int(raw), 0)
    except ValueError as exc:
        raise web.HTTPBadRequest(reason="Last-Event-ID must be an integer") from exc


async def _parse_request(request: web.Request) -> RunRequest | web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error(400, "body must be JSON")
    try:
        return RunRequest.model_validate(body)
    except ValidationError as exc:
        return _error(422, "invalid run request", detail=json.loads(exc.json()))


async def healthz(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    running = sum(1 for h in state.runs.values() if not h.task.done())
    return web.json_response(
        {"status": "ok", **state.info, "runs_in_memory": len(state.runs), "running": running}
    )


async def list_tools(request: web.Request) -> web.Response:
    return web.json_response({"tools": request.app[STATE_KEY].tools.specs()})


async def create_run(request: web.Request) -> web.Response:
    parsed = await _parse_request(request)
    if isinstance(parsed, web.Response):
        return parsed
    state = request.app[STATE_KEY]
    try:
        handle = state.start(parsed)
    except Exception as exc:  # e.g. the LLM planner without provider settings
        return _error(400, str(exc))
    return web.json_response(
        {
            "run_id": handle.run_id,
            "status_url": f"/v1/runs/{handle.run_id}",
            "events_url": f"/v1/runs/{handle.run_id}/events",
        },
        status=201,
    )


def _handle(request: web.Request) -> RunHandle:
    handle = request.app[STATE_KEY].runs.get(request.match_info["run_id"])
    if handle is None:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "unknown run"}), content_type="application/json"
        )
    return handle


async def run_status(request: web.Request) -> web.Response:
    handle = _handle(request)
    body: dict[str, Any] = {
        "run_id": handle.run_id,
        "state": handle.state,
        "events": len(handle.log),
    }
    if handle.state == "finished":
        result = handle.task.result()
        body.update(
            answer=result.answer,
            rounds=len(result.rounds),
            makespan_ms=round(result.makespan_ms, 3),
            model_calls=result.model_calls(),
        )
    elif handle.state == "failed":
        body["error"] = str(handle.task.exception())
    return web.json_response(body)


async def run_events(request: web.Request) -> web.StreamResponse:
    handle = _handle(request)
    state = request.app[STATE_KEY]
    return await _write_events(request, handle.log, _last_event_id(request), state.keepalive_s)


async def cancel_run(request: web.Request) -> web.Response:
    handle = _handle(request)
    if handle.task.done():
        return web.json_response({"run_id": handle.run_id, "state": handle.state})
    handle.task.cancel()
    return web.json_response({"run_id": handle.run_id, "state": "cancelling"}, status=202)


async def ask(request: web.Request) -> web.StreamResponse:
    parsed = await _parse_request(request)
    if isinstance(parsed, web.Response):
        return parsed
    state = request.app[STATE_KEY]
    try:
        handle = state.start(parsed)
    except Exception as exc:
        return _error(400, str(exc))
    try:
        return await _write_events(request, handle.log, 0, state.keepalive_s)
    except (ConnectionError, asyncio.CancelledError):
        # The client went away: stop the run so its tools stop consuming the pool.
        handle.task.cancel()
        raise


async def _shutdown(app: web.Application) -> None:
    state = app[STATE_KEY]
    tasks = [h.task for h in state.runs.values() if not h.task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def create_app(state: ServiceState) -> web.Application:
    app = web.Application()
    app[STATE_KEY] = state
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/v1/tools", list_tools)
    app.router.add_post("/v1/runs", create_run)
    app.router.add_get("/v1/runs/{run_id}", run_status)
    app.router.add_get("/v1/runs/{run_id}/events", run_events)
    app.router.add_delete("/v1/runs/{run_id}", cancel_run)
    app.router.add_post("/v1/ask", ask)
    app.on_shutdown.append(_shutdown)
    return app
