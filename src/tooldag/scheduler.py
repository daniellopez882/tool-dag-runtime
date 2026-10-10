"""Task Fetching Unit: dispatch plan tasks as soon as their dependencies resolve.

Two executors share one bookkeeping core:

* ``TaskFetchingUnit`` accepts tasks while the planner is still streaming and starts
  each one the moment its inputs exist, under a global concurrency limit and per-tool
  limits, with per-attempt timeouts and jittered exponential backoff between retries.
* ``SequentialExecutor`` is the ReAct-style baseline: it waits for the whole plan and
  runs one task at a time in plan order, with the same retry and timeout policy.

Failure handling is the same in both. A task that fails permanently cancels every task
that (transitively) depends on it, including dependents that arrive later in the
stream. ``cancel()`` stops the whole round: queued tasks are cancelled, pending retries
are dropped and running tasks receive ``CancelledError``.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from tooldag.dag import PlanGraph, call_signature
from tooldag.events import EventLog, preview
from tooldag.plan.ast import Arg, Call, ListArg, Ref, Template
from tooldag.plan.render import render_call
from tooldag.tools.base import ToolError, ToolRegistry, TransientToolError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Executor


class TaskState(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    BACKOFF = "backoff"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED})


@dataclass(slots=True)
class TaskRecord:
    call: Call
    state: TaskState = TaskState.WAITING
    attempts: int = 0
    result: Any = None
    error: str | None = None
    reused: bool = False
    signature: str | None = None
    submitted_ms: float = 0.0
    ready_ms: float | None = None
    started_ms: float | None = None
    finished_ms: float | None = None
    busy_ms: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


@dataclass(frozen=True, slots=True)
class Limits:
    max_concurrency: int = 4
    per_tool: Mapping[str, int] = field(default_factory=dict)
    timeout_s: float | None = 30.0
    tool_timeouts: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if any(v < 1 for v in self.per_tool.values()):
            raise ValueError("per-tool limits must be at least 1")

    def tool_limit(self, tool: str) -> int:
        return min(self.per_tool.get(tool, self.max_concurrency), self.max_concurrency)

    def timeout_for(self, tool: str) -> float | None:
        return self.tool_timeouts.get(tool, self.timeout_s)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter: delay ~ U(0, min(cap, base * mult**(n-1)))."""

    max_attempts: int = 3
    base_delay_s: float = 0.02
    max_delay_s: float = 0.5
    multiplier: float = 2.0

    def backoff(self, attempt: int, rng: random.Random) -> float:
        ceiling = min(self.max_delay_s, self.base_delay_s * self.multiplier ** (attempt - 1))
        return rng.uniform(0.0, ceiling)

    @staticmethod
    def retryable(exc: BaseException) -> bool:
        return isinstance(exc, TransientToolError | TimeoutError)


def resolve_arg(arg: Arg, results: Callable[[int], Any]) -> Any:
    match arg:
        case Ref(idx=idx):
            return results(idx)
        case Template(parts=parts):
            return "".join(p if isinstance(p, str) else str(results(p.idx)) for p in parts)
        case ListArg(items=items):
            return [resolve_arg(item, results) for item in items]
        case _:
            return arg


class _Attempt:
    """Outcome of one tool invocation."""

    __slots__ = ("error", "ok", "result", "retryable")

    def __init__(self, ok: bool, result: Any = None, error: str = "", retryable: bool = False):
        self.ok = ok
        self.result = result
        self.error = error
        self.retryable = retryable


class _ExecutorCore:
    """Bookkeeping shared by both executors: records, validation, events, propagation."""

    def __init__(
        self,
        tools: ToolRegistry,
        log: EventLog,
        *,
        round_: int = 1,
        limits: Limits | None = None,
        retry: RetryPolicy | None = None,
        executor: Executor | None = None,
        rng: random.Random | None = None,
        reuse: Mapping[str, Any] | None = None,
    ) -> None:
        self.tools = tools
        self.log = log
        self.round = round_
        self.limits = limits or Limits()
        self.retry = retry or RetryPolicy()
        self.executor = executor
        self.rng = rng or random.Random(0)
        self.reuse = reuse or {}
        self.graph = PlanGraph(validate_call=self._validate)
        self.records: dict[int, TaskRecord] = {}
        self.cancel_reason: str | None = None
        self.max_running = 0
        self.max_running_per_tool: Counter[str] = Counter()
        self._plan_closed = False
        self._done = asyncio.Event()
        self._running: dict[int, asyncio.Task[None]] = {}
        self._running_per_tool: Counter[str] = Counter()

    # -- intake -------------------------------------------------------------------------

    def _validate(self, call: Call) -> str | None:
        return self.tools.check_call(call.tool, len(call.args), [k for k, _ in call.kwargs])

    def submit(self, call: Call) -> None:
        """Accept one parsed task. Safe to call while earlier tasks are running."""
        if self._plan_closed:
            raise RuntimeError("submit() after end_of_plan()")
        if call.idx in self.records:
            self._emit("task_rejected", idx=call.idx, reason=f"task {call.idx} defined twice")
            return
        outcome = self.graph.add(call)
        record = TaskRecord(call, submitted_ms=self.log.elapsed_ms())
        self.records[call.idx] = record
        self._emit(
            "task_received",
            idx=call.idx,
            tool=call.tool,
            deps=sorted(call.deps),
            call=render_call(call),
        )
        if outcome.rejection is not None:
            self._settle(
                record, TaskState.FAILED, error=outcome.rejection.reason, event="task_rejected"
            )
            self._propagate([call.idx])
        elif outcome.cycle is not None:
            members = outcome.cycle.members
            for member in members:
                if not self.records[member].terminal:
                    self._settle(
                        self.records[member],
                        TaskState.FAILED,
                        error=self.graph.rejected[member],
                        event="task_rejected",
                    )
            self._propagate(list(members))
        else:
            self._evaluate(call.idx)
            if record.terminal:
                # Settled on arrival (cancelled or reused): wake tasks that referenced it early.
                self._propagate([call.idx])
        self._after_change()

    def end_of_plan(self) -> None:
        """The planner finished this round; tasks waiting on undefined numbers fail now."""
        if self._plan_closed:
            return
        dangling = self.graph.close()
        self._plan_closed = True
        failed: list[int] = []
        for ref, waiting in sorted(dangling.items()):
            for idx in sorted(waiting):
                record = self.records[idx]
                if not record.terminal:
                    reason = f"references ${ref}, which the plan never defined"
                    self._settle(record, TaskState.FAILED, error=reason, event="task_rejected")
                    failed.append(idx)
        self._propagate(failed)
        self._after_change()

    async def drain(self) -> dict[int, TaskRecord]:
        """Wait until every submitted task is terminal (requires ``end_of_plan``)."""
        await self._done.wait()
        return self.records

    def cancel(self, reason: str = "cancelled") -> None:
        if self.cancel_reason is not None:
            return
        self.cancel_reason = reason
        self._plan_closed = True
        self.graph.closed = True
        for record in self.records.values():
            if record.state in (TaskState.WAITING, TaskState.READY, TaskState.BACKOFF):
                self._drop_pending(record.call.idx)
                self._settle(record, TaskState.CANCELLED, error=reason)
        for task in list(self._running.values()):
            task.cancel()
        self._check_done()

    # -- state machine -----------------------------------------------------------------

    def _emit(self, type_: str, **data: Any) -> None:
        self.log.emit(type_, self.round, **data)

    def _result_of(self, idx: int) -> Any:
        return self.records[idx].result

    def _evaluate(self, idx: int) -> None:
        """Move a WAITING task forward if its dependencies allow it."""
        record = self.records[idx]
        if record.state is not TaskState.WAITING:
            return
        if self.cancel_reason is not None:
            self._settle(record, TaskState.CANCELLED, error=self.cancel_reason)
            return
        pending = False
        for dep in sorted(record.call.deps):
            dep_record = self.records.get(dep)
            if dep_record is None:
                pending = True
            elif dep_record.state in (TaskState.FAILED, TaskState.CANCELLED):
                reason = f"dependency ${dep} {dep_record.state.value}"
                self._settle(record, TaskState.CANCELLED, error=reason)
                return
            elif dep_record.state is not TaskState.SUCCEEDED:
                pending = True
        if pending:
            return
        record.signature = call_signature(record.call, lambda i: self.records[i].signature or "")
        record.ready_ms = self.log.elapsed_ms()
        if record.signature in self.reuse:
            record.reused = True
            self._settle(
                record,
                TaskState.SUCCEEDED,
                result=self.reuse[record.signature],
                event="task_reused",
            )
            return
        record.state = TaskState.READY
        self._on_ready(idx)

    def _settle(
        self,
        record: TaskRecord,
        state: TaskState,
        *,
        result: Any = None,
        error: str | None = None,
        event: str | None = None,
    ) -> None:
        record.state = state
        record.finished_ms = self.log.elapsed_ms()
        idx, tool = record.call.idx, record.call.tool
        if state is TaskState.SUCCEEDED:
            record.result = result
            if record.reused:
                self._emit(
                    event or "task_reused",
                    idx=idx,
                    tool=tool,
                    signature=record.signature,
                    result=preview(result),
                )
            else:
                self._emit(
                    event or "task_succeeded",
                    idx=idx,
                    tool=tool,
                    attempts=record.attempts,
                    busy_ms=round(record.busy_ms, 3),
                    result=preview(result),
                )
        else:
            record.error = error
            if state is TaskState.FAILED:
                self._emit(
                    event or "task_failed",
                    idx=idx,
                    tool=tool,
                    attempts=record.attempts,
                    error=error,
                )
            else:
                self._emit(event or "task_cancelled", idx=idx, tool=tool, reason=error)

    def _propagate(self, roots: list[int]) -> None:
        """Re-evaluate dependents of tasks that just became terminal (iteratively)."""
        work = deque(roots)
        while work:
            node = work.popleft()
            for child in sorted(self.graph.dependents.get(node, ())):
                record = self.records.get(child)
                if record is None or record.state is not TaskState.WAITING:
                    continue
                self._evaluate(child)
                if record.terminal:
                    work.append(child)

    def _check_done(self) -> None:
        if (
            self._plan_closed
            and not self._running
            and all(r.terminal for r in self.records.values())
        ):
            self._done.set()

    # -- running one attempt -------------------------------------------------------------

    async def _invoke(self, record: TaskRecord) -> _Attempt:
        call = record.call
        tool = self.tools[call.tool]
        try:
            args = [resolve_arg(a, self._result_of) for a in call.args]
            kwargs = {k: resolve_arg(v, self._result_of) for k, v in call.kwargs}
            bound = tool.bind(args, kwargs)
            async with asyncio.timeout(self.limits.timeout_for(call.tool)):
                result = await tool.invoke(bound, self.executor)
        except TimeoutError:
            return _Attempt(
                False,
                error=f"timed out after {self.limits.timeout_for(call.tool)}s",
                retryable=True,
            )
        except ToolError as exc:
            return _Attempt(
                False, error=str(exc) or type(exc).__name__, retryable=self.retry.retryable(exc)
            )
        except Exception as exc:  # a tool bug must fail the task, not the scheduler
            return _Attempt(False, error=f"{type(exc).__name__}: {exc}")
        return _Attempt(True, result=result)

    def _begin_attempt(self, record: TaskRecord) -> None:
        record.state = TaskState.RUNNING
        record.attempts += 1
        now = self.log.elapsed_ms()
        if record.started_ms is None:
            record.started_ms = now
        tool = record.call.tool
        self._running_per_tool[tool] += 1
        self.max_running = max(self.max_running, len(self._running) + 1)
        self.max_running_per_tool[tool] = max(
            self.max_running_per_tool[tool], self._running_per_tool[tool]
        )
        waited = now - (record.ready_ms if record.ready_ms is not None else now)
        self._emit(
            "task_started",
            idx=record.call.idx,
            tool=tool,
            attempt=record.attempts,
            waited_ms=round(waited, 3),
        )

    def _end_attempt(self, record: TaskRecord, began: float) -> None:
        self._running.pop(record.call.idx, None)
        self._running_per_tool[record.call.tool] -= 1
        record.busy_ms += (time.perf_counter() - began) * 1000.0

    # -- hooks for the concrete executors ----------------------------------------------

    def _on_ready(self, idx: int) -> None:
        raise NotImplementedError

    def _drop_pending(self, idx: int) -> None:
        raise NotImplementedError

    def _after_change(self) -> None:
        self._check_done()


class TaskFetchingUnit(_ExecutorCore):
    """Streaming DAG executor (the LLMCompiler Task Fetching Unit)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # One FIFO per tool; the global order is kept by a sequence number so a saturated
        # tool never blocks tasks of other tools that are behind it in the queue.
        self._ready: dict[str, deque[tuple[int, int]]] = {}
        self._ready_seq = itertools.count()
        self._backoff: dict[int, asyncio.TimerHandle] = {}

    def _on_ready(self, idx: int) -> None:
        tool = self.records[idx].call.tool
        self._ready.setdefault(tool, deque()).append((next(self._ready_seq), idx))

    def _drop_pending(self, idx: int) -> None:
        handle = self._backoff.pop(idx, None)
        if handle is not None:
            handle.cancel()
        tool = self.records[idx].call.tool
        queue = self._ready.get(tool)
        if queue:
            self._ready[tool] = deque(item for item in queue if item[1] != idx)

    def _after_change(self) -> None:
        self._pump()
        self._check_done()

    def _pump(self) -> None:
        while self.cancel_reason is None and len(self._running) < self.limits.max_concurrency:
            best: tuple[int, str] | None = None
            for tool, queue in self._ready.items():
                if queue and self._running_per_tool[tool] < self.limits.tool_limit(tool):
                    head = queue[0][0]
                    if best is None or head < best[0]:
                        best = (head, tool)
            if best is None:
                return
            _, idx = self._ready[best[1]].popleft()
            record = self.records[idx]
            self._begin_attempt(record)
            self._running[idx] = asyncio.create_task(self._run(record), name=f"task-{idx}")

    async def _run(self, record: TaskRecord) -> None:
        began = time.perf_counter()
        try:
            outcome = await self._invoke(record)
        except asyncio.CancelledError:
            self._end_attempt(record, began)
            self._settle(record, TaskState.CANCELLED, error=self.cancel_reason or "cancelled")
            self._check_done()
            raise
        self._end_attempt(record, began)
        if self.cancel_reason is not None:
            self._settle(record, TaskState.CANCELLED, error=self.cancel_reason)
        elif outcome.ok:
            self._settle(record, TaskState.SUCCEEDED, result=outcome.result)
            self._propagate([record.call.idx])
        elif outcome.retryable and record.attempts < self.retry.max_attempts:
            delay = self.retry.backoff(record.attempts, self.rng)
            record.state = TaskState.BACKOFF
            self._emit(
                "task_retry",
                idx=record.call.idx,
                tool=record.call.tool,
                attempt=record.attempts,
                error=outcome.error,
                backoff_ms=round(delay * 1000.0, 3),
            )
            loop = asyncio.get_running_loop()
            self._backoff[record.call.idx] = loop.call_later(delay, self._requeue, record.call.idx)
        else:
            self._settle(record, TaskState.FAILED, error=outcome.error)
            self._propagate([record.call.idx])
        self._after_change()

    def _requeue(self, idx: int) -> None:
        self._backoff.pop(idx, None)
        record = self.records[idx]
        if record.state is TaskState.BACKOFF and self.cancel_reason is None:
            record.state = TaskState.READY
            record.ready_ms = self.log.elapsed_ms()
            self._on_ready(idx)
            self._after_change()


class SequentialExecutor(_ExecutorCore):
    """ReAct-order baseline: wait for the full plan, then run one task at a time."""

    def _on_ready(self, idx: int) -> None:
        # Readiness is only recorded; drain() runs ready tasks in plan order.
        return

    def _drop_pending(self, idx: int) -> None:
        return

    def _check_done(self) -> None:
        # Completion is decided by drain() itself, which owns the execution loop.
        return

    async def drain(self) -> dict[int, TaskRecord]:
        if not self._plan_closed:
            raise RuntimeError("SequentialExecutor.drain() needs the whole plan first")
        for idx in self.graph.topological_order():
            record = self.records[idx]
            if self.cancel_reason is not None:
                break
            if record.state is not TaskState.READY:
                continue
            await self._run_to_completion(record)
        for record in self.records.values():
            if not record.terminal:
                self._settle(
                    record, TaskState.CANCELLED, error=self.cancel_reason or "not reached"
                )
        return self.records

    async def _run_to_completion(self, record: TaskRecord) -> None:
        while True:
            self._begin_attempt(record)
            began = time.perf_counter()
            task = asyncio.create_task(self._invoke(record), name=f"task-{record.call.idx}")
            self._running[record.call.idx] = task  # type: ignore[assignment]
            try:
                outcome = await task
            except asyncio.CancelledError:
                self._end_attempt(record, began)
                self._settle(record, TaskState.CANCELLED, error=self.cancel_reason or "cancelled")
                if self.cancel_reason is None:
                    raise
                return
            self._end_attempt(record, began)
            if outcome.ok:
                self._settle(record, TaskState.SUCCEEDED, result=outcome.result)
                self._propagate([record.call.idx])
                return
            if outcome.retryable and record.attempts < self.retry.max_attempts:
                delay = self.retry.backoff(record.attempts, self.rng)
                record.state = TaskState.BACKOFF
                self._emit(
                    "task_retry",
                    idx=record.call.idx,
                    tool=record.call.tool,
                    attempt=record.attempts,
                    error=outcome.error,
                    backoff_ms=round(delay * 1000.0, 3),
                )
                await asyncio.sleep(delay)
                if self.cancel_reason is not None:
                    self._settle(record, TaskState.CANCELLED, error=self.cancel_reason)
                    return
                continue
            self._settle(record, TaskState.FAILED, error=outcome.error)
            self._propagate([record.call.idx])
            return


type RoundExecutor = TaskFetchingUnit | SequentialExecutor
