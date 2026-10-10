from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tooldag.events import EventLog
from tooldag.plan import Call, ListArg, Ref
from tooldag.scheduler import (
    Limits,
    RetryPolicy,
    SequentialExecutor,
    TaskFetchingUnit,
    TaskState,
)
from tooldag.tools.base import Param, Tool, ToolError, ToolRegistry, TransientToolError

FAST_RETRY = RetryPolicy(max_attempts=3, base_delay_s=0.0005, max_delay_s=0.002)


@dataclass
class Trace:
    """Independent record of what the tools saw, used to check the scheduler's claims."""

    clock: int = 0
    starts: dict[int, int] = field(default_factory=dict)
    ends: dict[int, int] = field(default_factory=dict)
    running: Counter[str] = field(default_factory=Counter)
    peak: Counter[str] = field(default_factory=Counter)
    total: int = 0
    peak_total: int = 0
    attempts: Counter[int] = field(default_factory=Counter)
    cancelled: set[int] = field(default_factory=set)

    def tick(self) -> int:
        self.clock += 1
        return self.clock


def make_registry(trace: Trace, names: tuple[str, ...] = ("a", "b", "c")) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:

        async def fn(inputs: list[int], idx: int, spin: int, mode: str, _tool: str = name) -> int:
            trace.attempts[idx] += 1
            trace.starts.setdefault(idx, trace.tick())
            trace.running[_tool] += 1
            trace.total += 1
            trace.peak[_tool] = max(trace.peak[_tool], trace.running[_tool])
            trace.peak_total = max(trace.peak_total, trace.total)
            try:
                for _ in range(spin):
                    await asyncio.sleep(0)
                if mode == "hang":
                    await asyncio.Event().wait()
                if mode == "fail":
                    raise ToolError(f"task {idx} failed")
                if mode == "flaky" and trace.attempts[idx] == 1:
                    raise TransientToolError("try again")
                return idx + sum(inputs)
            except asyncio.CancelledError:
                trace.cancelled.add(idx)
                raise
            finally:
                trace.running[_tool] -= 1
                trace.total -= 1
                trace.ends[idx] = trace.tick()

        registry.register(
            Tool(
                name,
                "test tool",
                (
                    Param("inputs", list[int]),
                    Param("idx", int),
                    Param("spin", int),
                    Param("mode", str),
                ),
                fn,
                returns="int",
            )
        )
    return registry


@dataclass(frozen=True)
class Spec:
    idx: int
    tool: str
    deps: tuple[int, ...]
    spin: int
    mode: str

    def call(self) -> Call:
        return Call(
            self.idx,
            self.tool,
            (ListArg(tuple(Ref(d) for d in self.deps)), self.idx, self.spin, self.mode),
        )


@st.composite
def workloads(draw: st.DrawFn, allow_forward: bool = True) -> list[Spec]:
    n = draw(st.integers(1, 14))
    specs = []
    for idx in range(1, n + 1):
        upper = n + 1 if allow_forward else idx - 1
        pool = st.integers(1, max(upper, 1))
        deps = sorted(draw(st.sets(pool, max_size=3))) if upper >= 1 else []
        if not allow_forward:
            deps = [d for d in deps if d < idx]
        mode = draw(st.sampled_from(["ok"] * 6 + ["fail", "flaky"]))
        specs.append(
            Spec(idx, draw(st.sampled_from("abc")), tuple(deps), draw(st.integers(0, 4)), mode)
        )
    return specs


async def run_streaming(
    specs: list[Spec],
    limits: Limits,
    gaps: list[int],
    executor_cls: type = TaskFetchingUnit,
) -> tuple[dict, Trace, EventLog]:
    trace = Trace()
    log = EventLog()
    unit = executor_cls(
        make_registry(trace), log, limits=limits, retry=FAST_RETRY, rng=random.Random(7)
    )
    for spec, gap in zip(specs, gaps + [0] * len(specs), strict=False):
        unit.submit(spec.call())
        for _ in range(gap):
            await asyncio.sleep(0)
    unit.end_of_plan()
    records = await asyncio.wait_for(unit.drain(), timeout=10)
    return records, trace, log


def _on_cycle(specs: dict[int, Spec]) -> set[int]:
    out = set()
    for start in specs:
        stack, seen = list(specs[start].deps), set()
        while stack:
            node = stack.pop()
            if node == start:
                out.add(start)
                break
            if node in seen or node not in specs:
                continue
            seen.add(node)
            stack.extend(specs[node].deps)
    return out


def check_invariants(specs: list[Spec], records: dict, trace: Trace, limits: Limits) -> None:
    by_idx = {s.idx: s for s in specs}
    assert set(records) == set(by_idx)
    assert all(r.terminal for r in records.values())
    # Bounded concurrency, measured by the tools themselves.
    assert trace.peak_total <= limits.max_concurrency
    for tool, peak in trace.peak.items():
        assert peak <= limits.tool_limit(tool)
    expected: dict[int, int] = {}
    for idx in sorted(by_idx, key=lambda i: records[i].finished_ms or 0):
        record = records[idx]
        spec = by_idx[idx]
        if idx in trace.starts:
            # No task starts before every dependency has finished successfully.
            for dep in spec.deps:
                assert dep in by_idx, "task with an undefined dependency started"
                assert records[dep].state is TaskState.SUCCEEDED
                assert trace.ends[dep] < trace.starts[idx]
        if record.state is TaskState.SUCCEEDED:
            expected[idx] = idx + sum(expected[d] for d in spec.deps)
            assert record.result == expected[idx]
            assert spec.mode != "fail"
        if record.state is TaskState.FAILED and spec.mode == "fail" and idx in trace.starts:
            assert record.attempts == 1, "permanent errors are not retried"
    # Failure propagates: everything downstream of a failed task is cancelled, never run.
    failed = {i for i, r in records.items() if r.state is TaskState.FAILED}
    downstream: set[int] = set()
    frontier = list(failed)
    while frontier:
        node = frontier.pop()
        for spec in specs:
            if node in spec.deps and spec.idx not in downstream:
                downstream.add(spec.idx)
                frontier.append(spec.idx)
    for idx in downstream - failed:
        assert records[idx].state is TaskState.CANCELLED
        assert idx not in trace.starts
    for idx in _on_cycle(by_idx):
        assert idx not in trace.starts


@given(
    workloads(),
    st.integers(1, 6),
    st.dictionaries(st.sampled_from("abc"), st.integers(1, 3), max_size=3),
    st.lists(st.integers(0, 3), max_size=14),
)
@settings(max_examples=150, deadline=None)
def test_streaming_dag_invariants(
    specs: list[Spec], max_c: int, per_tool: dict[str, int], gaps: list[int]
) -> None:
    limits = Limits(max_concurrency=max_c, per_tool=per_tool, timeout_s=5)
    records, trace, _ = asyncio.run(run_streaming(specs, limits, gaps))
    check_invariants(specs, records, trace, limits)


@given(workloads(allow_forward=False))
@settings(max_examples=60, deadline=None)
def test_sequential_executor_obeys_the_same_invariants(specs: list[Spec]) -> None:
    limits = Limits(max_concurrency=1, timeout_s=5)
    records, trace, _ = asyncio.run(run_streaming(specs, limits, [], SequentialExecutor))
    check_invariants(specs, records, trace, limits)
    assert trace.peak_total <= 1


@given(workloads(allow_forward=False))
@settings(max_examples=40, deadline=None)
def test_dag_and_sequential_agree_on_outcomes(specs: list[Spec]) -> None:
    limits = Limits(max_concurrency=4, timeout_s=5)
    dag, _, _ = asyncio.run(run_streaming(specs, limits, []))
    seq, _, _ = asyncio.run(run_streaming(specs, limits, [], SequentialExecutor))
    assert {i: (r.state, r.result) for i, r in dag.items()} == {
        i: (r.state, r.result) for i, r in seq.items()
    }


def test_tasks_start_before_the_plan_is_complete() -> None:
    async def scenario() -> list[str]:
        trace = Trace()
        log = EventLog()
        unit = TaskFetchingUnit(make_registry(trace), log, limits=Limits(max_concurrency=2))
        unit.submit(Spec(1, "a", (), 0, "ok").call())
        await asyncio.sleep(0.01)
        started_early = 1 in trace.starts
        unit.submit(Spec(2, "a", (1,), 0, "ok").call())
        unit.end_of_plan()
        await unit.drain()
        return ["early" if started_early else "late"]

    assert asyncio.run(scenario()) == ["early"]


def test_cancel_propagates_to_running_and_queued_tasks() -> None:
    async def scenario() -> tuple[dict, Trace, EventLog]:
        trace = Trace()
        log = EventLog()
        unit = TaskFetchingUnit(make_registry(trace), log, limits=Limits(max_concurrency=2))
        for spec in (
            Spec(1, "a", (), 0, "hang"),
            Spec(2, "b", (), 0, "hang"),
            Spec(3, "c", (), 0, "ok"),
            Spec(4, "a", (1,), 0, "ok"),
        ):
            unit.submit(spec.call())
        await asyncio.sleep(0.01)
        unit.cancel("client went away")
        unit.end_of_plan()
        records = await asyncio.wait_for(unit.drain(), timeout=2)
        return records, trace, log

    records, trace, log = asyncio.run(scenario())
    assert {i: r.state for i, r in records.items()} == dict.fromkeys(
        (1, 2, 3, 4), TaskState.CANCELLED
    )
    assert trace.cancelled == {1, 2}
    assert set(trace.starts) == {1, 2}
    reasons = {e.data.get("reason") for e in log.events if e.type == "task_cancelled"}
    assert reasons == {"client went away"}


def test_submissions_after_a_failure_are_cancelled_on_arrival() -> None:
    async def scenario() -> dict:
        trace = Trace()
        unit = TaskFetchingUnit(make_registry(trace), EventLog(), limits=Limits())
        unit.submit(Spec(1, "a", (), 0, "fail").call())
        await asyncio.sleep(0.01)
        unit.submit(Spec(2, "b", (1,), 0, "ok").call())
        unit.submit(Spec(3, "c", (2,), 0, "ok").call())
        unit.end_of_plan()
        return await unit.drain()

    records = asyncio.run(scenario())
    assert records[1].state is TaskState.FAILED
    assert records[2].state is TaskState.CANCELLED
    assert records[2].error == "dependency $1 failed"
    assert records[3].error == "dependency $2 cancelled"


def _slow_registry(delay: float, calls: Counter[str]) -> ToolRegistry:
    async def slow(x: int) -> int:
        calls["slow"] += 1
        await asyncio.sleep(delay)
        return x

    registry = ToolRegistry()
    registry.register(Tool("slow", "sleeps", (Param("x", int),), slow))
    return registry


def test_timeouts_are_retried_then_fail() -> None:
    async def scenario() -> tuple[dict, EventLog]:
        calls: Counter[str] = Counter()
        log = EventLog()
        unit = TaskFetchingUnit(
            _slow_registry(1.0, calls),
            log,
            limits=Limits(timeout_s=0.02),
            retry=FAST_RETRY,
        )
        unit.submit(Call(1, "slow", (5,)))
        unit.end_of_plan()
        records = await unit.drain()
        assert calls["slow"] == 3
        return records, log

    records, log = asyncio.run(scenario())
    assert records[1].state is TaskState.FAILED
    assert "timed out" in (records[1].error or "")
    assert [e.data["attempt"] for e in log.events if e.type == "task_retry"] == [1, 2]


def test_unknown_tool_bad_arity_and_type_errors_fail_without_retry() -> None:
    async def scenario() -> dict:
        trace = Trace()
        unit = TaskFetchingUnit(make_registry(trace), EventLog(), retry=FAST_RETRY)
        unit.submit(Call(1, "nope", ()))
        unit.submit(Call(2, "a", (1,)))
        unit.submit(Call(3, "a", (ListArg(()), "not-an-int", 0, "ok")))
        unit.end_of_plan()
        return await unit.drain()

    records = asyncio.run(scenario())
    assert "unknown tool" in records[1].error
    assert "missing" in records[2].error
    assert "expects int" in records[3].error
    assert all(r.state is TaskState.FAILED for r in records.values())
    assert records[3].attempts == 1


def test_reuse_map_short_circuits_identical_work() -> None:
    async def scenario() -> tuple[dict, Trace]:
        trace = Trace()
        first = TaskFetchingUnit(make_registry(trace), EventLog())
        first.submit(Spec(1, "a", (), 0, "ok").call())
        first.end_of_plan()
        done = await first.drain()
        reuse = {done[1].signature: done[1].result}
        trace2 = Trace()
        second = TaskFetchingUnit(make_registry(trace2), EventLog(), reuse=reuse)
        # Same work under a different number is still recognised.
        second.submit(Call(5, "a", (ListArg(()), 1, 0, "ok")))
        second.submit(Call(6, "b", (ListArg((Ref(5),)), 6, 0, "ok")))
        second.end_of_plan()
        return await second.drain(), trace2

    records, trace = asyncio.run(scenario())
    assert records[5].reused and records[5].result == 1
    assert not records[6].reused and records[6].result == 7
    assert set(trace.starts) == {6}


@given(st.integers(1, 12), st.integers(0, 2**32 - 1))
def test_backoff_is_bounded_full_jitter(attempt: int, seed: int) -> None:
    policy = RetryPolicy(base_delay_s=0.01, max_delay_s=0.2, multiplier=3.0)
    delay = policy.backoff(attempt, random.Random(seed))
    assert 0.0 <= delay <= min(0.2, 0.01 * 3.0 ** (attempt - 1))


def test_per_tool_limit_does_not_block_other_tools() -> None:
    async def scenario() -> list[tuple[str, int]]:
        trace = Trace()
        log = EventLog()
        unit = TaskFetchingUnit(
            make_registry(trace), log, limits=Limits(max_concurrency=3, per_tool={"a": 1})
        )
        for idx in range(1, 4):
            unit.submit(Spec(idx, "a", (), 3, "ok").call())
        unit.submit(Spec(4, "b", (), 0, "ok").call())
        unit.end_of_plan()
        await unit.drain()
        return [(e.data["tool"], e.data["idx"]) for e in log.events if e.type == "task_started"]

    order = asyncio.run(scenario())
    # Task 4 (tool b) starts second although a-tasks 2 and 3 were queued before it.
    assert order[:2] == [("a", 1), ("b", 4)]


def test_limits_validation() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Limits(max_concurrency=0)
    with pytest.raises(ValueError, match="per-tool"):
        Limits(per_tool={"a": 0})


def test_event_stream_orders_lifecycle_per_task() -> None:
    specs = [Spec(1, "a", (), 1, "flaky"), Spec(2, "b", (1,), 0, "ok")]
    _, _, log = asyncio.run(run_streaming(specs, Limits(), []))
    per_task: dict[int, list[str]] = defaultdict(list)
    for event in log.events:
        if "idx" in event.data:
            per_task[event.data["idx"]].append(event.type)
    assert per_task[1] == [
        "task_received",
        "task_started",
        "task_retry",
        "task_started",
        "task_succeeded",
    ]
    assert per_task[2] == ["task_received", "task_started", "task_succeeded"]
    seqs = [e.seq for e in log.events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
