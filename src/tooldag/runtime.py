"""The LLMCompiler loop: stream a plan, dispatch tasks as they parse, join, maybe replan."""

from __future__ import annotations

import asyncio
import contextlib
import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tooldag.events import EventLog
from tooldag.joiner import diff_rounds, reusable_results
from tooldag.plan.ast import Call, JoinMarker, ParseIssue, Thought
from tooldag.plan.parser import StreamingPlanParser
from tooldag.rounds import JoinContext, JoinDecision, PlanContext, PlannerError, RoundSummary
from tooldag.scheduler import Limits, RetryPolicy, SequentialExecutor, TaskFetchingUnit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Executor

    from tooldag.plan.ast import PlanItem
    from tooldag.rounds import Joiner, Planner
    from tooldag.scheduler import RoundExecutor
    from tooldag.tools.base import ToolRegistry


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    limits: Limits = field(default_factory=Limits)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_rounds: int = 3
    mode: Literal["dag", "sequential"] = "dag"
    # False buffers the whole plan before submitting it (the "plan, then execute" baseline).
    stream_dispatch: bool = True
    seed: int = 0


@dataclass(slots=True)
class RunResult:
    run_id: str
    question: str
    answer: str | None
    answer_result: Any
    decision: JoinDecision | None
    rounds: list[RoundSummary]
    makespan_ms: float
    error: str | None = None

    @property
    def tasks_executed(self) -> int:
        return sum(len(s.executed()) for s in self.rounds)

    @property
    def tasks_reused(self) -> int:
        return sum(1 for s in self.rounds for r in s.records.values() if r.reused)

    def model_calls(self) -> dict[str, int]:
        """Structural count of chat-model calls, had a model been planning and joining.

        LLMCompiler: one planner call and one joiner call per round. A ReAct agent issues
        one model call per tool call plus one for the final answer.
        """
        joined = sum(1 for s in self.rounds if s.joiner_called)
        return {
            "llmcompiler": len(self.rounds) + joined,
            "react_equivalent": self.tasks_executed + 1,
        }


class Runtime:
    def __init__(
        self,
        tools: ToolRegistry,
        planner: Planner,
        joiner: Joiner,
        config: RuntimeConfig | None = None,
        *,
        executor: Executor | None = None,
    ) -> None:
        self.tools = tools
        self.planner = planner
        self.joiner = joiner
        self.config = config or RuntimeConfig()
        self.executor = executor

    def _unit(self, log: EventLog, round_: int, reuse: dict[str, Any]) -> RoundExecutor:
        cls = TaskFetchingUnit if self.config.mode == "dag" else SequentialExecutor
        return cls(
            self.tools,
            log,
            round_=round_,
            limits=self.config.limits,
            retry=self.config.retry,
            executor=self.executor,
            rng=random.Random(self.config.seed * 1000 + round_),
            reuse=dict(reuse),
        )

    async def run(
        self,
        question: str,
        options: Sequence[str] = (),
        *,
        log: EventLog | None = None,
    ) -> RunResult:
        if log is None:
            log = EventLog(uuid.uuid4().hex[:12])
        cfg = self.config
        log.emit(
            "run_started",
            question=question,
            options=list(options),
            planner=self.planner.name,
            joiner=self.joiner.name,
            mode=cfg.mode,
            stream_dispatch=cfg.stream_dispatch and cfg.mode == "dag",
            max_concurrency=cfg.limits.max_concurrency,
            max_rounds=cfg.max_rounds,
        )
        rounds: list[RoundSummary] = []
        reuse: dict[str, Any] = {}
        decision: JoinDecision | None = None
        answer_result: Any = None
        unit: RoundExecutor | None = None
        try:
            for round_ in range(1, cfg.max_rounds + 1):
                summary = RoundSummary(round_)
                unit = self._unit(log, round_, reuse)
                await self._plan_round(question, options, rounds, decision, summary, unit, log)
                summary.records = await unit.drain()
                started = [
                    r.started_ms for r in summary.records.values() if r.started_ms is not None
                ]
                summary.first_dispatch_ms = min(started, default=None)
                if rounds:
                    log.emit("plan_diff", round_, **diff_rounds(rounds, summary).to_json())
                rounds.append(summary)
                reuse.update(reusable_results(summary))
                if summary.planner_error is not None and not summary.records:
                    reason = f"planner failed: {summary.planner_error}"
                    decision = JoinDecision("finish", None, reason)
                else:
                    join_ctx = JoinContext(
                        question,
                        tuple(options),
                        round_,
                        cfg.max_rounds,
                        summary,
                        tuple(rounds[:-1]),
                    )
                    decision = await self.joiner.decide(join_ctx)
                    summary.joiner_called = True
                summary.decision = decision
                log.emit(
                    "join_decision",
                    round_,
                    action=decision.action,
                    answer=decision.answer,
                    reason=decision.reason,
                )
                if decision.action == "finish":
                    answer_result = self._answer_result(summary)
                    break
                log.emit("replan", round_ + 1, feedback=decision.reason, reusable=len(reuse))
        except asyncio.CancelledError:
            if unit is not None:
                unit.cancel("run cancelled")
                with contextlib.suppress(asyncio.CancelledError):
                    await unit.drain()
            log.emit("run_cancelled", reason="run cancelled")
            log.close()
            raise
        except Exception as exc:
            if unit is not None:
                unit.cancel(f"run failed: {exc}")
            log.emit("run_failed", error=f"{type(exc).__name__}: {exc}")
            log.close()
            raise
        result = RunResult(
            run_id=log.run_id,
            question=question,
            answer=decision.answer if decision else None,
            answer_result=answer_result,
            decision=decision,
            rounds=rounds,
            makespan_ms=log.elapsed_ms(),
        )
        log.emit(
            "run_finished",
            answer=result.answer,
            rounds=len(rounds),
            tasks_executed=result.tasks_executed,
            tasks_reused=result.tasks_reused,
            model_calls=result.model_calls(),
            makespan_ms=round(result.makespan_ms, 3),
        )
        log.close()
        return result

    def _answer_result(self, summary: RoundSummary) -> Any:
        records = sorted(summary.records.values(), key=lambda r: r.call.idx)
        for record in reversed(records):
            if record.call.tool == "rank" and record.result is not None:
                return record.result
        return records[-1].result if records else None

    async def _plan_round(
        self,
        question: str,
        options: Sequence[str],
        history: list[RoundSummary],
        decision: JoinDecision | None,
        summary: RoundSummary,
        unit: RoundExecutor,
        log: EventLog,
    ) -> None:
        round_ = summary.round
        ctx = PlanContext(
            question,
            tuple(options),
            self.tools,
            round_,
            tuple(history),
            decision.reason if decision else None,
        )
        parser = StreamingPlanParser()
        buffered: list[Call] = []
        stream = self.config.stream_dispatch and self.config.mode == "dag"
        log.emit("plan_started", round_)

        def handle(items: list[PlanItem]) -> None:
            for item in items:
                summary.items.append(item)
                match item:
                    case Call():
                        if stream:
                            unit.submit(item)
                        else:
                            buffered.append(item)
                    case Thought(text=text):
                        log.emit("thought", round_, text=text)
                    case JoinMarker(idx=idx):
                        log.emit("plan_join", round_, idx=idx)
                    case ParseIssue():
                        summary.issues.append(item)
                        log.emit(
                            "plan_issue",
                            round_,
                            line=item.line_no,
                            text=item.text[:120],
                            message=item.message,
                        )

        try:
            async with contextlib.aclosing(self.planner.stream_plan(ctx)) as chunks:
                async for chunk in chunks:
                    summary.plan_text += chunk
                    handle(parser.feed(chunk))
                    if parser.done:
                        break
        except PlannerError as exc:
            summary.planner_error = str(exc)
            log.emit("plan_error", round_, error=str(exc))
        handle(parser.close())
        summary.plan_complete_ms = log.elapsed_ms()
        log.emit(
            "plan_complete",
            round_,
            tasks=sum(1 for i in summary.items if isinstance(i, Call)),
            issues=len(summary.issues),
            join_seen=any(isinstance(i, JoinMarker) for i in summary.items),
        )
        for call in buffered:
            unit.submit(call)
        unit.end_of_plan()
