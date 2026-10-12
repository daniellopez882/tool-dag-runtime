"""Types shared by planners, joiners and the runtime: one plan/execute/join round."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from tooldag.scheduler import TaskState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tooldag.plan.ast import ParseIssue, PlanItem
    from tooldag.scheduler import TaskRecord
    from tooldag.tools.base import ToolRegistry


class PlannerError(RuntimeError):
    """The planner could not produce a plan (bad question, provider failure)."""


@dataclass(frozen=True, slots=True)
class JoinDecision:
    action: Literal["finish", "replan"]
    answer: str | None
    reason: str


@dataclass(slots=True)
class RoundSummary:
    round: int
    plan_text: str = ""
    items: list[PlanItem] = field(default_factory=list)
    records: dict[int, TaskRecord] = field(default_factory=dict)
    issues: list[ParseIssue] = field(default_factory=list)
    planner_error: str | None = None
    decision: JoinDecision | None = None
    joiner_called: bool = False
    plan_complete_ms: float | None = None
    first_dispatch_ms: float | None = None

    def failures(self) -> list[TaskRecord]:
        return [r for r in self.records.values() if r.state is TaskState.FAILED]

    def executed(self) -> list[TaskRecord]:
        return [r for r in self.records.values() if r.attempts > 0]


@dataclass(frozen=True, slots=True)
class PlanContext:
    question: str
    options: tuple[str, ...]
    tools: ToolRegistry
    round: int
    history: tuple[RoundSummary, ...] = ()
    feedback: str | None = None


@dataclass(frozen=True, slots=True)
class JoinContext:
    question: str
    options: tuple[str, ...]
    round: int
    max_rounds: int
    summary: RoundSummary
    history: tuple[RoundSummary, ...] = ()


class Planner(Protocol):
    name: str

    def stream_plan(self, ctx: PlanContext) -> AsyncIterator[str]:
        """Yield the plan as text chunks of any size."""
        ...


class Joiner(Protocol):
    name: str

    async def decide(self, ctx: JoinContext) -> JoinDecision: ...
