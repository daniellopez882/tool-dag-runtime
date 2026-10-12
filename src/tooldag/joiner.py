"""Joiner: after each round, answer or ask for a new plan; diff consecutive plans.

``RuleJoiner`` finishes when the answer task succeeded and otherwise replans with a
description of what failed. ``LLMJoiner`` asks a chat model for
``Action: Finish(...)`` or ``Action: Replan(...)``.

Plan diffing works on content signatures (``dag.call_signature``): a call in the new
plan with the same tool, arguments and upstream signatures as a call that already
succeeded is not executed again; the task fetching unit settles it from the reuse map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tooldag.events import preview
from tooldag.providers.base import ChatMessage, ChatProvider, ProviderError, complete
from tooldag.rounds import JoinContext, JoinDecision, RoundSummary
from tooldag.scheduler import TaskRecord, TaskState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_ACTION = re.compile(r"^\s*Action:\s*(Finish|Replan)\s*\((.*)\)\s*$", re.IGNORECASE | re.MULTILINE)
_THOUGHT = re.compile(r"Thought:\s*(.+)", re.IGNORECASE)


def problems_of(summary: RoundSummary) -> str:
    parts: list[str] = []
    if summary.planner_error:
        parts.append(f"planner failed: {summary.planner_error}")
    parts.extend(
        f"line {issue.line_no} not understood ({issue.message})" for issue in summary.issues
    )
    parts.extend(
        f"${r.call.idx} {r.call.tool} failed: {r.error}"
        for r in sorted(summary.records.values(), key=lambda r: r.call.idx)
        if r.state is TaskState.FAILED
    )
    cancelled = sum(1 for r in summary.records.values() if r.state is TaskState.CANCELLED)
    if cancelled:
        parts.append(f"{cancelled} dependent task(s) cancelled")
    if not summary.records and not summary.planner_error:
        parts.append("the plan contained no tasks")
    return "; ".join(parts) or "no problems recorded"


def answer_record(summary: RoundSummary, answer_tools: Sequence[str]) -> TaskRecord | None:
    """The task whose result answers the question: the last answer-tool call, or the last call."""
    records = sorted(summary.records.values(), key=lambda r: r.call.idx)
    if answer_tools:
        records = [r for r in records if r.call.tool in answer_tools]
    return records[-1] if records else None


class RuleJoiner:
    name = "rule"

    def __init__(self, answer_tools: Sequence[str] = ("rank",)) -> None:
        self.answer_tools = tuple(answer_tools)

    async def decide(self, ctx: JoinContext) -> JoinDecision:
        record = answer_record(ctx.summary, self.answer_tools)
        if record is not None and record.state is TaskState.SUCCEEDED:
            return JoinDecision(
                "finish", str(record.result), f"${record.call.idx} produced the answer"
            )
        problems = problems_of(ctx.summary)
        if ctx.round < ctx.max_rounds:
            return JoinDecision("replan", None, problems)
        return JoinDecision("finish", None, f"no answer after {ctx.round} round(s): {problems}")


JOIN_SYSTEM_PROMPT = """You check whether an agent's tool results answer a question.
Reply in exactly this form:
Thought: <one sentence>
Action: Finish(<the chosen option, copied exactly>)
or
Action: Replan(<what the next plan must change>)"""


def join_messages(ctx: JoinContext) -> list[ChatMessage]:
    lines = [f"Question: {ctx.question}", f"Options: {'; '.join(ctx.options)}", "Results:"]
    for idx in sorted(ctx.summary.records):
        record = ctx.summary.records[idx]
        detail = record.error if record.error else str(preview(record.result))
        lines.append(f"  ${idx} {record.call.tool} [{record.state.value}]: {detail}")
    lines.append(f"Round {ctx.round} of at most {ctx.max_rounds}.")
    return [
        {"role": "system", "content": JOIN_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_join_reply(text: str) -> JoinDecision:
    thought = _THOUGHT.search(text)
    reason = thought.group(1).strip().splitlines()[0] if thought else ""
    # Use the last Action line; models sometimes restate the format before answering.
    actions = list(_ACTION.finditer(text))
    action = actions[-1] if actions else None
    if action is None:
        return JoinDecision("replan", None, f"joiner reply not understood: {text.strip()[:120]}")
    verb, body = action.group(1).lower(), action.group(2).strip().strip('"').strip("'")
    if verb == "finish":
        return JoinDecision("finish", body or None, reason or "joiner chose an answer")
    return JoinDecision("replan", None, body or reason or "joiner asked for a new plan")


class LLMJoiner:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.name = f"llm via {provider.name}"

    async def decide(self, ctx: JoinContext) -> JoinDecision:
        try:
            reply = await complete(self.provider, join_messages(ctx), max_tokens=300)
        except ProviderError as exc:
            reply = f"Action: Replan(joiner call failed: {exc})"
        decision = parse_join_reply(reply)
        if decision.action == "replan" and ctx.round >= ctx.max_rounds:
            return JoinDecision("finish", None, f"out of rounds; last feedback: {decision.reason}")
        return decision


@dataclass(frozen=True, slots=True)
class PlanDiff:
    reused: tuple[int, ...]
    executed: tuple[int, ...]
    dropped: int

    def to_json(self) -> dict[str, Any]:
        return {
            "reused": list(self.reused),
            "executed": list(self.executed),
            "dropped": self.dropped,
        }


def diff_rounds(previous: Iterable[RoundSummary], current: RoundSummary) -> PlanDiff:
    """What the new round reused, what it actually ran, and how many old calls it dropped."""
    old = {
        r.signature
        for summary in previous
        for r in summary.records.values()
        if r.signature is not None
    }
    new = {r.signature for r in current.records.values() if r.signature is not None}
    reused = tuple(sorted(i for i, r in current.records.items() if r.reused))
    executed = tuple(sorted(i for i, r in current.records.items() if r.attempts > 0))
    return PlanDiff(reused, executed, len(old - new))


def reusable_results(summary: RoundSummary) -> Mapping[str, Any]:
    return {
        r.signature: r.result
        for r in summary.records.values()
        if r.state is TaskState.SUCCEEDED and r.signature is not None
    }
