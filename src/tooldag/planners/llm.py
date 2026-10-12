"""Planner backed by a chat model through the provider interface.

The model's token stream goes straight into the incremental parser, so tasks are
dispatched while the model is still writing the plan. Not exercised by the benchmarks
(no model is run there); covered by tests with the scripted fake provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tooldag.events import preview
from tooldag.providers.base import ChatMessage, ChatProvider, ProviderError
from tooldag.rounds import PlanContext, PlannerError, RoundSummary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SYSTEM_PROMPT = """You write execution plans for a movie-recommendation agent.

Tools:
{tools}

Write one item per line:
Thought: <optional reasoning>
<n>. <tool>(<arguments>)
...
<n>. join()
<END_OF_PLAN>

Rules:
- Number tasks from 1 in the order you write them.
- Arguments are "strings", numbers, [lists], true/false/null, name=value keyword
  arguments, or $k for the result of task k (lists may contain $k).
- Use only results of earlier tasks through $k; never invent values.
- Independent tasks run in parallel as soon as their inputs exist, so prefer wide
  plans over long chains.
- Finish with rank(...) over all options, then join().

Example:
Question: Find a movie similar to Heat, Ronin
Options: Collateral; Notting Hill
Thought: resolve every title, then compare each option with the liked films.
1. resolve_title("Heat")
2. resolve_title("Ronin")
3. resolve_title("Collateral")
4. resolve_title("Notting Hill")
5. cooccurrence([$1, $2], $3)
6. cooccurrence([$1, $2], $4)
7. genre_jaccard([$1, $2], $3)
8. genre_jaccard([$1, $2], $4)
9. rank([$3, $4], cooccurrence=[$5, $6], genre_jaccard=[$7, $8])
10. join()
<END_OF_PLAN>"""


def describe_round(summary: RoundSummary) -> str:
    lines = [f"Plan from round {summary.round}:", summary.plan_text.strip(), "Outcomes:"]
    for idx in sorted(summary.records):
        record = summary.records[idx]
        state = record.state.value
        detail = record.error or str(preview(record.result))
        lines.append(f"  ${idx} {record.call.tool}: {state} - {detail}")
    lines.extend(f"  line {issue.line_no}: {issue.message}" for issue in summary.issues)
    return "\n".join(lines)


def plan_messages(ctx: PlanContext) -> list[ChatMessage]:
    tools = "\n".join(f"- {t.signature()}: {t.description}" for t in ctx.tools)
    user = f"Question: {ctx.question}\nOptions: {'; '.join(ctx.options)}"
    if ctx.history:
        rounds = "\n\n".join(describe_round(s) for s in ctx.history)
        user += (
            f"\n\n{rounds}\n\nJoiner feedback: {ctx.feedback or 'none'}\n"
            "Write a new complete plan numbered from 1. Calls identical to ones that "
            "already succeeded are not re-run."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools)},
        {"role": "user", "content": user},
    ]


class LLMPlanner:
    def __init__(self, provider: ChatProvider, *, max_tokens: int = 1500) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.name = f"llm via {provider.name}"

    async def stream_plan(self, ctx: PlanContext) -> AsyncIterator[str]:
        try:
            async for delta in self.provider.stream(
                plan_messages(ctx), max_tokens=self.max_tokens
            ):
                yield delta
        except ProviderError as exc:
            raise PlannerError(str(exc)) from exc
