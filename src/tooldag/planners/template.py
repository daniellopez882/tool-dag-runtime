"""Deterministic planner for "Find a movie similar to A, B, C" questions.

It writes the plan an LLM planner would be asked to write: resolve every title, fetch
and embed plots, compute each signal per candidate, rank, join. The plan is emitted
one line at a time so the task fetching unit can start resolving titles while the rest
of the plan is still being written. On a replan it drops the titles whose resolution
failed and the signals whose tools failed; unchanged calls are reused by signature.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from tooldag.plan.ast import Call, JoinMarker, ListArg, PlanItem, Ref, Thought
from tooldag.plan.render import render_plan
from tooldag.rounds import PlanContext, PlannerError
from tooldag.scheduler import TaskState
from tooldag.tools.scorer import SIGNALS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

_QUESTION = re.compile(
    r"^\s*(?:find\s+(?:me\s+)?(?:a\s+|an\s+)?(?:movie|film)s?\s+similar\s+to|"
    r"(?:movies|films)\s+similar\s+to|similar\s+to)\s+(?P<liked>.+?)[\s:?.]*$",
    re.IGNORECASE | re.DOTALL,
)
_SIGNAL_OF_TOOL = {
    "plot": "plot_similarity",
    "embed": "plot_similarity",
    "plot_similarity": "plot_similarity",
    "cooccurrence": "cooccurrence",
    "genre_jaccard": "genre_jaccard",
}


def comma_split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def liked_text_of(question: str) -> str:
    match = _QUESTION.match(question)
    if match is None:
        raise PlannerError("the template planner expects 'Find a movie similar to A, B, C'")
    return match.group("liked")


class TemplatePlanner:
    name = "template"

    def __init__(
        self,
        segment: Callable[[str], list[str]] = comma_split,
        *,
        signals: Sequence[str] = SIGNALS,
        line_delay_s: float = 0.0,
    ) -> None:
        unknown = set(signals) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals {sorted(unknown)}")
        self.segment = segment
        self.signals = tuple(s for s in SIGNALS if s in signals)
        # Synthetic pacing for the streaming-overlap experiment; zero everywhere else.
        self.line_delay_s = line_delay_s

    @staticmethod
    def _exclusions(ctx: PlanContext) -> tuple[set[str], set[str]]:
        titles: set[str] = set()
        signals: set[str] = set()
        for summary in ctx.history:
            for record in summary.records.values():
                if record.state is not TaskState.FAILED:
                    continue
                call = record.call
                if call.tool == "resolve_title" and call.args and isinstance(call.args[0], str):
                    titles.add(call.args[0])
                elif call.tool in _SIGNAL_OF_TOOL:
                    signals.add(_SIGNAL_OF_TOOL[call.tool])
        return titles, signals

    def build(self, ctx: PlanContext) -> list[PlanItem]:
        liked = self.segment(liked_text_of(ctx.question))
        excluded_titles, dropped = self._exclusions(ctx)
        liked = [t for t in liked if t not in excluded_titles]
        candidates = [o for o in ctx.options if o not in excluded_titles]
        signals = [s for s in self.signals if s not in dropped]
        if not liked or not candidates or not signals:
            raise PlannerError(
                "nothing left to plan with after excluding "
                f"titles {sorted(excluded_titles)} and signals {sorted(dropped)}"
            )
        items: list[PlanItem] = []
        if ctx.round == 1:
            items.append(
                Thought(
                    f"Resolve {len(liked)} liked titles and {len(candidates)} options, embed "
                    f"plots, score each option on {', '.join(signals)}, then rank."
                )
            )
        else:
            dropped_text = ", ".join(sorted(excluded_titles | dropped)) or "nothing"
            items.append(Thought(f"Replan without {dropped_text}; unchanged calls are reused."))
        counter = 0

        def add(tool: str, *args: object, **kwargs: object) -> Ref:
            nonlocal counter
            counter += 1
            items.append(Call(counter, tool, tuple(args), tuple(kwargs.items())))  # type: ignore[arg-type]
            return Ref(counter)

        with_plots = "plot_similarity" in signals
        liked_refs: list[Ref] = []
        liked_vectors: list[Ref] = []
        for title in liked:
            ref = add("resolve_title", title)
            liked_refs.append(ref)
            if with_plots:
                liked_vectors.append(add("embed", add("plot", ref)))
        candidate_refs: list[Ref] = []
        per_signal: dict[str, list[Ref]] = {s: [] for s in signals}
        for title in candidates:
            ref = add("resolve_title", title)
            candidate_refs.append(ref)
            vector = add("embed", add("plot", ref)) if with_plots else None
            if "cooccurrence" in signals:
                per_signal["cooccurrence"].append(
                    add("cooccurrence", ListArg(tuple(liked_refs)), ref)
                )
            if "genre_jaccard" in signals:
                per_signal["genre_jaccard"].append(
                    add("genre_jaccard", ListArg(tuple(liked_refs)), ref)
                )
            if vector is not None:
                per_signal["plot_similarity"].append(
                    add("plot_similarity", ListArg(tuple(liked_vectors)), vector)
                )
        add(
            "rank",
            ListArg(tuple(candidate_refs)),
            **{s: ListArg(tuple(per_signal[s])) for s in signals},
        )
        items.append(JoinMarker(counter + 1))
        return items

    async def stream_plan(self, ctx: PlanContext) -> AsyncIterator[str]:
        text = render_plan(self.build(ctx))
        loop = asyncio.get_running_loop()
        start = loop.time()
        for n, line in enumerate(text.splitlines(keepends=True), start=1):
            if self.line_delay_s > 0:
                # Line n is due at start + n * delay; sleeping to a deadline keeps late
                # wake-ups (timer granularity, a busy machine) from adding up.
                await asyncio.sleep(max(0.0, start + n * self.line_delay_s - loop.time()))
            else:
                # Hand control back each line, as a network stream would.
                await asyncio.sleep(0)
            yield line
