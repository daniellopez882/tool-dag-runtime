"""Plan items produced by the streaming parser.

A plan is a numbered list of tool calls. Arguments are literals, lists, ``$N`` references
to the result of task ``N``, or string templates that interpolate ``$N`` into text.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Ref:
    """A reference to the result of task ``idx`` (written ``$idx`` or ``${idx}``)."""

    idx: int


@dataclass(frozen=True, slots=True)
class Template:
    """A string literal that contains ``$N`` references; resolved by ``str()`` of each result."""

    parts: tuple[str | Ref, ...]


@dataclass(frozen=True, slots=True)
class ListArg:
    items: tuple[Arg, ...]


type Scalar = str | int | float | bool | None
type Arg = Scalar | Ref | Template | ListArg


@dataclass(frozen=True, slots=True)
class Call:
    """One numbered tool call, e.g. ``3. embed($2)``."""

    idx: int
    tool: str
    args: tuple[Arg, ...] = ()
    kwargs: tuple[tuple[str, Arg], ...] = ()
    line_no: int = field(default=0, compare=False)

    @property
    def deps(self) -> frozenset[int]:
        found: set[int] = set()
        for arg in (*self.args, *(v for _, v in self.kwargs)):
            _collect_refs(arg, found)
        return frozenset(found)


@dataclass(frozen=True, slots=True)
class Thought:
    text: str
    line_no: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class JoinMarker:
    """``N. join()`` or ``<END_OF_PLAN>``: the planner has finished this round."""

    idx: int | None = None
    line_no: int = field(default=0, compare=False)


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """A line the parser could not turn into a plan item. The rest of the plan continues."""

    line_no: int
    text: str
    message: str


type PlanItem = Call | Thought | JoinMarker | ParseIssue


@dataclass(slots=True)
class Plan:
    """Everything parsed from one planner round, in arrival order."""

    items: list[PlanItem] = field(default_factory=list)

    @property
    def calls(self) -> list[Call]:
        return [item for item in self.items if isinstance(item, Call)]

    @property
    def issues(self) -> list[ParseIssue]:
        return [item for item in self.items if isinstance(item, ParseIssue)]

    @property
    def complete(self) -> bool:
        return any(isinstance(item, JoinMarker) for item in self.items)


def _collect_refs(arg: Arg, found: set[int]) -> None:
    match arg:
        case Ref(idx=idx):
            found.add(idx)
        case Template(parts=parts):
            found.update(p.idx for p in parts if isinstance(p, Ref))
        case ListArg(items=items):
            for item in items:
                _collect_refs(item, found)
        case _:
            pass
