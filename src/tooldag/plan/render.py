"""Write plan items back to the text form the parser reads."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tooldag.plan.ast import Arg, Call, JoinMarker, ListArg, ParseIssue, Ref, Template, Thought
from tooldag.plan.parser import END_OF_PLAN

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tooldag.plan.ast import PlanItem

_STRING_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r", "$": "$$"}


def _quote(text: str) -> str:
    return '"' + "".join(_STRING_ESCAPES.get(ch, ch) for ch in text) + '"'


def render_arg(arg: Arg) -> str:
    match arg:
        case bool() | None:
            return repr(arg)
        case int():
            return str(arg)
        case float():
            if not math.isfinite(arg):
                raise ValueError("plans cannot carry non-finite numbers")
            return repr(arg)
        case str():
            return _quote(arg)
        case Ref(idx=idx):
            return f"${idx}"
        case Template(parts=parts):
            # ${N} keeps a following digit from being read as part of the task number.
            body = "".join(
                f"${{{p.idx}}}" if isinstance(p, Ref) else _quote(p)[1:-1] for p in parts
            )
            return f'"{body}"'
        case ListArg(items=items):
            return "[" + ", ".join(render_arg(item) for item in items) + "]"
    raise TypeError(f"cannot render {type(arg).__name__}")


def render_call(call: Call) -> str:
    parts = [render_arg(a) for a in call.args]
    parts += [f"{name}={render_arg(value)}" for name, value in call.kwargs]
    return f"{call.idx}. {call.tool}({', '.join(parts)})"


def render_item(item: PlanItem) -> str:
    match item:
        case Call():
            return render_call(item)
        case Thought(text=text):
            return f"Thought: {text}"
        case JoinMarker(idx=None):
            return END_OF_PLAN
        case JoinMarker(idx=idx):
            return f"{idx}. join()"
        case ParseIssue(text=text):
            return text
    raise TypeError(f"cannot render {type(item).__name__}")


def render_plan(items: Iterable[PlanItem]) -> str:
    return "".join(render_item(item) + "\n" for item in items)
