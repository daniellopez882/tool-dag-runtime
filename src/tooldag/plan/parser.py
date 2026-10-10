"""Incremental parser for LLMCompiler-style plans.

The planner's output arrives as arbitrary text chunks (token deltas from a model, or
lines from the template planner). ``StreamingPlanParser.feed`` buffers the partial
line and returns every plan item whose line is complete, so the scheduler can dispatch
task 1 while the planner is still writing task 5.

Grammar, one item per line::

    Thought: free text
    3. tool_name("literal", $1, [$2, ${4}], k=0.5, label="from $1")
    9. join()
    <END_OF_PLAN>

``$N`` inside a string literal is interpolated (``$$`` writes a literal dollar sign).
Lines that do not parse become ``ParseIssue`` items; the parser never raises.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tooldag.plan.ast import (
    Arg,
    Call,
    JoinMarker,
    ListArg,
    ParseIssue,
    Plan,
    PlanItem,
    Ref,
    Template,
    Thought,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

END_OF_PLAN = "<END_OF_PLAN>"
JOIN_TOOL = "join"

_TASK_HEAD = re.compile(r"\s*(\d+)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_THOUGHT = re.compile(r"\s*thought\s*:\s*(.*)", re.IGNORECASE)
_NUMBER = re.compile(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS: dict[str, Arg] = {
    "True": True,
    "False": False,
    "None": None,
    "true": True,
    "false": False,
    "null": None,
}
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}
# Plans are short; a hard cap stops a runaway model from making one line arbitrarily expensive.
MAX_LINE_CHARS = 8192
MAX_NESTING = 16


class _MalformedError(Exception):
    pass


class _ArgReader:
    """Recursive-descent reader for the argument list of one call."""

    def __init__(self, text: str, pos: int) -> None:
        self.text = text
        self.pos = pos

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def read_call_args(self) -> tuple[tuple[Arg, ...], tuple[tuple[str, Arg], ...]]:
        args: list[Arg] = []
        kwargs: list[tuple[str, Arg]] = []
        seen: set[str] = set()
        self._skip_ws()
        if self._peek() == ")":
            self.pos += 1
            return (), ()
        while True:
            self._skip_ws()
            name = self._keyword_name()
            value = self._value(0)
            if name is None:
                if kwargs:
                    raise _MalformedError("positional argument after keyword argument")
                args.append(value)
            else:
                if name in seen:
                    raise _MalformedError(f"keyword argument {name!r} repeated")
                seen.add(name)
                kwargs.append((name, value))
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.pos += 1
                self._skip_ws()
                if self._peek() == ")":
                    self.pos += 1
                    return tuple(args), tuple(kwargs)
                continue
            if ch == ")":
                self.pos += 1
                return tuple(args), tuple(kwargs)
            if not ch:
                raise _MalformedError("unclosed argument list")
            raise _MalformedError(f"unexpected {ch!r} at column {self.pos + 1}")

    def _keyword_name(self) -> str | None:
        match = _IDENT.match(self.text, self.pos)
        if match is None or match.group(0) in _KEYWORDS:
            return None
        after = match.end()
        while after < len(self.text) and self.text[after].isspace():
            after += 1
        if (
            after < len(self.text)
            and self.text[after] == "="
            and self.text[after : after + 2] != "=="
        ):
            self.pos = after + 1
            self._skip_ws()
            return match.group(0)
        return None

    def _value(self, depth: int) -> Arg:
        if depth > MAX_NESTING:
            raise _MalformedError("lists nested too deeply")
        self._skip_ws()
        ch = self._peek()
        if ch in ('"', "'"):
            return self._string(ch)
        if ch == "$":
            return self._ref()
        if ch == "[":
            return self._list(depth)
        number = _NUMBER.match(self.text, self.pos)
        if number is not None:
            self.pos = number.end()
            token = number.group(0)
            if any(c in token for c in ".eE"):
                return float(token)
            return int(token)
        ident = _IDENT.match(self.text, self.pos)
        if ident is not None:
            word = ident.group(0)
            if word in _KEYWORDS:
                self.pos = ident.end()
                return _KEYWORDS[word]
            raise _MalformedError(f"bare word {word!r}; quote string arguments")
        if not ch:
            raise _MalformedError("argument missing")
        raise _MalformedError(f"unexpected {ch!r} at column {self.pos + 1}")

    def _ref_index(self) -> int:
        # Called with self.pos just after "$".
        if self._peek() == "{":
            end = self.text.find("}", self.pos)
            digits = self.text[self.pos + 1 : end] if end >= 0 else ""
            if not digits.isdigit():
                raise _MalformedError("malformed ${N} reference")
            self.pos = end + 1
            return int(digits)
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == start:
            raise _MalformedError("'$' must be followed by a task number")
        return int(self.text[start : self.pos])

    def _ref(self) -> Ref:
        self.pos += 1
        return Ref(self._ref_index())

    def _list(self, depth: int) -> ListArg:
        self.pos += 1
        items: list[Arg] = []
        self._skip_ws()
        if self._peek() == "]":
            self.pos += 1
            return ListArg(())
        while True:
            items.append(self._value(depth + 1))
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.pos += 1
                self._skip_ws()
                if self._peek() == "]":
                    self.pos += 1
                    return ListArg(tuple(items))
                continue
            if ch == "]":
                self.pos += 1
                return ListArg(tuple(items))
            if not ch:
                raise _MalformedError("unclosed list")
            raise _MalformedError(f"unexpected {ch!r} in list at column {self.pos + 1}")

    def _string(self, quote: str) -> str | Template:
        self.pos += 1
        parts: list[str | Ref] = []
        buf: list[str] = []
        text = self.text
        while True:
            if self.pos >= len(text):
                raise _MalformedError("unterminated string")
            ch = text[self.pos]
            if ch == quote:
                self.pos += 1
                break
            if ch == "\\":
                if self.pos + 1 >= len(text):
                    raise _MalformedError("unterminated string")
                nxt = text[self.pos + 1]
                buf.append(_ESCAPES.get(nxt, "\\" + nxt))
                self.pos += 2
                continue
            if ch == "$":
                if text[self.pos + 1 : self.pos + 2] == "$":
                    buf.append("$")
                    self.pos += 2
                    continue
                nxt = text[self.pos + 1 : self.pos + 2]
                if nxt.isdigit() or nxt == "{":
                    self.pos += 1
                    if buf:
                        parts.append("".join(buf))
                        buf = []
                    parts.append(Ref(self._ref_index()))
                    continue
            buf.append(ch)
            self.pos += 1
        if buf or not parts:
            parts.append("".join(buf))
        if any(isinstance(p, Ref) for p in parts):
            return Template(tuple(parts))
        return "".join(p for p in parts if isinstance(p, str))


def parse_line(line: str, line_no: int) -> list[PlanItem]:
    """Parse one complete line. Returns zero, one or two items (a call plus an end marker)."""
    text = line.rstrip("\r")
    stripped = text.strip()
    if not stripped or stripped.startswith("```"):
        return []
    if len(text) > MAX_LINE_CHARS:
        return [ParseIssue(line_no, text[:120], f"line longer than {MAX_LINE_CHARS} characters")]
    if END_OF_PLAN in text:
        head = text[: text.index(END_OF_PLAN)]
        items = parse_line(head, line_no) if head.strip() else []
        if not any(isinstance(item, JoinMarker) for item in items):
            items.append(JoinMarker(None, line_no))
        return items
    thought = _THOUGHT.fullmatch(text)
    if thought is not None:
        return [Thought(thought.group(1).strip(), line_no)]
    head_match = _TASK_HEAD.match(text)
    if head_match is None:
        return [ParseIssue(line_no, text, "expected 'N. tool(args)', 'Thought: ...' or join()")]
    idx = int(head_match.group(1))
    tool = head_match.group(2)
    reader = _ArgReader(text, head_match.end())
    try:
        args, kwargs = reader.read_call_args()
    except _MalformedError as exc:
        return [ParseIssue(line_no, text, str(exc))]
    trailing = text[reader.pos :].strip()
    if trailing:
        return [ParseIssue(line_no, text, f"unexpected text after call: {trailing[:40]!r}")]
    if idx <= 0:
        return [ParseIssue(line_no, text, "task numbers start at 1")]
    if tool == JOIN_TOOL:
        if args or kwargs:
            return [ParseIssue(line_no, text, "join() takes no arguments")]
        return [JoinMarker(idx, line_no)]
    return [Call(idx, tool, args, kwargs, line_no)]


class StreamingPlanParser:
    """Feed it text chunks of any size; it yields plan items as soon as each line is whole.

    After a join marker the parser is ``done`` and ignores further input, which lets the
    caller stop consuming the planner stream early.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._line_no = 0
        self._done = False
        # Set after an over-long line was reported; the rest of that line is discarded.
        self._skipping = False

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, chunk: str) -> list[PlanItem]:
        if self._done or not chunk:
            return []
        self._buf += chunk
        out: list[PlanItem] = []
        while not self._done:
            newline = self._buf.find("\n")
            if newline < 0:
                break
            line = self._buf[:newline]
            self._buf = self._buf[newline + 1 :]
            if self._skipping:
                self._skipping = False
                continue
            out.extend(self._emit(line))
        if self._skipping:
            self._buf = ""
        elif not self._done and len(self._buf.rstrip("\r")) > MAX_LINE_CHARS:
            # A newline may never come; report the line now instead of buffering without bound.
            self._line_no += 1
            message = f"line longer than {MAX_LINE_CHARS} characters"
            out.append(ParseIssue(self._line_no, self._buf[:120], message))
            self._buf = ""
            self._skipping = True
        return out

    def close(self) -> list[PlanItem]:
        """Flush the final line (plans need not end with a newline)."""
        if self._done:
            return []
        line, self._buf = self._buf, ""
        out = self._emit(line) if line and not self._skipping else []
        self._done = True
        return out

    def _emit(self, line: str) -> list[PlanItem]:
        self._line_no += 1
        items = parse_line(line, self._line_no)
        if any(isinstance(item, JoinMarker) for item in items):
            self._done = True
        return items


def parse_plan(text: str) -> Plan:
    """Parse a whole plan at once (same result as streaming it in any chunking)."""
    return Plan(list(stream_items([text])))


def stream_items(chunks: Iterable[str]) -> list[PlanItem]:
    parser = StreamingPlanParser()
    items: list[PlanItem] = []
    for chunk in chunks:
        items.extend(parser.feed(chunk))
        if parser.done:
            break
    items.extend(parser.close())
    return items
