from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tooldag.plan import (
    END_OF_PLAN,
    Call,
    JoinMarker,
    ListArg,
    ParseIssue,
    Ref,
    StreamingPlanParser,
    Template,
    Thought,
    parse_line,
    parse_plan,
    render_plan,
)
from tooldag.plan.parser import _KEYWORDS, MAX_LINE_CHARS, stream_items

IDENT = st.builds(
    str.__add__,
    st.sampled_from(string.ascii_letters + "_"),
    st.text(string.ascii_letters + string.digits + "_", max_size=11),
)
KWARG_NAME = IDENT.filter(lambda s: s not in _KEYWORDS)
TOOL_NAME = IDENT.filter(lambda s: s != "join")
SAFE_TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=20)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**12), 10**12),
    st.floats(allow_nan=False, allow_infinity=False),
    SAFE_TEXT,
)
refs = st.integers(1, 40).map(Ref)


@st.composite
def templates(draw: st.DrawFn) -> Template:
    # Canonical form only: refs separated by non-empty text, at least one ref.
    n_refs = draw(st.integers(1, 3))
    parts: list[str | Ref] = []
    if draw(st.booleans()):
        parts.append(draw(SAFE_TEXT.filter(bool)))
    for i in range(n_refs):
        parts.append(draw(refs))
        if i < n_refs - 1 or draw(st.booleans()):
            parts.append(draw(SAFE_TEXT.filter(bool)))
    return Template(tuple(parts))


args = st.recursive(
    st.one_of(scalars, refs, templates()),
    lambda inner: st.lists(inner, max_size=4).map(lambda xs: ListArg(tuple(xs))),
    max_leaves=8,
)


@st.composite
def calls(draw: st.DrawFn, idx: int) -> Call:
    positional = tuple(draw(st.lists(args, max_size=3)))
    names = draw(st.lists(KWARG_NAME, max_size=2, unique=True))
    kwargs = tuple((name, draw(args)) for name in names)
    return Call(idx, draw(TOOL_NAME), positional, kwargs)


thoughts = (
    st.text(st.characters(exclude_categories=("Cs", "Cc", "Zl", "Zp")), min_size=1, max_size=30)
    .map(str.strip)
    .filter(lambda s: s and END_OF_PLAN not in s)
    .map(Thought)
)


@st.composite
def plans(draw: st.DrawFn) -> list[Call | Thought | JoinMarker]:
    n = draw(st.integers(0, 6))
    items: list[Call | Thought | JoinMarker] = []
    for idx in range(1, n + 1):
        if draw(st.booleans()):
            items.append(draw(thoughts))
        items.append(draw(calls(idx)))
    if draw(st.booleans()):
        items.append(JoinMarker(n + 1))
    return items


def chunked(text: str, cuts: list[int]) -> list[str]:
    points = sorted({c % (len(text) + 1) for c in cuts})
    pieces, start = [], 0
    for point in points:
        pieces.append(text[start:point])
        start = point
    pieces.append(text[start:])
    return pieces


@given(plans())
@settings(max_examples=300)
def test_render_then_parse_round_trips(items: list[Call | Thought | JoinMarker]) -> None:
    assert parse_plan(render_plan(items)).items == items


@given(plans(), st.lists(st.integers(0, 10_000), max_size=12))
@settings(max_examples=300)
def test_streaming_matches_whole_text_for_any_chunking(
    items: list[Call | Thought | JoinMarker], cuts: list[int]
) -> None:
    text = render_plan(items)
    assert stream_items(chunked(text, cuts)) == parse_plan(text).items


@given(st.text(max_size=300), st.lists(st.integers(0, 400), max_size=10))
@settings(max_examples=400)
def test_parser_is_total_and_chunking_invariant_on_arbitrary_text(
    text: str, cuts: list[int]
) -> None:
    whole = parse_plan(text).items
    assert stream_items(chunked(text, cuts)) == whole
    for item in whole:
        assert isinstance(item, Call | Thought | JoinMarker | ParseIssue)


@given(
    st.lists(
        st.sampled_from(
            [
                '1. f("a"',
                "2. g($)",
                "3. h(x)",
                '4. k("unterminated)',
                "5. m([1, 2)",
                "6. n(a=1, 2)",
                "7. p(a=1, a=2)",
                "8. q() trailing",
                "free text line",
                "0. zero()",
                "9. join(1)",
                '10. ok("fine", $1)',
                "Thought: keep going",
            ]
        ),
        max_size=12,
    )
)
def test_malformed_lines_become_issues_and_do_not_stop_the_plan(lines: list[str]) -> None:
    items = parse_plan("\n".join(lines)).items
    good = [i for i in items if isinstance(i, Call)]
    assert all(c.tool == "ok" for c in good)
    expected_issues = sum(1 for line in lines if not line.startswith(("10.", "Thought")))
    assert len([i for i in items if isinstance(i, ParseIssue)]) == expected_issues


def test_parses_lllmcompiler_style_plan() -> None:
    text = (
        "Thought: resolve both titles first\n"
        '1. resolve_title("Batman")\n'
        "2. resolve_title('The Mask')\n"
        "3. cooccurrence([$1], $2)\n"
        '4. note("score was $3 for ${1}0 and $$5")\n'
        "5. join()<END_OF_PLAN>\n"
        '6. ignored("after join")\n'
    )
    items = parse_plan(text).items
    assert items[0] == Thought("resolve both titles first")
    assert items[1] == Call(1, "resolve_title", ("Batman",))
    assert items[2] == Call(2, "resolve_title", ("The Mask",))
    assert items[3] == Call(3, "cooccurrence", (ListArg((Ref(1),)), Ref(2)))
    assert items[3].deps == frozenset({1, 2})
    assert items[4] == Call(
        4, "note", (Template(("score was ", Ref(3), " for ", Ref(1), "0 and $5")),)
    )
    assert items[5] == JoinMarker(5)
    assert len(items) == 6


def test_keyword_arguments_literals_and_trailing_commas() -> None:
    (call,) = parse_line("7. rank([$1, $2,], weights=[0.5, -2, 1e-3], label=None, on=true,)", 1)
    assert isinstance(call, Call)
    assert call.args == (ListArg((Ref(1), Ref(2))),)
    assert dict(call.kwargs) == {
        "weights": ListArg((0.5, -2, 0.001)),
        "label": None,
        "on": True,
    }


def test_end_of_plan_marker_alone_and_code_fences_are_handled() -> None:
    items = parse_plan('```\n1. a("x")\n```\n<END_OF_PLAN>\n2. b()').items
    assert items == [Call(1, "a", ("x",)), JoinMarker(None)]


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ('1. f("x"', "unclosed"),
        ("1. f($)", "task number"),
        ("1. f(${x})", "malformed"),
        ("1. f(bare)", "bare word"),
        ("1. f(a=1, 2)", "positional argument after keyword"),
        ("1. f(a=1, a=2)", "repeated"),
        ("1. f() extra", "unexpected text"),
        ("0. f()", "start at 1"),
        ("3. join($1)", "no arguments"),
        ("hello", "expected"),
        ("1. f(" + "[" * 40 + "]" * 40 + ")", "nested"),
    ],
)
def test_issue_messages(line: str, fragment: str) -> None:
    (issue,) = parse_line(line, 4)
    assert isinstance(issue, ParseIssue)
    assert issue.line_no == 4
    assert fragment in issue.message


def test_over_long_line_is_reported_once_and_skipped_when_streamed() -> None:
    long_line = '1. f("' + "x" * (MAX_LINE_CHARS + 10) + '")'
    text = long_line + '\n2. g("ok")\n'
    pieces = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    streamed = stream_items(pieces)
    assert streamed == parse_plan(text).items
    assert isinstance(streamed[0], ParseIssue)
    assert streamed[1] == Call(2, "g", ("ok",))


def test_parser_stops_after_join_and_close_is_idempotent() -> None:
    parser = StreamingPlanParser()
    assert parser.feed("1. a()\n2. join()\n3. b()\n") == [Call(1, "a"), JoinMarker(2)]
    assert parser.done
    assert parser.feed("4. c()\n") == []
    assert parser.close() == []


def test_final_line_without_newline_is_flushed_by_close() -> None:
    parser = StreamingPlanParser()
    assert parser.feed('1. a("x")') == []
    assert parser.close() == [Call(1, "a", ("x",))]


@given(st.text(alphabet=string.printable, max_size=60))
def test_thought_lines_keep_their_text(text: str) -> None:
    clean = " ".join(text.split())
    if not clean or END_OF_PLAN in clean:
        return
    assert parse_line(f"Thought: {clean}", 1) == [Thought(clean)]
