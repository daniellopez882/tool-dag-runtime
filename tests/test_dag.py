from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tooldag.dag import PlanGraph, call_signature
from tooldag.plan import Call, ListArg, Ref, Template


def make(idx: int, *deps: int, tool: str = "t") -> Call:
    return Call(idx, tool, tuple(Ref(d) for d in deps))


def test_forward_reference_waits_and_is_not_dangling_once_defined() -> None:
    graph = PlanGraph()
    assert graph.add(make(1, 2)).accepted
    assert graph.missing_refs() == {2: {1}}
    assert graph.add(make(2)).accepted
    assert graph.close() == {}
    assert graph.topological_order() == [2, 1]


def test_cycle_is_detected_when_the_closing_edge_arrives() -> None:
    graph = PlanGraph()
    assert graph.add(make(1, 3)).accepted
    assert graph.add(make(2, 1)).accepted
    outcome = graph.add(make(3, 2))
    assert not outcome.accepted
    assert outcome.cycle is not None
    assert set(outcome.cycle.members) == {1, 2, 3}
    assert all("cycle" in graph.rejected[i] for i in (1, 2, 3))


def test_self_reference_is_a_cycle_of_one() -> None:
    graph = PlanGraph()
    outcome = graph.add(make(4, 4))
    assert outcome.cycle is not None
    assert outcome.cycle.members == (4,)


def test_dangling_reference_reported_at_close() -> None:
    graph = PlanGraph()
    graph.add(make(1))
    graph.add(make(2, 1, 9))
    assert graph.close() == {9: {2}}


def test_duplicate_index_keeps_first_definition() -> None:
    graph = PlanGraph()
    graph.add(make(1, tool="a"))
    outcome = graph.add(make(1, tool="b"))
    assert outcome.rejection is not None
    assert graph.calls[1].tool == "a"


def test_validator_rejection_is_recorded_and_linked() -> None:
    graph = PlanGraph(validate_call=lambda c: "unknown tool" if c.tool == "bad" else None)
    graph.add(make(1, tool="bad"))
    graph.add(make(2, 1))
    assert graph.rejected == {1: "unknown tool"}
    assert graph.transitive_dependents([1]) == {2}


@st.composite
def random_graphs(draw: st.DrawFn) -> tuple[list[Call], list[int]]:
    n = draw(st.integers(1, 12))
    calls = []
    for idx in range(1, n + 1):
        deps = draw(st.sets(st.integers(1, n + 2), max_size=3))
        calls.append(make(idx, *sorted(deps)))
    order = draw(st.permutations(list(range(n))))
    return calls, order


def _on_cycle(calls: dict[int, Call]) -> set[int]:
    """Ground truth: nodes on some directed cycle (reachable from themselves)."""
    out = set()
    for start in calls:
        stack, seen = list(calls[start].deps), set()
        while stack:
            node = stack.pop()
            if node == start:
                out.add(start)
                break
            if node in seen or node not in calls:
                continue
            seen.add(node)
            stack.extend(calls[node].deps)
    return out


@given(random_graphs())
def test_no_cycle_member_survives_and_accepted_part_is_acyclic(
    case: tuple[list[Call], list[int]],
) -> None:
    calls, order = case
    graph = PlanGraph()
    for i in order:
        graph.add(calls[i])
    dangling = graph.close()
    blocked = set(graph.rejected) | graph.transitive_dependents(graph.rejected)
    blocked |= {w for waiting in dangling.values() for w in waiting}
    blocked |= graph.transitive_dependents(blocked)
    by_idx = {c.idx: c for c in calls}
    # Every node on a cycle is unrunnable: rejected or downstream of a rejection.
    assert _on_cycle(by_idx) <= blocked
    runnable = {i: c for i, c in by_idx.items() if i not in blocked}
    assert _on_cycle(runnable) == set()
    order_ = graph.topological_order()
    position = {node: k for k, node in enumerate(order_)}
    for idx in runnable:
        for dep in by_idx[idx].deps:
            if dep in runnable:
                assert position[dep] < position[idx]


def test_signature_ignores_numbering_but_tracks_content() -> None:
    a1, a2 = Call(1, "resolve", ("Heat",)), Call(7, "resolve", ("Heat",))
    sig_a1 = call_signature(a1, lambda i: "")
    assert sig_a1 == call_signature(a2, lambda i: "")
    downstream_1 = Call(2, "plot", (Ref(1),))
    downstream_2 = Call(8, "plot", (Ref(7),))
    sigs = {1: sig_a1, 7: sig_a1}
    assert call_signature(downstream_1, sigs.__getitem__) == call_signature(
        downstream_2, sigs.__getitem__
    )
    other = Call(2, "plot", (Template(("x ", Ref(1))),))
    assert call_signature(other, sigs.__getitem__) != call_signature(
        downstream_1, sigs.__getitem__
    )
    assert call_signature(Call(3, "f", (ListArg((1.0,)),)), lambda i: "") != call_signature(
        Call(3, "f", (ListArg((1,)),)), lambda i: ""
    )


def test_depth_counts_longest_chain() -> None:
    graph = PlanGraph()
    for call in (make(1), make(2, 1), make(3, 1), make(4, 2, 3), make(5)):
        graph.add(call)
    assert graph.depth() == 3
