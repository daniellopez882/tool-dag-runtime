"""Incremental validation of a plan's dependency graph.

Tasks arrive one at a time while the planner streams. A task may reference a task
number that has not arrived yet (a forward reference); it simply waits. Every arrival
is checked for a cycle through the new node, and the end of the plan reports
references that never arrived. Cycle members can never be dispatched, because each of
them waits on another member, so rejecting them on arrival loses nothing.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tooldag.plan.ast import Arg, Call, ListArg, Ref, Template

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass(frozen=True, slots=True)
class Rejection:
    idx: int
    reason: str


@dataclass(frozen=True, slots=True)
class CycleFound:
    members: tuple[int, ...]


@dataclass(slots=True)
class AddOutcome:
    accepted: bool
    rejection: Rejection | None = None
    cycle: CycleFound | None = None


@dataclass(slots=True)
class PlanGraph:
    """Structure of one plan round: nodes, edges, forward references and cycles.

    ``validate_call`` lets the owner reject calls for reasons outside the graph (unknown
    tool, wrong arity) using the same bookkeeping.
    """

    validate_call: Callable[[Call], str | None] | None = None
    calls: dict[int, Call] = field(default_factory=dict)
    dependents: dict[int, set[int]] = field(default_factory=dict)
    rejected: dict[int, str] = field(default_factory=dict)
    closed: bool = False

    def add(self, call: Call) -> AddOutcome:
        if self.closed:
            raise RuntimeError("plan graph is closed")
        idx = call.idx
        if idx in self.calls or idx in self.rejected:
            # Keep the first definition; the duplicate is reported and dropped.
            return AddOutcome(False, Rejection(idx, f"task {idx} defined twice"))
        if self.validate_call is not None and (reason := self.validate_call(call)) is not None:
            self.rejected[idx] = reason
            self.calls[idx] = call
            self._link(call)
            return AddOutcome(False, Rejection(idx, reason))
        self.calls[idx] = call
        self._link(call)
        cycle = self._cycle_through(idx)
        if cycle is not None:
            for member in cycle:
                self.rejected.setdefault(member, f"dependency cycle {_fmt_cycle(cycle)}")
            return AddOutcome(False, cycle=CycleFound(cycle))
        return AddOutcome(True)

    def _link(self, call: Call) -> None:
        for dep in call.deps:
            self.dependents.setdefault(dep, set()).add(call.idx)

    def _cycle_through(self, start: int) -> tuple[int, ...] | None:
        """Search from ``start`` along "is needed by" edges for a way back to ``start``.

        Only tasks that referenced ``start`` before it arrived can close a cycle, so with
        no forward references this returns at once instead of walking every upstream
        task (which made long chains quadratic to submit).
        """
        parent: dict[int, int] = {}
        stack = [start]
        while stack:
            node = stack.pop()
            for child in sorted(self.dependents.get(node, ())):
                if child == start:
                    # start depends on node, node on parent[node], ... back to start.
                    members = [start]
                    cur = node
                    while cur != start:
                        members.append(cur)
                        cur = parent[cur]
                    return tuple(members)
                # Rejected tasks are already dead; anything that reaches them is cancelled.
                if child in parent or child in self.rejected:
                    continue
                parent[child] = node
                stack.append(child)
        return None

    def missing_refs(self) -> dict[int, set[int]]:
        """Map each referenced-but-undefined task number to the tasks waiting on it."""
        return {
            ref: waiting
            for ref, waiting in self.dependents.items()
            if ref not in self.calls and ref not in self.rejected
        }

    def close(self) -> dict[int, set[int]]:
        """End of plan: return dangling references (``ref -> waiting tasks``)."""
        self.closed = True
        return self.missing_refs()

    def transitive_dependents(self, roots: Iterable[int]) -> set[int]:
        out: set[int] = set()
        frontier = list(roots)
        while frontier:
            node = frontier.pop()
            for child in self.dependents.get(node, ()):
                if child not in out:
                    out.add(child)
                    frontier.append(child)
        return out

    def topological_order(self) -> list[int]:
        """Accepted tasks in dependency order, lowest task number first among peers."""
        accepted = {i for i in self.calls if i not in self.rejected}
        indegree = {i: sum(1 for d in self.calls[i].deps if d in accepted) for i in accepted}
        ready = [i for i, n in indegree.items() if n == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            node = heapq.heappop(ready)
            order.append(node)
            for child in self.dependents.get(node, ()):
                if child in indegree:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        heapq.heappush(ready, child)
        return order


def _fmt_cycle(cycle: tuple[int, ...]) -> str:
    return " -> ".join(f"${i}" for i in (*cycle, cycle[0]))


def _canonical(arg: Arg, sig_of: Callable[[int], str]) -> object:
    match arg:
        case Ref(idx=idx):
            return {"ref": sig_of(idx)}
        case Template(parts=parts):
            return {"tpl": [p if isinstance(p, str) else {"ref": sig_of(p.idx)} for p in parts]}
        case ListArg(items=items):
            return {"list": [_canonical(i, sig_of) for i in items]}
        case float():
            return {"float": repr(arg)}
        case _:
            return arg


def call_signature(call: Call, sig_of: Callable[[int], str]) -> str:
    """Content address of a call: tool, arguments, and the signatures of what it reads.

    Two calls from different plan rounds get the same signature exactly when they would
    compute the same thing, even if the planner renumbered them. The joiner uses this to
    reuse results across a replan.
    """
    payload = {
        "tool": call.tool,
        "args": [_canonical(a, sig_of) for a in call.args],
        "kwargs": sorted((k, _canonical(v, sig_of)) for k, v in call.kwargs),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode(), digest_size=12).hexdigest()
