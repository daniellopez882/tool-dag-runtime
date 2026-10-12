"""Helpers shared by the benchmarks: statistics, plan shapes and record inspection."""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from tooldag.scheduler import TaskState, resolve_arg
from tooldag.tools.movies import MovieHit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tooldag.rounds import RoundSummary
    from tooldag.runtime import RunResult
    from tooldag.scheduler import TaskRecord


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def summary_stats(values: Sequence[float]) -> dict[str, float]:
    """p50, p95, mean, min and max (linear-interpolated percentiles), rounded to 0.1."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "mean": round(float(arr.mean()), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
    }


def wilson_interval(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion observed on ``n`` trials."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tie_aware_hit(scores: np.ndarray, correct: int, tol: float = 1e-12) -> float:
    """1/k when the correct row is among k rows tied for the top score, else 0.

    This is the expected accuracy of breaking ties uniformly at random, so a signal that
    scores every option the same earns chance level rather than "first option wins".
    """
    top = float(np.max(scores))
    tied = np.flatnonzero(np.abs(scores - top) <= tol)
    return 1.0 / len(tied) if correct in tied else 0.0


def final_rank_record(result: RunResult) -> TaskRecord | None:
    if not result.rounds:
        return None
    summary = result.rounds[-1]
    ranks = [
        r
        for r in summary.records.values()
        if r.call.tool == "rank" and r.state is TaskState.SUCCEEDED
    ]
    return max(ranks, key=lambda r: r.call.idx) if ranks else None


def rank_inputs(
    summary: RoundSummary, record: TaskRecord
) -> tuple[list[MovieHit], dict[str, list[float | None]]]:
    """The candidates and raw signal columns the rank task was called with."""
    records = summary.records

    def result_of(idx: int) -> Any:
        return records[idx].result

    call = record.call
    candidates = resolve_arg(call.args[0], result_of)
    signals = {name: resolve_arg(value, result_of) for name, value in call.kwargs}
    return [c for c in candidates if isinstance(c, MovieHit)], signals


def critical_path_ms(summary: RoundSummary) -> float:
    """Longest dependency chain weighted by each task's measured busy time."""
    records = summary.records
    memo: dict[int, float] = {}

    def finish(idx: int) -> float:
        if idx in memo:
            return memo[idx]
        record = records[idx]
        # Provisional value first, so a rejected cycle cannot recurse forever.
        memo[idx] = record.busy_ms
        deps = [finish(d) for d in record.call.deps if d in records]
        memo[idx] = record.busy_ms + max(deps, default=0.0)
        return memo[idx]

    for idx in sorted(records):
        finish(idx)
    return max(memo.values(), default=0.0)
