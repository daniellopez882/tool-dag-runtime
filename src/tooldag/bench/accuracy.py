"""Accuracy on BIG-bench movie_recommendation and on its popularity-matched variant.

For each question set, 100 questions fit the scorer weights and the other 400 are held
out. Every question runs through the full runtime (template planner, task fetching unit,
rule joiner). The held-out runs use the fitted scorer, so the "runtime answer" row is
what the system actually answered. The other rows rescore the same per-option signal
values that the runtime's tasks computed, with ties split evenly (expected accuracy
under random tie-breaking).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from tooldag.bench.common import (
    final_rank_record,
    log,
    rank_inputs,
    tie_aware_hit,
    wilson_interval,
)
from tooldag.bigbench import load_questions, popularity_matched, split_fit_eval
from tooldag.config import weights_path
from tooldag.fetch import asset_path
from tooldag.joiner import RuleJoiner
from tooldag.planners import TemplatePlanner
from tooldag.runtime import Runtime, RuntimeConfig
from tooldag.scheduler import Limits
from tooldag.tools.embed import CachingEmbedder, MiniLMEmbedder
from tooldag.tools.movies import MATCH_THRESHOLD, MovieTools
from tooldag.tools.scorer import SIGNALS, LinearScorer, features_matrix, fit_conditional_logit

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from tooldag.bigbench import Question
    from tooldag.runtime import RunResult
    from tooldag.tools.catalog import Catalog
    from tooldag.tools.embed import Embedder

CONCURRENT_QUESTIONS = 4


@dataclass(slots=True)
class Observed:
    """What one runtime run produced for one question."""

    qid: int
    correct: int | None  # row of the correct option among the ranked candidates
    features: np.ndarray  # (n_candidates, len(SIGNALS)), NaN where a value is missing
    n_ratings: np.ndarray
    runtime_hit: bool
    rounds: int
    tasks_executed: int
    tasks_reused: int
    model_calls: dict[str, int] = field(default_factory=dict)


def _observe(q: Question, result: RunResult, catalog: Catalog, answer_id: int | None) -> Observed:
    record = final_rank_record(result)
    features = np.full((0, len(SIGNALS)), np.nan)
    n_ratings = np.zeros(0)
    correct: int | None = None
    hit = False
    if record is not None:
        candidates, signals = rank_inputs(result.rounds[-1], record)
        columns = {name: signals.get(name, [None] * len(candidates)) for name in SIGNALS}
        features = features_matrix(SIGNALS, columns, len(candidates))
        n_ratings = np.array([catalog.movies[c.id].n_ratings for c in candidates], dtype=float)
        ids = [c.id for c in candidates]
        correct = ids.index(answer_id) if answer_id in ids else None
        ranking = record.result
        hit = correct is not None and ranking.ids[ranking.best_index] == answer_id
    return Observed(
        q.qid,
        correct,
        features,
        n_ratings,
        hit,
        len(result.rounds),
        result.tasks_executed,
        result.tasks_reused,
        result.model_calls(),
    )


async def _run_all(
    questions: Sequence[Question],
    catalog: Catalog,
    embedder: Embedder,
    scorer: LinearScorer,
    segment: Callable[[str], list[str]],
    pool: ThreadPoolExecutor,
    label: str,
) -> list[Observed]:
    tools = MovieTools(catalog, embedder, scorer).registry()
    runtime = Runtime(
        tools,
        TemplatePlanner(segment),
        RuleJoiner(),
        RuntimeConfig(limits=Limits(max_concurrency=6)),
        executor=pool,
    )
    gate = asyncio.Semaphore(CONCURRENT_QUESTIONS)
    done = 0

    def answer_id(q: Question) -> int | None:
        found = catalog.index.best(q.answer)
        return found.movie_id if found and found.score >= MATCH_THRESHOLD else None

    async def one(q: Question) -> Observed:
        nonlocal done
        async with gate:
            result = await runtime.run(q.text, q.options)
        done += 1
        if done % 50 == 0 or done == len(questions):
            log(f"  {label}: {done}/{len(questions)} questions")
        return _observe(q, result, catalog, answer_id(q))

    return list(await asyncio.gather(*(one(q) for q in questions)))


def _fit(observed: Sequence[Observed], signals: Sequence[str]) -> LinearScorer:
    usable = [o for o in observed if o.correct is not None]
    cols = [SIGNALS.index(s) for s in signals]
    groups = [o.features[:, cols] for o in usable]
    return fit_conditional_logit(groups, [o.correct for o in usable], signals)  # type: ignore[misc]


def _accuracy(observed: Sequence[Observed], score: Callable[[Observed], np.ndarray]) -> float:
    hits = [tie_aware_hit(score(o), o.correct) if o.correct is not None else 0.0 for o in observed]
    return float(np.mean(hits)) if hits else 0.0


def _row(name: str, accuracy: float, n: int, note: str = "") -> dict[str, Any]:
    lo, hi = wilson_interval(accuracy, n)
    return {
        "scorer": name,
        "accuracy": round(accuracy, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n": n,
        "note": note,
    }


def _evaluate(
    fit_obs: Sequence[Observed], held: Sequence[Observed], fitted: LinearScorer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(held)
    rows = [
        _row(
            "random choice (expected)",
            float(np.mean([1.0 / max(len(o.n_ratings), 1) for o in held])),
            n,
            "1 / number of options",
        ),
        _row(
            "most-rated option",
            _accuracy(held, lambda o: o.n_ratings),
            n,
            "MovieLens rating count; no signal tools",
        ),
    ]
    for signal in SIGNALS:
        single = fitted.restrict((signal,))
        unit = LinearScorer((signal,), (1.0,), single.mean, single.scale, "single signal")
        col = SIGNALS.index(signal)
        rows.append(
            _row(
                f"{signal} only",
                _accuracy(held, lambda o, u=unit, c=col: u.scores(o.features[:, [c]])),
                n,
                "missing values scored at the fitting-split mean",
            )
        )
    equal = LinearScorer.equal()
    rows.append(
        _row(
            "equal weights, raw signals",
            _accuracy(held, lambda o: equal.scores(o.features)),
            n,
            "the scorer used before any fitting",
        )
    )
    no_co_signals = tuple(s for s in SIGNALS if s != "cooccurrence")
    no_co = _fit(fit_obs, no_co_signals)
    cols = [SIGNALS.index(s) for s in no_co_signals]
    rows.append(
        _row(
            "fitted, without co-occurrence (ablation)",
            _accuracy(held, lambda o: no_co.scores(o.features[:, cols])),
            n,
            "refitted on the same 100 questions",
        )
    )
    rows.append(
        _row(
            "fitted, all three signals",
            _accuracy(held, lambda o: fitted.scores(o.features)),
            n,
            "rescored offline, ties split",
        )
    )
    runtime_acc = float(np.mean([o.runtime_hit for o in held])) if held else 0.0
    rows.append(
        _row(
            "runtime answer (fitted scorer)",
            runtime_acc,
            n,
            "what the rule joiner returned; argmax, first option on ties",
        )
    )
    extra = {"weights_without_cooccurrence": no_co.to_json()}
    return rows, extra


def _structure(observed: Sequence[Observed]) -> dict[str, Any]:
    return {
        "questions": len(observed),
        "answer_not_among_candidates": sum(1 for o in observed if o.correct is None),
        "replanned_questions": sum(1 for o in observed if o.rounds > 1),
        "mean_tasks_executed": round(float(np.mean([o.tasks_executed for o in observed])), 2),
        "mean_tasks_reused": round(float(np.mean([o.tasks_reused for o in observed])), 2),
        "mean_model_calls_llmcompiler": round(
            float(np.mean([o.model_calls["llmcompiler"] for o in observed])), 2
        ),
        "mean_model_calls_react_equivalent": round(
            float(np.mean([o.model_calls["react_equivalent"] for o in observed])), 2
        ),
        "candidates_without_plot_vector": int(
            sum(
                int(np.isnan(o.features[:, SIGNALS.index("plot_similarity")]).sum())
                for o in observed
            )
        ),
    }


async def _dataset(
    name: str,
    fit_q: Sequence[Question],
    held_q: Sequence[Question],
    catalog: Catalog,
    embedder: Embedder,
    segment: Callable[[str], list[str]],
    pool: ThreadPoolExecutor,
) -> tuple[dict[str, Any], LinearScorer]:
    log(f"accuracy [{name}]: fitting split ({len(fit_q)} questions, equal weights)")
    fit_obs = await _run_all(
        fit_q, catalog, embedder, LinearScorer.equal(), segment, pool, f"{name}/fit"
    )
    fitted = _fit(fit_obs, SIGNALS)
    log(f"accuracy [{name}]: held-out split ({len(held_q)} questions, fitted weights)")
    held_obs = await _run_all(held_q, catalog, embedder, fitted, segment, pool, f"{name}/held-out")
    rows, extra = _evaluate(fit_obs, held_obs, fitted)
    return (
        {
            "dataset": name,
            "n_fit": len(fit_q),
            "n_held_out": len(held_q),
            "weights": fitted.to_json(),
            **extra,
            "rows": rows,
            "structure": _structure(held_obs),
        },
        fitted,
    )


def bench_accuracy(
    root: Path,
    catalog: Catalog,
    segment: Callable[[str], list[str]],
    *,
    smoke: bool,
    save_weights: bool,
) -> dict[str, Any]:
    questions = load_questions(asset_path(root, "bigbench-movie-recommendation"))
    n_fit = 10 if smoke else 100
    if smoke:
        questions = questions[:24]
    fit_q, held_q = split_fit_eval(questions, n_fit=n_fit)
    fit_ids = {q.qid for q in fit_q}
    variant = popularity_matched(questions, catalog)
    v_fit = [q for q in variant if q.qid in fit_ids]
    v_held = [q for q in variant if q.qid not in fit_ids]

    minilm = MiniLMEmbedder(root / "models" / "all-MiniLM-L6-v2", intra_op_threads=1)
    embedder = CachingEmbedder(minilm)

    async def go() -> tuple[list[dict[str, Any]], LinearScorer]:
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="tool") as pool:
            original, fitted = await _dataset(
                "BIG-bench movie_recommendation", fit_q, held_q, catalog, embedder, segment, pool
            )
            matched, _ = await _dataset(
                "popularity-matched variant", v_fit, v_held, catalog, embedder, segment, pool
            )
        return [original, matched], fitted

    datasets, fitted = asyncio.run(go())
    if save_weights:
        fitted.save(weights_path(root))
        log(f"accuracy: fitted weights written to {weights_path(root)}")
    return {
        "benchmark": "accuracy",
        "split": {"n_fit": n_fit, "seed": 2026, "method": "random sample of question indices"},
        "variant": {
            "construction": "distractors redrawn among films rated within +/-20% (widened "
            "by 1.5x until 3 exist) as often as the answer, seed 7",
            "questions_built": len(variant),
        },
        "embedder": embedder.name,
        "embeddings_computed": embedder.misses,
        "planner": "template",
        "joiner": "rule",
        "max_concurrency": 6,
        "datasets": datasets,
    }
