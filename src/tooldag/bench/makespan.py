"""Makespan of one question: sequential (ReAct order) against DAG dispatch.

Every run makes real ONNX calls (no embedding cache). Configurations are interleaved per
question, and their order rotates from question to question, so drift on the machine
and the tokenizer's word cache do not favour one configuration.

The streaming experiment paces the template planner at a fixed delay per plan line, as
a stand-in for a model writing the plan. It compares dispatching tasks as their lines
parse against waiting for the whole plan. The pacing is synthetic; no model is run.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tooldag.bench.common import critical_path_ms, log, summary_stats
from tooldag.bigbench import load_questions, split_fit_eval
from tooldag.fetch import asset_path
from tooldag.joiner import RuleJoiner
from tooldag.planners import TemplatePlanner
from tooldag.runtime import Runtime, RuntimeConfig
from tooldag.scheduler import Limits
from tooldag.tools.embed import MiniLMEmbedder
from tooldag.tools.movies import MovieTools

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from tooldag.bigbench import Question
    from tooldag.runtime import RunResult
    from tooldag.tools.catalog import Catalog
    from tooldag.tools.embed import Embedder
    from tooldag.tools.scorer import LinearScorer


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    mode: Literal["dag", "sequential"]
    concurrency: int
    intra_op_threads: int = 1
    line_delay_ms: float = 0.0
    stream_dispatch: bool = True


MAKESPAN_CONFIGS = (
    Config("sequential (ReAct order)", "sequential", 1),
    Config("sequential, ONNX 6 intra-op threads", "sequential", 1, intra_op_threads=6),
    Config("DAG, concurrency 1", "dag", 1),
    Config("DAG, concurrency 2", "dag", 2),
    Config("DAG, concurrency 4", "dag", 4),
    Config("DAG, concurrency 6", "dag", 6),
)


def streaming_configs(delays_ms: Sequence[float]) -> tuple[Config, ...]:
    out: list[Config] = []
    for delay in delays_ms:
        out.append(Config(f"stream dispatch, {delay:g} ms/line", "dag", 6, line_delay_ms=delay))
        out.append(
            Config(
                f"plan then execute, {delay:g} ms/line",
                "dag",
                6,
                line_delay_ms=delay,
                stream_dispatch=False,
            )
        )
    return tuple(out)


def _questions(root: Path, n: int) -> list[Question]:
    questions = load_questions(asset_path(root, "bigbench-movie-recommendation"))
    _, held_out = split_fit_eval(questions)
    return held_out[:n]


def _embedders(root: Path, configs: Sequence[Config]) -> dict[int, Embedder]:
    model_dir = root / "models" / "all-MiniLM-L6-v2"
    threads = sorted({c.intra_op_threads for c in configs})
    return {t: MiniLMEmbedder(model_dir, intra_op_threads=t) for t in threads}


async def _run_once(
    config: Config,
    q: Question,
    catalog: Catalog,
    embedder: Embedder,
    scorer: LinearScorer,
    segment: Callable[[str], list[str]],
    pool: ThreadPoolExecutor,
) -> RunResult:
    runtime = Runtime(
        MovieTools(catalog, embedder, scorer).registry(),
        TemplatePlanner(segment, line_delay_s=config.line_delay_ms / 1000.0),
        RuleJoiner(),
        RuntimeConfig(
            limits=Limits(max_concurrency=config.concurrency),
            mode=config.mode,
            stream_dispatch=config.stream_dispatch,
        ),
        executor=pool,
    )
    return await runtime.run(q.text, q.options)


def _measure(
    configs: Sequence[Config],
    questions: Sequence[Question],
    catalog: Catalog,
    embedders: dict[int, Embedder],
    scorer: LinearScorer,
    segment: Callable[[str], list[str]],
    label: str,
    warmup: int = 2,
) -> dict[str, list[RunResult]]:
    results: dict[str, list[RunResult]] = {c.name: [] for c in configs}
    pools = {c.name: ThreadPoolExecutor(max_workers=c.concurrency) for c in configs}

    async def go() -> None:
        for i, q in enumerate([*questions[:warmup], *questions]):
            recording = i >= warmup
            shift = i % len(configs)
            for config in [*configs[shift:], *configs[:shift]]:
                result = await _run_once(
                    config,
                    q,
                    catalog,
                    embedders[config.intra_op_threads],
                    scorer,
                    segment,
                    pools[config.name],
                )
                if result.answer is None:
                    raise RuntimeError(f"{config.name}: no answer for question {q.qid}")
                if recording:
                    results[config.name].append(result)
            if recording and ((i - warmup + 1) % 10 == 0 or i - warmup + 1 == len(questions)):
                log(f"  {label}: {i - warmup + 1}/{len(questions)} questions")

    try:
        asyncio.run(go())
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    return results


def _config_row(config: Config, runs: Sequence[RunResult], baseline_p50: float | None) -> dict:
    makespans = [r.makespan_ms for r in runs]
    embeds = [
        rec.busy_ms
        for r in runs
        for rec in r.rounds[-1].records.values()
        if rec.call.tool == "embed" and rec.attempts > 0
    ]
    stats = summary_stats(makespans)
    row: dict[str, Any] = {
        "config": config.name,
        "mode": config.mode,
        "max_concurrency": config.concurrency,
        "onnx_intra_op_threads": config.intra_op_threads,
        "makespan_ms": stats,
        "critical_path_ms": summary_stats([critical_path_ms(r.rounds[-1]) for r in runs]),
        "embed_call_ms": summary_stats(embeds),
        "tasks_per_run": round(float(np.mean([r.tasks_executed for r in runs])), 2),
    }
    if baseline_p50:
        row["speedup_p50_vs_sequential"] = round(baseline_p50 / stats["p50"], 2)
    return row


def bench_makespan(
    root: Path,
    catalog: Catalog,
    scorer: LinearScorer,
    segment: Callable[[str], list[str]],
    *,
    smoke: bool,
) -> dict[str, Any]:
    n = 3 if smoke else 40
    questions = _questions(root, n)
    embedders = _embedders(root, MAKESPAN_CONFIGS)
    log(f"makespan: {len(questions)} questions x {len(MAKESPAN_CONFIGS)} configurations")
    results = _measure(
        MAKESPAN_CONFIGS,
        questions,
        catalog,
        embedders,
        scorer,
        segment,
        "makespan",
        1 if smoke else 2,
    )
    baseline = summary_stats([r.makespan_ms for r in results[MAKESPAN_CONFIGS[0].name]])["p50"]
    rows = [_config_row(c, results[c.name], baseline) for c in MAKESPAN_CONFIGS]
    agree = all(
        len({results[c.name][i].answer for c in MAKESPAN_CONFIGS}) == 1
        for i in range(len(questions))
    )
    return {
        "benchmark": "makespan",
        "questions": len(questions),
        "question_source": "first held-out BIG-bench questions",
        "embedder": "all-MiniLM-L6-v2 int8 ONNX, no cache",
        "planner": "template (no pacing)",
        "warmup_questions": 1 if smoke else 2,
        "answers_identical_across_configs": agree,
        "configs": rows,
    }


def bench_streaming(
    root: Path,
    catalog: Catalog,
    scorer: LinearScorer,
    segment: Callable[[str], list[str]],
    *,
    smoke: bool,
) -> dict[str, Any]:
    n = 2 if smoke else 15
    delays = (5.0, 25.0)
    configs = streaming_configs(delays)
    questions = _questions(root, n)
    embedders = _embedders(root, configs)
    log(f"streaming: {len(questions)} questions x {len(configs)} configurations")
    results = _measure(configs, questions, catalog, embedders, scorer, segment, "streaming", 1)
    rows = []
    for config in configs:
        runs = results[config.name]
        plan_done = [r.rounds[-1].plan_complete_ms or 0.0 for r in runs]
        first = [r.rounds[-1].first_dispatch_ms or 0.0 for r in runs]
        rows.append(
            {
                "config": config.name,
                "line_delay_ms": config.line_delay_ms,
                "stream_dispatch": config.stream_dispatch,
                "makespan_ms": summary_stats([r.makespan_ms for r in runs]),
                "plan_complete_ms": summary_stats(plan_done),
                "first_dispatch_ms": summary_stats(first),
                "after_plan_ms": summary_stats(
                    [r.makespan_ms - p for r, p in zip(runs, plan_done, strict=True)]
                ),
            }
        )
    return {
        "benchmark": "streaming",
        "questions": len(questions),
        "pacing": "synthetic: the template planner sleeps a fixed time before each plan "
        "line; no model is run",
        "max_concurrency": 6,
        "configs": rows,
    }
