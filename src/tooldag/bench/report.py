"""Render the README result tables from bench/results/*.json."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

PENDING = "_pending benchmark_"


def _load(results_dir: Path, name: str) -> dict[str, Any] | None:
    path = results_dir / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def measured_on(result: dict[str, Any]) -> str:
    m = result["machine"]
    packages = m["packages"]
    return (
        f"Measured on: {m['cpu']}, {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, "
        f"{m['os']}, CPU only, Python {m['python']}, onnxruntime {packages['onnxruntime']}, "
        f"{result['date']}, commit `{result['git_sha'][:12]}`, `{result['command']}`."
    )


def _md(header: list[str], align: str, rows: list[list[Any]]) -> str:
    """Markdown table; ``align`` has one character per column, 'l' or 'r'."""
    rule = "|" + "|".join("---" if a == "l" else "---:" for a in align) + "|"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(["| " + " | ".join(header) + " |", rule, *body])


def trace_table(result: dict[str, Any]) -> str:
    return _md(
        ["task", "tool", "line parsed (ms)", "started (ms)", "finished (ms)"],
        "rlrrr",
        [
            [
                f"${row['idx']}",
                row["tool"],
                row.get("received_ms", ""),
                row.get("started_ms", ""),
                row.get("finished_ms", ""),
            ]
            for row in result["tasks"]
        ],
    )


def makespan_table(result: dict[str, Any]) -> str:
    return _md(
        [
            "configuration",
            "makespan p50 (ms)",
            "p95 (ms)",
            "speedup at p50",
            "critical path p50 (ms)",
            "one embed call p50 (ms)",
        ],
        "lrrrrr",
        [
            [
                row["config"],
                row["makespan_ms"]["p50"],
                row["makespan_ms"]["p95"],
                f"{row.get('speedup_p50_vs_sequential', 1.0):.2f}x",
                row["critical_path_ms"]["p50"],
                row["embed_call_ms"]["p50"],
            ]
            for row in result["configs"]
        ],
    )


def streaming_table(result: dict[str, Any]) -> str:
    return _md(
        [
            "configuration",
            "makespan p50 (ms)",
            "plan complete p50 (ms)",
            "first task started p50 (ms)",
            "work left after plan p50 (ms)",
        ],
        "lrrrr",
        [
            [
                row["config"],
                row["makespan_ms"]["p50"],
                row["plan_complete_ms"]["p50"],
                row["first_dispatch_ms"]["p50"],
                row["after_plan_ms"]["p50"],
            ]
            for row in result["configs"]
        ],
    )


def accuracy_table(result: dict[str, Any]) -> str:
    datasets = result["datasets"]
    header = ["scorer", *(f"{d['dataset']} ({d['n_held_out']} held out)" for d in datasets)]
    rows = []
    for name in (row["scorer"] for row in datasets[0]["rows"]):
        cells: list[Any] = [name]
        for d in datasets:
            row = next(r for r in d["rows"] if r["scorer"] == name)
            lo, hi = row["ci95"]
            cells.append(f"{_pct(row['accuracy'])} ({_pct(lo)} to {_pct(hi)})")
        rows.append(cells)
    return _md(header, "l" + "r" * len(datasets), rows)


def weights_table(result: dict[str, Any]) -> str:
    signals = result["datasets"][0]["weights"]["signals"]
    return _md(
        ["dataset", *signals],
        "l" + "r" * len(signals),
        [
            [d["dataset"], *(f"{w:+.3f}" for w in d["weights"]["weights"])]
            for d in result["datasets"]
        ],
    )


def structure_table(result: dict[str, Any]) -> str:
    d = result["datasets"][0]["structure"]
    return _md(
        ["per held-out BIG-bench question", "value"],
        "lr",
        [
            ["tool tasks executed (mean)", d["mean_tasks_executed"]],
            ["questions that needed a replan", d["replanned_questions"]],
            [
                "model calls, LLMCompiler (planner + joiner per round)",
                d["mean_model_calls_llmcompiler"],
            ],
            [
                "model calls, ReAct equivalent (one per tool call + answer)",
                d["mean_model_calls_react_equivalent"],
            ],
        ],
    )


def overhead_table(result: dict[str, Any]) -> str:
    parser = result["parser"]
    rows = [
        [row["shape"], row["executor"], row["us_per_task_median"], row["events_per_task"]]
        for row in result["executors"]
    ]
    rows.append(
        [
            f"parser, {parser['lines']}-line plan in {parser['chunk_chars']}-char chunks",
            "-",
            f"{parser['us_per_line_median']} per line",
            "-",
        ]
    )
    return _md(["plan shape", "executor", "us per task (median)", "events per task"], "llrr", rows)


def replan_table(result: dict[str, Any]) -> str:
    return _md(
        ["scenario", "defect", "rounds", "tasks run", "tasks reused", "retries", "answer"],
        "llrrrrl",
        [
            [
                row["scenario"],
                row["defect"],
                row["rounds"],
                row["tasks_executed"],
                row["tasks_reused"],
                row["retries"],
                row["answer"] or "none",
            ]
            for row in result["scenarios"]
        ],
    )


def render(results_dir: Path) -> str:
    sections: list[tuple[str, str, Any]] = [
        ("trace", "Per-task timing of the captured run", trace_table),
        ("makespan", "Makespan", makespan_table),
        ("streaming", "Streaming dispatch", streaming_table),
        ("accuracy", "Accuracy", accuracy_table),
        ("accuracy", "Fitted weights", weights_table),
        ("accuracy", "Structural model-call counts", structure_table),
        ("overhead", "Scheduler overhead", overhead_table),
        ("replan", "Replan scenarios", replan_table),
    ]
    out: list[str] = []
    for name, title, fn in sections:
        result = _load(results_dir, name)
        out.append(f"### {title}\n")
        if result is None:
            out.append(PENDING + "\n")
            continue
        out.append(fn(result) + "\n")
        out.append(measured_on(result) + "\n")
    return "\n".join(out)
