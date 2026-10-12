"""Run the benchmarks and write one provenance-stamped JSON file per benchmark."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from tooldag.bench.common import log
from tooldag.bench.machine import write_result
from tooldag.config import data_dir
from tooldag.service import full_data_present, load_domain

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

ORDER = ("accuracy", "makespan", "streaming", "overhead", "replan", "trace")
NEEDS_DATA = frozenset({"accuracy", "makespan", "streaming"})


def run_suite(
    out_dir: Path,
    *,
    smoke: bool = False,
    only: Sequence[str] | None = None,
    command: str = "tooldag bench",
) -> list[Path]:
    selected = [name for name in ORDER if not only or name in only]
    unknown = set(only or ()) - set(ORDER)
    if unknown:
        raise ValueError(f"unknown benchmark(s) {sorted(unknown)}; choose from {ORDER}")
    root = data_dir()
    have_data = full_data_present(root)
    missing = [name for name in selected if name in NEEDS_DATA and not have_data]
    if missing and not smoke:
        raise SystemExit(
            f"{', '.join(missing)} need the fetched data under {root}; run `tooldag fetch`"
        )
    prefix = "smoke-" if smoke else ""
    written: list[Path] = []

    def domain_loader() -> Any:
        # Loaded lazily and again after accuracy, which may have written new weights.
        return load_domain("full" if have_data else "fixture", root=root)

    for name in selected:
        if name in missing:
            log(f"{name}: skipped (data not fetched)")
            continue
        started = time.perf_counter()
        metrics = _run_one(name, root, domain_loader, smoke)
        metrics["wall_seconds"] = round(time.perf_counter() - started, 1)
        metrics["smoke"] = smoke
        path = write_result(out_dir / f"{prefix}{name}.json", command, metrics)
        log(f"{name}: {metrics['wall_seconds']} s -> {path}")
        written.append(path)
    return written


def _run_one(
    name: str, root: Path, domain_loader: Callable[[], Any], smoke: bool
) -> dict[str, Any]:
    if name == "accuracy":
        from tooldag.bench.accuracy import bench_accuracy

        domain = domain_loader()
        return bench_accuracy(
            root, domain.catalog, domain.segmenter(), smoke=smoke, save_weights=not smoke
        )
    if name in ("makespan", "streaming"):
        from tooldag.bench.makespan import bench_makespan, bench_streaming

        domain = domain_loader()
        fn = bench_makespan if name == "makespan" else bench_streaming
        return fn(root, domain.catalog, domain.scorer, domain.segmenter(), smoke=smoke)
    if name == "overhead":
        from tooldag.bench.overhead import bench_overhead

        return bench_overhead(smoke=smoke)
    if name == "replan":
        from tooldag.bench.replan import bench_replan

        return bench_replan()
    from tooldag.bench.trace import bench_trace

    domain = domain_loader()
    return bench_trace(root if domain.catalog.source != "fixture" else None, domain)
