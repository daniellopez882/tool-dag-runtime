from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from tooldag.bench.common import summary_stats, tie_aware_hit, wilson_interval
from tooldag.bench.machine import provenance, write_result
from tooldag.bench.overhead import bench_overhead
from tooldag.bench.replan import bench_replan
from tooldag.bench.report import PENDING, render
from tooldag.cli import app
from tooldag.events import EventLog, preview

if TYPE_CHECKING:
    from pathlib import Path


def test_follow_replays_then_streams_live_events() -> None:
    async def scenario() -> list[int]:
        log = EventLog("r")
        log.emit("a")
        log.emit("b")
        seen: list[int] = []

        async def reader() -> None:
            async for event in log.follow(after_seq=1):
                assert event is not None
                seen.append(event.seq)

        task = asyncio.create_task(reader())
        await asyncio.sleep(0)
        log.emit("c")
        log.close()
        await task
        return seen

    assert asyncio.run(scenario()) == [2, 3]


def test_follow_yields_heartbeats_while_idle() -> None:
    async def scenario() -> list[object]:
        log = EventLog()
        out: list[object] = []
        async for event in log.follow(heartbeat_s=0.01):
            out.append(event)
            if len(out) == 2:
                log.emit("late")
            if len(out) == 3:
                log.close()
        return out

    out = asyncio.run(scenario())
    assert out[:2] == [None, None]
    assert out[2].type == "late"


def test_closed_log_rejects_events() -> None:
    log = EventLog()
    log.close()
    with pytest.raises(RuntimeError):
        log.emit("x")


@dataclass
class _Point:
    x: float
    label: str


def test_preview_is_compact_and_json_safe() -> None:
    value = {
        "long": "x" * 200,
        "list": list(range(10)),
        "nan": float("nan"),
        "point": _Point(1.23456789, "p"),
        "array": np.zeros((2, 3)),
    }
    out = preview(value)
    json.dumps(out)
    assert out["long"].endswith("(200 chars)")
    assert out["list"][-1] == "... (+4)"
    assert out["nan"] is None
    assert out["point"] == {"x": 1.234568, "label": "p"}
    assert out["array"] == {"shape": [2, 3], "dtype": "float64"}


def test_statistics_helpers() -> None:
    stats = summary_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["p50"] == 2.5 and stats["n"] == 4
    assert summary_stats([]) == {"n": 0}
    lo, hi = wilson_interval(0.5, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.192, abs=0.002)
    assert tie_aware_hit(np.array([1.0, 1.0, 0.0]), 0) == 0.5
    assert tie_aware_hit(np.array([1.0, 2.0, 0.0]), 0) == 0.0
    assert tie_aware_hit(np.array([1.0, 2.0, 0.0]), 1) == 1.0


def test_result_files_carry_provenance(tmp_path: Path) -> None:
    path = write_result(tmp_path / "x.json", "tooldag bench", {"metric": 1})
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("machine", "date", "git_sha", "command", "metric"):
        assert key in data
    assert data["machine"]["cpu"]
    assert data["machine"]["logical_cores"] >= 1
    assert provenance("c")["command"] == "c"


def test_replan_benchmark_reaches_the_expected_answers() -> None:
    result = bench_replan()
    assert result["all_expected"]
    rows = {r["scenario"]: r for r in result["scenarios"]}
    assert rows["transient tool failure"]["retries"] == 2
    assert rows["malformed line"]["rounds"] == 2
    assert rows["malformed line"]["tasks_reused"] > 0
    assert rows["never recovers"]["answer"] is None


def test_overhead_benchmark_smoke() -> None:
    result = bench_overhead(smoke=True)
    assert {r["executor"] for r in result["executors"]} == {"dag", "sequential"}
    assert all(r["us_per_task_median"] > 0 for r in result["executors"])
    assert result["parser"]["lines"] == 34


def test_report_renders_present_and_pending_sections(tmp_path: Path) -> None:
    write_result(tmp_path / "replan.json", "tooldag bench", bench_replan())
    text = render(tmp_path)
    assert "| malformed line |" in text
    assert "Measured on:" in text
    assert PENDING in text


def test_cli_ask_and_demo_on_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOOLDAG_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ask",
            "Find a movie similar to The Vault at Midnight, Glass City Heist",
            "-o",
            "Silent Orchard",
            "-o",
            "The Forger's Daughter",
            "--catalog",
            "fixture",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.splitlines()]
    assert events[-1]["data"]["answer"] == "The Forger's Daughter"
    demo = runner.invoke(app, ["demo"])
    assert demo.exit_code == 0, demo.output
    assert "never recovers" in demo.output


def test_cli_bench_refuses_without_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOOLDAG_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["bench", "--only", "accuracy", "--out-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "tooldag fetch" in str(result.output) + str(result.exception)
