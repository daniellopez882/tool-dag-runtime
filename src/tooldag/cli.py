"""Command line: fetch data, build the catalogue, ask questions, serve, benchmark."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from tooldag.config import data_dir

if TYPE_CHECKING:
    from tooldag.events import Event

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Streaming tool-DAG runtime (LLMCompiler pattern) for movie-recommendation questions.",
)

CatalogOpt = Annotated[
    str, typer.Option("--catalog", help="auto | full (fetched data) | fixture (bundled, offline)")
]


def describe(event: Event) -> str:
    """One terminal line per lifecycle event."""
    d = event.data
    head = f"{event.t_ms:9.1f} ms  {event.type:<15}"
    match event.type:
        case "task_received":
            return f"{head} {d['call']}"
        case "task_started":
            attempt, queued = d["attempt"], d["waited_ms"]
            return f"{head} ${d['idx']} {d['tool']} (attempt {attempt}, queued {queued:.1f} ms)"
        case "task_succeeded":
            result = json.dumps(d["result"], ensure_ascii=False)
            return f"{head} ${d['idx']} {d['tool']} {d['busy_ms']:.1f} ms -> {result[:70]}"
        case "task_failed" | "task_rejected":
            return f"{head} ${d['idx']} {d['tool']}: {d.get('error')}"
        case "task_cancelled":
            return f"{head} ${d['idx']} {d['tool']}: {d.get('reason')}"
        case "task_retry":
            return f"{head} ${d['idx']} {d['tool']} after {d['backoff_ms']:.1f} ms: {d['error']}"
        case "task_reused":
            return f"{head} ${d['idx']} {d['tool']} (signature {d['signature']})"
        case "thought":
            return f"{head} {d['text']}"
        case "join_decision":
            return f"{head} {d['action']}: {d.get('answer') or d.get('reason')}"
        case _:
            compact = json.dumps(d, ensure_ascii=False, default=str)
            return f"{head} {compact[:110]}" if d else head


async def _ask(
    question: str,
    options: list[str],
    catalog: str,
    concurrency: int,
    mode: str,
    planner: str,
    joiner: str,
    as_json: bool,
) -> int:
    from tooldag.events import EventLog
    from tooldag.runtime import Runtime, RuntimeConfig
    from tooldag.scheduler import Limits
    from tooldag.service import load_domain, make_joiner, make_planner
    from tooldag.tools.base import tool_pool

    domain = load_domain(catalog)  # type: ignore[arg-type]
    log = EventLog("cli")
    with tool_pool(concurrency) as pool:
        runtime = Runtime(
            domain.tools,
            make_planner(planner, domain),
            make_joiner(joiner),
            RuntimeConfig(limits=Limits(max_concurrency=concurrency), mode=mode),  # type: ignore[arg-type]
            executor=pool,
        )
        task = asyncio.create_task(runtime.run(question, options, log=log))
        async for event in log.follow():
            if event is None:
                continue
            if as_json:
                print(event.model_dump_json())
            else:
                print(describe(event))
        result = await task
    if not as_json:
        source = f"{domain.catalog.source} catalogue, {domain.embedder.name}"
        print(f"\nanswer: {result.answer}  ({source})")
    return 0 if result.answer else 1


@app.command()
def fetch(
    only: Annotated[list[str] | None, typer.Option(help="asset name(s) to fetch")] = None,
) -> None:
    """Download datasets and the MiniLM model into $TOOLDAG_DATA_DIR, verified by sha256."""
    from tooldag.fetch import ASSETS
    from tooldag.fetch import fetch as do_fetch

    root = data_dir()
    print(f"data dir: {root}")
    do_fetch(root, only)
    for asset in ASSETS:
        if not only or asset.name in only:
            print(f"  {asset.name}: {asset.license}")


@app.command()
def index(
    rebuild: Annotated[bool, typer.Option(help="rebuild even if up to date")] = False,
) -> None:
    """Build the joined MovieLens + CMU catalogue used by the tools."""
    from tooldag.index import ensure_catalog

    catalog = ensure_catalog(data_dir(), rebuild=rebuild)
    print(json.dumps(catalog.stats, indent=2))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="e.g. 'Find a movie similar to Heat, Ronin'")],
    option: Annotated[list[str], typer.Option("--option", "-o", help="answer option (repeat)")],
    catalog: CatalogOpt = "auto",
    concurrency: Annotated[int, typer.Option(min=1, max=64)] = 6,
    mode: Annotated[str, typer.Option(help="dag | sequential")] = "dag",
    planner: Annotated[str, typer.Option(help="template | llm")] = "template",
    joiner: Annotated[str, typer.Option(help="rule | llm")] = "rule",
    as_json: Annotated[
        bool, typer.Option("--json", help="print raw events as JSON lines")
    ] = False,
) -> None:
    """Answer one question and print the task lifecycle as it happens."""
    code = asyncio.run(
        _ask(question, option, catalog, concurrency, mode, planner, joiner, as_json)
    )
    raise typer.Exit(code)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    catalog: CatalogOpt = "auto",
    workers: Annotated[int, typer.Option(help="threads for blocking tools (ONNX)")] = 6,
) -> None:
    """Run the HTTP/SSE service."""
    from aiohttp import web

    from tooldag.server import ServiceState, create_app
    from tooldag.service import load_domain, make_joiner, make_planner
    from tooldag.tools.base import tool_pool

    domain = load_domain(catalog)  # type: ignore[arg-type]
    pool = tool_pool(workers)
    state = ServiceState(
        tools=domain.tools,
        planner_factory=lambda kind: make_planner(kind, domain),
        joiner_factory=make_joiner,
        executor=pool,
        info={
            "catalog": domain.catalog.source,
            "movies": len(domain.catalog.movies),
            "embedder": domain.embedder.name,
            "scorer": domain.scorer.origin,
        },
    )
    print(f"serving on http://{host}:{port} ({domain.catalog.source}, {domain.embedder.name})")
    try:
        web.run_app(create_app(state), host=host, port=port, handler_cancellation=True, print=None)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


@app.command()
def demo() -> None:
    """Offline demo on the bundled fixture: one normal run and one replan after a bad plan."""
    from tooldag.bench.replan import run_scenarios

    with open(Path(__file__).parent / "fixtures" / "mini_catalog.json", encoding="utf-8") as fh:
        q = json.load(fh)["questions"][0]
    code = asyncio.run(
        _ask(q["question"], q["options"], "fixture", 4, "dag", "template", "rule", False)
    )
    print("\nfault scenarios (scripted planner, fixture catalogue):")
    failures = 0
    for row in run_scenarios():
        ok = row["answer"] == row["expected_answer"]
        failures += not ok
        print(
            f"  {row['scenario']:<26} rounds={row['rounds']} executed={row['tasks_executed']} "
            f"reused={row['tasks_reused']} retries={row['retries']} answer={row['answer']!r}"
            f"{'' if ok else '  (unexpected)'}"
        )
    raise typer.Exit(1 if code or failures else 0)


@app.command()
def bench(
    out_dir: Annotated[Path, typer.Option(help="where result JSON files go")] = Path(
        "bench/results"
    ),
    smoke: Annotated[bool, typer.Option(help="tiny subset; files are named smoke-*.json")] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(help="accuracy | makespan | streaming | overhead | replan | trace"),
    ] = None,
) -> None:
    """Run the benchmark suite and write bench/results/*.json."""
    from tooldag.bench.suite import run_suite

    command = "tooldag " + " ".join(sys.argv[1:])
    run_suite(out_dir, smoke=smoke, only=only, command=command)


@app.command()
def report(
    results_dir: Annotated[Path, typer.Option()] = Path("bench/results"),
) -> None:
    """Print the README tables from the stored result files."""
    from tooldag.bench.report import render

    print(render(results_dir))


def main() -> None:
    # Windows consoles default to a legacy code page; titles contain non-ASCII letters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    app()


if __name__ == "__main__":
    main()
