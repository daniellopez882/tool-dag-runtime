from __future__ import annotations

import asyncio
import json
from importlib import resources

import pytest

from tooldag.events import EventLog
from tooldag.joiner import LLMJoiner, RuleJoiner, diff_rounds, parse_join_reply, problems_of
from tooldag.planners import LLMPlanner, ScriptedPlanner, TemplatePlanner
from tooldag.planners.llm import plan_messages
from tooldag.planners.template import liked_text_of
from tooldag.providers import FakeProvider
from tooldag.rounds import PlanContext, PlannerError, RoundSummary
from tooldag.runtime import Runtime, RuntimeConfig
from tooldag.scheduler import Limits, RetryPolicy, TaskState
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder
from tooldag.tools.movies import MovieTools

FAST = RetryPolicy(max_attempts=2, base_delay_s=0.0005, max_delay_s=0.001)


def fixture_questions() -> list[dict]:
    raw = resources.files("tooldag.fixtures").joinpath("mini_catalog.json").read_text("utf-8")
    return json.loads(raw)["questions"]


@pytest.fixture(scope="module")
def tools():
    return MovieTools(Catalog.fixture(), HashingEmbedder()).registry()


def run(runtime: Runtime, question: str, options: list[str], log: EventLog | None = None):
    return asyncio.run(runtime.run(question, options, log=log))


def runtime_for(tools, planner, joiner=None, **config) -> Runtime:
    cfg = RuntimeConfig(limits=Limits(max_concurrency=4), retry=FAST, **config)
    return Runtime(tools, planner, joiner or RuleJoiner(), cfg)


@pytest.mark.parametrize("mode", ["dag", "sequential"])
@pytest.mark.parametrize("q", fixture_questions(), ids=lambda q: q["answer"])
def test_template_planner_answers_fixture_questions(tools, q, mode) -> None:
    result = run(runtime_for(tools, TemplatePlanner(), mode=mode), q["question"], q["options"])
    assert result.answer == q["answer"]
    assert len(result.rounds) == 1
    assert result.model_calls() == {
        "llmcompiler": 2,
        "react_equivalent": result.tasks_executed + 1,
    }


def test_dag_and_sequential_produce_identical_rankings(tools) -> None:
    q = fixture_questions()[0]
    dag = run(runtime_for(tools, TemplatePlanner()), q["question"], q["options"])
    seq = run(
        runtime_for(tools, TemplatePlanner(), mode="sequential"), q["question"], q["options"]
    )
    assert dag.answer_result.scores == seq.answer_result.scores


def test_streaming_dispatch_starts_tasks_before_the_plan_is_complete(tools) -> None:
    q = fixture_questions()[0]
    log = EventLog()
    result = run(
        runtime_for(tools, TemplatePlanner(line_delay_s=0.002)), q["question"], q["options"], log
    )
    summary = result.rounds[0]
    assert summary.first_dispatch_ms is not None and summary.plan_complete_ms is not None
    assert summary.first_dispatch_ms < summary.plan_complete_ms
    types = [e.type for e in log.events]
    assert types.index("task_succeeded") < types.index("plan_complete")


def test_buffered_dispatch_waits_for_the_whole_plan(tools) -> None:
    q = fixture_questions()[0]
    log = EventLog()
    runtime = runtime_for(tools, TemplatePlanner(line_delay_s=0.001), stream_dispatch=False)
    run(runtime, q["question"], q["options"], log)
    types = [e.type for e in log.events]
    assert types.index("plan_complete") < types.index("task_started")


def clean_plan(tools, q) -> str:
    from tooldag.plan.render import render_plan

    ctx = PlanContext(q["question"], tuple(q["options"]), tools, 1)
    return render_plan(TemplatePlanner().build(ctx))


def test_replan_after_malformed_plan_reuses_finished_work(tools) -> None:
    q = fixture_questions()[1]
    good = clean_plan(tools, q)
    lines = good.splitlines()
    rank_line = next(i for i, line in enumerate(lines) if ". rank(" in line)
    broken = "\n".join([*lines[:rank_line], "99. rank([$1, oops)", *lines[rank_line + 1 :]])
    planner = ScriptedPlanner([broken, good], chunk_chars=5)
    log = EventLog()
    result = run(runtime_for(tools, planner), q["question"], q["options"], log)
    assert result.answer == q["answer"]
    assert len(result.rounds) == 2
    assert result.rounds[0].issues
    second = result.rounds[1]
    executed = [r.call.tool for r in second.executed()]
    assert executed == ["rank"]
    assert result.tasks_reused == len(second.records) - 1
    assert planner.contexts[1].feedback is not None
    assert "not understood" in planner.contexts[1].feedback
    diff = next(e for e in log.events if e.type == "plan_diff")
    assert diff.data["executed"] == [second.records[max(second.records)].call.idx]


def test_cyclic_and_dangling_plans_are_rejected_then_replanned(tools) -> None:
    q = fixture_questions()[2]
    cyclic = (
        '1. resolve_title("The Vault at Midnight")\n'
        "2. cooccurrence([$1], $3)\n"
        "3. genre_jaccard([$2], $1)\n"
        "4. rank([$1], cooccurrence=[$2], genre_jaccard=[$7])\n"
        "5. join()\n"
    )
    planner = ScriptedPlanner([cyclic, clean_plan(tools, q)])
    result = run(runtime_for(tools, planner), q["question"], q["options"])
    first = result.rounds[0].records
    assert first[2].state is TaskState.FAILED and "cycle" in (first[2].error or "")
    assert first[3].state is TaskState.FAILED
    assert first[4].state in (TaskState.FAILED, TaskState.CANCELLED)
    assert result.answer == q["answer"]


def test_joiner_gives_up_after_max_rounds(tools) -> None:
    planner = ScriptedPlanner(['1. imdb("x")\n2. join()\n'] * 2)
    result = run(runtime_for(tools, planner, max_rounds=2), "Find a movie similar to X", ["A"])
    assert result.answer is None
    assert len(result.rounds) == 2
    assert "no answer after 2 round(s)" in result.decision.reason


def test_planner_error_ends_the_run(tools) -> None:
    result = run(runtime_for(tools, TemplatePlanner()), "What should I watch?", ["A"])
    assert result.answer is None
    assert "planner failed" in (result.decision.reason if result.decision else "")


def test_template_planner_drops_failed_titles_on_replan(tools) -> None:
    q = fixture_questions()[0]
    question = q["question"].replace("similar to ", "similar to Nowhere Man, ")
    result = run(runtime_for(tools, TemplatePlanner()), question, q["options"])
    assert result.answer == q["answer"]
    assert len(result.rounds) == 2
    round2_titles = [
        r.call.args[0] for r in result.rounds[1].records.values() if r.call.tool == "resolve_title"
    ]
    assert "Nowhere Man" not in round2_titles
    assert result.tasks_reused > 0
    diff = diff_rounds(result.rounds[:1], result.rounds[1])
    assert diff.dropped > 0


def test_liked_text_of_rejects_other_questions() -> None:
    assert liked_text_of("Find a movie similar to Heat, Ronin:") == "Heat, Ronin"
    with pytest.raises(PlannerError):
        liked_text_of("Recommend something")


def test_llm_planner_and_joiner_with_fake_provider(tools) -> None:
    q = fixture_questions()[0]
    plan = "Thought: resolve and compare.\n" + clean_plan(tools, q).split("\n", 1)[1]
    plan += "<END_OF_PLAN>\nThis trailing prose is ignored."
    provider = FakeProvider(
        [plan, "Thought: rank says so.\nAction: Finish(Comet Wardens)"], chunk_chars=3
    )
    runtime = runtime_for(tools, LLMPlanner(provider), LLMJoiner(provider))
    result = run(runtime, q["question"], q["options"])
    assert result.answer == "Comet Wardens"
    assert len(provider.calls) == 2
    system = provider.calls[0][0]["content"]
    assert "resolve_title(title: str) -> movie" in system


def test_llm_joiner_replan_then_out_of_rounds(tools) -> None:
    provider = FakeProvider(
        [
            '1. resolve_title("Orbit of Ash")\n2. join()\n',
            "Action: Replan(rank the options)",
            '1. resolve_title("Orbit of Ash")\n2. join()\n',
            "Action: Replan(still no ranking)",
        ]
    )
    runtime = runtime_for(tools, LLMPlanner(provider), LLMJoiner(provider), max_rounds=2)
    result = run(runtime, "Find a movie similar to Orbit of Ash", ["Comet Wardens"])
    assert result.answer is None
    assert result.tasks_reused == 1
    assert "out of rounds" in result.decision.reason
    replan_prompt = provider.calls[2][1]["content"]
    assert "Plan from round 1" in replan_prompt and "rank the options" in replan_prompt


def test_provider_failure_becomes_planner_error(tools) -> None:
    runtime = runtime_for(tools, LLMPlanner(FakeProvider([])))
    result = run(runtime, "Find a movie similar to Orbit of Ash", ["Comet Wardens"])
    assert result.answer is None
    assert result.rounds[0].planner_error


@pytest.mark.parametrize(
    ("reply", "action", "answer"),
    [
        ("Thought: ok\nAction: Finish(Heat)", "finish", "Heat"),
        ('Action: Finish("Heat")', "finish", "Heat"),
        ("Action: Finish(<x>)\nThought: a\nAction: Replan(use genres)", "replan", None),
        ("I think Heat.", "replan", None),
    ],
)
def test_parse_join_reply(reply: str, action: str, answer: str | None) -> None:
    decision = parse_join_reply(reply)
    assert (decision.action, decision.answer) == (action, answer)


def test_cancelling_a_run_cancels_running_tools(tools) -> None:
    async def scenario() -> list[str]:
        started = asyncio.Event()

        async def slow_plan(ctx):
            yield '1. resolve_title("Orbit of Ash")\n'
            started.set()
            await asyncio.sleep(10)

        class Slow:
            name = "slow"

            def stream_plan(self, ctx):
                return slow_plan(ctx)

        log = EventLog()
        runtime = runtime_for(tools, Slow())
        task = asyncio.create_task(runtime.run("Find a movie similar to X", ["A"], log=log))
        await started.wait()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return [e.type for e in log.events]

    types = asyncio.run(scenario())
    assert types[-1] == "run_cancelled"


def test_problems_of_summarises_failures() -> None:
    summary = RoundSummary(1, planner_error="boom")
    assert "planner failed: boom" in problems_of(summary)
    assert problems_of(RoundSummary(1)) == "the plan contained no tasks"


def test_plan_messages_include_history(tools) -> None:
    summary = RoundSummary(1, plan_text='1. resolve_title("A")\n')
    ctx = PlanContext("Find a movie similar to A", ("B",), tools, 2, (summary,), "fix it")
    messages = plan_messages(ctx)
    assert messages[0]["role"] == "system"
    assert "Joiner feedback: fix it" in messages[1]["content"]
