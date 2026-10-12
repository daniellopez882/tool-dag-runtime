# ADR 0001: Dispatch each task when its plan line completes

Status: accepted

## Context

In the LLMCompiler pattern a planner writes a numbered plan, and a Task Fetching Unit
runs the tasks. A chat model writes such a plan at tens of tokens per second, so a
40-task plan takes seconds to arrive. If execution waits for the whole plan, every
tool call sits behind the slowest part of the system, the model. If it starts tasks
as their lines arrive, three questions come up that a "parse the finished plan"
design never has to answer:

1. A line may reference a task that has not been written yet (`$7` on line 3).
2. A cycle can only be seen once its closing edge arrives, possibly after some of its
   members would otherwise have been ready.
3. A reference to a task that never appears is only known to be dangling at the end
   of the plan.

The executor also has to respect a global concurrency limit and per-tool limits
(ONNX inference is the expensive tool; title lookups are cheap) without one saturated
tool holding back ready tasks of other tools.

## Decision

- The parser (`plan/parser.py`) is incremental: `feed()` takes text chunks of any size
  and returns every item whose line is complete. Malformed lines become `ParseIssue`
  items rather than exceptions, so one bad line does not stop the rest of the plan.
- The graph (`dag.py`) accepts forward references. A task whose dependency has not
  arrived stays `WAITING`. On every arrival the graph searches for a cycle through the
  new node, following the tasks that referenced it early; cycle members are rejected
  and everything downstream of them is cancelled. At end of plan, tasks still waiting
  on undefined numbers fail with a "never defined" error.
- The Task Fetching Unit (`scheduler.py`) keeps one FIFO queue per tool and a global
  arrival sequence number. The pump repeatedly takes the oldest ready task among
  tools that are under their own limit, until the global limit is reached. Failure
  and cancellation propagate to dependents, including dependents that arrive after
  the failure.
- The same bookkeeping core drives the `SequentialExecutor` baseline, so the two
  executors differ only in when tasks start.

## Consequences

- Tasks start while the planner is still writing; the streaming benchmark measures how
  much of the tool time this hides under a paced planner.
- Cycle detection costs nothing for plans without forward references, which is what
  every planner in this repository writes.
- The scheduler has more states (waiting, ready, running, backoff, terminal) and more
  invariants to test. These are checked with hypothesis over random plans, random
  arrival gaps and random failures: no task starts before its dependencies succeed,
  global and per-tool limits hold, and failures and cancellation reach every
  dependent.

## Alternatives rejected

- **Parse the whole plan, then execute.** Simpler, and kept as a configuration
  (`stream_dispatch=False`) so the difference can be measured, but it serialises
  planning and execution.
- **Reject forward references.** Would let the graph assume a topological arrival
  order, but models do write them, and rejecting a whole plan for it forces a replan.
- **One global FIFO queue.** A burst of ready `embed` tasks at the per-tool limit would
  block cheap tasks queued behind them.
