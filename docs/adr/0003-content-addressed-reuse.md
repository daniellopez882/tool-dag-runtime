# ADR 0003: Reuse work across replans by content signature, not task number

Status: accepted

## Context

When the joiner asks for a new plan, the planner writes a complete plan again,
numbered from 1. Most of it is usually the same work as before: the failed title is
dropped, but the other titles, plots and embeddings are unchanged. Re-running them
would make a replan cost as much as the first round. The new plan's task numbers
cannot be trusted to line up with the old ones; removing one title shifts every
later number.

## Decision

Each call gets a content signature (`dag.call_signature`): a hash of the tool name,
the literal arguments and, for every `$N` reference, the signature of task `N`. Two
calls in different rounds have the same signature exactly when they apply the same
tool to the same inputs computed the same way, regardless of numbering. When a task
becomes ready, the Task Fetching Unit looks its signature up in the results of
earlier rounds and, on a hit, settles the task from that result (`task_reused`)
instead of calling the tool. A `plan_diff` event reports what was reused, what ran
and how many old calls the new plan dropped.

## Consequences

- Renumbering, reordering and inserting tasks do not defeat reuse.
- A change upstream invalidates everything downstream of it automatically, because
  the downstream signatures include it. Dropping one liked title re-runs the signal
  tasks that read the liked list, but not the title lookups, plots or embeddings.
- Reuse assumes tools are deterministic for equal inputs. That holds for every tool
  here; a tool with side effects or randomness would need to opt out.
- Signatures are computed from the plan, not from result values, so no result (a
  vector, a ranking) ever has to be hashed.

## Alternatives rejected

- **Diff plans by task number.** Breaks as soon as the planner renumbers.
- **Cache by tool name and resolved argument values.** Needs a stable hash of every
  result type, including numpy vectors, and would treat two different calls that
  happen to produce equal values as interchangeable.
- **Ask the planner to write only the new tasks.** Shorter replans, but the planner
  then has to refer to results it cannot see by number, and a model gets that wrong.
