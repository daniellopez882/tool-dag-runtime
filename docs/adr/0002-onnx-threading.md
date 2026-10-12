# ADR 0002: One ONNX intra-op thread per call, parallelism from the scheduler

Status: accepted

## Context

The only expensive tool is `embed`: a MiniLM forward pass over a plot summary of up to
256 WordPiece tokens. The runtime is a single asyncio event loop. CPU-bound work on
the loop would stall every other task and the SSE streams, so blocking tools must run
somewhere else. onnxruntime can parallelise a single inference across threads
(intra-op parallelism), and it releases the GIL while it runs.

There are two ways to use six cores for a question that needs eight embeddings:

- run the embeddings one after another, each using six intra-op threads, or
- run up to six embeddings at once, each using one intra-op thread.

The second is what a DAG scheduler is for, but it only pays off if single-threaded
inference scales across Python threads, and if intra-op threading does not already
capture the same speedup for a sequential executor.

## Decision

- Tools marked `blocking=True` (only `embed`) run on a `ThreadPoolExecutor` sized to the
  scheduler's concurrency limit. Pure-Python tools run inline on the event loop; their
  cost is tens of microseconds and a thread hop would cost more.
- The pool's threads are all started before the first dispatch (`tools.base.tool_pool`).
  `ThreadPoolExecutor` otherwise creates each thread inside `submit()`, and
  `Thread.start()` blocks until the OS schedules the new thread; called from a tool
  dispatch, that wait happens on the event loop. On a loaded machine asyncio's debug
  mode showed single dispatch steps taking 30 to 60 ms, enough to delay plan parsing
  behind every early `embed` call.
- The `InferenceSession` is created with `intra_op_num_threads=1` and
  `ORT_SEQUENTIAL` execution, and is shared by all worker threads
  (`InferenceSession.run` is thread-safe).
- The makespan benchmark includes a sequential configuration whose session uses six
  intra-op threads, so the comparison between "parallel inside one call" and
  "parallel across calls" is measured on the same machine rather than assumed.

## Consequences

- Parallelism is visible to the scheduler: the per-tool limit and the global limit
  bound CPU use, and the event log shows how many embeddings overlapped.
- The pure-Python WordPiece tokenizer runs on the worker thread and holds the GIL
  while it runs. Its cost per plot is small next to inference, but it is serial across
  workers.
- Accuracy runs memoise embeddings by text (`CachingEmbedder`); timing runs never do,
  so every timed `embed` is a real ONNX call.

## Alternatives rejected

- **Process pool.** Avoids the GIL for tokenisation, but every worker loads its own
  model copy, and arguments and vectors have to be pickled across processes.
- **Batch all embeddings of a plan into one call.** Fewer, larger inferences, but it
  turns independent tasks into a barrier and defeats the point of dispatching each
  task as soon as its input exists.
- **Default onnxruntime threading.** Each session would start one thread per core;
  six concurrent calls would then run 36 threads on six cores.
