"""Tool descriptions, argument binding and runtime type checks."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from concurrent.futures import Executor

POOL_START_TIMEOUT_S = 30.0


def tool_pool(workers: int) -> ThreadPoolExecutor:
    """Thread pool for blocking tools, with every worker thread already running.

    ``ThreadPoolExecutor`` starts threads lazily inside ``submit()``, and
    ``Thread.start()`` waits until the OS has scheduled the new thread. Called from a
    tool dispatch, that wait happens on the event loop and stalls plan parsing and
    every other task; on a busy machine it was measured in tens of milliseconds.
    """
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool")
    barrier = threading.Barrier(workers + 1, timeout=POOL_START_TIMEOUT_S)
    # Each submit finds no idle thread (the earlier ones block on the barrier), so the
    # pool has to create all ``workers`` threads before the barrier opens.
    futures = [pool.submit(barrier.wait) for _ in range(workers)]
    barrier.wait()
    for future in futures:
        future.result()
    return pool


class ToolError(Exception):
    """The call failed and retrying the same arguments will not help."""


class TransientToolError(ToolError):
    """The call failed in a way a retry might fix (the scheduler retries these)."""


class ToolInputError(ToolError):
    """The arguments do not fit the tool's parameters."""


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    annotation: Any
    description: str = ""
    default: Any = REQUIRED

    @property
    def required(self) -> bool:
        return self.default is REQUIRED


@dataclass(frozen=True, slots=True)
class Tool:
    """A named function the planner can call.

    ``blocking`` tools run on the runtime's thread pool. That is only worth it for work
    that releases the GIL (ONNX inference); pure-Python tools run inline on the loop.
    """

    name: str
    description: str
    params: tuple[Param, ...]
    fn: Callable[..., Any]
    returns: str = "object"
    blocking: bool = False

    def arity_problem(self, n_args: int, kw_names: Sequence[str]) -> str | None:
        names = [p.name for p in self.params]
        if n_args > len(names):
            return f"{self.name}() takes {len(names)} arguments, got {n_args}"
        for kw in kw_names:
            if kw not in names:
                return f"{self.name}() has no parameter {kw!r}"
            if names.index(kw) < n_args:
                return f"{self.name}() got {kw!r} twice"
        given = set(names[:n_args]) | set(kw_names)
        missing = [p.name for p in self.params if p.required and p.name not in given]
        if missing:
            return f"{self.name}() missing {', '.join(missing)}"
        return None

    def bind(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        problem = self.arity_problem(len(args), list(kwargs))
        if problem is not None:
            raise ToolInputError(problem)
        bound: dict[str, Any] = {}
        for param, value in zip(self.params, args, strict=False):
            bound[param.name] = value
        bound.update(kwargs)
        for param in self.params:
            if param.name not in bound:
                bound[param.name] = param.default
            elif not conforms(bound[param.name], param.annotation):
                got = type(bound[param.name]).__name__
                want = describe_type(param.annotation)
                raise ToolInputError(f"{self.name}({param.name}=...) expects {want}, got {got}")
        return bound

    async def invoke(self, bound: Mapping[str, Any], executor: Executor | None) -> Any:
        if self.blocking:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, functools.partial(self.fn, **bound))
        result = self.fn(**bound)
        if inspect.isawaitable(result):
            result = await result
        return result

    def signature(self) -> str:
        parts = []
        for p in self.params:
            text = f"{p.name}: {describe_type(p.annotation)}"
            if not p.required:
                text += f" = {p.default!r}"
            parts.append(text)
        return f"{self.name}({', '.join(parts)}) -> {self.returns}"

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature(),
            "description": self.description,
            "params": [
                {
                    "name": p.name,
                    "type": describe_type(p.annotation),
                    "required": p.required,
                    "description": p.description,
                }
                for p in self.params
            ],
            "returns": self.returns,
            "blocking": self.blocking,
        }


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        if tool.name == "join":
            raise ValueError("'join' is reserved for the end-of-plan marker")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def check_call(self, tool: str, n_args: int, kw_names: Sequence[str]) -> str | None:
        found = self._tools.get(tool)
        if found is None:
            return f"unknown tool {tool!r}"
        return found.arity_problem(n_args, kw_names)


def conforms(value: Any, annotation: Any) -> bool:
    """Structural ``isinstance`` for the annotations tools use (unions and ``list[T]``)."""
    if annotation is Any or annotation is object:
        return True
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(conforms(value, arm) for arm in get_args(annotation))
    if origin is list:
        (inner,) = get_args(annotation) or (Any,)
        return isinstance(value, list | tuple) and all(conforms(v, inner) for v in value)
    if annotation is None or annotation is type(None):
        return value is None
    if annotation is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, annotation)


def describe_type(annotation: Any) -> str:
    if annotation is Any or annotation is object:
        return "any"
    if annotation is None or annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return " | ".join(describe_type(a) for a in get_args(annotation))
    if origin is list:
        return f"list[{', '.join(describe_type(a) for a in get_args(annotation))}]"
    return getattr(annotation, "__name__", str(annotation))
