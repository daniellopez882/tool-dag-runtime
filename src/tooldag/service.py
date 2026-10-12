"""Assemble catalogue, embedder, scorer and tools for the CLI, the server and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tooldag.config import data_dir, weights_path
from tooldag.fetch import ASSETS_BY_NAME, is_present
from tooldag.index import ensure_catalog
from tooldag.joiner import LLMJoiner, RuleJoiner
from tooldag.planners import LLMPlanner, TemplatePlanner
from tooldag.providers.openai_compat import OpenAICompatibleProvider
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder, MiniLMEmbedder
from tooldag.tools.movies import MovieTools
from tooldag.tools.scorer import LinearScorer
from tooldag.tools.titles import segment_titles

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tooldag.rounds import Joiner, Planner
    from tooldag.tools.base import ToolRegistry
    from tooldag.tools.embed import Embedder

CatalogMode = Literal["auto", "full", "fixture"]
_FULL_ASSETS = (
    "movielens-latest-small",
    "cmu-movie-summaries",
    "minilm-onnx-quantized",
    "minilm-vocab",
)


class MissingDataError(RuntimeError):
    pass


def full_data_present(root: Path) -> bool:
    return all(is_present(root, ASSETS_BY_NAME[n], verify=False) for n in _FULL_ASSETS)


@dataclass(slots=True)
class Domain:
    catalog: Catalog
    embedder: Embedder
    scorer: LinearScorer
    movie_tools: MovieTools

    @property
    def tools(self) -> ToolRegistry:
        return self.movie_tools.registry()

    def segmenter(self) -> Callable[[str], list[str]]:
        index = self.catalog.index

        def score(text: str) -> float:
            found = index.best(text)
            return found.score if found else 0.0

        return lambda text: segment_titles(text, score)


def load_domain(
    mode: CatalogMode = "auto",
    *,
    root: Path | None = None,
    intra_op_threads: int = 1,
    embedder: Embedder | None = None,
) -> Domain:
    root = root or data_dir()
    if mode == "auto":
        mode = "full" if full_data_present(root) else "fixture"
    if mode == "full":
        if not full_data_present(root):
            raise MissingDataError(f"data not found under {root}; run `tooldag fetch` first")
        catalog = ensure_catalog(root)
        if embedder is None:
            embedder = MiniLMEmbedder(
                root / "models" / "all-MiniLM-L6-v2", intra_op_threads=intra_op_threads
            )
        scorer = LinearScorer.load_or_equal(weights_path(root))
    else:
        catalog = Catalog.fixture()
        embedder = embedder or HashingEmbedder()
        scorer = LinearScorer.equal()
    return Domain(catalog, embedder, scorer, MovieTools(catalog, embedder, scorer))


def make_planner(kind: str, domain: Domain, *, line_delay_s: float = 0.0) -> Planner:
    if kind == "template":
        return TemplatePlanner(domain.segmenter(), line_delay_s=line_delay_s)
    if kind == "llm":
        return LLMPlanner(OpenAICompatibleProvider.from_env())
    raise ValueError(f"unknown planner {kind!r}")


def make_joiner(kind: str) -> Joiner:
    if kind == "rule":
        return RuleJoiner()
    if kind == "llm":
        return LLMJoiner(OpenAICompatibleProvider.from_env())
    raise ValueError(f"unknown joiner {kind!r}")
