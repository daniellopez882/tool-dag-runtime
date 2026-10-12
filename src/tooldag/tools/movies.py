"""The movie-recommendation tools the planner can call."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tooldag.tools.base import Param, Tool, ToolError, ToolRegistry
from tooldag.tools.embed import Vector
from tooldag.tools.scorer import SIGNALS, LinearScorer, features_matrix

if TYPE_CHECKING:
    from tooldag.tools.catalog import Catalog
    from tooldag.tools.embed import Embedder

MATCH_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class MovieHit:
    id: int
    title: str
    year: int | None
    match: float

    def preview(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "match": round(self.match, 3),
        }

    def __str__(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


@dataclass(frozen=True, slots=True)
class Ranking:
    titles: tuple[str, ...]
    ids: tuple[int, ...]
    scores: tuple[float, ...]
    signals: tuple[str, ...]

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def best(self) -> str:
        return self.titles[self.best_index]

    def preview(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "scores": {t: round(s, 4) for t, s in zip(self.titles, self.scores, strict=True)},
            "signals": list(self.signals),
        }

    def __str__(self) -> str:
        return self.best


class MovieTools:
    def __init__(
        self,
        catalog: Catalog,
        embedder: Embedder,
        scorer: LinearScorer | None = None,
        *,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.scorer = scorer or LinearScorer.equal()
        self.match_threshold = match_threshold

    def resolve_title(self, title: str) -> MovieHit:
        found = self.catalog.index.best(title)
        if found is None or found.score < self.match_threshold:
            best = f" (closest match scored {found.score:.2f})" if found else ""
            raise ToolError(f"no catalogue title matches {title!r}{best}")
        movie = self.catalog.movies[found.movie_id]
        return MovieHit(movie.id, movie.title, movie.year, found.score)

    def plot(self, movie: MovieHit) -> str:
        return self.catalog.movies[movie.id].plot

    def embed(self, text: str) -> Vector:
        return self.embedder.embed(text)

    @staticmethod
    def plot_similarity(liked: list[Vector], candidate: Vector) -> float | None:
        sims = [s for v in liked if (s := candidate.cosine(v)) is not None]
        return float(np.mean(sims)) if sims else None

    def cooccurrence(self, liked: list[MovieHit], candidate: MovieHit) -> float:
        """Mean cosine between the candidate's and each liked film's set of likers."""
        users_c = self.catalog.likers.get(candidate.id, frozenset())
        if not users_c or not liked:
            return 0.0
        total = 0.0
        for movie in liked:
            users_l = self.catalog.likers.get(movie.id, frozenset())
            if users_l:
                total += len(users_c & users_l) / math.sqrt(len(users_c) * len(users_l))
        return total / len(liked)

    def genre_jaccard(self, liked: list[MovieHit], candidate: MovieHit) -> float:
        genres_c = set(self.catalog.movies[candidate.id].genres)
        if not liked:
            return 0.0
        total = 0.0
        for movie in liked:
            genres_l = set(self.catalog.movies[movie.id].genres)
            union = genres_c | genres_l
            total += len(genres_c & genres_l) / len(union) if union else 0.0
        return total / len(liked)

    def rank(
        self,
        candidates: list[MovieHit],
        cooccurrence: list[float | None] | None = None,
        plot_similarity: list[float | None] | None = None,
        genre_jaccard: list[float | None] | None = None,
    ) -> Ranking:
        given = {
            name: values
            for name, values in (
                ("cooccurrence", cooccurrence),
                ("plot_similarity", plot_similarity),
                ("genre_jaccard", genre_jaccard),
            )
            if values is not None
        }
        if not candidates:
            raise ToolError("rank() needs at least one candidate")
        if not given:
            raise ToolError("rank() needs at least one signal")
        signals = tuple(s for s in SIGNALS if s in given)
        try:
            matrix = features_matrix(signals, given, len(candidates))
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        scores = self.scorer.restrict(signals).scores(matrix)
        return Ranking(
            tuple(c.title for c in candidates),
            tuple(c.id for c in candidates),
            tuple(float(s) for s in scores),
            signals,
        )

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        hit_list = list[MovieHit]
        maybe_floats = list[float | None] | None
        reg.register(
            Tool(
                "resolve_title",
                "Find the catalogue film whose title best matches the text (trigram match).",
                (Param("title", str, "film title as written in the question"),),
                self.resolve_title,
                returns="movie",
            )
        )
        reg.register(
            Tool(
                "plot",
                "Plot summary of a resolved film (empty string when the corpus has none).",
                (Param("movie", MovieHit, "result of resolve_title"),),
                self.plot,
                returns="str",
            )
        )
        reg.register(
            Tool(
                "embed",
                "MiniLM sentence embedding of a text (CPU ONNX inference).",
                (Param("text", str),),
                self.embed,
                returns="vector",
                blocking=True,
            )
        )
        reg.register(
            Tool(
                "plot_similarity",
                "Mean cosine similarity between a candidate's plot vector and liked plots.",
                (Param("liked", list[Vector]), Param("candidate", Vector)),
                self.plot_similarity,
                returns="float | None",
            )
        )
        reg.register(
            Tool(
                "cooccurrence",
                "How often users who liked the given films also liked the candidate.",
                (Param("liked", hit_list), Param("candidate", MovieHit)),
                self.cooccurrence,
                returns="float",
            )
        )
        reg.register(
            Tool(
                "genre_jaccard",
                "Mean Jaccard overlap between the candidate's genres and each liked film's.",
                (Param("liked", hit_list), Param("candidate", MovieHit)),
                self.genre_jaccard,
                returns="float",
            )
        )
        reg.register(
            Tool(
                "rank",
                "Score candidates with the weighted scorer; signal lists align with candidates.",
                (
                    Param("candidates", hit_list),
                    Param("cooccurrence", maybe_floats, default=None),
                    Param("plot_similarity", maybe_floats, default=None),
                    Param("genre_jaccard", maybe_floats, default=None),
                ),
                self.rank,
                returns="ranking",
            )
        )
        return reg
