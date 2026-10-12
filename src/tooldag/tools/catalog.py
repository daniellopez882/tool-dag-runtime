"""The movie catalogue the tools read: titles, genres, likers and plot summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

from tooldag.tools.titles import TrigramIndex

if TYPE_CHECKING:
    from pathlib import Path

CATALOG_FORMAT = 1


@dataclass(frozen=True, slots=True)
class Movie:
    id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    names: tuple[str, ...]
    n_ratings: int
    plot: str = ""
    wiki_id: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "genres": list(self.genres),
            "names": list(self.names),
            "n_ratings": self.n_ratings,
            "plot": self.plot,
            "wiki_id": self.wiki_id,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Movie:
        return cls(
            id=int(row["id"]),
            title=row["title"],
            year=row.get("year"),
            genres=tuple(row.get("genres", ())),
            names=tuple(row.get("names", ())),
            n_ratings=int(row.get("n_ratings", 0)),
            plot=row.get("plot", ""),
            wiki_id=row.get("wiki_id"),
        )


@dataclass(slots=True)
class Catalog:
    movies: dict[int, Movie]
    likers: dict[int, frozenset[int]]
    source: str
    stats: dict[str, Any] = field(default_factory=dict)
    _index: TrigramIndex | None = None

    @property
    def index(self) -> TrigramIndex:
        if self._index is None:
            self._index = TrigramIndex.build(
                ((m.id, m.names) for m in self.movies.values()),
                {m.id: m.n_ratings for m in self.movies.values()},
            )
        return self._index

    def save(self, directory: Path, meta: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "movies.jsonl.part"
        with tmp.open("w", encoding="utf-8") as fh:
            for movie in self.movies.values():
                row = movie.to_json()
                row["likers"] = sorted(self.likers.get(movie.id, ()))
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(directory / "movies.jsonl")
        payload = {"format": CATALOG_FORMAT, "source": self.source, "stats": self.stats, **meta}
        (directory / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Catalog:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != CATALOG_FORMAT:
            raise ValueError(f"catalogue at {directory} has an old format; run `tooldag index`")
        movies: dict[int, Movie] = {}
        likers: dict[int, frozenset[int]] = {}
        with (directory / "movies.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                movie = Movie.from_json(row)
                movies[movie.id] = movie
                if row.get("likers"):
                    likers[movie.id] = frozenset(row["likers"])
        return cls(movies, likers, meta["source"], meta.get("stats", {}))

    @classmethod
    def fixture(cls) -> Catalog:
        """Twelve invented films bundled with the package, for tests and the offline demo."""
        raw = resources.files("tooldag.fixtures").joinpath("mini_catalog.json").read_text("utf-8")
        data = json.loads(raw)
        movies = {row["id"]: Movie.from_json(row) for row in data["movies"]}
        likers = {row["id"]: frozenset(row.get("likers", ())) for row in data["movies"]}
        return cls(movies, likers, "fixture", {"movies": len(movies)})
