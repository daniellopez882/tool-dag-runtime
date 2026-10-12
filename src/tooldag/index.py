"""Build the catalogue from the fetched MovieLens and CMU files.

Each MovieLens film keeps its title, alternate titles, genres, rating count and the
set of users who rated it 4.0 or higher. A CMU plot summary is attached when a title
variant and the release year match (allowing one year of drift between the sources).
The result is written under ``$TOOLDAG_DATA_DIR/catalog`` and never committed.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from tooldag.config import catalog_dir
from tooldag.fetch import ASSETS_BY_NAME, asset_path
from tooldag.tools.catalog import CATALOG_FORMAT, Catalog, Movie
from tooldag.tools.titles import normalize_title, parse_movielens_title

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LIKE_THRESHOLD = 4.0
_SOURCES = ("movielens-latest-small", "cmu-movie-summaries")


def _movielens(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("ml-latest-small/movies.csv") as fh:
            movies = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        with archive.open("ml-latest-small/ratings.csv") as fh:
            ratings = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
    return movies, ratings


def _cmu_members(path: Path) -> Iterator[tuple[str, str]]:
    wanted = {"MovieSummaries/plot_summaries.txt", "MovieSummaries/movie.metadata.tsv"}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if member.name in wanted:
                handle = archive.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read().decode("utf-8")


def _year(release: str) -> int | None:
    head = release[:4]
    return int(head) if head.isdigit() else None


def build_catalog(root: Path) -> Catalog:
    movie_rows, rating_rows = _movielens(asset_path(root, "movielens-latest-small"))
    counts: Counter[int] = Counter()
    likers: dict[int, set[int]] = defaultdict(set)
    for row in rating_rows:
        movie_id = int(row["movieId"])
        counts[movie_id] += 1
        if float(row["rating"]) >= LIKE_THRESHOLD:
            likers[movie_id].add(int(row["userId"]))

    plots: dict[int, str] = {}
    names_years: list[tuple[int, str, int | None]] = []
    for name, text in _cmu_members(asset_path(root, "cmu-movie-summaries")):
        if name.endswith("plot_summaries.txt"):
            for line in text.splitlines():
                wiki, _, summary = line.partition("\t")
                if wiki.isdigit() and summary.strip():
                    plots[int(wiki)] = summary.strip()
        else:
            for line in text.splitlines():
                cols = line.split("\t")
                if len(cols) >= 4 and cols[0].isdigit():
                    names_years.append((int(cols[0]), cols[2], _year(cols[3])))
    by_key: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    for wiki, title, year in names_years:
        if wiki in plots:
            by_key[(normalize_title(title), year)].append(wiki)

    join_rule: Counter[str] = Counter()
    movies: dict[int, Movie] = {}
    for row in movie_rows:
        movie_id = int(row["movieId"])
        parsed = parse_movielens_title(row["title"])
        names = (parsed.title, *parsed.alternates)
        wiki_id = None
        if parsed.year is not None:
            for rule, year in (
                ("exact year", parsed.year),
                ("year -1", parsed.year - 1),
                ("year +1", parsed.year + 1),
            ):
                hits = sorted(
                    {w for n in names for w in by_key.get((normalize_title(n), year), ())}
                )
                if hits:
                    wiki_id = hits[0]
                    join_rule[rule] += 1
                    break
        if wiki_id is None:
            join_rule["no plot"] += 1
        genres = tuple(g for g in row["genres"].split("|") if g and g != "(no genres listed)")
        movies[movie_id] = Movie(
            id=movie_id,
            title=parsed.title,
            year=parsed.year,
            genres=genres,
            names=names,
            n_ratings=counts.get(movie_id, 0),
            plot=plots.get(wiki_id, "") if wiki_id is not None else "",
            wiki_id=wiki_id,
        )
    stats = {
        "movies": len(movies),
        "ratings": len(rating_rows),
        "likes": sum(len(v) for v in likers.values()),
        "users": len({int(r["userId"]) for r in rating_rows}),
        "cmu_plots": len(plots),
        "plot_join": dict(join_rule),
    }
    frozen = {m: frozenset(users) for m, users in likers.items()}
    return Catalog(movies, frozen, "movielens-latest-small+cmu-movie-summaries", stats)


def _source_fingerprint() -> dict[str, str]:
    return {name: ASSETS_BY_NAME[name].sha256 for name in _SOURCES}


def ensure_catalog(root: Path, *, rebuild: bool = False) -> Catalog:
    """Load the derived catalogue, rebuilding it when missing or built from other sources."""
    directory = catalog_dir(root)
    meta_path = directory / "meta.json"
    if not rebuild and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") == CATALOG_FORMAT and meta.get("sources") == _source_fingerprint():
            return Catalog.load(directory)
    catalog = build_catalog(root)
    catalog.save(directory, {"sources": _source_fingerprint(), "like_threshold": LIKE_THRESHOLD})
    return catalog
