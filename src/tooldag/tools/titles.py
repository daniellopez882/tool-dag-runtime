"""Title normalisation and a character-trigram title resolver.

BIG-bench writes titles without punctuation ("Star Wars Episode IV - A New Hope"),
MovieLens writes them with suffixed articles and alternate titles in parentheses
("Léon: The Professional (a.k.a. The Professional) (Léon) (1994)"). Both sides are
normalised to lowercase ASCII words, every alternate title becomes its own entry, and
a query is matched first exactly and then by trigram Jaccard similarity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_ARTICLES = "The|A|An|Les|La|Le|L'|Il|El|Das|Der|Die|Los|Las|Lo|Gli|Den|Det|De|Het|Un|Une|Ein|Eine"
_SUFFIX_ARTICLE = re.compile(rf"^(?P<body>.+), (?P<article>{_ARTICLES})$")
_YEAR = re.compile(r"\s*\((?P<year>\d{4})(?:\s*[-–]\s*\d{0,4})?\)\s*$")
_TRAILING_PAREN = re.compile(r"\s*\((?P<alt>[^()]+)\)\s*$")
_AKA = re.compile(r"^(?:a\.k\.a\.|aka)\s+", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase ASCII words: articles fronted, accents dropped, '&' read as 'and'.

    Punctuation is deleted rather than turned into a space because BIG-bench deletes it:
    "L.A. Confidential" -> "LA Confidential", "8 1/2" -> "8 12", "R.I.P.D." -> "RIPD".
    """
    text = unicodedata.normalize("NFKD", front_article(title))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = "".join(ch for ch in text if ch.isspace() or ch.isalnum())
    return " ".join(_NON_ALNUM.sub(" ", text).split())


def front_article(title: str) -> str:
    """'Matrix, The' -> 'The Matrix'; "Auberge espagnole, L'" -> "L'Auberge espagnole"."""
    match = _SUFFIX_ARTICLE.match(title.strip())
    if match is None:
        return title.strip()
    article = match.group("article")
    joiner = "" if article.endswith("'") else " "
    return f"{article}{joiner}{match.group('body')}"


@dataclass(frozen=True, slots=True)
class ParsedTitle:
    title: str
    alternates: tuple[str, ...]
    year: int | None


def parse_movielens_title(raw: str) -> ParsedTitle:
    text = raw.strip()
    year: int | None = None
    match = _YEAR.search(text)
    if match is not None:
        year = int(match.group("year"))
        text = text[: match.start()]
    alternates: list[str] = []
    # Peel trailing "(alt title)" groups; a title that is only a parenthesis stays whole.
    while True:
        match = _TRAILING_PAREN.search(text)
        if match is None or match.start() == 0:
            break
        alternates.insert(0, front_article(_AKA.sub("", match.group("alt").strip())))
        text = text[: match.start()]
    return ParsedTitle(front_article(text), tuple(a for a in alternates if a), year)


def trigrams(normalized: str) -> frozenset[str]:
    padded = f"  {normalized} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


@dataclass(frozen=True, slots=True)
class TitleMatch:
    movie_id: int
    matched_name: str
    score: float


@dataclass(slots=True)
class TrigramIndex:
    """Inverted index from trigram to title variants; ties go to the more-rated movie.

    Posting lists are frozen into numpy arrays after the build, so a fuzzy query is one
    ``bincount`` over the concatenated postings of its trigrams instead of a Python loop
    over every shared entry (common trigrams such as " th" have thousands of entries).
    """

    popularity: dict[int, int] = field(default_factory=dict)
    _names: list[tuple[int, str, int]] = field(default_factory=list)
    _postings: dict[str, np.ndarray] = field(default_factory=dict)
    _exact: dict[str, list[int]] = field(default_factory=dict)
    _sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    _movie_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _pops: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def build(
        cls, entries: Iterable[tuple[int, Iterable[str]]], popularity: dict[int, int]
    ) -> TrigramIndex:
        index = cls(popularity=dict(popularity))
        postings: dict[str, list[int]] = {}
        for movie_id, names in entries:
            for name in names:
                norm = normalize_title(name)
                if not norm:
                    continue
                key = len(index._names)
                grams = trigrams(norm)
                index._names.append((movie_id, norm, len(grams)))
                index._exact.setdefault(norm, []).append(key)
                for gram in grams:
                    postings.setdefault(gram, []).append(key)
        index._postings = {g: np.asarray(keys, dtype=np.int32) for g, keys in postings.items()}
        index._sizes = np.asarray([size for _, _, size in index._names], dtype=np.int32)
        index._movie_ids = np.asarray([m for m, _, _ in index._names], dtype=np.int64)
        index._pops = np.asarray(
            [index.popularity.get(m, 0) for m, _, _ in index._names], dtype=np.int64
        )
        return index

    def __len__(self) -> int:
        return len(self._names)

    def search(self, query: str, limit: int = 1) -> list[TitleMatch]:
        norm = normalize_title(query)
        if not norm or not self._names:
            return []
        exact = self._exact.get(norm)
        if exact and limit == 1:
            key = min(
                exact,
                key=lambda k: (-int(self._pops[k]), int(self._movie_ids[k])),
            )
            return [TitleMatch(int(self._movie_ids[key]), self._names[key][1], 1.0)]
        grams = trigrams(norm)
        lists = [self._postings[g] for g in grams if g in self._postings]
        if not lists:
            return []
        overlap = np.bincount(np.concatenate(lists), minlength=len(self._names))
        keys = np.flatnonzero(overlap)
        shared = overlap[keys]
        scores = shared / (len(grams) + self._sizes[keys] - shared)
        movie_ids = self._movie_ids[keys]
        # Primary key last: score descending, then popularity descending, then movie id.
        order = np.lexsort((movie_ids, -self._pops[keys], -scores))
        out: list[TitleMatch] = []
        seen: set[int] = set()
        for pos in order:
            movie_id = int(movie_ids[pos])
            if movie_id in seen:
                continue
            seen.add(movie_id)
            out.append(TitleMatch(movie_id, self._names[int(keys[pos])][1], float(scores[pos])))
            if len(out) == limit:
                break
        return out

    def best(self, query: str) -> TitleMatch | None:
        found = self.search(query, 1)
        return found[0] if found else None

    def exact_ids(self, query: str) -> set[int]:
        """Every movie with a title variant that normalises to the same text as ``query``."""
        return {int(self._movie_ids[k]) for k in self._exact.get(normalize_title(query), ())}


def segment_titles(text: str, score: Callable[[str], float], max_join: int = 3) -> list[str]:
    """Split 'A, B, C' into titles when some titles contain commas ('Monsters, Inc').

    Every way of grouping adjacent comma-separated fragments is scored by the mean
    resolver score of its groups; the best grouping wins, preferring more groups on ties.
    Plans have four or five fragments, so exhaustive search is cheap.
    """
    fragments = [f.strip() for f in text.split(",") if f.strip()]
    if len(fragments) <= 1:
        return fragments
    cache: dict[tuple[int, int], float] = {}

    def group_score(i: int, j: int) -> float:
        if (i, j) not in cache:
            cache[(i, j)] = float(score(", ".join(fragments[i:j])))
        return cache[(i, j)]

    best_key = (-1.0, 0)
    best_groups: list[tuple[int, int]] = [(i, i + 1) for i in range(len(fragments))]

    def walk(start: int, groups: list[tuple[int, int]]) -> None:
        nonlocal best_key, best_groups
        if start == len(fragments):
            mean = sum(group_score(i, j) for i, j in groups) / len(groups)
            key = (round(mean, 9), len(groups))
            if key > best_key:
                best_key, best_groups = key, list(groups)
            return
        for end in range(start + 1, min(start + max_join, len(fragments)) + 1):
            walk(end, [*groups, (start, end)])

    walk(0, [])
    return [", ".join(fragments[i:j]) for i, j in best_groups]
