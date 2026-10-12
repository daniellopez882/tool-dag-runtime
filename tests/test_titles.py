from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tooldag.tools.titles import (
    TrigramIndex,
    front_article,
    normalize_title,
    parse_movielens_title,
    segment_titles,
    trigrams,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Matrix, The", "the matrix"),
        ("L.A. Confidential", "la confidential"),
        ("Star Wars: Episode IV - A New Hope", "star wars episode iv a new hope"),
        ("Amélie", "amelie"),
        ("Lock, Stock & Two Smoking Barrels", "lock stock and two smoking barrels"),
        ("8 1/2", "8 12"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_front_article_handles_elided_articles() -> None:
    assert front_article("Auberge espagnole, L'") == "L'Auberge espagnole"
    assert front_article("Heat") == "Heat"


def test_parse_movielens_title_peels_year_and_alternates() -> None:
    parsed = parse_movielens_title(
        "Léon: The Professional (a.k.a. The Professional) (Léon) (1994)"
    )
    assert parsed.year == 1994
    assert parsed.title == "Léon: The Professional"
    assert parsed.alternates == ("The Professional", "Léon")


def test_parse_movielens_title_without_year_or_with_range() -> None:
    assert parse_movielens_title("Babylon 5").year is None
    assert parse_movielens_title("Fawlty Towers (1975-1979)").year == 1975
    assert parse_movielens_title("(500) Days of Summer (2009)").title == "(500) Days of Summer"


def _index() -> TrigramIndex:
    entries = [
        (1, ["The Matrix"]),
        (2, ["The Matrix Reloaded"]),
        (3, ["Monsters, Inc."]),
        (4, ["Monsters"]),
        (5, ["Heat"]),
        (6, ["Heat"]),  # a remake with the same title and fewer ratings
    ]
    return TrigramIndex.build(entries, {1: 300, 2: 100, 3: 200, 4: 5, 5: 150, 6: 3})


def test_exact_match_prefers_the_more_rated_film() -> None:
    index = _index()
    hit = index.best("heat")
    assert hit is not None
    assert (hit.movie_id, hit.score) == (5, 1.0)
    assert index.exact_ids("Heat") == {5, 6}


def test_fuzzy_match_ranks_by_jaccard_and_dedupes_movies() -> None:
    index = _index()
    found = index.search("Matrix Reloaded", limit=3)
    assert found[0].movie_id == 2
    assert len({m.movie_id for m in found}) == len(found)
    assert all(found[i].score >= found[i + 1].score for i in range(len(found) - 1))
    assert index.search("", limit=3) == []
    assert index.search("qqqq zzzz", limit=1) == []


@given(st.text(min_size=1, max_size=30))
def test_fuzzy_scores_are_jaccard_values(query: str) -> None:
    index = _index()
    for match in index.search(query, limit=6):
        assert 0.0 < match.score <= 1.0
        expected = len(trigrams(normalize_title(query)) & trigrams(match.matched_name)) / len(
            trigrams(normalize_title(query)) | trigrams(match.matched_name)
        )
        assert match.score == pytest.approx(expected)


def test_segment_titles_keeps_commas_inside_titles() -> None:
    index = _index()

    def score(text: str) -> float:
        hit = index.best(text)
        return hit.score if hit else 0.0

    assert segment_titles("Monsters, Inc, The Matrix, Heat", score) == [
        "Monsters, Inc",
        "The Matrix",
        "Heat",
    ]
    assert segment_titles("Heat", score) == ["Heat"]
    assert segment_titles("", score) == []
