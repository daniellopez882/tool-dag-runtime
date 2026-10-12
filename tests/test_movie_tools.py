from __future__ import annotations

import math

import pytest

from tooldag.tools.base import ToolError
from tooldag.tools.catalog import Catalog
from tooldag.tools.embed import HashingEmbedder, Vector
from tooldag.tools.movies import MovieTools


@pytest.fixture
def tools() -> MovieTools:
    return MovieTools(Catalog.fixture(), HashingEmbedder())


def test_resolve_title_exact_fuzzy_and_unknown(tools: MovieTools) -> None:
    assert tools.resolve_title("orbit of ash").id == 1
    assert tools.resolve_title("Starfal Protocol").id == 3
    with pytest.raises(ToolError, match="no catalogue title"):
        tools.resolve_title("Completely Different Thing")


def test_cooccurrence_is_mean_cosine_of_liker_sets(tools: MovieTools) -> None:
    a, b, c = (tools.resolve_title(t) for t in ("Orbit of Ash", "Comet Wardens", "Tidewater"))
    likers = tools.catalog.likers
    expected = len(likers[a.id] & likers[b.id]) / math.sqrt(len(likers[a.id]) * len(likers[b.id]))
    assert tools.cooccurrence([a], b) == pytest.approx(expected)
    assert tools.cooccurrence([a, c], b) == pytest.approx(
        (expected + tools.cooccurrence([c], b)) / 2
    )
    assert tools.cooccurrence([], b) == 0.0


def test_genre_jaccard(tools: MovieTools) -> None:
    orbit = tools.resolve_title("Orbit of Ash")  # Sci-Fi, Adventure
    wardens = tools.resolve_title("Comet Wardens")  # Sci-Fi, Adventure, Children
    assert tools.genre_jaccard([orbit], wardens) == pytest.approx(2 / 3)
    assert tools.genre_jaccard([orbit], orbit) == 1.0


def test_plot_similarity_skips_empty_vectors(tools: MovieTools) -> None:
    v = tools.embed("a colony ship drifts near a dying star")
    empty = tools.embed("   ")
    assert empty.empty
    assert tools.plot_similarity([v, empty], v) == pytest.approx(1.0, abs=1e-5)
    assert tools.plot_similarity([empty], v) is None
    assert str(empty) == "<empty vector>"
    assert Vector(None).cosine(v) is None


def test_rank_prefers_the_highest_weighted_score(tools: MovieTools) -> None:
    a, b = tools.resolve_title("Comet Wardens"), tools.resolve_title("Tidewater")
    ranking = tools.rank([a, b], cooccurrence=[0.4, 0.1], genre_jaccard=[0.5, None])
    assert ranking.best == "Comet Wardens"
    assert ranking.signals == ("cooccurrence", "genre_jaccard")
    assert ranking.preview()["best"] == "Comet Wardens"
    with pytest.raises(ToolError, match="at least one signal"):
        tools.rank([a, b])
    with pytest.raises(ToolError, match="values for"):
        tools.rank([a, b], cooccurrence=[0.4])


def test_registry_exposes_typed_signatures(tools: MovieTools) -> None:
    registry = tools.registry()
    assert set(registry.names()) == {
        "resolve_title",
        "plot",
        "embed",
        "plot_similarity",
        "cooccurrence",
        "genre_jaccard",
        "rank",
    }
    assert registry["embed"].blocking
    assert "resolve_title(title: str) -> movie" in registry["resolve_title"].signature()
    assert registry.check_call("rank", 0, []) == "rank() missing candidates"
    assert registry.check_call("nope", 1, []) == "unknown tool 'nope'"
