"""Fetching, catalogue building and question loading, on tiny synthetic files."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tarfile
import urllib.error
import zipfile
from typing import TYPE_CHECKING

import pytest

from tooldag import fetch as fetch_mod
from tooldag.bigbench import load_questions, popularity_matched, split_fit_eval
from tooldag.index import build_catalog, ensure_catalog
from tooldag.tools.catalog import Catalog

if TYPE_CHECKING:
    from pathlib import Path


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _asset(payload: bytes, **overrides: object) -> fetch_mod.Asset:
    asset = fetch_mod.Asset(
        name="tiny",
        relpath="raw/tiny.bin",
        url="https://example.invalid/tiny.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        license="test",
        purpose="test",
    )
    return dataclasses.replace(asset, **overrides)


@pytest.fixture
def one_asset(monkeypatch: pytest.MonkeyPatch) -> fetch_mod.Asset:
    asset = _asset(b"hello world")
    monkeypatch.setattr(fetch_mod, "ASSETS_BY_NAME", {asset.name: asset})
    return asset


def test_fetch_verifies_and_skips_when_present(tmp_path: Path, one_asset) -> None:
    calls: list[str] = []

    def opener(url: str) -> _Response:
        calls.append(url)
        return _Response(b"hello world")

    lines: list[str] = []
    fetch_mod.fetch(tmp_path, ["tiny"], opener=opener, log=lines.append)
    assert (tmp_path / "raw/tiny.bin").read_bytes() == b"hello world"
    fetch_mod.fetch(tmp_path, ["tiny"], opener=opener, log=lines.append)
    assert len(calls) == 1
    assert all("sha256 verified" in line for line in lines if line.startswith("ok"))


def test_checksum_mismatch_raises_and_leaves_nothing(tmp_path: Path, monkeypatch) -> None:
    asset = _asset(b"expected bytes")
    monkeypatch.setattr(fetch_mod, "ASSETS_BY_NAME", {asset.name: asset})
    with pytest.raises(fetch_mod.ChecksumError, match="expected 14 bytes"):
        fetch_mod.fetch(tmp_path, ["tiny"], opener=lambda url: _Response(b"tampered byte"))
    assert list((tmp_path / "raw").iterdir()) == []


def test_network_errors_are_retried_then_reported(tmp_path: Path, one_asset, monkeypatch) -> None:
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    attempts = []

    def opener(url: str) -> _Response:
        attempts.append(url)
        raise urllib.error.URLError("offline")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        fetch_mod.fetch(tmp_path, ["tiny"], opener=opener, attempts=2, log=lambda s: None)
    assert len(attempts) == 2


def test_pinned_assets_have_hashes_sizes_and_licences() -> None:
    for asset in fetch_mod.ASSETS:
        assert len(asset.sha256) == 64
        assert asset.size > 0
        assert asset.license
    assert {a.name for a in fetch_mod.ASSETS} >= {
        "bigbench-movie-recommendation",
        "movielens-latest-small",
        "cmu-movie-summaries",
        "minilm-onnx-quantized",
    }


def _write_sources(root: Path) -> None:
    movies = (
        "movieId,title,genres\n"
        '1,"Matrix, The (1999)",Action|Sci-Fi\n'
        "2,Heat (1995),Action|Crime|Thriller\n"
        '3,"Léon: The Professional (a.k.a. The Professional) (Léon) (1994)",Action|Crime\n'
        "4,Obscure Film (2001),(no genres listed)\n"
    )
    ratings = (
        "userId,movieId,rating,timestamp\n1,1,5.0,0\n1,2,4.0,0\n2,1,4.5,0\n2,3,3.0,0\n3,2,2.0,0\n"
    )
    zpath = root / "raw/movielens/ml-latest-small.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as archive:
        archive.writestr("ml-latest-small/movies.csv", movies)
        archive.writestr("ml-latest-small/ratings.csv", ratings)
    plots = (
        "10\tA hacker learns reality is simulated.\n"
        "20\tA thief plans one last job.\n"
        "30\tAn assassin protects a girl.\n"
    )
    meta = (
        "10\t/m/x\tThe Matrix\t1999-03-31\n"
        "20\t/m/y\tHeat\t1996\n"  # one year of drift between the sources
        "30\t/m/z\tLéon\t1994\n"
    )
    tpath = root / "raw/cmu/MovieSummaries.tar.gz"
    tpath.parent.mkdir(parents=True)
    with tarfile.open(tpath, "w:gz") as archive:
        for name, text in (("plot_summaries.txt", plots), ("movie.metadata.tsv", meta)):
            data = text.encode()
            info = tarfile.TarInfo(f"MovieSummaries/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_build_catalog_joins_movielens_and_cmu(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    catalog = build_catalog(tmp_path)
    matrix, heat, leon, obscure = (catalog.movies[i] for i in (1, 2, 3, 4))
    assert matrix.title == "The Matrix" and matrix.plot.startswith("A hacker")
    assert heat.wiki_id == 20  # matched with year +1
    assert leon.names == ("Léon: The Professional", "The Professional", "Léon")
    assert leon.plot.startswith("An assassin")
    assert obscure.genres == () and obscure.plot == ""
    assert catalog.likers[1] == frozenset({1, 2})
    assert catalog.likers[2] == frozenset({1})
    assert matrix.n_ratings == 2
    assert catalog.stats["plot_join"] == {"exact year": 2, "year +1": 1, "no plot": 1}


def test_ensure_catalog_caches_and_rebuilds_on_source_change(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    first = ensure_catalog(tmp_path)
    loaded = Catalog.load(tmp_path / "catalog")
    assert loaded.movies == first.movies
    assert loaded.likers == first.likers
    meta_path = tmp_path / "catalog/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sources"]["movielens-latest-small"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    rebuilt = ensure_catalog(tmp_path)
    assert json.loads(meta_path.read_text())["sources"] != meta["sources"]
    assert rebuilt.movies == first.movies


def _task_json(path: Path) -> None:
    examples = [
        {
            "input": "Orbit of Ash, Starfall Protocol:",
            "target_scores": {"Comet Wardens": 1, "Tidewater": 0, "The Wedding Season": 0},
        },
        {
            "input": "Paper Hearts in Lisbon, Second Chances at Maple Street",
            "target_scores": {"The Wedding Season": 1, "Glass City Heist": 0, "Tidewater": 0},
        },
    ]
    path.write_text(json.dumps({"examples": examples}))


def test_load_questions_and_split(tmp_path: Path) -> None:
    _task_json(tmp_path / "task.json")
    questions = load_questions(tmp_path / "task.json")
    assert questions[0].liked_text == "Orbit of Ash, Starfall Protocol"
    assert questions[0].text == "Find a movie similar to Orbit of Ash, Starfall Protocol"
    assert questions[1].answer_index == 0
    fit, held = split_fit_eval(questions, n_fit=1, seed=3)
    assert {q.qid for q in fit} | {q.qid for q in held} == {0, 1}
    assert split_fit_eval(questions, n_fit=1, seed=3) == (fit, held)


def test_load_questions_rejects_ambiguous_answers(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"examples": [{"input": "A", "target_scores": {"x": 1, "y": 1}}]}))
    with pytest.raises(ValueError, match="2 correct options"):
        load_questions(path)


def test_popularity_matched_keeps_answer_and_redraws_distractors(tmp_path: Path) -> None:
    _task_json(tmp_path / "task.json")
    questions = load_questions(tmp_path / "task.json")
    catalog = Catalog.fixture()
    variant = popularity_matched(questions, catalog, tolerance=0.3, min_ratings=1)
    assert [q.qid for q in variant] == [0, 1]
    for original, new in zip(questions, variant, strict=True):
        assert new.answer == original.answer
        assert len(new.options) == 4 and new.answer in new.options
        answer_n = catalog.movies[catalog.index.best(new.answer).movie_id].n_ratings
        for option in new.options:
            movie = catalog.movies[catalog.index.best(option).movie_id]
            assert movie.title not in original.liked_text
            assert 0.7 * answer_n <= movie.n_ratings <= 1.3 * answer_n
    assert popularity_matched(questions, catalog, tolerance=0.3, min_ratings=1) == variant
