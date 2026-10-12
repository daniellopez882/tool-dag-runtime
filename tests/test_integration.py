"""Checks against the fetched data and the MiniLM model (``pytest -m integration``)."""

from __future__ import annotations

import numpy as np
import pytest

from tooldag.bigbench import load_questions
from tooldag.config import data_dir
from tooldag.fetch import ASSETS, asset_path, is_present
from tooldag.service import full_data_present, load_domain
from tooldag.tools.embed import MiniLMEmbedder
from tooldag.tools.wordpiece import WordPieceTokenizer

pytestmark = pytest.mark.integration

ROOT = data_dir()
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
needs_data = pytest.mark.skipif(
    not full_data_present(ROOT) or not (MODEL_DIR / "tokenizer.json").is_file(),
    reason="run `tooldag fetch` first",
)


@needs_data
def test_fetched_files_match_pinned_checksums() -> None:
    assert all(is_present(ROOT, asset) for asset in ASSETS)


@needs_data
def test_wordpiece_matches_reference_tokenizer_on_plot_summaries() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    reference = tokenizers.Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    reference.no_truncation()
    reference.no_padding()
    ours = WordPieceTokenizer.from_vocab_file(MODEL_DIR / "vocab.txt")
    domain = load_domain("full")
    texts = [m.plot for m in domain.catalog.movies.values() if m.plot][:400]
    texts += [m.title for m in list(domain.catalog.movies.values())[:2000]]
    texts += ["Amélie", "Léon", "8½", "Crouching Tiger, 卧虎藏龙", "naïve café déjà vu", ""]
    mismatches = []
    for text in texts:
        want = reference.encode(text).ids
        got = ours.encode(text, max_length=10**6).ids
        if got != want:
            mismatches.append(text[:60])
    assert mismatches == []


@needs_data
def test_minilm_embeddings_are_unit_length_and_semantic() -> None:
    embedder = MiniLMEmbedder(MODEL_DIR)
    heist = embedder.embed("A crew of thieves plans a bank robbery.")
    robbery = embedder.embed("Criminals rob a bank vault in an elaborate heist.")
    romance = embedder.embed("Two strangers fall in love during a summer in Paris.")
    assert heist.values is not None
    assert float(np.linalg.norm(heist.values)) == pytest.approx(1.0, abs=1e-5)
    assert heist.cosine(robbery) > heist.cosine(romance) + 0.2  # type: ignore[operator]


@needs_data
def test_every_bigbench_option_resolves_exactly() -> None:
    domain = load_domain("full")
    questions = load_questions(asset_path(ROOT, "bigbench-movie-recommendation"))
    assert len(questions) == 500
    scores = [domain.catalog.index.best(o).score for q in questions for o in q.options]
    assert min(scores) == 1.0
