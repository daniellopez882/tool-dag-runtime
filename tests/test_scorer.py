from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tooldag.tools.scorer import SIGNALS, LinearScorer, features_matrix, fit_conditional_logit


def test_missing_values_score_at_the_mean() -> None:
    scorer = LinearScorer(("a", "b"), (1.0, 2.0), (0.5, 1.0), (0.5, 2.0))
    features = np.array([[1.0, np.nan], [0.5, 3.0]])
    assert scorer.scores(features).tolist() == pytest.approx([1.0, 2.0])


def test_restrict_keeps_matching_weights() -> None:
    scorer = LinearScorer(SIGNALS, (1.0, 2.0, 3.0), (0.0, 0.1, 0.2), (1.0, 1.0, 2.0))
    sub = scorer.restrict(("genre_jaccard", "cooccurrence"))
    assert sub.weights == (3.0, 1.0)
    assert sub.mean == (0.2, 0.0)
    with pytest.raises(ValueError, match="no weights"):
        scorer.restrict(("popularity",))


def test_features_matrix_checks_lengths() -> None:
    m = features_matrix(("a",), {"a": [1.0, None]}, 2)
    assert m[0, 0] == 1.0
    assert np.isnan(m[1, 0])
    with pytest.raises(ValueError, match="2 values for 3"):
        features_matrix(("a",), {"a": [1.0, None]}, 3)


def test_json_round_trip(tmp_path) -> None:
    scorer = LinearScorer(SIGNALS, (1.0, -2.0, 0.5), (0.1, 0.2, 0.3), (1.0, 2.0, 3.0), "x")
    path = tmp_path / "w.json"
    scorer.save(path)
    assert LinearScorer.load_or_equal(path) == scorer
    assert LinearScorer.load_or_equal(tmp_path / "missing.json") == LinearScorer.equal()


def _synthetic(true_w: np.ndarray, n: int, seed: int) -> tuple[list[np.ndarray], list[int]]:
    rng = np.random.default_rng(seed)
    groups, answers = [], []
    for _ in range(n):
        x = rng.normal(size=(4, len(true_w)))
        logits = x @ true_w
        p = np.exp(logits - logits.max())
        answers.append(int(rng.choice(4, p=p / p.sum())))
        groups.append(x)
    return groups, answers


def test_conditional_logit_recovers_the_direction_of_true_weights() -> None:
    true_w = np.array([2.0, 0.0, -1.0])
    groups, answers = _synthetic(true_w, 3000, seed=1)
    fitted = fit_conditional_logit(groups, answers, SIGNALS, l2=0.1)
    w = np.asarray(fitted.weights) * 1.0 / np.asarray(fitted.scale)  # undo standardisation
    cosine = float(w @ true_w / (np.linalg.norm(w) * np.linalg.norm(true_w)))
    assert cosine > 0.98


@given(st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_fit_is_finite_and_beats_chance_on_its_own_data(seed: int) -> None:
    groups, answers = _synthetic(np.array([1.5, 1.0, 0.0]), 200, seed)
    # Knock out some values the way missing plots do.
    groups[0][1, 1] = np.nan
    fitted = fit_conditional_logit(groups, answers, SIGNALS)
    assert all(np.isfinite(fitted.weights))
    hits = [int(np.argmax(fitted.scores(g)) == a) for g, a in zip(groups, answers, strict=True)]
    assert np.mean(hits) > 0.3


def test_fit_needs_groups() -> None:
    with pytest.raises(ValueError, match="no training groups"):
        fit_conditional_logit([], [], SIGNALS)
