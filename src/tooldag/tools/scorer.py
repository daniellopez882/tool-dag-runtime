"""Weighted linear scorer over per-candidate signals, and its conditional-logit fit.

Each candidate gets a feature vector (co-occurrence, plot similarity, genre overlap).
Features are standardised with statistics from the fitting split; a missing value
(no plot to embed) becomes 0 after standardisation, i.e. the fitting-set mean.
Weights maximise the multinomial likelihood of the correct option among a question's
options (a conditional logit), fitted with Newton's method and an L2 penalty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

SIGNALS = ("cooccurrence", "plot_similarity", "genre_jaccard")


@dataclass(frozen=True, slots=True)
class LinearScorer:
    signals: tuple[str, ...]
    weights: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    origin: str = "equal weights"

    @classmethod
    def equal(cls, signals: Sequence[str] = SIGNALS) -> LinearScorer:
        n = len(signals)
        return cls(tuple(signals), (1.0,) * n, (0.0,) * n, (1.0,) * n)

    def restrict(self, signals: Sequence[str]) -> LinearScorer:
        """The same scorer using only ``signals`` (weights of the others are dropped)."""
        unknown = [s for s in signals if s not in self.signals]
        if unknown:
            raise ValueError(f"scorer has no weights for {unknown}")
        keep = [self.signals.index(s) for s in signals]
        return LinearScorer(
            tuple(signals),
            tuple(self.weights[i] for i in keep),
            tuple(self.mean[i] for i in keep),
            tuple(self.scale[i] for i in keep),
            self.origin,
        )

    def standardize(self, features: np.ndarray) -> np.ndarray:
        z = (features - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.where(np.isnan(z), 0.0, z)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """``features``: (n_candidates, n_signals) with NaN for missing values."""
        return self.standardize(features) @ np.asarray(self.weights)

    def to_json(self) -> dict[str, Any]:
        return {
            "signals": list(self.signals),
            "weights": list(self.weights),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "origin": self.origin,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LinearScorer:
        return cls(
            tuple(data["signals"]),
            tuple(float(x) for x in data["weights"]),
            tuple(float(x) for x in data["mean"]),
            tuple(float(x) for x in data["scale"]),
            str(data.get("origin", "loaded")),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load_or_equal(cls, path: Path) -> LinearScorer:
        if path.is_file():
            return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
        return cls.equal()


def features_matrix(
    signals: Sequence[str], values: Mapping[str, Sequence[float | None]], n: int
) -> np.ndarray:
    matrix = np.full((n, len(signals)), np.nan)
    for col, name in enumerate(signals):
        column = values[name]
        if len(column) != n:
            raise ValueError(f"{name} has {len(column)} values for {n} candidates")
        for row, value in enumerate(column):
            if value is not None:
                matrix[row, col] = float(value)
    return matrix


def fit_conditional_logit(
    groups: Sequence[np.ndarray],
    answers: Sequence[int],
    signals: Sequence[str],
    *,
    l2: float = 1.0,
    iterations: int = 50,
    tol: float = 1e-10,
) -> LinearScorer:
    """Fit weights so the correct option of each group gets the highest softmax mass.

    ``groups[q]`` is (n_options, n_signals) with NaN for missing; ``answers[q]`` indexes
    the correct row.
    """
    if not groups:
        raise ValueError("no training groups")
    stacked = np.vstack(groups)
    mean = np.nanmean(stacked, axis=0)
    scale = np.nanstd(stacked, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    scale = np.where((scale > 0) & ~np.isnan(scale), scale, 1.0)
    base = LinearScorer(tuple(signals), (0.0,) * len(signals), tuple(mean), tuple(scale))
    zs = [base.standardize(g) for g in groups]
    d = len(signals)
    w = np.zeros(d)
    for _ in range(iterations):
        grad = -l2 * w
        hess = -l2 * np.eye(d)
        for z, y in zip(zs, answers, strict=True):
            logits = z @ w
            p = np.exp(logits - logits.max())
            p /= p.sum()
            expected = p @ z
            grad += z[y] - expected
            hess -= (z * p[:, None]).T @ z - np.outer(expected, expected)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if float(np.abs(step).max()) < tol:
            break
    return LinearScorer(
        tuple(signals),
        tuple(float(x) for x in w),
        tuple(float(x) for x in mean),
        tuple(float(x) for x in scale),
        origin=f"conditional logit, {len(groups)} questions, l2={l2}",
    )
