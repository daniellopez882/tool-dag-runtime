"""BIG-bench movie_recommendation questions and a popularity-matched variant.

In the original task the correct option is a widely rated film and the three
distractors come from the long tail, so "pick the most-rated option" already scores
near 100%. The variant keeps each question's liked films and correct answer but redraws
the distractors from catalogue films with a similar rating count, which makes the
signals have to do the work. The variant is constructed here; it is not part of
BIG-bench.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tooldag.tools.titles import segment_titles

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tooldag.tools.catalog import Catalog

QUESTION_PREFIX = "Find a movie similar to "


@dataclass(frozen=True, slots=True)
class Question:
    qid: int
    liked_text: str
    options: tuple[str, ...]
    answer: str

    @property
    def text(self) -> str:
        return QUESTION_PREFIX + self.liked_text

    @property
    def answer_index(self) -> int:
        return self.options.index(self.answer)


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for qid, example in enumerate(data["examples"]):
        options = tuple(example["target_scores"])
        answers = [o for o, s in example["target_scores"].items() if s == 1]
        if len(answers) != 1:
            raise ValueError(f"question {qid} has {len(answers)} correct options")
        liked = example["input"].strip().rstrip(":").strip()
        questions.append(Question(qid, liked, options, answers[0]))
    return questions


def split_fit_eval(
    questions: Sequence[Question], n_fit: int = 100, seed: int = 2026
) -> tuple[list[Question], list[Question]]:
    """Deterministic split: ``n_fit`` questions to fit weights, the rest held out."""
    chosen = set(random.Random(seed).sample(range(len(questions)), n_fit))
    fit = [q for i, q in enumerate(questions) if i in chosen]
    held_out = [q for i, q in enumerate(questions) if i not in chosen]
    return fit, held_out


def popularity_matched(
    questions: Sequence[Question],
    catalog: Catalog,
    *,
    seed: int = 7,
    tolerance: float = 0.2,
    min_ratings: int = 5,
) -> list[Question]:
    """Redraw distractors among films rated about as often as the answer (within ``tolerance``)."""
    index = catalog.index

    def resolve(title: str) -> int | None:
        found = index.best(title)
        return found.movie_id if found is not None and found.score >= 0.5 else None

    def score(text: str) -> float:
        found = index.best(text)
        return found.score if found else 0.0

    # A distractor whose title also names another film (remakes, alternate titles) would
    # resolve to the more-rated one, so only films with a title of their own are drawn.
    by_count = sorted(
        (m for m in catalog.movies.values() if index.exact_ids(m.title) == {m.id}),
        key=lambda m: (m.n_ratings, m.id),
    )
    rng = random.Random(seed)
    out: list[Question] = []
    for q in questions:
        answer_id = resolve(q.answer)
        if answer_id is None:
            continue
        liked_ids = {resolve(t) for t in segment_titles(q.liked_text, score)}
        target = catalog.movies[answer_id].n_ratings
        width = tolerance
        pool: list[int] = []
        while len(pool) < 3 and width < 4.0:
            lo, hi = target * (1 - width), target * (1 + width)
            pool = [
                m.id
                for m in by_count
                if lo <= m.n_ratings <= hi
                and m.n_ratings >= min_ratings
                and m.id != answer_id
                and m.id not in liked_ids
            ]
            width *= 1.5
        if len(pool) < 3:
            continue
        distractors = [catalog.movies[i].title for i in rng.sample(pool, 3)]
        options = [q.answer, *distractors]
        rng.shuffle(options)
        out.append(Question(q.qid, q.liked_text, tuple(options), q.answer))
    return out
