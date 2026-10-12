# ADR 0004: Report accuracy on a popularity-matched variant as well as on BIG-bench

Status: accepted

## Context

The benchmark task is BIG-bench `movie_recommendation`: given four films someone
liked, pick the similar film among four options. Every option title in the task
matches a MovieLens title exactly, and the correct option is usually a widely rated
film while the three distractors come from the long tail. A check during development
showed that choosing the option with the most MovieLens ratings, without looking at
the liked films at all, answers nearly every question. A scorer that learns "pick the
popular one" would look excellent on this task while saying little about whether the
signals (co-occurrence, plot similarity, genre overlap) measure similarity.

## Decision

- The accuracy benchmark reports a "most-rated option" baseline next to the signal
  scorers, so the popularity shortcut is visible in the same table.
- It also evaluates every scorer on a variant built in `bigbench.py`: each question
  keeps its liked films and its correct answer, and the three distractors are redrawn
  among catalogue films rated about as often as the answer (within 20%, widened by
  1.5x until three candidates exist; films with fewer than 5 ratings, liked films, the
  answer and films whose title is shared with another film are excluded). The draw is
  seeded.
- Both sets use the same question split: the same 100 question indices fit the
  weights and the same 400 are held out.
- The variant is labelled as constructed here, not as part of BIG-bench.

## Consequences

- Readers can see how much of the original accuracy the popularity baseline alone
  explains, and how the signals do when popularity is controlled.
- Co-occurrence is computed from MovieLens likers, and rating counts come from the
  same ratings, so even on the variant co-occurrence is not independent of
  popularity. The ablation without co-occurrence is reported for that reason.
- The variant depends on the catalogue (MovieLens ml-latest-small); it is rebuilt
  deterministically from the fetched files and never stored in the repository.

## Alternatives rejected

- **Report BIG-bench accuracy alone.** Honest but misleading: a near-ceiling number
  that a one-line baseline matches.
- **Drop the popular answers.** Would change which questions are asked and shrink the
  set; redrawing distractors keeps all questions and their answers.
- **Match on rating count only within the question's genres.** Makes the genre signal
  useless by construction.
