"""Pure metric functions for the RAG retrieval benchmark -- no I/O, unit-testable.

Retrieval metrics operate on an *ordered* list of retrieved chunk ids plus the
set of gold (relevant) ids. Keeping them side-effect free mirrors the project's
style (see ``core/chunking.py``) and lets the runner stay a thin orchestration
layer on top.
"""

import math
import statistics
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold chunks that appear in the top-``k`` retrieved ids.

    Returns 0.0 when there are no gold chunks (an empty gold set is not scored
    here; the runner excludes such queries from recall averages).
    """
    if not gold:
        return 0.0
    found = sum(1 for cid in retrieved[:k] if cid in gold)
    return found / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are gold. 0.0 if ``k`` <= 0."""
    if k <= 0:
        return 0.0
    found = sum(1 for cid in retrieved[:k] if cid in gold)
    return found / k


def hit_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> bool:
    """True if *any* gold chunk is in the top-``k`` (a.k.a. success@k)."""
    return any(cid in gold for cid in retrieved[:k])


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """1 / (rank of the first gold hit), or 0.0 if no gold id was retrieved."""
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean / median / p95 / min / max of ``values`` (zeros for an empty input)."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    # Nearest-rank p95: index ceil(0.95*n)-1, clamped to the last element.
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }
