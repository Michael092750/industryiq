from ragproject.core.chat.adapters.filtering import ThresholdFilter
from ragproject.core.vectorstore import Hit


def _hit(score: float) -> Hit:
    return Hit(id=f"id{score}", score=score, metadata={"text": "x"})


def test_keeps_hits_at_or_above_threshold() -> None:
    keep = ThresholdFilter(threshold=0.5).keep([_hit(0.9), _hit(0.5)])
    assert keep == [_hit(0.9), _hit(0.5)]


def test_drops_hits_below_threshold_per_hit() -> None:
    keep = ThresholdFilter(threshold=0.5).keep([_hit(0.9), _hit(0.3)])
    assert keep == [_hit(0.9)]  # the 0.3 hit is dropped individually


def test_default_threshold_keeps_nonnegative() -> None:
    assert ThresholdFilter().keep([_hit(0.0), _hit(0.8)]) == [_hit(0.0), _hit(0.8)]


def test_empty_in_empty_out() -> None:
    assert ThresholdFilter(0.5).keep([]) == []
