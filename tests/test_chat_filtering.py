from industryiq.core.retrieval.adapters.filtering import ThresholdFilter
from industryiq.core.vectorstore import Hit, ScoreKind


def _hit(score: float, kind: ScoreKind = ScoreKind.COSINE) -> Hit:
    return Hit(id=f"id{score}-{kind.value}", score=score, metadata={"text": "x"}, score_kind=kind)


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


def test_cosine_threshold_does_not_apply_to_bm25() -> None:
    # A BM25 score of 0.3 would fall below the cosine cutoff, but BM25 is not on
    # the cosine scale, so it must pass through unfiltered by default.
    bm25 = _hit(0.3, ScoreKind.BM25)
    assert ThresholdFilter(threshold=0.5).keep([bm25]) == [bm25]


def test_normalized_scores_pass_through_by_default() -> None:
    # Query-relative [0, 1] blend: a cosine cutoff must not touch it.
    normalized = _hit(0.1, ScoreKind.NORMALIZED)
    assert ThresholdFilter(threshold=0.5).keep([normalized]) == [normalized]


def test_mixed_list_filters_only_the_matching_kind() -> None:
    cosine_low = _hit(0.3, ScoreKind.COSINE)  # dropped by the 0.5 cosine cutoff
    cosine_high = _hit(0.9, ScoreKind.COSINE)  # kept
    bm25_low = _hit(0.3, ScoreKind.BM25)  # kept: no BM25 cutoff configured
    keep = ThresholdFilter(threshold=0.5).keep([cosine_low, cosine_high, bm25_low])
    assert keep == [cosine_high, bm25_low]


def test_per_kind_threshold_filters_that_kind() -> None:
    # Once calibrated, a BM25 cutoff can be supplied and is applied to BM25 hits.
    filt = ThresholdFilter(threshold=0.5, thresholds={ScoreKind.BM25: 2.0})
    kept = _hit(3.0, ScoreKind.BM25)
    dropped = _hit(1.0, ScoreKind.BM25)
    assert filt.keep([kept, dropped]) == [kept]


def test_from_settings_unset_kinds_keep_everything() -> None:
    # Matches the config default (bm25/normalized = None): only cosine is gated.
    filt = ThresholdFilter.from_settings(0.5)
    bm25 = _hit(0.1, ScoreKind.BM25)
    normalized = _hit(0.1, ScoreKind.NORMALIZED)
    cosine_low = _hit(0.3, ScoreKind.COSINE)
    assert filt.keep([bm25, normalized, cosine_low]) == [bm25, normalized]


def test_from_settings_applies_supplied_per_kind_cutoffs() -> None:
    filt = ThresholdFilter.from_settings(0.5, bm25=2.0, normalized=0.2)
    assert filt.keep([_hit(3.0, ScoreKind.BM25)]) == [_hit(3.0, ScoreKind.BM25)]
    assert filt.keep([_hit(1.0, ScoreKind.BM25)]) == []
    assert filt.keep([_hit(0.1, ScoreKind.NORMALIZED)]) == []
