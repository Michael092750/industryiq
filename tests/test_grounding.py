"""Unit tests for the grounding gate -- pure, offline.

Covers the pure citation helpers and both gate adapters (deterministic + LLM, the
latter over a ``FakeLLM``), so the whole faithfulness check runs with no network.
"""

from industryiq.core.generation import FakeLLM
from industryiq.core.grounding import (
    DEFAULT_ABSTENTION,
    DeterministicGroundingGate,
    GroundingGate,
    LlmGroundingGate,
    citation_caveat,
    cited_indices,
    renumber_citations,
    verify_citations,
)
from industryiq.core.vectorstore import Hit


def _hits(n: int = 2) -> list[Hit]:
    return [Hit(id=str(i), score=0.9, metadata={"text": f"chunk {i}"}) for i in range(1, n + 1)]


# --- pure helpers -----------------------------------------------------------------


def test_cited_indices_are_distinct_and_ordered() -> None:
    assert cited_indices("first [2], then [1], again [2].") == [2, 1]


def test_cited_indices_empty_when_no_markers() -> None:
    assert cited_indices("no citations here") == []


def test_verify_citations_flags_out_of_range() -> None:
    # Two hits are numbered [1]..[2]; [3] and [0] are fabricated.
    assert verify_citations("see [1], [3], [0]", _hits(2)) == (3, 0)


def test_verify_citations_clean_when_all_in_range() -> None:
    assert verify_citations("see [1] and [2]", _hits(2)) == ()


def test_verify_citations_all_invalid_when_no_context() -> None:
    assert verify_citations("see [1]", []) == (1,)


def test_citation_caveat_names_every_unmatched_marker() -> None:
    caveat = citation_caveat((3, 7))
    assert "[3], [7]" in caveat
    assert caveat.startswith("\n\n")  # a suffix, so it cannot corrupt the answer above it


# --- renumbering into a shared namespace ------------------------------------------


def test_renumber_citations_maps_local_markers_to_global_ones() -> None:
    # This text numbered its own context [1]..[2]; globally those are [4] and [2].
    text = "first [1], then [2]."
    assert renumber_citations(text, ["mck", "bcg"], {"mck": 4, "bcg": 2}) == "first [4], then [2]."


def test_renumber_citations_drops_a_marker_past_the_local_context() -> None:
    assert renumber_citations("real [1] invented [9].", ["mck"], {"mck": 1}) == "real [1] invented."


def test_renumber_citations_drops_a_marker_whose_label_is_not_in_the_index() -> None:
    # A label that never made it into the merged list must not point at position 1.
    renumbered = renumber_citations("gone [1] kept [2]", ["dropped", "bcg"], {"bcg": 2})
    assert renumbered == "gone kept [2]"


def test_renumber_citations_is_a_no_op_without_markers() -> None:
    assert renumber_citations("no citations here", ["mck"], {"mck": 1}) == "no citations here"


# --- deterministic gate -----------------------------------------------------------


def test_deterministic_gate_satisfies_the_port() -> None:
    assert isinstance(DeterministicGroundingGate(), GroundingGate)


def test_deterministic_gate_abstains_on_empty_context() -> None:
    verdict = DeterministicGroundingGate().check("q", "any answer", [])
    assert verdict.grounded is False
    assert verdict.abstention == DEFAULT_ABSTENTION


def test_deterministic_gate_flags_fabricated_citation_without_abstaining() -> None:
    verdict = DeterministicGroundingGate().check("q", "claim [5]", _hits(2))
    assert verdict.grounded is False
    assert verdict.invalid_citations == (5,)
    assert verdict.abstention is None  # a caveat, not a replacement


def test_deterministic_gate_passes_a_clean_grounded_answer() -> None:
    verdict = DeterministicGroundingGate().check("q", "it is [1] and [2]", _hits(2))
    assert verdict.grounded is True
    assert verdict.invalid_citations == ()


def test_deterministic_gate_uses_a_custom_abstention() -> None:
    verdict = DeterministicGroundingGate(abstention="nope").check("q", "a", [])
    assert verdict.abstention == "nope"


# --- llm gate ---------------------------------------------------------------------


def test_llm_gate_short_circuits_on_empty_context_without_calling_the_model() -> None:
    llm = FakeLLM(response="GROUNDED")
    verdict = LlmGroundingGate(llm).check("q", "answer", [])
    assert verdict.grounded is False and verdict.abstention == DEFAULT_ABSTENTION
    assert llm.last_prompt is None  # the model was never consulted


def test_llm_gate_abstains_when_the_model_says_unsupported() -> None:
    verdict = LlmGroundingGate(FakeLLM(response="UNSUPPORTED")).check("q", "a [1]", _hits(2))
    assert verdict.grounded is False
    assert verdict.abstention == DEFAULT_ABSTENTION


def test_llm_gate_passes_a_grounded_answer() -> None:
    verdict = LlmGroundingGate(FakeLLM(response="GROUNDED")).check("q", "a [1]", _hits(2))
    assert verdict.grounded is True


def test_llm_gate_still_flags_bad_citations_when_model_says_grounded() -> None:
    verdict = LlmGroundingGate(FakeLLM(response="GROUNDED")).check("q", "a [9]", _hits(2))
    assert verdict.grounded is False
    assert verdict.invalid_citations == (9,)
    assert verdict.abstention is None
