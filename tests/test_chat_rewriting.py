from industryiq.core.chat.adapters.rewriting import LlmQueryRewriter, NoOpQueryRewriter
from industryiq.core.chat.models import Turn
from industryiq.core.generation import FakeLLM


def test_llm_rewriter_returns_question_unchanged_on_first_turn() -> None:
    # No history -> no LLM call needed; the question already stands alone.
    rewriter = LlmQueryRewriter(FakeLLM(response="SHOULD NOT BE USED"))
    assert rewriter.condense([], "what is X") == "what is X"


def test_llm_rewriter_condenses_followups_with_the_llm() -> None:
    rewriter = LlmQueryRewriter(FakeLLM(response="What is the price of X?"))
    history = [Turn("tell me about X", "X is a product")]
    assert rewriter.condense(history, "what about its price") == "What is the price of X?"


def test_llm_rewriter_falls_back_to_question_when_llm_returns_blank() -> None:
    rewriter = LlmQueryRewriter(FakeLLM(response="   "))
    assert rewriter.condense([Turn("a", "b")], "follow up") == "follow up"


def test_noop_rewriter_passes_question_through() -> None:
    assert NoOpQueryRewriter().condense([Turn("a", "b")], "q") == "q"
