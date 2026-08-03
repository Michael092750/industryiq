from industryiq.core.chat.adapters.routing import AlwaysRetrieveRouter, LlmRouter
from industryiq.core.chat.models import Turn
from industryiq.core.generation import FakeLLM

KB = "industry analysis reports"


def test_always_router_retrieves_without_planning() -> None:
    decision = AlwaysRetrieveRouter().route([], "anything")
    assert decision.should_retrieve is True
    assert decision.needs_planning is False


def test_llm_router_simple_retrieves_without_planning() -> None:
    decision = LlmRouter(FakeLLM(response="SIMPLE"), KB).route([], "outlook for the EV market?")
    assert decision.should_retrieve is True
    assert decision.needs_planning is False


def test_llm_router_complex_escalates_to_planning() -> None:
    decision = LlmRouter(FakeLLM(response="COMPLEX"), KB).route([], "compare AI and semis")
    assert decision.should_retrieve is True
    assert decision.needs_planning is True


def test_llm_router_none_skips_retrieval() -> None:
    decision = LlmRouter(FakeLLM(response="NONE"), KB).route([], "hello there")
    assert decision.should_retrieve is False
    assert decision.needs_planning is False


def test_llm_router_unrecognized_verdict_defaults_to_simple_retrieve() -> None:
    decision = LlmRouter(FakeLLM(response="???"), KB).route([], "something")
    assert decision.should_retrieve is True
    assert decision.needs_planning is False


def test_llm_router_injects_kb_description_into_the_prompt() -> None:
    llm = FakeLLM(response="SIMPLE")
    LlmRouter(llm, KB).route([Turn("hi", "hello")], "and the pricing?")
    assert llm.last_prompt is not None
    assert KB in llm.last_prompt  # the model is told what the KB holds
    assert "and the pricing?" in llm.last_prompt
    assert "COMPLEX" in llm.last_prompt  # the 3-way vocabulary is in the prompt
