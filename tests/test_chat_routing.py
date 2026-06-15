from ragproject.core.chat.adapters.routing import AlwaysRetrieveRouter, LlmRouter
from ragproject.core.chat.models import Turn
from ragproject.core.generation import FakeLLM

KB = "industry analysis reports"


def test_always_router_always_retrieves() -> None:
    decision = AlwaysRetrieveRouter().route([], "anything")
    assert decision.should_retrieve is True


def test_llm_router_retrieves_on_yes() -> None:
    decision = LlmRouter(FakeLLM(response="yes"), KB).route([], "outlook for the EV market?")
    assert decision.should_retrieve is True


def test_llm_router_skips_on_no() -> None:
    decision = LlmRouter(FakeLLM(response="No."), KB).route([], "hello there")
    assert decision.should_retrieve is False


def test_llm_router_injects_kb_description_into_the_prompt() -> None:
    llm = FakeLLM(response="yes")
    LlmRouter(llm, KB).route([Turn("hi", "hello")], "and the pricing?")
    assert llm.last_prompt is not None
    assert KB in llm.last_prompt  # the model is told what the KB holds
    assert "and the pricing?" in llm.last_prompt
    assert "yes/no" in llm.last_prompt
