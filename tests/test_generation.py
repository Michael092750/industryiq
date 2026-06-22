from industryiq.core.generation import LLM, FakeLLM, build_prompt, generate_answer
from industryiq.core.vectorstore import Hit


def _hits() -> list[Hit]:
    return [
        Hit(id="1", score=0.9, metadata={"text": "The sky is blue."}),
        Hit(id="2", score=0.8, metadata={"text": "Grass is green."}),
    ]


def test_fake_llm_satisfies_interface() -> None:
    assert isinstance(FakeLLM(), LLM)


def test_build_prompt_includes_question() -> None:
    prompt = build_prompt("What color is the sky?", _hits())
    assert "What color is the sky?" in prompt


def test_build_prompt_numbers_context() -> None:
    prompt = build_prompt("q", _hits())
    assert "[1] The sky is blue." in prompt
    assert "[2] Grass is green." in prompt


def test_build_prompt_with_no_hits_states_missing_context() -> None:
    prompt = build_prompt("q", [])
    assert "no relevant context" in prompt


def test_generate_answer_returns_llm_output() -> None:
    llm = FakeLLM(response="It is blue [1].")
    answer = generate_answer("What color is the sky?", _hits(), llm)
    assert answer == "It is blue [1]."


def test_generate_answer_sends_assembled_prompt_to_llm() -> None:
    llm = FakeLLM()
    generate_answer("What color is the sky?", _hits(), llm)
    assert llm.last_prompt is not None
    assert "What color is the sky?" in llm.last_prompt
    assert "[1] The sky is blue." in llm.last_prompt
