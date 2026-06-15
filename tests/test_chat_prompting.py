from ragproject.core.chat.models import Turn
from ragproject.core.chat.prompting import (
    build_chat_prompt,
    build_condense_prompt,
    format_history,
)
from ragproject.core.vectorstore import Hit


def _hit(text: str, score: float = 1.0, id_: str = "x") -> Hit:
    return Hit(id=id_, score=score, metadata={"text": text})


def test_format_history_alternates_user_and_assistant() -> None:
    history = [Turn("hi", "hello"), Turn("who are you", "a bot")]
    assert format_history(history) == (
        "User: hi\nAssistant: hello\nUser: who are you\nAssistant: a bot"
    )


def test_condense_prompt_includes_history_and_followup() -> None:
    history = [Turn("tell me about cats", "cats are great")]
    prompt = build_condense_prompt(history, "what about dogs")
    assert "cats are great" in prompt
    assert "what about dogs" in prompt
    assert prompt.endswith("Standalone question:")


def test_chat_prompt_grounds_in_hits_and_history() -> None:
    prompt = build_chat_prompt([Turn("hi", "hello")], "what is X", [_hit("X is a thing")])
    assert "[1] X is a thing" in prompt
    assert "Conversation so far:" in prompt
    assert "Question: what is X" in prompt


def test_chat_prompt_without_history_omits_history_block() -> None:
    prompt = build_chat_prompt([], "what is X", [_hit("X is a thing")])
    assert "Conversation so far:" not in prompt


def test_chat_prompt_without_hits_says_no_context() -> None:
    prompt = build_chat_prompt([], "what is X", [])
    assert "(no relevant context found)" in prompt
