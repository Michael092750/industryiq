from ragproject.core.chat.models import Turn
from ragproject.core.chat.store_memory import InMemoryConversationStore


def test_create_assigns_id_and_persists() -> None:
    store = InMemoryConversationStore()
    convo = store.create("My chat")
    assert convo.id
    assert convo.title == "My chat"
    assert store.get(convo.id) == convo


def test_get_unknown_returns_none() -> None:
    assert InMemoryConversationStore().get("nope") is None


def test_history_returns_turns_in_order() -> None:
    store = InMemoryConversationStore()
    convo = store.create("c")
    store.append(convo.id, Turn("q1", "a1"))
    store.append(convo.id, Turn("q2", "a2"))
    assert store.history(convo.id) == [Turn("q1", "a1"), Turn("q2", "a2")]


def test_history_limit_returns_most_recent_oldest_first() -> None:
    store = InMemoryConversationStore()
    convo = store.create("c")
    for i in range(5):
        store.append(convo.id, Turn(f"q{i}", f"a{i}"))
    assert store.history(convo.id, limit=2) == [Turn("q3", "a3"), Turn("q4", "a4")]


def test_history_of_unknown_conversation_is_empty() -> None:
    assert InMemoryConversationStore().history("nope") == []
