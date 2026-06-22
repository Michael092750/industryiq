from industryiq.core.chat.adapters.store_memory import InMemoryConversationStore
from industryiq.core.chat.models import Turn


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


def test_list_all_returns_all_conversations() -> None:
    store = InMemoryConversationStore()
    a = store.create("a")
    b = store.create("b")
    assert {c.id for c in store.list_all()} == {a.id, b.id}


def test_list_all_empty_store_is_empty() -> None:
    assert InMemoryConversationStore().list_all() == []
