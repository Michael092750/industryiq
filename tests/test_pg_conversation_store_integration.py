"""Integration tests for PgConversationStore against a real Postgres.

Run with a database available (e.g. `docker compose up -d`) via:

    pytest -m integration

These are skipped from the default unit run. They assert PgConversationStore
honors the same ConversationStore contract as InMemoryConversationStore.
"""

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from industryiq.config import get_settings
from industryiq.core.chat.adapters.store_pg import PgConversationStore
from industryiq.core.chat.models import Turn
from industryiq.core.chat.ports import ConversationStore

pytestmark = pytest.mark.integration

DATABASE_URL = get_settings().database_url


@pytest.fixture
def store() -> Iterator[PgConversationStore]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    suffix = uuid.uuid4().hex[:8]
    conv_table = f"test_conversations_{suffix}"
    msg_table = f"test_messages_{suffix}"
    yield PgConversationStore(
        DATABASE_URL, conversations_table=conv_table, messages_table=msg_table
    )
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {msg_table}")
        conn.execute(f"DROP TABLE IF EXISTS {conv_table}")
        conn.commit()


def test_satisfies_interface(store: PgConversationStore) -> None:
    assert isinstance(store, ConversationStore)


def test_create_and_get(store: PgConversationStore) -> None:
    convo = store.create("My chat")
    fetched = store.get(convo.id)
    assert fetched is not None
    assert fetched.id == convo.id
    assert fetched.title == "My chat"


def test_get_unknown_returns_none(store: PgConversationStore) -> None:
    assert store.get("nope") is None


def test_history_returns_turns_oldest_first(store: PgConversationStore) -> None:
    convo = store.create("c")
    store.append(convo.id, Turn("q1", "a1"))
    store.append(convo.id, Turn("q2", "a2"))
    assert store.history(convo.id) == [Turn("q1", "a1"), Turn("q2", "a2")]


def test_history_limit_returns_most_recent_oldest_first(store: PgConversationStore) -> None:
    convo = store.create("c")
    for i in range(5):
        store.append(convo.id, Turn(f"q{i}", f"a{i}"))
    assert store.history(convo.id, limit=2) == [Turn("q3", "a3"), Turn("q4", "a4")]


def test_history_is_isolated_per_conversation(store: PgConversationStore) -> None:
    first = store.create("a")
    second = store.create("b")
    store.append(first.id, Turn("qa", "aa"))
    assert store.history(second.id) == []


def test_list_all_returns_created_conversations(store: PgConversationStore) -> None:
    a = store.create("a")
    b = store.create("b")
    assert {c.id for c in store.list_all()} == {a.id, b.id}
