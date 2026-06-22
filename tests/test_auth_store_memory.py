from industryiq.core.auth.adapters.store_memory import InMemoryUserStore


def test_create_assigns_id_and_persists() -> None:
    store = InMemoryUserStore()
    user = store.create("a@example.com", "hash")
    assert user.id
    assert user.email == "a@example.com"
    assert store.get_by_id(user.id) == user


def test_get_by_email_returns_user() -> None:
    store = InMemoryUserStore()
    user = store.create("a@example.com", "hash")
    assert store.get_by_email("a@example.com") == user


def test_get_unknown_returns_none() -> None:
    store = InMemoryUserStore()
    assert store.get_by_email("nobody@example.com") is None
    assert store.get_by_id("nope") is None
