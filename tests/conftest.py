"""Shared pytest fixtures and test doubles."""

from collections.abc import Callable

import pytest


class _FakeClock:
    """Deterministic monotonic clock: advances a fixed step on every call."""

    def __init__(self, step: float = 0.001) -> None:
        self._n = 0
        self._step = step

    def __call__(self) -> float:
        self._n += 1
        return self._n * self._step


@pytest.fixture
def fake_clock() -> Callable[..., _FakeClock]:
    """Return a factory for deterministic clocks (each call advances ``step``)."""
    return _FakeClock
