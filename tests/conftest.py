"""Shared pytest fixtures and test doubles."""

from collections.abc import Callable

import pytest


@pytest.fixture(autouse=True)
def _pin_pdf_parser_to_pypdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unit suite fast and hermetic.

    The app default is now ``PDF_PARSER=docling`` (heavy, loads ML models). Pin
    every test to pypdf so PDF-loading tests don't invoke Docling when the extra
    happens to be installed locally. Tests that exercise Docling override this
    with ``monkeypatch.setenv("PDF_PARSER", "docling")``.
    """
    monkeypatch.setenv("PDF_PARSER", "pypdf")


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
