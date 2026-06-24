"""Tests for the Docling PDF parser path in :mod:`industryiq.core.loaders`.

Docling itself is a heavy optional dependency (PyTorch + models), so these tests
never import it: they stub the converter to prove the dispatch and per-page glue,
keeping the unit suite fast and dependency-free.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from industryiq.core import loaders

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeDoclingDoc:
    """Minimal stand-in for a ``DoclingDocument`` (no torch/models needed)."""

    def __init__(self, page_md: dict[int, str], whole: str | None = None) -> None:
        # Docling stores pages as a {page_no: PageItem} mapping; we only need len().
        self.pages = dict.fromkeys(page_md)
        self._page_md = page_md
        self._whole = whole if whole is not None else "\n".join(page_md.values())

    def export_to_markdown(self, page_no: int | None = None) -> str:
        if page_no is None:
            return self._whole
        return self._page_md[page_no]


class _FakeConverter:
    def __init__(self, doc: _FakeDoclingDoc) -> None:
        self._doc = doc

    def convert(self, source: str) -> SimpleNamespace:
        return SimpleNamespace(document=self._doc)


@pytest.fixture
def use_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_PARSER", "docling")


def test_config_default_is_docling() -> None:
    # The autouse conftest fixture pins the *env* to pypdf for speed; the app's
    # real default lives on the Settings dataclass, which is docling.
    from industryiq.config import Settings

    assert Settings().pdf_parser == "docling"


def test_explicit_pypdf_skips_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    # The autouse fixture already sets PDF_PARSER=pypdf; docling must not be built.
    def boom() -> object:
        raise AssertionError("docling must not be built when PDF_PARSER=pypdf")

    monkeypatch.setattr(loaders, "_get_docling_converter", boom)
    pages = loaders.load_pdf_pages(FIXTURES / "sample.pdf")
    assert any("Hello from a PDF document." in page for page in pages)


def test_docling_failure_falls_back_to_pypdf(
    use_docling: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_p: Path) -> list[str]:
        raise RuntimeError("conversion exploded")

    monkeypatch.setattr(loaders, "_load_pdf_pages_docling", boom)
    with caplog.at_level("WARNING"):
        pages = loaders.load_pdf_pages(FIXTURES / "sample.pdf")

    assert any("Hello from a PDF document." in page for page in pages)  # pypdf result
    assert "falling back to pypdf" in caplog.text


def test_docling_returns_one_markdown_string_per_page(
    use_docling: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _FakeDoclingDoc({1: "# Page one\n\n| a | b |", 2: "## Page two"})
    monkeypatch.setattr(loaders, "_get_docling_converter", lambda: _FakeConverter(doc))

    pages = loaders.load_pdf_pages(FIXTURES / "sample.pdf")

    assert pages == ["# Page one\n\n| a | b |", "## Page two"]


def test_docling_no_pages_falls_back_to_whole_document(
    use_docling: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _FakeDoclingDoc({}, whole="# Whole document")
    monkeypatch.setattr(loaders, "_get_docling_converter", lambda: _FakeConverter(doc))

    pages = loaders.load_pdf_pages(FIXTURES / "sample.pdf")

    assert pages == ["# Whole document"]


def test_docling_flows_through_load_pages_utf8_safe(
    use_docling: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # load_pages wraps each page in _to_utf8_safe; a stray surrogate must be dropped.
    doc = _FakeDoclingDoc({1: "good" + chr(0xD835) + "text"})
    monkeypatch.setattr(loaders, "_get_docling_converter", lambda: _FakeConverter(doc))

    pages = loaders.load_pages(FIXTURES / "sample.pdf")

    assert pages == ["goodtext"]
    pages[0].encode("utf-8")  # must not raise


def test_docling_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("docling") is not None:
        pytest.skip("docling is installed; the missing-dependency path can't be exercised")
    monkeypatch.setattr(loaders, "_docling_converter", None)
    with pytest.raises(ModuleNotFoundError, match=r"industryiq\[docling\]"):
        loaders._get_docling_converter()
