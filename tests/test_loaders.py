from pathlib import Path

import pytest

from ragproject.core import loaders
from ragproject.core.loaders import load, load_docx, load_pdf, load_text

FIXTURES = Path(__file__).parent / "fixtures"


# --- load_text -------------------------------------------------------------


def test_load_text_reads_content() -> None:
    text = load_text(FIXTURES / "sample.txt")
    assert "Hello, RAG world." in text
    assert "testing loaders" in text


def test_load_text_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_text(FIXTURES / "does_not_exist.txt")


def test_load_text_wrong_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "data.csv"
    bad.write_text("a,b,c")
    with pytest.raises(ValueError):
        load_text(bad)


# --- load_pdf --------------------------------------------------------------


def test_load_pdf_reads_all_pages() -> None:
    text = load_pdf(FIXTURES / "sample.pdf")
    assert "Hello from a PDF document." in text
    assert "second page" in text


def test_load_pdf_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(FIXTURES / "missing.pdf")


def test_load_pdf_wrong_extension_raises() -> None:
    with pytest.raises(ValueError):
        load_pdf(FIXTURES / "sample.txt")


# --- load_docx -------------------------------------------------------------


def test_load_docx_reads_all_paragraphs() -> None:
    text = load_docx(FIXTURES / "sample.docx")
    assert "Hello from a Word document." in text
    assert "second paragraph" in text


def test_load_docx_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_docx(FIXTURES / "missing.docx")


def test_load_docx_wrong_extension_raises() -> None:
    with pytest.raises(ValueError):
        load_docx(FIXTURES / "sample.txt")


# --- load (dispatcher) -----------------------------------------------------


def test_load_dispatches_txt() -> None:
    assert load(FIXTURES / "sample.txt") == load_text(FIXTURES / "sample.txt")


def test_load_dispatches_pdf() -> None:
    assert "Hello from a PDF document." in load(FIXTURES / "sample.pdf")


def test_load_dispatches_docx() -> None:
    assert "Hello from a Word document." in load(FIXTURES / "sample.docx")


def test_load_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "data.csv"
    bad.write_text("a,b,c")
    with pytest.raises(ValueError):
        load(bad)


def test_load_strips_non_utf8_surrogates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A loader (e.g. a broken PDF font map) can emit lone surrogates; load() must
    # drop them so the result is always UTF-8 encodable for embedding/storage.
    bad_text = "good" + chr(0xD835) + chr(0xDF0F) + "text"  # lone surrogates
    monkeypatch.setitem(loaders._LOADERS, ".surro", lambda _p: bad_text)
    result = load(tmp_path / "weird.surro")
    assert result == "goodtext"
    result.encode("utf-8")  # must not raise
