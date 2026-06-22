import pytest

from industryiq.core.chunking import chunk_text


def test_short_text_returns_single_chunk() -> None:
    chunks = chunk_text("one two three", chunk_size=10, overlap=2)
    assert chunks == ["one two three"]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("", chunk_size=10, overlap=2) == []
    assert chunk_text("   \n  ", chunk_size=10, overlap=2) == []


def test_long_text_splits_into_multiple_chunks() -> None:
    words = " ".join(str(i) for i in range(10))  # "0 1 2 ... 9"
    chunks = chunk_text(words, chunk_size=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_no_chunk_exceeds_chunk_size() -> None:
    words = " ".join(str(i) for i in range(100))
    for chunk in chunk_text(words, chunk_size=15, overlap=3):
        assert len(chunk.split()) <= 15


def test_consecutive_chunks_overlap() -> None:
    words = " ".join(str(i) for i in range(20))
    chunks = chunk_text(words, chunk_size=5, overlap=2)
    # last 2 words of chunk[0] must equal first 2 words of chunk[1]
    first_tail = chunks[0].split()[-2:]
    second_head = chunks[1].split()[:2]
    assert first_tail == second_head


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_params_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=chunk_size, overlap=overlap)
