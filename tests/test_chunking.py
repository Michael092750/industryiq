import pytest

from industryiq.core.chunking import chunk_text, split_sections


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


def test_split_sections_no_heading_yields_single_block() -> None:
    assert split_sections("plain text, no heading") == [(None, "plain text, no heading")]


def test_split_sections_empty_text_yields_nothing() -> None:
    assert split_sections("") == []


def test_split_sections_tags_each_heading_block() -> None:
    text = "# Title\nintro\n## Outlook\nbody\n### Risks\nmore"
    assert split_sections(text) == [
        ("Title", "# Title\nintro"),
        ("Outlook", "## Outlook\nbody"),
        ("Risks", "### Risks\nmore"),
    ]


def test_split_sections_keeps_heading_line_inside_its_block() -> None:
    # The heading words stay in the block so they still embed/index as content.
    assert split_sections("## Scope 3 Emissions\ndetail") == [
        ("Scope 3 Emissions", "## Scope 3 Emissions\ndetail")
    ]


def test_split_sections_leading_content_uses_initial_section() -> None:
    # Body before the first heading inherits the carried-in section.
    assert split_sections("carried body\n## New\nbody", initial_section="Prev") == [
        ("Prev", "carried body"),
        ("New", "## New\nbody"),
    ]


def test_split_sections_ignores_non_atx_hashes() -> None:
    # No space after '#', and a bare '###', are not headings.
    assert split_sections("#nospace\n###\ntext") == [(None, "#nospace\n###\ntext")]


def test_split_sections_coalesces_consecutive_headings() -> None:
    # Stacked headings with no body between them must not each become a bodyless
    # block; they ride into the next section, tagged by the nearest (last) heading.
    text = "## Title\n## 2.1 Overview\nreal body here"
    assert split_sections(text) == [("2.1 Overview", "## Title\n## 2.1 Overview\nreal body here")]


def test_split_sections_heading_only_text_stays_single_block() -> None:
    # A block that is only headings (e.g. a table-of-contents page) yields one
    # block, not one bodyless block per heading line.
    assert split_sections("## A\n## B\n## C") == [("C", "## A\n## B\n## C")]
