import pytest

from industryiq.core.chunking import chunk_markdown, chunk_text, split_sections


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


# --- chunk_markdown: keep tables / charts / figures whole -------------------- #


def test_chunk_markdown_plain_prose_matches_chunk_text() -> None:
    # With no tables/fences it must behave exactly like chunk_text.
    words = " ".join(str(i) for i in range(20))
    assert chunk_markdown(words, chunk_size=5, overlap=2) == chunk_text(
        words, chunk_size=5, overlap=2
    )


def test_chunk_markdown_keeps_table_whole_even_when_oversized() -> None:
    # A table longer than chunk_size words must still land in ONE chunk, intact.
    table = "\n".join(["| a | b |", "|---|---|"] + [f"| {i} | {i * 2} |" for i in range(30)])
    chunks = chunk_markdown(table, chunk_size=5, overlap=1)
    assert chunks == [table]  # single, unsplit chunk
    assert len(chunks[0].split()) > 5  # exceeds chunk_size, deliberately not split


def test_chunk_markdown_table_not_split_from_surrounding_prose() -> None:
    text = "Intro sentence before.\n| a | b |\n|---|---|\n| 1 | 2 |\nTrailing prose after."
    chunks = chunk_markdown(text, chunk_size=100, overlap=10)
    # The whole table survives verbatim in exactly one chunk...
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert any(c == table for c in chunks)
    # ...and no chunk contains only part of it (a lone row).
    assert not any("| 1 | 2 |" in c and "| a | b |" not in c for c in chunks)


def test_chunk_markdown_keeps_fenced_block_whole() -> None:
    text = "before\n```\nrow1,10\nrow2,20\nrow3,30\n```\nafter"
    chunks = chunk_markdown(text, chunk_size=2, overlap=0)
    assert "```\nrow1,10\nrow2,20\nrow3,30\n```" in chunks


def test_chunk_markdown_single_pipe_line_is_not_a_table() -> None:
    # A lone '|' line (not a real table) is prose, chunked normally.
    chunks = chunk_markdown("a | b is prose here", chunk_size=100, overlap=0)
    assert chunks == ["a | b is prose here"]


# --- chunk_markdown: min-size coalescing (no orphan short chunks) ------------- #

_SMALL_TABLE = "| a | b |\n|---|---|\n| 1 | 2 |"


def test_chunk_markdown_min_chars_zero_leaves_small_pieces_separate() -> None:
    # Default (no coalescing): a small table and a short prose line stay separate chunks.
    text = f"{_SMALL_TABLE}\n\nShort caption prose."
    assert len(chunk_markdown(text, chunk_size=100, overlap=10)) == 2


def test_chunk_markdown_coalesces_small_pieces_up_to_floor() -> None:
    # With a floor, the small table + short prose merge into one non-orphan chunk.
    text = f"{_SMALL_TABLE}\n\nShort caption prose."
    merged = chunk_markdown(text, chunk_size=100, overlap=10, min_chars=40)
    assert len(merged) == 1
    assert "| a | b |" in merged[0] and "Short caption prose." in merged[0]


def test_chunk_markdown_coalesce_keeps_table_whole() -> None:
    # Even when merged into a bigger chunk, the table is never split.
    text = f"Intro sentence.\n\n{_SMALL_TABLE}\n\nOutro prose after the table."
    merged = chunk_markdown(text, chunk_size=100, overlap=10, min_chars=1000)
    assert any(_SMALL_TABLE in c for c in merged)  # whole table in one chunk
    assert not any("| 1 | 2 |" in c and "| a | b |" not in c for c in merged)  # no lone row


def test_chunk_markdown_coalesce_does_not_dilute_substantial_prose() -> None:
    # A small table after a substantial prose chunk must NOT be merged into it (that
    # dilutes the prose embedding); the prose stands alone and the table stays whole.
    long_prose = "word " * 120  # one ~600-char prose chunk, over the floor
    text = f"{long_prose}\n\n{_SMALL_TABLE}"
    merged = chunk_markdown(text, chunk_size=200, overlap=20, min_chars=400)
    assert len(merged) == 2
    assert merged[0].strip() == long_prose.strip()  # prose un-diluted
    assert _SMALL_TABLE in merged[1]  # table whole, on its own


def test_chunk_markdown_coalesce_merges_a_run_of_small_tables() -> None:
    # Consecutive small tables (a figure-dense page) merge into one substantial block.
    t2 = "| c | d |\n|---|---|\n| 3 | 4 |"
    merged = chunk_markdown(f"{_SMALL_TABLE}\n\n{t2}", chunk_size=100, overlap=10, min_chars=1000)
    assert len(merged) == 1
    assert _SMALL_TABLE in merged[0] and t2 in merged[0]  # both tables whole, together


def test_chunk_markdown_coalesce_whole_text_under_floor_is_one_chunk() -> None:
    # If the whole text is under the floor, keep it as a single chunk (don't drop it).
    assert chunk_markdown("Tiny.", chunk_size=100, overlap=10, min_chars=400) == ["Tiny."]
