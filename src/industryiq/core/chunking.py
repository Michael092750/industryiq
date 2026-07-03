"""Chunking: split a long text into smaller, overlapping pieces.

Chunks are measured in *words* (whitespace-separated tokens). Overlap repeats
the last few words of one chunk at the start of the next so that context
spanning a boundary isn't lost during retrieval.

:func:`split_sections` is a Markdown-aware pre-pass: it carves text into blocks
under their nearest ATX heading, so a caller can word-chunk each block and tag
every chunk with the section it came from (used for the ``section`` metadata
field). On text with no headings -- plain ``.txt`` or pypdf output -- it yields a
single block, so word-chunking is identical to chunking the text directly.
"""

import re

# An ATX Markdown heading: 1-6 leading '#', a space, then the heading text
# (optional trailing '#'s are dropped). The leading ``\S`` requires real content,
# so a bare ``###`` or a spaceless ``#tag`` is not treated as a heading. Docling
# emits exactly this form; pypdf/plain text emit none, so they fall through as a
# single section.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")


def split_sections(
    text: str, *, initial_section: str | None = None
) -> list[tuple[str | None, str]]:
    """Split Markdown ``text`` into ``(section, block)`` pairs by ATX heading.

    Each block is the heading line plus the body beneath it, up to (but not
    including) the next heading; ``section`` is that heading's text. The heading
    line is kept *inside* the block so its words still embed/index -- it doubles
    as both retrievable content and the ``section`` tag. Any content before the
    first heading forms a leading block tagged with ``initial_section``.

    Consecutive headings with no body between them (title pages, tables of
    contents, stacked figure/table captions) are **coalesced**: they accumulate
    into the next section's block rather than each becoming its own bodyless chunk.
    A bare heading embeds very close to any topical query but answers nothing, so
    left standalone it would crowd real paragraphs out of the top results. The
    coalesced block is tagged with the nearest (last) heading.

    ``initial_section`` lets a caller carry the active heading across a boundary
    it splits on independently (e.g. page boundaries): a section that opened on
    one page still tags the body that continues onto the next. The last pair's
    ``section`` is the heading in force at the end of ``text`` -- pass it as the
    next call's ``initial_section`` to continue the carry.

    Text with no headings yields a single ``(initial_section, text)`` pair, so
    downstream word-chunking matches chunking the raw text.
    """
    blocks: list[tuple[str | None, str]] = []
    current_section = initial_section
    current_lines: list[str] = []
    has_body = False  # current_lines holds real content, not just heading lines
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            # Only close the block if it has body text; otherwise let the heading
            # accumulate so consecutive headings never split into bodyless chunks.
            if current_lines and has_body:
                blocks.append((current_section, "\n".join(current_lines)))
                current_lines = []
                has_body = False
            current_section = match.group(2)
        elif line.strip():
            has_body = True
        current_lines.append(line)
    if current_lines:
        blocks.append((current_section, "\n".join(current_lines)))
    return blocks


# A GFM table row starts and ends with a pipe -- Docling renders tables (and, with
# chart extraction, chart-to-CSV data) this way, including the ``|---|---|`` rule
# row. A run of >=2 consecutive such lines is a table.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# A fenced code block delimiter (``` or ~~~) -- the other multi-line structure a
# chart/table may serialize to.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _segment_markdown(text: str) -> list[tuple[bool, str]]:
    """Split ``text`` into ordered ``(is_atomic, segment)`` runs.

    Atomic segments are structures that must not be split across chunks -- GFM
    tables and fenced code blocks (how Docling serializes tables and extracted
    chart data). Everything else is prose. Consecutive prose lines merge into one
    prose segment; each table / fenced block is its own atomic segment, in order.
    """
    lines = text.splitlines()
    segments: list[tuple[bool, str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            segments.append((False, "\n".join(prose)))
            prose.clear()

    i, n = 0, len(lines)
    while i < n:
        if _FENCE_RE.match(lines[i]):
            flush_prose()
            block = [lines[i]]
            i += 1
            while i < n:  # consume up to and including the closing fence (or EOF)
                block.append(lines[i])
                closed = bool(_FENCE_RE.match(lines[i]))
                i += 1
                if closed:
                    break
            segments.append((True, "\n".join(block)))
        elif _TABLE_ROW_RE.match(lines[i]) and i + 1 < n and _TABLE_ROW_RE.match(lines[i + 1]):
            flush_prose()
            block = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            segments.append((True, "\n".join(block)))
        else:
            prose.append(lines[i])
            i += 1
    flush_prose()
    return segments


def chunk_markdown(text: str, *, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Word-chunk Markdown ``text`` while keeping tables/charts/figures intact.

    Prose runs are split exactly like :func:`chunk_text` (overlapping word chunks).
    GFM tables and fenced code blocks -- how Docling renders tables and extracted
    chart-to-CSV data -- are emitted **whole**, never cut across a chunk boundary,
    even when one such block exceeds ``chunk_size`` words. Segments keep document
    order, so a table stays adjacent to the prose (its caption/heading) around it.

    Text with no such structures chunks identically to :func:`chunk_text`, so plain
    ``.txt`` / pypdf output is unaffected.
    """
    chunks: list[str] = []
    for is_atomic, segment in _segment_markdown(text):
        if is_atomic:
            block = segment.strip("\n")
            if block.strip():
                chunks.append(block)
        else:
            chunks.extend(chunk_text(segment, chunk_size=chunk_size, overlap=overlap))
    return chunks


def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` words.

    Args:
        text: The input text to split.
        chunk_size: Maximum number of words per chunk. Must be positive.
        overlap: Number of words each chunk shares with the previous one.
            Must be non-negative and smaller than ``chunk_size``.

    Returns:
        A list of chunk strings. An empty or whitespace-only input yields ``[]``.

    Raises:
        ValueError: If ``chunk_size`` or ``overlap`` are out of range.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks
