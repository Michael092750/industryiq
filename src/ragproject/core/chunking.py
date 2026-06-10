"""Chunking: split a long text into smaller, overlapping pieces.

Chunks are measured in *words* (whitespace-separated tokens). Overlap repeats
the last few words of one chunk at the start of the next so that context
spanning a boundary isn't lost during retrieval.
"""


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
