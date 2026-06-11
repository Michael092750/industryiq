"""Document loaders: turn a file on disk into plain text.

Each format has its own small, directly testable function (:func:`load_text`,
:func:`load_pdf`). :func:`load` is a dispatcher that picks the right one based
on the file extension, so callers don't need to care about the format.
"""

from collections.abc import Callable
from pathlib import Path

import pypdf


def load_text(path: str | Path) -> str:
    """Read a plain-text ``.txt`` file and return its contents (UTF-8).

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
        ValueError: If the file is not a ``.txt`` file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() != ".txt":
        raise ValueError(f"load_text expects a .txt file, got {p.suffix!r}")
    return p.read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    """Extract text from a ``.pdf`` file, joining pages with newlines.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
        ValueError: If the file is not a ``.pdf`` file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"load_pdf expects a .pdf file, got {p.suffix!r}")
    reader = pypdf.PdfReader(str(p))
    return "\n".join(page.extract_text() for page in reader.pages)


_LOADERS: dict[str, Callable[[str | Path], str]] = {
    ".txt": load_text,
    ".pdf": load_pdf,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load(path: str | Path) -> str:
    """Load any supported file by dispatching on its extension.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
        ValueError: If the file's extension is not supported.
    """
    p = Path(path)
    loader = _LOADERS.get(p.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type {p.suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return loader(path)
