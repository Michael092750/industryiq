"""Document loaders: turn a file on disk into plain text.

Each format has its own small, directly testable function (:func:`load_text`,
:func:`load_pdf`). :func:`load` is a dispatcher that picks the right one based
on the file extension, so callers don't need to care about the format.
"""

from collections.abc import Callable
from pathlib import Path

import docx
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


def load_docx(path: str | Path) -> str:
    """Extract text from a ``.docx`` file, joining paragraphs with newlines.

    Raises:
        FileNotFoundError: If ``path`` does not point to an existing file.
        ValueError: If the file is not a ``.docx`` file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() != ".docx":
        raise ValueError(f"load_docx expects a .docx file, got {p.suffix!r}")
    document = docx.Document(str(p))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


_LOADERS: dict[str, Callable[[str | Path], str]] = {
    ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def _to_utf8_safe(text: str) -> str:
    """Drop characters that cannot be encoded as UTF-8.

    PDF extraction can emit lone surrogate code points (from broken font maps)
    that are valid in a Python ``str`` but not encodable to UTF-8. Left in, they
    crash every UTF-8 consumer downstream -- the embedding tokenizer, JSON
    payloads to Bedrock, and Postgres text columns. Stripping them here keeps
    each loader's output safe to embed and store.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


def load(path: str | Path) -> str:
    """Load any supported file by dispatching on its extension.

    The returned text is guaranteed UTF-8 encodable (see :func:`_to_utf8_safe`).

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
    return _to_utf8_safe(loader(path))
