"""Read an uploaded file's text via the path-based loaders.

Shared by the routes that accept file uploads. Writes the upload to a temp file
so the path-based loaders can read it, validates the extension, and cleans up.
"""

import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from industryiq.core.loaders import SUPPORTED_EXTENSIONS, load


def load_upload(file: UploadFile) -> str:
    """Return the text of an uploaded file, or raise 415 for an unsupported type."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        return load(tmp_path)
    finally:
        os.unlink(tmp_path)
