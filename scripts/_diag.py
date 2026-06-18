import sys
from pathlib import Path

from ragproject.core.chunking import chunk_text
from ragproject.core.loaders import SUPPORTED_EXTENSIONS, load

root = Path(sys.argv[1])
for path in sorted(root.rglob("*")):
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        continue
    try:
        text = load(path)
    except Exception as e:  # noqa: BLE001
        print(f"LOAD-FAIL {path.name}: {type(e).__name__}: {e}")
        continue
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    bad = [(i, c) for i, c in enumerate(chunks) if not isinstance(c, str) or not c.strip()]
    flag = "  <-- BAD" if bad else ""
    print(f"{path.name}: textlen={len(text)} chunks={len(chunks)} bad={len(bad)}{flag}")
    for i, c in bad[:5]:
        print(f"    chunk[{i}] type={type(c).__name__} repr={c!r}")
