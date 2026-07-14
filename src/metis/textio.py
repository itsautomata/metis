"""text i/o helpers for reading vault notes."""

from pathlib import Path


def read_note_text(path: Path | str) -> str:
    """read a vault note as text, tolerating non-UTF-8 files.

    tries UTF-8, then CP-1252, then Latin-1
    """
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")
