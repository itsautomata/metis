"""a corrupt or truncated PDF must fail as a clean ValueError, not crash the batch."""

import pytest

from metis.ingest.extract import extract_from_pdf


def test_garbage_pdf_raises_valueerror(tmp_path):
    """a .pdf whose bytes are not a real PDF must raise ValueError (which cli.py skips cleanly)."""
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 this is not actually a pdf document at all")
    with pytest.raises(ValueError):
        extract_from_pdf(p)


def test_empty_pdf_raises_valueerror(tmp_path):
    """a zero-byte .pdf must raise ValueError, not fitz's EmptyFileError."""
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        extract_from_pdf(p)
