"""a non-2xx on a direct arxiv/PDF download must become a clean ValueError (which ingest skips)."""

import httpx
import pytest

from metis.ingest import extract


def _resp(status):
    return httpx.Response(status, request=httpx.Request("GET", "https://example.com"))


def test_pdf_url_http_error_becomes_valueerror(monkeypatch):
    """a dead `.pdf` link (404) raises ValueError, not a bare httpx.HTTPStatusError."""
    monkeypatch.setattr(extract, "_safe_get", lambda url, **k: _resp(404))
    with pytest.raises(ValueError):
        extract.extract_from_pdf_url("https://example.com/gone.pdf")


def test_arxiv_http_error_becomes_valueerror(monkeypatch):
    """a withdrawn/typo'd arxiv id (404) raises ValueError, not httpx.HTTPStatusError."""
    monkeypatch.setattr(extract, "_safe_get", lambda url, **k: _resp(404))
    with pytest.raises(ValueError):
        extract.extract_from_arxiv("https://arxiv.org/abs/0000.00000")
