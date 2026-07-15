"""a PDF URL must serve real PDF bytes, not HTML mislabeled as application/pdf."""

from types import SimpleNamespace

import pytest

from metis.ingest import extract


def _resp(content, content_type="application/pdf"):
    return SimpleNamespace(
        headers={"content-type": content_type},
        content=content,
        raise_for_status=lambda: None,
    )


def test_html_served_as_pdf_is_rejected(monkeypatch):
    """an HTML paywall/error page labeled application/pdf must raise, not be extracted as the paper."""
    html = b"<html><body><h1>Paywall</h1><p>Please subscribe.</p></body></html>"
    monkeypatch.setattr(extract, "_safe_get", lambda url, **kw: _resp(html))
    with pytest.raises(ValueError):
        extract.extract_from_pdf_url("https://example.com/paper.pdf")


def test_real_pdf_bytes_pass_the_guard(monkeypatch):
    """real PDF magic bytes pass the guard and reach extraction."""
    monkeypatch.setattr(extract, "_safe_get", lambda url, **kw: _resp(b"%PDF-1.4\nfake body"))
    monkeypatch.setattr(extract, "extract_from_pdf", lambda path: ("title", "body text"))
    _, text = extract.extract_from_pdf_url("https://example.com/paper.pdf")
    assert text == "body text"


def test_pdf_with_leading_whitespace_accepted(monkeypatch):
    """some servers prepend whitespace before %PDF-; that is still a real PDF."""
    monkeypatch.setattr(extract, "_safe_get", lambda url, **kw: _resp(b"\r\n  %PDF-1.7\nbody"))
    monkeypatch.setattr(extract, "extract_from_pdf", lambda path: ("t", "ok"))
    _, text = extract.extract_from_pdf_url("https://example.com/paper.pdf")
    assert text == "ok"
