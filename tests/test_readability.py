"""tests for readable article bodies: structured markdown extraction + heading demotion."""

from metis.ingest.process import ProcessedContent
from metis.ingest.write import _demote_headings, build_markdown


def test_demote_headings_shifts_by_two():
    out = _demote_headings("# A\n\ntext\n\n## B")
    assert "### A" in out
    assert "#### B" in out


def test_demote_headings_caps_at_h6():
    assert "###### deep" in _demote_headings("##### deep")
    assert "###### x" in _demote_headings("###### x")


def test_demote_headings_leaves_non_headings():
    src = "a line with # a hash mid-text\n- a bullet\nplain text"
    assert _demote_headings(src) == src


def test_demote_headings_skips_code_fences():
    src = "## Real\n\n```python\n# a comment\nx = 1\n```\n\n## After"
    out = _demote_headings(src)
    assert "#### Real" in out and "#### After" in out  # real headings demoted
    assert "# a comment" in out and "### a comment" not in out  # fence comment untouched


def test_build_markdown_nests_article_under_content():
    md = build_markdown(
        "My Note", "# Art Title\n\nbody\n\n## Sub", "https://x.com", "url",
        ProcessedContent(summary="s", key_points=["k1"], tags=["t"], chunks=[]),
    )
    headings = [ln for ln in md.splitlines() if ln.startswith("#")]
    assert "# My Note" in headings
    assert "## Summary" in headings
    assert "## Content" in headings
    # the article's headings are demoted strictly below the note's h2 sections
    assert "### Art Title" in headings
    assert "#### Sub" in headings


def test_extract_from_url_requests_markdown(monkeypatch):
    """extract_from_url must ask trafilatura for markdown, not flat text."""
    from metis.ingest import extract

    calls = {}

    def _fake_extract(content, **kwargs):
        calls.update(kwargs)
        return "# H\n\nbody"

    class _Resp:
        text = "<html><body><article><h1>H</h1><p>body</p></article></body></html>"

    monkeypatch.setattr(extract, "_safe_get", lambda url, **k: _Resp())
    monkeypatch.setattr("trafilatura.extract", _fake_extract)
    monkeypatch.setattr("trafilatura.extract_metadata", lambda c: None)

    extract.extract_from_url("https://example.com/article")

    assert calls.get("output_format") == "markdown"
