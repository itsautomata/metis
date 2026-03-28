"""tests for text extraction."""

import pytest
from pathlib import Path

from metis.ingest.extract import (
    is_url,
    is_arxiv,
    is_youtube,
    _youtube_video_id,
    _arxiv_to_pdf_url,
    _arxiv_abs_url,
    extract_from_markdown,
)


# --- URL detection ---

def test_is_url_https():
    assert is_url("https://example.com") is True

def test_is_url_http():
    assert is_url("http://example.com") is True

def test_is_url_local_path():
    assert is_url("/home/user/file.pdf") is False

def test_is_url_relative_path():
    assert is_url("notes.md") is False

def test_is_url_empty():
    assert is_url("") is False

def test_is_url_ftp():
    assert is_url("ftp://files.example.com") is False


# --- arxiv detection ---

def test_is_arxiv_abs():
    assert is_arxiv("https://arxiv.org/abs/2401.12345") is True

def test_is_arxiv_pdf():
    assert is_arxiv("https://arxiv.org/pdf/2401.12345") is True

def test_is_arxiv_www():
    assert is_arxiv("https://www.arxiv.org/abs/2401.12345") is True

def test_is_arxiv_false():
    assert is_arxiv("https://example.com/paper") is False

def test_arxiv_to_pdf_url():
    assert _arxiv_to_pdf_url("https://arxiv.org/abs/2401.12345") == "https://arxiv.org/pdf/2401.12345"

def test_arxiv_to_pdf_url_versioned():
    assert _arxiv_to_pdf_url("https://arxiv.org/abs/2401.12345v2") == "https://arxiv.org/pdf/2401.12345v2"

def test_arxiv_abs_url_from_pdf():
    assert _arxiv_abs_url("https://arxiv.org/pdf/2401.12345") == "https://arxiv.org/abs/2401.12345"


# --- youtube detection ---

def test_is_youtube_watch():
    assert is_youtube("https://www.youtube.com/watch?v=abc123def45") is True

def test_is_youtube_short():
    assert is_youtube("https://youtu.be/abc123def45") is True

def test_is_youtube_false():
    assert is_youtube("https://example.com/video") is False

def test_youtube_video_id():
    assert _youtube_video_id("https://www.youtube.com/watch?v=abc123def45") == "abc123def45"

def test_youtube_video_id_short():
    assert _youtube_video_id("https://youtu.be/abc123def45") == "abc123def45"

def test_youtube_video_id_with_params():
    assert _youtube_video_id("https://www.youtube.com/watch?v=abc123def45&list=PLxyz") == "abc123def45"

def test_youtube_video_id_invalid():
    with pytest.raises(ValueError):
        _youtube_video_id("https://example.com/not-youtube")

def test_is_youtube_playlist_only():
    assert is_youtube("https://www.youtube.com/playlist?list=PLxyz") is False


# --- markdown extraction ---

def test_extract_markdown_with_title(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# My Title\n\nsome content here")
    title, text = extract_from_markdown(md)
    assert title == "My Title"
    assert "some content here" in text

def test_extract_markdown_no_title(tmp_path):
    md = tmp_path / "my-note.md"
    md.write_text("just some text without a heading")
    title, text = extract_from_markdown(md)
    assert title == "my note"
    assert "just some text" in text

def test_extract_markdown_with_frontmatter(tmp_path):
    """title should come from # heading, not frontmatter."""
    md = tmp_path / "test.md"
    md.write_text("---\nsource: url\ntags: [a]\n---\n\n# Real Title\n\ncontent")
    title, text = extract_from_markdown(md)
    assert title == "Real Title"

def test_extract_markdown_empty_file(tmp_path):
    md = tmp_path / "empty.md"
    md.write_text("")
    title, text = extract_from_markdown(md)
    assert title == "empty"
    assert text == ""
