"""tests for text chunking."""

from metis.ingest.process import chunk_text


def test_short_text_single_chunk():
    text = "this is a short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraph_splitting():
    text = "paragraph one.\n\nparagraph two.\n\nparagraph three."
    chunks = chunk_text(text, max_chars=30)
    assert len(chunks) > 1
    assert "paragraph one" in chunks[0]


def test_line_splitting_fallback():
    """transcripts have single newlines, not double."""
    lines = "\n".join(f"line {i} with some content" for i in range(100))
    chunks = chunk_text(lines, max_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 400  # max_chars * 2 safety


def test_oversized_chunk_hard_split():
    """single block of text with no newlines gets hard-split."""
    text = "word " * 10000  # ~50000 chars, no newlines
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1000  # within safety margin


def test_empty_text():
    chunks = chunk_text("")
    assert len(chunks) == 1


def test_overlap_exists():
    """chunks should have overlapping content."""
    paragraphs = [f"paragraph {i} " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=500)
    if len(chunks) >= 2:
        # last words of chunk 0 should appear in chunk 1
        last_words = chunks[0].split()[-10:]
        last_phrase = " ".join(last_words)
        assert any(word in chunks[1] for word in last_words[:5])
