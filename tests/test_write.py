"""tests for vault writing."""

from pathlib import Path
from datetime import date

from metis.config import MetisConfig
from metis.ingest.process import ProcessedContent
from metis.ingest.write import slugify, build_markdown, write_to_vault, write_link_only


# --- slugify ---

def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"

def test_slugify_special_chars():
    assert slugify("What's up? (test)") == "whats-up-test"

def test_slugify_underscores():
    assert slugify("my_file_name") == "my-file-name"

def test_slugify_long_title():
    title = "a" * 100
    assert len(slugify(title)) <= 80

def test_slugify_dashes():
    assert slugify("multi---dash") == "multi-dash"

def test_slugify_empty():
    assert slugify("") == ""

def test_slugify_unicode():
    result = slugify("café résumé")
    assert "caf" in result  # unicode letters preserved or stripped, no crash


# --- build markdown ---

def test_build_markdown_has_frontmatter():
    processed = ProcessedContent(
        summary="test summary",
        key_points=["point 1", "point 2"],
        tags=["tag1", "tag2"],
        chunks=[],
    )
    md = build_markdown("Title", "body text", "https://example.com", "url", processed)
    assert "---" in md
    assert "source: \"https://example.com\"" in md
    assert "tags: [tag1, tag2]" in md
    assert "type: url" in md


def test_build_markdown_has_source_link():
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])
    md = build_markdown("Title", "body", "https://example.com", "url", processed)
    assert "> [source](https://example.com)" in md


def test_build_markdown_youtube_uses_transcript_label():
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])
    md = build_markdown("Title", "transcript text", "https://youtube.com", "youtube", processed)
    assert "## Transcript" in md
    assert "## Content" not in md


def test_build_markdown_url_uses_content_label():
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])
    md = build_markdown("Title", "article text", "https://example.com", "url", processed)
    assert "## Content" in md
    assert "## Transcript" not in md


def test_build_markdown_extra_metadata():
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])
    md = build_markdown("Title", "text", "url", "youtube", processed, extra={"channel": "TestChannel"})
    assert 'channel: "TestChannel"' in md


# --- write to vault ---

def test_write_to_vault_creates_file(tmp_path):
    config = MetisConfig(vault_path=tmp_path, output_folder="test-output")
    processed = ProcessedContent(summary="s", key_points=["p1"], tags=["t1"], chunks=[])

    path = write_to_vault("Test Note", "content", "https://example.com", "url", processed, config)

    assert path.exists()
    assert path.parent.name == "test-output"
    text = path.read_text()
    assert "# Test Note" in text


def test_write_to_vault_no_overwrite(tmp_path):
    config = MetisConfig(vault_path=tmp_path, output_folder="test-output")
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])

    path1 = write_to_vault("Same Title", "content 1", "url1", "url", processed, config)
    path2 = write_to_vault("Same Title", "content 2", "url2", "url", processed, config)

    assert path1 != path2
    assert path1.exists()
    assert path2.exists()


# --- write link only ---

def test_write_to_vault_nested_folder(tmp_path):
    config = MetisConfig(vault_path=tmp_path, output_folder="research/ai/papers")
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])

    path = write_to_vault("Deep Note", "content", "url", "url", processed, config)

    assert path.exists()
    assert "research/ai/papers" in str(path)


def test_dedup_detects_existing(tmp_path, monkeypatch):
    """check_duplicate returns path when source was already ingested."""
    from metis.ingest.write import check_duplicate, _save_sources_index, SOURCES_INDEX_PATH
    index_path = tmp_path / "sources.json"
    monkeypatch.setattr("metis.ingest.write.SOURCES_INDEX_PATH", index_path)

    # simulate an existing note
    note = tmp_path / "existing.md"
    note.write_text("# test")
    _save_sources_index({"https://example.com": str(note)})

    result = check_duplicate("https://example.com")
    assert result == note


def test_dedup_returns_none_for_new(tmp_path, monkeypatch):
    """check_duplicate returns None for never-ingested source."""
    from metis.ingest.write import check_duplicate
    index_path = tmp_path / "sources.json"
    monkeypatch.setattr("metis.ingest.write.SOURCES_INDEX_PATH", index_path)

    result = check_duplicate("https://new-url.com")
    assert result is None


def test_dedup_cleans_stale_entry(tmp_path, monkeypatch):
    """if indexed file was deleted, check_duplicate cleans the entry."""
    from metis.ingest.write import check_duplicate, _save_sources_index, _load_sources_index
    index_path = tmp_path / "sources.json"
    monkeypatch.setattr("metis.ingest.write.SOURCES_INDEX_PATH", index_path)

    _save_sources_index({"https://example.com": "/nonexistent/path.md"})

    result = check_duplicate("https://example.com")
    assert result is None
    # stale entry should be removed
    index = _load_sources_index()
    assert "https://example.com" not in index


def test_write_link_only(tmp_path):
    config = MetisConfig(vault_path=tmp_path, output_folder="test-output")

    path = write_link_only("https://example.com/page", config)

    assert path.exists()
    text = path.read_text()
    assert "source: \"https://example.com/page\"" in text
    assert "type: link" in text
    assert "link only" in text
