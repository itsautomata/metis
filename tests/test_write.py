"""tests for vault writing."""

import re

import yaml

from metis.config import MetisConfig
from metis.ingest.process import ProcessedContent
from metis.ingest.write import build_markdown, slugify, write_link_only, write_to_vault


def _frontmatter(md: str) -> dict:
    """parse the YAML frontmatter block out of a built note."""
    m = re.match(r"^---\n(.*?)\n---\n", md, re.DOTALL)
    assert m, "no frontmatter block found"
    return yaml.safe_load(m.group(1))


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
    assert slugify("") == "note"  # empty title must not become a hidden ".md" dotfile


def test_slugify_emoji_only():
    assert slugify("🎉🔥💯") == "note"  # symbol-only title must not become a ".md" dotfile


def test_slugify_punctuation_only():
    assert slugify("!!!") == "note"

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
    fm = _frontmatter(md)
    assert fm["source"] == "https://example.com"
    assert fm["tags"] == ["tag1", "tag2"]
    assert fm["type"] == "url"
    assert fm["summary"] == "test summary"


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
    fm = _frontmatter(md)
    assert fm["channel"] == "TestChannel"


def test_build_markdown_summary_with_quotes_roundtrips():
    summary = 'the essay analyzes the film "Her": memory and loss'
    processed = ProcessedContent(summary=summary, key_points=[], tags=["ai"], chunks=[])
    md = build_markdown("Title", "body", "https://example.com", "url", processed)
    fm = _frontmatter(md)
    assert fm["summary"] == summary
    assert fm["tags"] == ["ai"]


def test_build_markdown_extra_with_quotes_roundtrips():
    processed = ProcessedContent(summary="s", key_points=[], tags=[], chunks=[])
    md = build_markdown("Title", "text", "url", "tweet", processed, extra={"author": 'the "real" one'})
    fm = _frontmatter(md)
    assert fm["author"] == 'the "real" one'


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


def test_dedup_detects_existing(tmp_path):
    """check_duplicate returns path when source was already ingested into this vault."""
    from metis.ingest.write import _save_sources_index, check_duplicate
    cfg = MetisConfig(vault_path=tmp_path)

    # simulate an existing note
    note = tmp_path / "existing.md"
    note.write_text("# test")
    _save_sources_index({"https://example.com": str(note)}, cfg)

    result = check_duplicate("https://example.com", cfg)
    assert result == note


def test_dedup_returns_none_for_new(tmp_path):
    """check_duplicate returns None for never-ingested source."""
    from metis.ingest.write import check_duplicate

    result = check_duplicate("https://new-url.com", MetisConfig(vault_path=tmp_path))
    assert result is None


def test_dedup_cleans_stale_entry(tmp_path):
    """if indexed file was deleted, check_duplicate cleans the entry."""
    from metis.ingest.write import (
        _load_sources_index,
        _save_sources_index,
        check_duplicate,
    )
    cfg = MetisConfig(vault_path=tmp_path)

    _save_sources_index({"https://example.com": "/nonexistent/path.md"}, cfg)

    result = check_duplicate("https://example.com", cfg)
    assert result is None
    # stale entry should be removed
    index = _load_sources_index(cfg)
    assert "https://example.com" not in index


def test_write_link_only(tmp_path):
    config = MetisConfig(vault_path=tmp_path, output_folder="test-output")

    path = write_link_only("https://example.com/page", config)

    assert path.exists()
    text = path.read_text()
    fm = _frontmatter(text)
    assert fm["source"] == "https://example.com/page"
    assert fm["type"] == "link"
    assert "link only" in fm["summary"]
