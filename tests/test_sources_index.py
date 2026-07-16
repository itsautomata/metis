"""the ingest dedup index must survive a corrupt file and save atomically."""

from metis.ingest import write


def test_load_tolerates_corrupt_file(monkeypatch, tmp_path):
    """a truncated sources.json must not crash ingest -- it reads as no index."""
    p = tmp_path / "sources.json"
    monkeypatch.setattr(write, "SOURCES_INDEX_PATH", p)
    p.write_text("{ truncated", encoding="utf-8")
    assert write._load_sources_index() == {}


def test_load_tolerates_empty_file(monkeypatch, tmp_path):
    """a 0-byte file (interrupted write) reads as no index, not JSONDecodeError."""
    p = tmp_path / "sources.json"
    monkeypatch.setattr(write, "SOURCES_INDEX_PATH", p)
    p.write_text("", encoding="utf-8")
    assert write._load_sources_index() == {}


def test_load_rejects_non_dict(monkeypatch, tmp_path):
    """valid JSON of the wrong shape reads as no index, not a broken dict."""
    p = tmp_path / "sources.json"
    monkeypatch.setattr(write, "SOURCES_INDEX_PATH", p)
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert write._load_sources_index() == {}


def test_save_roundtrips_and_leaves_no_temp(monkeypatch, tmp_path):
    """the atomic save round-trips and cleans up its temp file via the rename."""
    p = tmp_path / "sources.json"
    monkeypatch.setattr(write, "SOURCES_INDEX_PATH", p)
    write._save_sources_index({"https://example.com/a": "/vault/a.md"})
    assert write._load_sources_index() == {"https://example.com/a": "/vault/a.md"}
    assert not (tmp_path / "sources.tmp").exists()  # temp was renamed into place
