"""sync state must survive a corrupt/empty file and save atomically."""

from metis.index import sync


def test_load_tolerates_corrupt_file(monkeypatch, tmp_path):
    """a truncated state file must not crash sync -- it reads as no state."""
    p = tmp_path / "sync_state.json"
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", p)
    p.write_text("{ truncated", encoding="utf-8")  # invalid JSON, as a killed save leaves
    assert sync._load_sync_state() == {}


def test_load_tolerates_empty_file(monkeypatch, tmp_path):
    """a 0-byte file (interrupted write) reads as no state, not JSONDecodeError."""
    p = tmp_path / "sync_state.json"
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", p)
    p.write_text("", encoding="utf-8")
    assert sync._load_sync_state() == {}


def test_load_rejects_non_dict(monkeypatch, tmp_path):
    """valid JSON of the wrong shape (a list) reads as no state, not a broken dict."""
    p = tmp_path / "sync_state.json"
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", p)
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert sync._load_sync_state() == {}


def test_save_roundtrips_and_leaves_no_temp(monkeypatch, tmp_path):
    """the atomic save round-trips and cleans up its temp file via the rename."""
    p = tmp_path / "sync_state.json"
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", p)
    sync._save_sync_state({"a.md": "hash1", "b.md": "hash2"})
    assert sync._load_sync_state() == {"a.md": "hash1", "b.md": "hash2"}
    assert not (tmp_path / "sync_state.tmp").exists()  # temp was renamed into place
