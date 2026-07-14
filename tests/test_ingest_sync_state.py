"""ingest registers notes in sync_state so a later sync doesn't re-embed them."""

from metis.index import sync


def test_mark_file_synced_records_current_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    note = tmp_path / "note.md"
    note.write_text("# hello\n\nbody", encoding="utf-8")

    sync.mark_file_synced(note)

    state = sync._load_sync_state()
    assert state[str(note)] == sync._file_hash(note)  # present + current -> sync's 'unchanged' branch


def test_edit_after_mark_is_still_detected(monkeypatch, tmp_path):
    """editing a marked note changes its hash, so sync will re-embed it (not silently skip)."""
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    note = tmp_path / "note.md"
    note.write_text("original", encoding="utf-8")
    sync.mark_file_synced(note)
    recorded = sync._load_sync_state()[str(note)]

    note.write_text("edited content", encoding="utf-8")
    assert sync._file_hash(note) != recorded  # changed -> sync's 'changed' branch fires


def test_mark_preserves_other_entries(monkeypatch, tmp_path):
    """recording one note must not drop notes already tracked in the state."""
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    sync._save_sync_state({"/prior/a.md": "deadbeef"})
    note = tmp_path / "b.md"
    note.write_text("body", encoding="utf-8")

    sync.mark_file_synced(note)

    state = sync._load_sync_state()
    assert state["/prior/a.md"] == "deadbeef"      # existing entry kept
    assert str(note) in state                       # new one added
