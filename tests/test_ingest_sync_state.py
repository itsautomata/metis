"""ingest registers notes in sync_state so a later sync doesn't re-embed them."""

from metis.config import MetisConfig
from metis.index import sync


def _cfg(tmp_path):
    return MetisConfig(vault_path=tmp_path)


def test_mark_file_synced_records_current_hash(tmp_path):
    cfg = _cfg(tmp_path)
    note = tmp_path / "note.md"
    note.write_text("# hello\n\nbody", encoding="utf-8")

    sync.mark_file_synced(note, cfg)

    state = sync._load_sync_state(cfg)
    assert state[str(note)] == sync._file_hash(note)  # present + current -> sync's 'unchanged' branch


def test_edit_after_mark_is_still_detected(tmp_path):
    """editing a marked note changes its hash, so sync will re-embed it (not silently skip)."""
    cfg = _cfg(tmp_path)
    note = tmp_path / "note.md"
    note.write_text("original", encoding="utf-8")
    sync.mark_file_synced(note, cfg)
    recorded = sync._load_sync_state(cfg)[str(note)]

    note.write_text("edited content", encoding="utf-8")
    assert sync._file_hash(note) != recorded  # changed -> sync's 'changed' branch fires


def test_mark_preserves_other_entries(tmp_path):
    """recording one note must not drop notes already tracked in this vault's state."""
    cfg = _cfg(tmp_path)
    sync._save_sync_state({"/prior/a.md": "deadbeef"}, cfg)
    note = tmp_path / "b.md"
    note.write_text("body", encoding="utf-8")

    sync.mark_file_synced(note, cfg)

    state = sync._load_sync_state(cfg)
    assert state["/prior/a.md"] == "deadbeef"      # existing entry kept
    assert str(note) in state                       # new one added
