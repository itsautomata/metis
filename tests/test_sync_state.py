"""per-vault sync state must survive a corrupt/empty file and save atomically."""

from metis import config
from metis.config import MetisConfig
from metis.index import sync


def _cfg(tmp_path):
    return MetisConfig(vault_path=tmp_path)


def test_load_tolerates_corrupt_file(tmp_path):
    """a truncated state file must not crash sync -- it reads as no state."""
    config.SYNC_STATE_PATH.write_text("{ truncated", encoding="utf-8")
    assert sync._load_sync_state(_cfg(tmp_path)) == {}


def test_load_tolerates_empty_file(tmp_path):
    """a 0-byte file (interrupted write) reads as no state, not JSONDecodeError."""
    config.SYNC_STATE_PATH.write_text("", encoding="utf-8")
    assert sync._load_sync_state(_cfg(tmp_path)) == {}


def test_load_rejects_non_dict(tmp_path):
    """valid JSON of the wrong shape (a list) reads as no state, not a broken dict."""
    config.SYNC_STATE_PATH.write_text("[1, 2, 3]", encoding="utf-8")
    assert sync._load_sync_state(_cfg(tmp_path)) == {}


def test_save_roundtrips_and_leaves_no_temp(tmp_path):
    """the atomic save round-trips this vault's slice and cleans up its temp file."""
    cfg = _cfg(tmp_path)
    sync._save_sync_state({"a.md": "hash1", "b.md": "hash2"}, cfg)
    assert sync._load_sync_state(cfg) == {"a.md": "hash1", "b.md": "hash2"}
    assert not config.SYNC_STATE_PATH.with_suffix(".tmp").exists()
