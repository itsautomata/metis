"""the per-vault ingest dedup index must survive a corrupt file and save atomically."""

from metis import config
from metis.config import MetisConfig
from metis.ingest import write


def _cfg(tmp_path):
    return MetisConfig(vault_path=tmp_path)


def test_load_tolerates_corrupt_file(tmp_path):
    """a truncated sources.json must not crash ingest -- it reads as no index."""
    config.SOURCES_INDEX_PATH.write_text("{ truncated", encoding="utf-8")
    assert write._load_sources_index(_cfg(tmp_path)) == {}


def test_load_tolerates_empty_file(tmp_path):
    """a 0-byte file (interrupted write) reads as no index, not JSONDecodeError."""
    config.SOURCES_INDEX_PATH.write_text("", encoding="utf-8")
    assert write._load_sources_index(_cfg(tmp_path)) == {}


def test_load_rejects_non_dict(tmp_path):
    """valid JSON of the wrong shape reads as no index, not a broken dict."""
    config.SOURCES_INDEX_PATH.write_text("[1, 2, 3]", encoding="utf-8")
    assert write._load_sources_index(_cfg(tmp_path)) == {}


def test_save_roundtrips_and_leaves_no_temp(tmp_path):
    """the atomic save round-trips this vault's slice and cleans up its temp file."""
    cfg = _cfg(tmp_path)
    write._save_sources_index({"https://example.com/a": "/vault/a.md"}, cfg)
    assert write._load_sources_index(cfg) == {"https://example.com/a": "/vault/a.md"}
    assert not config.SOURCES_INDEX_PATH.with_suffix(".tmp").exists()
