"""a truncated categorization.json must not crash ingest; the save is atomic and vault-scoped."""

from metis import classify, config
from metis.config import MetisConfig

_DEFAULT = {"folder_descriptions": {}, "folder_embeddings": {}, "feedback": []}


def _cfg(tmp_path):
    return MetisConfig(vault_path=tmp_path)


def test_load_tolerates_truncated_file(tmp_path):
    """a save killed mid-write leaves invalid JSON; the load falls back to defaults."""
    config.CATEGORIZATION_PATH.write_text('{"folder_embeddings": {"a": [0.1, 0.2', encoding="utf-8")
    assert classify._load_categorization(_cfg(tmp_path)) == _DEFAULT


def test_load_rejects_non_dict(tmp_path):
    """valid JSON of the wrong shape reads as defaults, not a broken dict."""
    config.CATEGORIZATION_PATH.write_text("[1, 2, 3]", encoding="utf-8")
    assert classify._load_categorization(_cfg(tmp_path)) == _DEFAULT


def test_save_is_atomic_and_roundtrips(tmp_path):
    """the atomic save round-trips this vault's slice and leaves no temp file behind."""
    cfg = _cfg(tmp_path)
    payload = {"folder_descriptions": {"a": "x"}, "folder_embeddings": {}, "feedback": []}
    classify._save_categorization(payload, cfg)
    assert classify._load_categorization(cfg) == payload
    assert not config.CATEGORIZATION_PATH.with_suffix(".tmp").exists()
