"""a truncated categorization.json (interrupted save) must not crash ingest; the save is atomic."""

from metis import classify

_DEFAULT = {"folder_descriptions": {}, "folder_embeddings": {}, "feedback": []}


def test_load_tolerates_truncated_file(monkeypatch, tmp_path):
    """a save killed mid-write leaves invalid JSON; the load falls back to defaults."""
    p = tmp_path / "categorization.json"
    monkeypatch.setattr(classify, "CATEGORIZATION_PATH", p)
    p.write_text('{"folder_embeddings": {"a": [0.1, 0.2', encoding="utf-8")
    assert classify._load_categorization() == _DEFAULT


def test_load_rejects_non_dict(monkeypatch, tmp_path):
    """valid JSON of the wrong shape reads as defaults, not a broken dict."""
    p = tmp_path / "categorization.json"
    monkeypatch.setattr(classify, "CATEGORIZATION_PATH", p)
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert classify._load_categorization() == _DEFAULT


def test_save_is_atomic_and_roundtrips(monkeypatch, tmp_path):
    """the atomic save round-trips and leaves no temp file behind."""
    p = tmp_path / "categorization.json"
    monkeypatch.setattr(classify, "CATEGORIZATION_PATH", p)
    payload = {"folder_descriptions": {"a": "x"}, "folder_embeddings": {}, "feedback": []}
    classify._save_categorization(payload)
    assert classify._load_categorization() == payload
    assert not (tmp_path / "categorization.tmp").exists()
