"""tests for vault health analysis"""

from metis.config import MetisConfig
from metis.health import run_health


def test_health_survives_duplicate_notes(tmp_path, monkeypatch):
    """two identical (distance-0) notes must not crash DBSCAN with eps=0."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("---\ntags:\n  - x\n---\nsame content")
    (vault / "b.md").write_text("---\ntags:\n  - x\n---\nsame content")

    class _FakeCollection:
        def count(self):
            return 2

        def get(self, include=None):
            return {
                "metadatas": [{"file_path": str(vault / "a.md")}, {"file_path": str(vault / "b.md")}],
                "embeddings": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
            }

    monkeypatch.setattr("metis.health.get_collection", lambda c: _FakeCollection())

    report = run_health(MetisConfig(vault_path=vault, chromadb_path=tmp_path / "cdb"))

    assert report is not None
