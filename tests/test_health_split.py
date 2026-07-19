"""analyze_split must not propose a one-group 'split' on a homogeneous folder."""

from metis import health
from metis.config import MetisConfig


def test_split_returns_none_on_homogeneous_folder(tmp_path, monkeypatch):
    """four notes with identical embeddings collapse to one KMeans cluster, so there is
    nothing to split and analyze_split must return None, not a single group of all four."""
    fps = [f"/vault/papers/n{i}.md" for i in range(4)]
    embs = [[0.1, 0.2, 0.3]] * 4          # identical -> KMeans yields a single label
    folders = ["papers"] * 4
    monkeypatch.setattr(health, "_extract_vault_data", lambda config: (fps, embs, folders))

    cfg = MetisConfig(vault_path=tmp_path, chromadb_path=tmp_path / "cdb")
    assert health.analyze_split("papers", cfg) is None


def test_split_returns_two_groups_when_folder_is_heterogeneous(tmp_path, monkeypatch):
    """two well-separated clusters still split into two groups (no regression)."""
    fps = [f"/vault/papers/n{i}.md" for i in range(4)]
    embs = [[0.0, 0.0], [0.01, 0.0], [9.0, 9.0], [9.0, 9.01]]
    folders = ["papers"] * 4
    monkeypatch.setattr(health, "_extract_vault_data", lambda config: (fps, embs, folders))
    monkeypatch.setattr(health, "_label_cluster", lambda file_paths: "topic")

    cfg = MetisConfig(vault_path=tmp_path, chromadb_path=tmp_path / "cdb")
    result = health.analyze_split("papers", cfg)
    assert result is not None
    assert len(result) == 2
