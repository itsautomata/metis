"""a same-name model that returns a different vector width is drift, and must not crash _cosine."""

from metis.config import MetisConfig, OpenAIConfig
from metis.index import canary


def _cfg(tmp_path, model="mymodel"):
    return MetisConfig(openai=OpenAIConfig(embedding_model=model), chromadb_path=tmp_path / "cdb")


def test_width_change_reports_drift_without_crashing(tmp_path, monkeypatch):
    """a 3-dim probe against a 2-dim baseline under one model id reads as drift, not a numpy error."""
    monkeypatch.setattr(canary, "CANARY_PATH", tmp_path / "canary.json")
    two_dim = [[1.0, 0.0], [0.0, 1.0]]
    monkeypatch.setattr("metis.index.embed.embed_texts", lambda texts, config: [list(v) for v in two_dim])
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)

    three_dim = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    monkeypatch.setattr("metis.index.embed.embed_texts", lambda texts, config: [list(v) for v in three_dim])
    assert canary.check_drift(cfg).status == "drift"
