"""tests for folder placement during batch ingest."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig
from metis.ingest.process import ProcessedContent

runner = CliRunner()


def test_batch_cancel_falls_to_default_not_previous_folder(tmp_path, monkeypatch):
    """source 1 picks a folder, source 2 cancels: source 2 must go to the default."""
    vault = tmp_path / "vault"
    vault.mkdir()
    config = MetisConfig(
        vault_path=vault,
        output_folder="metis-ingested",
        chromadb_path=tmp_path / "chromadb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: config)
    monkeypatch.setattr("metis.ingest.write.check_duplicate", lambda src: None)
    monkeypatch.setattr(
        "metis.ingest.extract.extract",
        lambda source, **k: (source, "body text here", "url", source, None),
    )
    monkeypatch.setattr(
        "metis.ingest.process.process",
        lambda text, cfg: ProcessedContent(summary="s", key_points=[], tags=[], chunks=["chunk"]),
    )
    monkeypatch.setattr("metis.index.embed.embed_texts", lambda chunks, cfg: [[0.1]])
    monkeypatch.setattr("metis.index.store.store_chunks_with_embeddings", lambda *a, **k: 0)
    monkeypatch.setattr("metis.classify.suggest_folder", lambda emb, cfg: [("research/ai", 0.8)])
    monkeypatch.setattr("metis.classify.record_feedback", lambda *a, **k: None)

    picks = iter(["research/ai", None])  # source one picks, source two cancels
    monkeypatch.setattr("metis.pick.pick_suggested_folder", lambda suggestions, cfg: next(picks))

    result = runner.invoke(app, ["ingest", "one", "two"])

    assert result.exit_code == 0
    assert (vault / "research/ai" / "one.md").exists()
    assert (vault / "metis-ingested" / "two.md").exists()
    assert not (vault / "research/ai" / "two.md").exists()
