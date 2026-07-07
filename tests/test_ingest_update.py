"""tests for the ingest update path."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig
from metis.ingest.process import ProcessedContent

runner = CliRunner()


def _make_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    existing = vault / "metis-ingested" / "existing-note.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# existing note\n\noriginal content worth keeping")

    config = MetisConfig(
        vault_path=vault,
        output_folder="metis-ingested",
        chromadb_path=tmp_path / "chromadb",
    )
    return config, existing


def test_failed_embed_preserves_existing_note(tmp_path, monkeypatch):
    """embedding fails while updating an already-ingested source: the old note stays."""
    config, existing = _make_config(tmp_path)
    monkeypatch.setattr("metis.cli.load_config", lambda: config)
    # source already ingested -> enters the update branch
    monkeypatch.setattr("metis.ingest.write.check_duplicate", lambda src: existing)
    monkeypatch.setattr("metis.index.sync._remove_file_from_index", lambda *a, **k: 0)
    monkeypatch.setattr(
        "metis.ingest.extract.extract",
        lambda *a, **k: ("existing note", "new text", "url", "https://example.com", None),
    )
    monkeypatch.setattr(
        "metis.ingest.process.process",
        lambda text, cfg: ProcessedContent(summary="s", key_points=[], tags=[], chunks=["chunk"]),
    )

    def _boom(*a, **k):
        raise RuntimeError("openai down")
    monkeypatch.setattr("metis.index.embed.embed_texts", _boom)

    result = runner.invoke(app, ["ingest", "https://example.com"], input="y\n")

    assert result.exit_code == 0
    assert existing.exists(), "existing note was destroyed by a failed update"
    assert "original content worth keeping" in existing.read_text()


def test_failed_extract_preserves_existing_note(tmp_path, monkeypatch):
    """extraction fails while updating (e.g. URL now dead): the old note stays."""
    config, existing = _make_config(tmp_path)
    monkeypatch.setattr("metis.cli.load_config", lambda: config)
    monkeypatch.setattr("metis.ingest.write.check_duplicate", lambda src: existing)
    monkeypatch.setattr("metis.index.sync._remove_file_from_index", lambda *a, **k: 0)

    def _dead(*a, **k):
        raise ValueError("could not extract text from: https://example.com")
    monkeypatch.setattr("metis.ingest.extract.extract", _dead)

    result = runner.invoke(app, ["ingest", "https://example.com"], input="y\n")

    assert result.exit_code == 0
    assert existing.exists(), "existing note was destroyed by a failed update"
    assert "original content worth keeping" in existing.read_text()


def test_dedup_keys_on_canonical_source_link(tmp_path, monkeypatch):
    """the duplicate check must use extract()'s canonical source_link, not the raw input."""
    config, existing = _make_config(tmp_path)
    monkeypatch.setattr("metis.cli.load_config", lambda: config)

    seen = {}

    def _record(src):
        seen["arg"] = src
        return existing  # a duplicate; decline the update below to stop right here

    monkeypatch.setattr("metis.ingest.write.check_duplicate", _record)
    monkeypatch.setattr(
        "metis.ingest.extract.extract",
        lambda *a, **k: ("t", "text", "file", "file:///abs/notes.md", None),
    )

    result = runner.invoke(app, ["ingest", "notes.md"], input="n\n")

    assert result.exit_code == 0
    assert seen["arg"] == "file:///abs/notes.md", "dedup used the raw input, not the canonical source_link"
