"""a --note path with '..' (or a symlinked vault) must scope to the note, not silently miss."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def _vault_with_note(tmp_path):
    vault = tmp_path / "vault"
    (vault / "metis-ingested").mkdir(parents=True)
    note = vault / "metis-ingested" / "mynote.md"
    note.write_text("# my note\n\ncontent", encoding="utf-8")
    cfg = MetisConfig(vault_path=vault, output_folder="metis-ingested", chromadb_path=tmp_path / "cdb")
    return cfg, vault


def test_dotdot_note_path_canonicalizes_to_stored_key(tmp_path, monkeypatch):
    cfg, vault = _vault_with_note(tmp_path)
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)

    got = {}

    def _ask(q, config, note_path=None, history=None):
        got["note_path"] = note_path
        return ("answer", [], 0.9)

    monkeypatch.setattr("metis.chat.ask", _ask)

    # a non-canonical path with '..' pointing at the same note
    dotdot = str(vault / "metis-ingested" / ".." / "metis-ingested" / "mynote.md")
    result = runner.invoke(app, ["chat", "q", "--note", dotdot], input="n\n")

    assert result.exit_code == 0
    # must equal the exact key the index stores: vault_path + clean relative
    assert got["note_path"] == str(vault / "metis-ingested" / "mynote.md")
