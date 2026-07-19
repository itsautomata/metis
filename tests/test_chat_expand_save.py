"""chat --save with a low-confidence answer plus expansion must leave one Q&A entry, not two."""

from types import SimpleNamespace

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def _vault_note(tmp_path):
    vault = tmp_path / "vault"
    (vault / "metis-ingested").mkdir(parents=True)
    note = vault / "metis-ingested" / "n.md"
    note.write_text("# n\n\n## Content\n\nbody\n", encoding="utf-8")
    cfg = MetisConfig(vault_path=vault, output_folder="metis-ingested", chromadb_path=tmp_path / "cdb")
    return cfg, note


def _patch_common(monkeypatch, cfg, saves):
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr("metis.chat.ask", lambda *a, **k: ("weak answer", ["/s.md"], 0.1))  # low conf
    monkeypatch.setattr("metis.chat.save_qa_to_note", lambda *a, **k: saves.append((a, k)))


def test_low_confidence_expand_accept_saves_once(tmp_path, monkeypatch):
    """accepting the wikipedia expansion saves only the expanded answer, not the weak one too."""
    cfg, note = _vault_note(tmp_path)
    saves = []
    _patch_common(monkeypatch, cfg, saves)

    monkeypatch.setattr("metis.expand.extract_search_keywords", lambda q, c: "kw")
    monkeypatch.setattr(
        "metis.expand.search_wikipedia",
        lambda kw: [SimpleNamespace(title="T", preview="p", source_type="wikipedia")],
    )
    monkeypatch.setattr("metis.pick.pick_wikipedia", lambda choices: "T")
    monkeypatch.setattr("metis.expand.ingest_external", lambda best, config: (tmp_path / "ext.md", None))

    result = runner.invoke(app, ["chat", "q", "--note", str(note), "--save"], input="y\n")
    assert result.exit_code == 0, result.output
    assert len(saves) == 1  # exactly one entry, the expanded answer


def test_low_confidence_expand_decline_saves_once(tmp_path, monkeypatch):
    """declining the expansion still saves the original answer once (no lost save)."""
    cfg, note = _vault_note(tmp_path)
    saves = []
    _patch_common(monkeypatch, cfg, saves)

    result = runner.invoke(app, ["chat", "q", "--note", str(note), "--save"], input="n\n")
    assert result.exit_code == 0, result.output
    assert len(saves) == 1  # the original answer, saved as a fallback
