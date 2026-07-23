"""the machine-readable + preview surfaces: search --json, reindex --dry-run."""

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def test_search_json_emits_parseable_array(monkeypatch):
    """--json prints a JSON array on stdout, no rich prose, and skips the interactive pick."""
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    rows = [
        SimpleNamespace(file_path="/v/alpha.md", score=0.91, text="alpha body"),
        SimpleNamespace(file_path="/v/beta.md", score=0.80, text="beta body"),
    ]
    monkeypatch.setattr("metis.search.search_vault", lambda q, c, limit=5: rows)

    result = runner.invoke(app, ["search", "hi", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert [d["note"] for d in data] == ["alpha.md", "beta.md"]
    assert data[0]["score"] == 0.91
    assert "searching:" not in result.stdout


def test_search_json_empty_is_valid_json(monkeypatch):
    """no results in --json is an empty array on stdout, not a yellow prose line."""
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr("metis.search.search_vault", lambda q, c, limit=5: [])

    result = runner.invoke(app, ["search", "hi", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_bare_metis_splash_suppresses_art_when_piped():
    """no-args prints the two starter examples, but the wordmark art is suppressed off a terminal (J18)."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "metis ingest" in result.output
    assert "metis --help" in result.output
    assert "M E T I S" not in result.output  # the wordmark never reaches a pipe


def test_reindex_dry_run_calls_no_provider(monkeypatch, tmp_path):
    """--dry-run reports the note count and never reaches reindex_vault (no embedding cost)."""
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "a.md").write_text("a")
    (vault / "b.md").write_text("b")
    cfg = MetisConfig(vault_path=vault, chromadb_path=tmp_path / "c")
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)

    def _boom(*a, **k):
        raise AssertionError("reindex_vault ran during a dry run")
    monkeypatch.setattr("metis.index.sync.reindex_vault", _boom)

    result = runner.invoke(app, ["reindex", "--dry-run"])

    assert result.exit_code == 0
    assert "would re-embed" in result.output
    assert "2" in result.output
