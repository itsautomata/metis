"""ingest --pick-folder then cancel must abort, not silently ingest into the default folder."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def test_cancelled_folder_pick_aborts_before_extract(tmp_path, monkeypatch):
    """when the folder picker is cancelled (returns None), no extraction or ingest may run."""
    vault = tmp_path / "vault"
    (vault / "metis-ingested").mkdir(parents=True)
    cfg = MetisConfig(vault_path=vault, output_folder="metis-ingested", chromadb_path=tmp_path / "cdb")

    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr("metis.pick.pick_folder", lambda config: None)  # user hits Ctrl-C

    def _extract(*args, **kwargs):
        raise AssertionError("extract must not run after a cancelled folder pick")

    monkeypatch.setattr("metis.ingest.extract.extract", _extract)

    result = runner.invoke(app, ["ingest", "https://example.com/x", "--pick-folder"])
    assert result.exit_code == 0
    assert "cancelled" in result.output
