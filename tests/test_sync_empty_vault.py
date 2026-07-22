"""sync must not wipe the index when the vault resolves to zero files by mistake."""

import pytest
from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig
from metis.index import sync

runner = CliRunner()


class _FakeCollection:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n

    def get(self, **kwargs):
        return {"metadatas": []}

    def delete(self, ids):
        pass


def _empty_vault_cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()  # exists but holds no .md files
    return MetisConfig(vault_path=vault, chromadb_path=tmp_path / "cdb")


def test_sync_aborts_when_vault_empty_but_index_not(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "get_collection", lambda c: _FakeCollection(5))
    with pytest.raises(sync.EmptyVaultError):
        sync.sync_vault(_empty_vault_cfg(tmp_path))


def test_force_bypasses_the_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "get_collection", lambda c: _FakeCollection(5))
    report = sync.sync_vault(_empty_vault_cfg(tmp_path), force=True)  # must not raise
    assert report.total_files == 0


def test_empty_vault_allowed_when_index_also_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "get_collection", lambda c: _FakeCollection(0))
    report = sync.sync_vault(_empty_vault_cfg(tmp_path))  # nothing to protect -> no abort
    assert report.total_files == 0


def test_cli_sync_aborts_with_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setattr("metis.cli.load_config", lambda: _empty_vault_cfg(tmp_path))
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr(sync, "get_collection", lambda c: _FakeCollection(5))

    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 1
    assert "would delete" in result.output
    assert "--force" in result.output


def test_cli_sync_force_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr("metis.cli.load_config", lambda: _empty_vault_cfg(tmp_path))
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr(sync, "get_collection", lambda c: _FakeCollection(5))

    result = runner.invoke(app, ["sync", "--force"])
    assert result.exit_code == 0
