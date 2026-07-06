"""tests for `metis secret list"""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def _line(output: str, key: str) -> str:
    return next(ln for ln in output.splitlines() if f"{key}:" in ln)


def test_list_reports_env_var_key_as_set(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.secrets.keyring.get_password", lambda *a, **k: None)
    monkeypatch.setenv("METIS_OPENAI_KEY", "sk-live-from-env")
    monkeypatch.delenv("METIS_X_BEARER", raising=False)

    result = runner.invoke(app, ["secret", "list"])

    assert result.exit_code == 0
    assert "not set" not in _line(result.stdout, "openai-key")


def test_list_reports_absent_key_as_not_set(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.secrets.keyring.get_password", lambda *a, **k: None)
    monkeypatch.delenv("METIS_X_BEARER", raising=False)

    result = runner.invoke(app, ["secret", "list"])

    assert result.exit_code == 0
    assert "not set" in _line(result.stdout, "x-token")


def test_list_includes_embedding_key(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.secrets.keyring.get_password", lambda *a, **k: None)
    monkeypatch.setenv("METIS_EMBEDDING_KEY", "sk-embed-from-env")

    result = runner.invoke(app, ["secret", "list"])

    assert result.exit_code == 0
    assert "not set" not in _line(result.stdout, "embedding-key")
