"""fixed-set command arguments are rejected at parse time (exit 2), not validated in-body."""

from typer.testing import CliRunner

from metis.cli import app

runner = CliRunner()


def test_secret_invalid_action_rejected_before_picker(monkeypatch):
    shown = []
    monkeypatch.setattr("metis.pick.pick_secret", lambda names: shown.append(1) or None)
    result = runner.invoke(app, ["secret", "random"])
    assert result.exit_code == 2   # click choice rejection, not a silent exit 0 that mirrors 'set'
    assert not shown               # the key picker is never reached for a bad action


def test_secret_invalid_name_rejected(monkeypatch):
    result = runner.invoke(app, ["secret", "set", "badname"])
    assert result.exit_code == 2


def test_config_invalid_key_rejected():
    result = runner.invoke(app, ["config", "badkey"])
    assert result.exit_code == 2


def test_secret_list_still_works(monkeypatch):
    monkeypatch.setattr("metis.secrets.get_provider_key", lambda: "")
    monkeypatch.setattr("metis.secrets.get_embedding_key", lambda: "")
    monkeypatch.setattr("metis.secrets.get_x_bearer", lambda: "")
    result = runner.invoke(app, ["secret", "list"])
    assert result.exit_code == 0
    assert "provider-key" in result.output


def test_config_no_args_shows_settings(monkeypatch, tmp_path):
    import metis.config as cfg
    cfgpath = tmp_path / "config.yaml"
    cfgpath.write_text("vault_path: /v\nchromadb:\n  path: /db\n")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_PATH", cfgpath)
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "vault:" in result.output
