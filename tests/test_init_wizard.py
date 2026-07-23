"""the guided A-to-Z setup wizard: metis init walks the whole config, every step skippable."""

import yaml
from typer.testing import CliRunner

from metis.cli import app

runner = CliRunner()


def _isolate_config(monkeypatch, tmp_path):
    """point config at a temp dir with a pre-seeded chromadb path so nothing touches the real home."""
    cfgdir = tmp_path / "dotmetis"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yaml"
    db = tmp_path / "db"
    cfgpath.write_text(f"vault_path: {tmp_path / 'seedvault'}\nchromadb:\n  path: {db}\n")
    monkeypatch.setattr("metis.config.CONFIG_DIR", cfgdir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", cfgpath)
    return cfgpath


def _wire(monkeypatch, prompts, picks, confirms, provider_key=""):
    """script the wizard: typer.prompt for free text, pick_from for menus, confirm_menu for gates."""
    pq, kq, cq = list(prompts), list(picks), list(confirms)
    monkeypatch.setattr("metis.cli.typer.prompt", lambda *a, **k: pq.pop(0))
    monkeypatch.setattr("metis.pick.pick_from", lambda *a, **k: kq.pop(0))
    monkeypatch.setattr("metis.pick.confirm_menu", lambda *a, **k: cq.pop(0))
    monkeypatch.setattr("metis.cli._interactive", lambda: True)
    monkeypatch.setattr("metis.cli.doctor", lambda *a, **k: None)
    monkeypatch.setattr("metis.secrets.get_provider_key", lambda: provider_key)


def test_wizard_base_url_preset_and_custom(monkeypatch):
    """the provider menu maps a preset pick to its endpoint; the last option prompts for a custom one."""
    from metis import cli

    monkeypatch.setattr("metis.pick.pick_from", lambda prompt, options, default=None: options[1][1])
    assert cli._wizard_base_url("") == "https://openrouter.ai/api/v1"

    monkeypatch.setattr("metis.pick.pick_from", lambda prompt, options, default=None: options[-1][1])
    monkeypatch.setattr("metis.cli.typer.prompt", lambda *a, **k: "http://x/v1")
    assert cli._wizard_base_url("") == "http://x/v1"

    monkeypatch.setattr("metis.pick.pick_from", lambda prompt, options, default=None: None)
    assert cli._wizard_base_url("https://keep/v1") == "https://keep/v1"


def test_wizard_preset_provider_full_config(monkeypatch, tmp_path):
    cfgpath = _isolate_config(monkeypatch, tmp_path)
    vault = tmp_path / "myvault"
    _wire(
        monkeypatch,
        prompts=[str(vault), "gpt-4o", "text-embedding-3-small", "metis-ingested"],
        picks=["https://openrouter.ai/api/v1", "auto"],
        confirms=[False, True],
        provider_key="already-set",  # skip the key prompt
    )

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(cfgpath.read_text())
    assert raw["vault_path"] == str(vault)
    assert raw["openai"]["base_url"] == "https://openrouter.ai/api/v1"
    assert raw["openai"]["chat_model"] == "gpt-4o"
    assert raw["openai"]["embedding_model"] == "text-embedding-3-small"
    assert raw["output_folder"] == "metis-ingested"
    assert "link_style" not in raw


def test_wizard_custom_provider_stores_key_and_forces_link(monkeypatch, tmp_path):
    from metis.cli import _CUSTOM_BASE_URL

    cfgpath = _isolate_config(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("metis.secrets.set_secret", lambda name, value: calls.append((name, value)))
    vault = tmp_path / "v2"
    _wire(
        monkeypatch,
        prompts=[str(vault), "http://localhost:11434/v1", "sk-test", "gpt-4o", "text-embedding-3-small", "notes"],
        picks=[_CUSTOM_BASE_URL, "markdown"],
        confirms=[False, True],
        provider_key="",  # empty -> the key prompt fires
    )

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(cfgpath.read_text())
    assert raw["openai"]["base_url"] == "http://localhost:11434/v1"
    assert raw["output_folder"] == "notes"
    assert raw["link_style"] == "markdown"
    assert ("provider-key", "sk-test") in calls


def test_wizard_advanced_branch_embedding_and_extra_keys(monkeypatch, tmp_path):
    cfgpath = _isolate_config(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("metis.secrets.set_secret", lambda name, value: calls.append((name, value)))
    vault = tmp_path / "v3"
    _wire(
        monkeypatch,
        prompts=[
            str(vault), "sk-main", "gpt-4o", "text-embedding-3-small", "metis-ingested",
            "https://api.openai.com/v1", "text-embedding-3-small", "sk-embed", "xtok",
        ],
        picks=["", "auto"],           # provider=openai, link=auto
        confirms=[True, True],        # advanced=yes, doctor=yes (stubbed)
        provider_key="",
    )

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(cfgpath.read_text())
    assert raw["embedding"]["base_url"] == "https://api.openai.com/v1"
    assert raw["embedding"]["model"] == "text-embedding-3-small"
    assert ("provider-key", "sk-main") in calls
    assert ("embedding-api-key", "sk-embed") in calls
    assert ("x-bearer-token", "xtok") in calls


def test_wizard_survives_bare_null_openai_config(monkeypatch, tmp_path):
    """a hand-edited config with a bare 'openai:' (null) line must not crash the wizard."""
    cfgdir = tmp_path / "dotmetis"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yaml"
    cfgpath.write_text(f"vault_path: {tmp_path / 'v'}\nchromadb:\n  path: {tmp_path / 'db'}\nopenai:\n")
    monkeypatch.setattr("metis.config.CONFIG_DIR", cfgdir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", cfgpath)
    _wire(
        monkeypatch,
        prompts=[str(tmp_path / "v2"), "gpt-4o", "text-embedding-3-small", "metis-ingested"],
        picks=["https://openrouter.ai/api/v1", "auto"],
        confirms=[False, True],
        provider_key="already-set",
    )

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    raw = yaml.safe_load(cfgpath.read_text())
    assert raw["openai"]["base_url"] == "https://openrouter.ai/api/v1"


def test_init_non_interactive_writes_defaults_no_prompt(monkeypatch, tmp_path):
    """off a terminal, init must not launch the wizard or hang; it writes defaults and prints hints."""
    _isolate_config(monkeypatch, tmp_path)
    # _interactive() is already False under CliRunner (stdin is not a tty)
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "metis initialized" in result.output
    assert "metis setup" not in result.output
    assert "metis config vault" in result.output
