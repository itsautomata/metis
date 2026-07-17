"""a malformed config.yaml must fail loud (clean abort)."""

import pytest
import typer

from metis import config


def test_load_config_rejects_list_yaml(monkeypatch, tmp_path):
    """a top-level list (stray leading '- ') aborts cleanly instead of AttributeError."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("- vault_path\n- output_folder\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_load_config_rejects_scalar_yaml(monkeypatch, tmp_path):
    """a top-level string (a line with no colon) aborts cleanly."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("just some notes\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_load_config_rejects_malformed_yaml(monkeypatch, tmp_path):
    """invalid YAML syntax aborts cleanly instead of an uncaught ParserError."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("openai: [unclosed\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_load_config_accepts_valid_dict(monkeypatch, tmp_path):
    """a normal mapping still loads (no regression)."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("vault_path: /tmp/v\noutput_folder: notes\n", encoding="utf-8")
    cfg = config.load_config()
    assert cfg.output_folder == "notes"


def test_init_config_rejects_non_dict(monkeypatch, tmp_path):
    """init_config's merge path also refuses a non-dict existing file."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    p.write_text("42\n", encoding="utf-8")  # parses to int
    with pytest.raises(typer.Exit):
        config.init_config()
