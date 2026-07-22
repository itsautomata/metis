"""a config section flattened to a scalar/list, or a non-string path, must abort clean not crash."""

import pytest
import typer

from metis import config


def _write(monkeypatch, tmp_path, text):
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text(text, encoding="utf-8")
    return p


def test_scalar_openai_section_aborts_cleanly(monkeypatch, tmp_path):
    """`openai: gpt-4o` (the block flattened to a scalar) aborts instead of AttributeError."""
    _write(monkeypatch, tmp_path, "vault_path: /tmp/v\nopenai: gpt-4o\n")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_scalar_chromadb_section_aborts_cleanly(monkeypatch, tmp_path):
    """`chromadb: /some/path` (the two-line block collapsed to one) aborts, not AttributeError."""
    _write(monkeypatch, tmp_path, "vault_path: /tmp/v\nchromadb: /tmp/db\n")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_list_embedding_section_aborts_cleanly(monkeypatch, tmp_path):
    """a section written as a list aborts cleanly."""
    _write(monkeypatch, tmp_path, "embedding:\n  - a\n  - b\n")
    with pytest.raises(typer.Exit):
        config.load_config()


def test_null_section_still_falls_back_to_defaults(monkeypatch, tmp_path):
    """a present-but-empty section is not an error: it uses defaults (no regression)."""
    _write(monkeypatch, tmp_path, "vault_path: /tmp/v\nopenai:\n")
    cfg = config.load_config()
    assert cfg.openai.chat_model == "gpt-4o"
    assert cfg.openai.embedding_model == "text-embedding-3-small"


def test_non_string_path_does_not_crash(monkeypatch, tmp_path):
    """a bare number for a path key coerces to a string Path instead of raising TypeError."""
    _write(monkeypatch, tmp_path, "vault_path: 2024\n")
    cfg = config.load_config()
    assert str(cfg.vault_path).endswith("2024")
