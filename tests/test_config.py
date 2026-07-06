"""tests for configuration loading."""


import yaml

from metis.config import (
    MetisConfig,
    init_config,
    load_config,
    vault_folders,
)


def test_vault_folders_excludes_symlink_escaping_vault(tmp_path):
    """a symlinked dir pointing outside the vault is not listed, so it can't be picked or suggested."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "real").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "escape").symlink_to(outside)

    folders = vault_folders(MetisConfig(vault_path=vault))

    assert "real" in folders
    assert "escape" not in folders


def test_vault_folders_missing_vault_returns_empty(tmp_path):
    assert vault_folders(MetisConfig(vault_path=tmp_path / "nope")) == []


def test_default_config_values():
    config = MetisConfig()
    assert config.output_folder == "metis-ingested"
    assert config.openai.chat_model == "gpt-4o"
    assert config.openai.embedding_model == "text-embedding-3-small"
    assert config.openai.base_url == ""


def test_init_creates_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_path = config_dir / "config.yaml"
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    result = init_config()
    assert result == config_path
    assert config_path.exists()

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    assert "openai" in raw
    assert "base_url" in raw["openai"]
    assert "api_key" not in raw["openai"]  # secrets live in the keychain, not the yaml
    assert "x_api" not in raw
    assert "azure_openai" not in raw


def test_init_preserves_existing_and_adds_missing(tmp_path, monkeypatch):
    """init should keep existing keys and add missing ones."""
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("custom: true")
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    init_config()
    text = config_path.read_text()
    assert "custom: true" in text
    assert "openai:" in text


def test_load_config_openai(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "vault_path": str(tmp_path / "vault"),
        "openai": {"chat_model": "gpt-4o"},
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert config.openai.chat_model == "gpt-4o"


def test_load_config_ignores_yaml_key(tmp_path, monkeypatch):
    """a key left in an old yaml is ignored: secrets live in the keychain, not the config."""
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "openai": {"api_key": "sk-should-be-ignored"},
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert not hasattr(config.openai, "api_key")


def test_load_config_base_url(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "vault_path": str(tmp_path / "vault"),
        "openai": {"base_url": "https://openrouter.ai/api/v1"},
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert config.openai.base_url == "https://openrouter.ai/api/v1"


def test_load_config_legacy_azure_ignored(tmp_path, monkeypatch):
    """a stale provider: azure and azure_openai block don't break loading; they're ignored."""
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "vault_path": str(tmp_path / "vault"),
        "provider": "azure",
        "azure_openai": {"endpoint": "https://old.azure.com/", "api_key": "x"},
        "openai": {"chat_model": "gpt-4o"},
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert config.openai.chat_model == "gpt-4o"
    assert not hasattr(config, "provider")
