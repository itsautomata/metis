"""tests for configuration loading."""

import yaml
from pathlib import Path

from metis.config import MetisConfig, load_config, init_config, CONFIG_DIR


def test_default_config_values():
    config = MetisConfig()
    assert config.provider == "openai"
    assert config.output_folder == "metis-ingested"
    assert config.openai.chat_model == "gpt-4o"
    assert config.openai.embedding_model == "text-embedding-3-small"
    assert config.azure.chat_model == "gpt-4o"


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
    assert "provider" in raw
    assert "openai" in raw
    assert "azure_openai" in raw


def test_init_does_not_overwrite(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("custom: true")
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    init_config()
    assert config_path.read_text() == "custom: true"


def test_load_config_openai(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "vault_path": str(tmp_path / "vault"),
        "provider": "openai",
        "openai": {"api_key": "sk-test", "chat_model": "gpt-4o"},
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert config.provider == "openai"
    assert config.openai.api_key == "sk-test"


def test_load_config_azure(tmp_path, monkeypatch):
    config_dir = tmp_path / ".metis"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "vault_path": str(tmp_path / "vault"),
        "provider": "azure",
        "azure_openai": {
            "endpoint": "https://test.openai.azure.com/",
            "api_key": "azure-test",
        },
    }))
    monkeypatch.setattr("metis.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("metis.config.CONFIG_PATH", config_path)

    config = load_config()
    assert config.provider == "azure"
    assert config.azure.api_key == "azure-test"
    assert config.azure.endpoint == "https://test.openai.azure.com/"
