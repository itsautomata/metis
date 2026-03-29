"""configuration loading for metis."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


CONFIG_DIR = Path.home() / ".metis"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "vault_path": str(Path.home() / "obsidian" / "vault"),
    "output_folder": "metis-ingested",
    "provider": "openai",
    "openai": {
        "api_key": "",
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o",
    },
    "azure_openai": {
        "endpoint": "",
        "api_key": "",
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o",
    },
    "x_api": {
        "bearer_token": "",
    },
    "chromadb": {
        "path": str(CONFIG_DIR / "chromadb"),
    },
}


@dataclass
class OpenAIConfig:
    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"


@dataclass
class AzureConfig:
    endpoint: str = ""
    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"


@dataclass
class XApiConfig:
    bearer_token: str = ""


@dataclass
class MetisConfig:
    vault_path: Path = field(default_factory=lambda: Path.home() / "obsidian" / "vault")
    output_folder: str = "metis-ingested"
    provider: str = "openai"
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    azure: AzureConfig = field(default_factory=AzureConfig)
    x_api: XApiConfig = field(default_factory=XApiConfig)
    chromadb_path: Path = field(default_factory=lambda: CONFIG_DIR / "chromadb")


def init_config() -> Path:
    """create or update config file with any missing sections. returns config path."""
    import os
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        os.chmod(CONFIG_PATH, 0o600)  # owner-only read/write
    else:
        # merge missing keys into existing config
        with open(CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

        updated = False
        for key, value in DEFAULT_CONFIG.items():
            if key not in existing:
                existing[key] = value
                updated = True

        if updated:
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    return CONFIG_PATH


def load_config() -> MetisConfig:
    """load config from ~/.metis/config.yaml."""
    if not CONFIG_PATH.exists():
        init_config()

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    provider = raw.get("provider", "openai")

    openai_raw = raw.get("openai", {})
    openai_cfg = OpenAIConfig(
        api_key=openai_raw.get("api_key", ""),
        embedding_model=openai_raw.get("embedding_model", "text-embedding-3-small"),
        chat_model=openai_raw.get("chat_model", "gpt-4o"),
    )

    azure_raw = raw.get("azure_openai", {})
    azure_cfg = AzureConfig(
        endpoint=azure_raw.get("endpoint", ""),
        api_key=azure_raw.get("api_key", ""),
        embedding_model=azure_raw.get("embedding_model", "text-embedding-3-small"),
        chat_model=azure_raw.get("chat_model", "gpt-4o"),
    )

    x_raw = raw.get("x_api", {})
    x_cfg = XApiConfig(
        bearer_token=x_raw.get("bearer_token", ""),
    )

    chromadb_raw = raw.get("chromadb", {})

    return MetisConfig(
        vault_path=Path(raw.get("vault_path", DEFAULT_CONFIG["vault_path"])).expanduser(),
        output_folder=raw.get("output_folder", "metis-ingested"),
        provider=provider,
        openai=openai_cfg,
        azure=azure_cfg,
        x_api=x_cfg,
        chromadb_path=Path(chromadb_raw.get("path", str(CONFIG_DIR / "chromadb"))).expanduser(),
    )
