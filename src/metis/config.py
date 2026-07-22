"""configuration loading for metis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import yaml

CONFIG_DIR = Path.home() / ".metis"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "vault_path": str(Path.home() / "obsidian" / "vault"),
    "output_folder": "metis-ingested",
    "openai": {
        "base_url": "",
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o",
    },
    "chromadb": {
        "path": str(CONFIG_DIR / "chromadb"),
    },
}


@dataclass
class OpenAIConfig:
    base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"


@dataclass
class EmbeddingConfig:
    """optional override to run embeddings on a different provider than chat.

    inactive unless base_url is set; then embeddings use this endpoint instead of openai's.
    """
    base_url: str = ""
    model: str = ""


@dataclass
class MetisConfig:
    vault_path: Path = field(default_factory=lambda: Path.home() / "obsidian" / "vault")
    output_folder: str = "metis-ingested"
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chromadb_path: Path = field(default_factory=lambda: CONFIG_DIR / "chromadb")


def _default_config_yaml() -> str:
    """the starter config with inline guidance. values come from DEFAULT_CONFIG so they can't drift."""
    d = DEFAULT_CONFIG
    o = d["openai"]
    return f"""\
# metis config. api keys live in the OS keychain (`metis secret set`), not here.

vault_path: {d["vault_path"]}
output_folder: {d["output_folder"]}

openai:
  # base_url points at any OpenAI-compatible provider. empty = OpenAI. other providers:
  #   openrouter: https://openrouter.ai/api/v1
  #   ollama:     http://localhost:11434/v1   (local; key can be anything)
  # put that provider's key in the provider-key slot: `metis secret set provider-key`
  base_url: "{o["base_url"]}"

  # chat/summary model. provider-specific id (gpt-4o, anthropic/claude-3.5-sonnet, llama3.1).
  # must support JSON mode, or summaries and tags come back empty.
  chat_model: {o["chat_model"]}

  # embedding model. changing this re-spaces the whole index:
  # metis refuses search/link/health until you run `metis reindex`.
  embedding_model: {o["embedding_model"]}

# optional: run embeddings on a DIFFERENT provider than chat (e.g. chat via
# openrouter, embeddings direct on openai). omit this block to embed via `openai` above.
# embedding:
#   base_url: https://api.openai.com/v1
#   model: text-embedding-3-small
#   # key: `metis secret set embedding-key` (falls back to provider-key if unset)

chromadb:
  path: {d["chromadb"]["path"]}
"""


def _config_error(detail: str) -> NoReturn:
    """abort with a clean message when the config file is unusable."""
    import typer
    from rich.console import Console

    console = Console()
    console.print(f"[red]✗ {CONFIG_PATH} is not a usable config.[/red]")
    console.print(f"[dim]{detail}[/dim]")
    console.print(f"[dim]fix it, or delete it to regenerate defaults: rm {CONFIG_PATH}[/dim]")
    raise typer.Exit(1)


def _require_mapping(value, name: str) -> dict:
    """a present-but-null section falls back to defaults; a scalar or list is a clear mistake."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        _config_error(f"the `{name}:` section must be a mapping of key: value, not a {type(value).__name__}.")
    return value


def init_config() -> Path:
    """create or update config file with any missing sections. returns config path."""
    import os
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as f:
            f.write(_default_config_yaml())
        os.chmod(CONFIG_PATH, 0o600)  # owner-only read/write
    else:
        # merge missing keys into existing config
        with open(CONFIG_PATH) as f:
            try:
                existing = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                _config_error(f"it is not valid YAML: {e}")
        if not isinstance(existing, dict):
            _config_error(f"its top level must be a mapping of key: value, but it parsed as a {type(existing).__name__}.")

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
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            _config_error(f"it is not valid YAML: {e}")
    if not isinstance(raw, dict):
        _config_error(f"its top level must be a mapping of key: value, but it parsed as a {type(raw).__name__}.")

    # `.get(k) or default` (not `.get(k, default)`) so a present-but-null value
    # (`embedding_model:` with nothing after it) falls back instead of becoming None.
    openai_raw = _require_mapping(raw.get("openai"), "openai")
    openai_cfg = OpenAIConfig(
        base_url=openai_raw.get("base_url") or "",
        embedding_model=openai_raw.get("embedding_model") or "text-embedding-3-small",
        chat_model=openai_raw.get("chat_model") or "gpt-4o",
    )

    embedding_raw = _require_mapping(raw.get("embedding"), "embedding")
    embedding_cfg = EmbeddingConfig(
        base_url=embedding_raw.get("base_url") or "",
        model=embedding_raw.get("model") or "",
    )

    chromadb_raw = _require_mapping(raw.get("chromadb"), "chromadb")

    return MetisConfig(
        vault_path=Path(str(raw.get("vault_path") or DEFAULT_CONFIG["vault_path"])).expanduser(),
        output_folder=str(raw.get("output_folder") or "metis-ingested"),
        openai=openai_cfg,
        embedding=embedding_cfg,
        chromadb_path=Path(str(chromadb_raw.get("path") or (CONFIG_DIR / "chromadb"))).expanduser(),
    )


def vault_folders(config: MetisConfig) -> list[str]:
    """vault subfolders as sorted relative paths, excluding symlinks that escape the vault."""
    vault = config.vault_path
    if not vault.exists():
        return []
    vault_resolved = vault.resolve()
    return sorted(
        str(p.relative_to(vault))
        for p in vault.rglob("*")
        if p.is_dir()
        and not p.name.startswith(".")
        and p.resolve().is_relative_to(vault_resolved)
    )
