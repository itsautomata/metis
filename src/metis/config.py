"""configuration loading for metis."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import yaml

CONFIG_DIR = Path.home() / ".metis"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

# vault-scoped sidecars. each file holds one slice per vault, keyed by vault_key(); chromadb is
# partitioned by collection name the same way. the canary is not here: it is model-scoped, global.
SOURCES_INDEX_PATH = CONFIG_DIR / "sources.json"
SYNC_STATE_PATH = CONFIG_DIR / "sync_state.json"
CATEGORIZATION_PATH = CONFIG_DIR / "categorization.json"

# parks pre-per-vault (flat) sidecar state until its owning vault reclaims it
LEGACY_STATE_KEY = "__legacy__"

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
    link_style: str = ""  # "" auto-detects from the vault; "wikilink" / "markdown" force it


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
    from rich.markup import escape

    from metis.ui import err_console

    err_console.print(f"[err]✗ {escape(str(CONFIG_PATH))} is not a usable config.[/err]")
    err_console.print(f"[muted]{escape(detail)}[/muted]")
    err_console.print(f"[muted]fix it, or delete it to regenerate defaults: rm {escape(str(CONFIG_PATH))}[/muted]")
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

    link_style = raw.get("link_style") or ""
    if link_style not in ("wikilink", "markdown"):
        link_style = ""  # anything else (or absent) means auto-detect

    return MetisConfig(
        vault_path=Path(str(raw.get("vault_path") or DEFAULT_CONFIG["vault_path"])).expanduser(),
        output_folder=str(raw.get("output_folder") or "metis-ingested"),
        openai=openai_cfg,
        embedding=embedding_cfg,
        chromadb_path=Path(str(chromadb_raw.get("path") or (CONFIG_DIR / "chromadb"))).expanduser(),
        link_style=link_style,
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


# --- per-vault state: key, sidecar io, and the one-time migration off the old global layout ---

def vault_key(vault_path: Path) -> str:
    """a stable, collection-name-safe key for a vault, so per-vault state never collides."""
    canonical = str(Path(vault_path).expanduser())
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def read_json(path: Path) -> dict:
    """load a JSON sidecar, tolerating a missing or corrupt file (returns {})."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict) -> None:
    """persist a JSON sidecar atomically so an interrupted write can't corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _is_flat_map(data: dict) -> bool:
    """a pre-per-vault sources/sync_state file: a flat map whose values are not per-vault dicts."""
    return bool(data) and all(not isinstance(v, dict) for v in data.values())


def _is_flat_categorization(data: dict) -> bool:
    return "folder_descriptions" in data or "folder_embeddings" in data or "feedback" in data


def legacy_owned_by(config: MetisConfig) -> bool:
    """the pre-per-vault global state belongs to this vault iff the legacy sync_state's file paths
    are all inside it."""
    data = read_json(SYNC_STATE_PATH)
    payload = data if _is_flat_map(data) else data.get(LEGACY_STATE_KEY)
    if not isinstance(payload, dict) or not payload:
        return False
    vault = str(config.vault_path)
    return all(str(p).startswith(vault) for p in payload)


def migrate_state(config: MetisConfig) -> None:
    """fold pre-per-vault flat sidecars into a vault-keyed shape, once. adopt them for this vault
    when its sync_state proves ownership; otherwise park them under LEGACY_STATE_KEY so no per-vault
    write destroys them. idempotent and cheap once migrated. chromadb adoption lives in get_collection.
    """
    key = vault_key(config.vault_path)
    owned = legacy_owned_by(config)
    target = key if owned else LEGACY_STATE_KEY

    for path, is_flat in (
        (SOURCES_INDEX_PATH, _is_flat_map),
        (SYNC_STATE_PATH, _is_flat_map),
        (CATEGORIZATION_PATH, _is_flat_categorization),
    ):
        data = read_json(path)
        if is_flat(data):
            write_json(path, {target: data})
        elif owned and LEGACY_STATE_KEY in data and key not in data:
            data[key] = data[LEGACY_STATE_KEY]
            write_json(path, data)
