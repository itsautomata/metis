"""vault sync — re-index changed, new, and deleted files."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from metis.config import MetisConfig, CONFIG_DIR
from metis.ingest.process import chunk_text
from metis.index.store import store_chunks, _get_collection

SYNC_STATE_PATH = CONFIG_DIR / "sync_state.json"


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    total_files: int = 0
    total_chunks: int = 0


def _file_hash(path: Path) -> str:
    """fast hash of file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_sync_state() -> dict[str, str]:
    """load previous sync state: {file_path: hash}."""
    if SYNC_STATE_PATH.exists():
        return json.loads(SYNC_STATE_PATH.read_text())
    return {}


def _save_sync_state(state: dict[str, str]) -> None:
    """persist sync state."""
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(json.dumps(state, indent=2))


def _find_vault_files(config: MetisConfig) -> list[Path]:
    """find all markdown files in the vault."""
    return sorted(config.vault_path.rglob("*.md"))


def _remove_file_from_index(file_path: str, config: MetisConfig) -> int:
    """remove all chunks for a file from chromadb. returns count removed."""
    collection = _get_collection(config)

    # find all chunk IDs for this file
    results = collection.get(
        where={"file_path": file_path},
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


def sync_vault(config: MetisConfig) -> SyncReport:
    """sync the vault with chromadb. returns a report of what changed."""
    report = SyncReport()
    old_state = _load_sync_state()
    new_state = {}

    vault_files = _find_vault_files(config)
    current_paths = set()

    for path in vault_files:
        file_key = str(path)
        current_paths.add(file_key)
        current_hash = _file_hash(path)
        new_state[file_key] = current_hash

        if file_key not in old_state:
            # new file
            text = path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            n = store_chunks(chunks, path, config)
            report.added += 1
            report.total_chunks += n

        elif old_state[file_key] != current_hash:
            # changed file — remove old chunks, add new
            _remove_file_from_index(file_key, config)
            text = path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            n = store_chunks(chunks, path, config)
            report.updated += 1
            report.total_chunks += n

        else:
            # unchanged
            report.unchanged += 1

    # deleted files — in old state but not in current vault
    for old_path in old_state:
        if old_path not in current_paths:
            _remove_file_from_index(old_path, config)
            report.deleted += 1

    report.total_files = len(vault_files)
    _save_sync_state(new_state)

    return report
