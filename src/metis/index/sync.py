"""vault sync: re-index changed, new, and deleted files."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb

from metis.config import CONFIG_DIR, MetisConfig
from metis.index.store import get_collection, store_chunks
from metis.ingest.process import chunk_text
from metis.textio import read_note_text

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
    """load previous sync state: {file_path: hash}. tolerates a missing or corrupt file.
    """
    if not SYNC_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(SYNC_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_sync_state(state: dict[str, str]) -> None:
    """persist sync state atomically so an interrupted write can't corrupt it."""
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SYNC_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(SYNC_STATE_PATH)


def mark_file_synced(file_path: Path) -> None:
    """record a file's current content hash in the sync state.

    ingest calls this after storing a note so a later `metis sync` treats it as
    already indexed instead of re-embedding it. an edit in obsidian changes the
    hash, so sync still re-embeds edited notes.
    """
    state = _load_sync_state()
    state[str(file_path)] = _file_hash(file_path)
    _save_sync_state(state)


def _find_vault_files(config: MetisConfig) -> list[Path]:
    """find all markdown files in the vault."""
    return sorted(config.vault_path.rglob("*.md"))


def _remove_file_from_index(file_path: str, config: MetisConfig) -> int:
    """remove all chunks for a file from chromadb. returns count removed."""
    collection = get_collection(config)

    # find all chunk IDs for this file
    results = collection.get(
        where={"file_path": file_path},
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


class EmptyVaultError(Exception):
    """the vault resolved to zero files while the index still holds notes.

    likely an unmounted drive or a wrong vault_path, not an intentional wipe.
    """

    def __init__(self, vault_path: Path, reason: str, indexed: int) -> None:
        self.vault_path = vault_path
        super().__init__(f"{reason}: {vault_path}. syncing would delete all {indexed} indexed notes.")


def sync_vault(
    config: MetisConfig,
    on_progress: Callable[[int, int, str], None] | None = None,
    force: bool = False,
) -> SyncReport:
    """sync the vault with chromadb. returns a report of what changed.

    on_progress, if given, is called as on_progress(done, total, filename) as each
    file is processed, so a caller can render progress.
    """
    report = SyncReport()
    old_state = _load_sync_state()
    new_state = {}

    vault_files = _find_vault_files(config)
    total = len(vault_files)

    if not vault_files and not force:
        # zero files can mean "the vault was emptied" OR "vault_path is wrong / unmounted".
        # if the index still holds notes, refuse to wipe it on this ambiguous empty result.
        collection = get_collection(config)
        if collection.count() > 0:
            reason = "vault path does not exist" if not config.vault_path.exists() else "vault has no markdown files"
            raise EmptyVaultError(config.vault_path, reason, collection.count())

    current_paths = set()

    for i, path in enumerate(vault_files):
        if on_progress:
            on_progress(i, total, path.name)
        file_key = str(path)
        current_paths.add(file_key)
        current_hash = _file_hash(path)
        new_state[file_key] = current_hash

        if file_key not in old_state:
            # new file
            text = read_note_text(path)
            chunks = chunk_text(text)
            n = store_chunks(chunks, path, config)
            report.added += 1
            report.total_chunks += n

        elif old_state[file_key] != current_hash:
            # changed file — remove old chunks, add new
            _remove_file_from_index(file_key, config)
            text = read_note_text(path)
            chunks = chunk_text(text)
            n = store_chunks(chunks, path, config)
            report.updated += 1
            report.total_chunks += n

        else:
            # unchanged
            report.unchanged += 1

    if on_progress:
        on_progress(total, total, "")

    # deleted files — in old state but not in current vault
    for old_path in old_state:
        if old_path not in current_paths:
            _remove_file_from_index(old_path, config)
            report.deleted += 1

    # also clean chromadb of orphaned entries (ingested before sync tracking)
    collection = get_collection(config)
    all_meta = collection.get(include=["metadatas"])
    indexed_paths = set()
    for meta in all_meta["metadatas"]:
        fp = meta.get("file_path", "")
        if fp:
            indexed_paths.add(fp)

    for indexed_path in indexed_paths:
        if indexed_path not in current_paths:
            _remove_file_from_index(indexed_path, config)
            report.deleted += 1

    report.total_files = len(vault_files)
    _save_sync_state(new_state)

    return report


def reindex_vault(config: MetisConfig) -> SyncReport:
    """drop the index and re-embed the whole vault with the current embedding model.

    used after changing `embedding_model`: the old vectors live in a different space,
    so everything must be rebuilt. also drops state tied to the old space.
    """
    from metis.classify import clear_folder_embeddings
    from metis.index.store import COLLECTION_NAME

    # drop the collection so its embedding-model stamp resets to the current model
    client = chromadb.PersistentClient(path=str(config.chromadb_path))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to drop on a fresh vault

    # forget sync state so every file re-embeds; drop the folder-embedding cache (old space)
    if SYNC_STATE_PATH.exists():
        SYNC_STATE_PATH.unlink()
    clear_folder_embeddings()

    return sync_vault(config)
