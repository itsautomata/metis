"""vault sync: re-index changed, new, and deleted files."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb

from metis import config as _cfg
from metis.config import MetisConfig
from metis.index.store import get_collection, store_chunks
from metis.ingest.process import chunk_text
from metis.textio import read_note_text


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped: int = 0
    total_files: int = 0
    total_chunks: int = 0


def _file_hash(path: Path) -> str:
    """fast hash of file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_sync_state(config: MetisConfig) -> dict[str, str]:
    """this vault's sync state: {file_path: hash}. tolerates a missing or corrupt file."""
    _cfg.migrate_state(config)
    slice_ = _cfg.read_json(_cfg.SYNC_STATE_PATH).get(_cfg.vault_key(config.vault_path), {})
    return slice_ if isinstance(slice_, dict) else {}


def _save_sync_state(state: dict[str, str], config: MetisConfig) -> None:
    """persist this vault's slice, leaving other vaults' slices in the same file untouched."""
    data = _cfg.read_json(_cfg.SYNC_STATE_PATH)
    data[_cfg.vault_key(config.vault_path)] = state
    _cfg.write_json(_cfg.SYNC_STATE_PATH, data)


def mark_file_synced(file_path: Path, config: MetisConfig) -> None:
    """record a file's current content hash in the sync state.

    ingest calls this after storing a note so a later `metis sync` treats it as
    already indexed instead of re-embedding it. an edit changes the hash, so sync
    still re-embeds edited notes.
    """
    state = _load_sync_state(config)
    state[str(file_path)] = _file_hash(file_path)
    _save_sync_state(state, config)


def _find_vault_files(config: MetisConfig) -> list[Path]:
    """markdown files in the vault, skipping dot-directories (.obsidian, .trash) and dotfiles.

    a bare rglob would index Obsidian's deleted notes (.trash) and stray .md under .obsidian,
    resurfacing them in search and chat.
    """
    vault = config.vault_path
    return sorted(
        p for p in vault.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(vault).parts)
    )


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
    old_state = _load_sync_state(config)
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

        try:
            current_hash = _file_hash(path)

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
        except (FileNotFoundError, IsADirectoryError):
            # a broken symlink, or a file moved/deleted between the scan and now: leave it out of
            # current_paths so its stale vectors get pruned below, and move on.
            report.skipped += 1
            continue

        current_paths.add(file_key)
        new_state[file_key] = current_hash

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
    _save_sync_state(new_state, config)

    # baseline the drift canary for any non-empty index (idempotent; also covers a pre-existing
    # index whose sync is a no-op and would otherwise never get a baseline)
    if collection.count() > 0:
        from metis.index.canary import ensure_baseline
        ensure_baseline(config)

    return report


def reindex_vault(config: MetisConfig) -> SyncReport:
    """drop the index and re-embed the whole vault with the current embedding model.

    used after changing `embedding_model`: the old vectors live in a different space,
    so everything must be rebuilt. also drops state tied to the old space.
    """
    from metis.classify import clear_folder_embeddings
    from metis.index.store import collection_name

    # drop this vault's collection so its embedding-model stamp resets to the current model
    client = chromadb.PersistentClient(path=str(config.chromadb_path))
    try:
        client.delete_collection(collection_name(config))
    except Exception:
        pass  # nothing to drop on a fresh vault

    # forget this vault's sync state so every file re-embeds; drop the folder-embedding cache
    _save_sync_state({}, config)
    clear_folder_embeddings(config)
    # drop the drift baseline; sync_vault below re-captures it against the new model
    from metis.index.canary import reset as reset_canary
    reset_canary()

    return sync_vault(config)
