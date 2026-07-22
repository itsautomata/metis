"""auto-categorization — suggests which folder a note belongs in.

two-signal blended classifier:
  1. semantic matching: compare note embedding against folder identity embeddings
  2. KNN voting: find nearest existing notes, count their folders

the blend shifts from semantic (early, few notes) to KNN (later, many notes)
as your vault grows.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from metis import config as _cfg
from metis.config import MetisConfig, vault_folders
from metis.index.embed import embed_texts
from metis.index.store import get_collection, query_collection
from metis.textio import read_note_text

# --- data storage ---

def _default_categorization() -> dict:
    return {"folder_descriptions": {}, "folder_embeddings": {}, "feedback": []}


def _load_categorization(config: MetisConfig) -> dict:
    """this vault's categorization slice: folder descriptions, embeddings, and feedback.

    a truncated write (an interrupted save of this multi-MB file) falls back to defaults instead of
    bricking ingest; the next save rewrites it.
    """
    _cfg.migrate_state(config)
    slice_ = _cfg.read_json(_cfg.CATEGORIZATION_PATH).get(_cfg.vault_key(config.vault_path))
    return slice_ if isinstance(slice_, dict) else _default_categorization()


def _save_categorization(data: dict, config: MetisConfig) -> None:
    """persist this vault's slice, leaving other vaults' slices in the same file untouched."""
    all_data = _cfg.read_json(_cfg.CATEGORIZATION_PATH)
    all_data[_cfg.vault_key(config.vault_path)] = data
    _cfg.write_json(_cfg.CATEGORIZATION_PATH, all_data)


def clear_folder_embeddings(config: MetisConfig) -> None:
    """drop cached folder embeddings so they recompute; called when the embedding model changes."""
    data = _load_categorization(config)
    if data.get("folder_embeddings"):
        data["folder_embeddings"] = {}
        _save_categorization(data, config)


# --- folder descriptions ---

def _auto_describe_folder(folder: str, config: MetisConfig) -> str:
    """generate a description from frontmatter of existing notes.

    extracts tags and summaries — the metadata metis already generated.
    """
    import yaml

    folder_path = config.vault_path / folder
    if not folder_path.exists():
        return folder.replace("_", " ").replace("-", " ")

    all_tags = set()
    summaries = []

    for md in sorted(folder_path.glob("*.md"))[:10]:
        text = read_note_text(md)
        if not text.startswith("---"):
            continue

        parts = text.split("---", 2)
        if len(parts) < 3:
            continue

        try:
            fm = yaml.safe_load(parts[1])
            if not fm:
                continue
            tags = fm.get("tags", [])
            if isinstance(tags, list):
                all_tags.update(str(t) for t in tags)
            summary = fm.get("summary", "")
            if summary and isinstance(summary, str):
                summaries.append(summary[:150])
        except Exception:
            continue

    parts = [folder.replace("_", " ").replace("-", " ")]
    if all_tags:
        parts.append(f"topics: {', '.join(sorted(all_tags))}")
    if summaries:
        parts.append(" | ".join(summaries))

    return ". ".join(parts)


def get_folder_embeddings(config: MetisConfig) -> dict[str, list[float]]:
    """get or compute embeddings for all vault folders.

    uses cached embeddings if available, recomputes for new folders.
    """
    data = _load_categorization(config)
    cached = data.get("folder_embeddings", {})
    descriptions = data.get("folder_descriptions", {})

    folders = vault_folders(config)
    if not folders:
        return {}

    # find folders that need embedding
    to_embed = []
    to_embed_names = []
    for f in folders:
        if f not in cached:
            desc = descriptions.get(f) or _auto_describe_folder(f, config)
            descriptions[f] = desc
            to_embed.append(desc)
            to_embed_names.append(f)

    # batch embed new folders
    if to_embed:
        new_embeddings = embed_texts(to_embed, config)
        for name, emb in zip(to_embed_names, new_embeddings):
            cached[name] = emb

    # save updated cache
    data["folder_embeddings"] = cached
    data["folder_descriptions"] = descriptions
    _save_categorization(data, config)

    # return only current folders (prune deleted ones)
    return {f: cached[f] for f in folders if f in cached}


# --- signal 1: semantic matching ---

def semantic_scores(note_embedding: list[float], folder_embeddings: dict[str, list[float]]) -> dict[str, float]:
    """cosine similarity between a note and each folder's identity embedding.

    returns {folder: score} sorted by score descending.
    """
    if not folder_embeddings:
        return {}

    folders = list(folder_embeddings.keys())
    folder_matrix = np.array([folder_embeddings[f] for f in folders])
    note_vector = np.array([note_embedding])

    similarities = cosine_similarity(note_vector, folder_matrix)[0]

    scores = {f: round(float(s), 4) for f, s in zip(folders, similarities)}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# --- signal 2: KNN voting ---

KNN_OVERFETCH = 10


def _tally_folder_votes(ordered_file_paths: list[str], vault: Path, k: int) -> dict[str, int]:
    """collapse nearest-first chunk hits to distinct notes, one vote per note."""
    seen: set[str] = set()
    votes: dict[str, int] = {}
    for file_path in ordered_file_paths:
        if not file_path or file_path in seen:
            continue
        try:
            rel = Path(file_path).relative_to(vault)
        except ValueError:
            continue
        seen.add(file_path)
        folder = str(rel.parent)
        if folder == ".":
            folder = "metis-ingested"
        votes[folder] = votes.get(folder, 0) + 1
        if len(seen) >= k:
            break
    return votes


def knn_scores(note_embedding: list[float], config: MetisConfig, k: int = 5) -> dict[str, float]:
    """find the k nearest notes in chromadb, count their folders.

    chunks are collapsed to their parent note, so a note's size does not inflate
    its vote. returns {folder: vote_fraction} sorted by votes descending.
    """
    collection = get_collection(config)
    count = collection.count()
    if count == 0:
        return {}

    results = query_collection(
        collection,
        config,
        query_embeddings=[note_embedding],
        n_results=min(count, k * KNN_OVERFETCH),
    )

    metadatas = results["metadatas"][0] if results["metadatas"] else []
    ordered = [meta.get("file_path", "") for meta in metadatas]
    folder_votes = _tally_folder_votes(ordered, config.vault_path, k)

    total = sum(folder_votes.values())
    if total == 0:
        return {}

    scores = {f: round(v / total, 4) for f, v in folder_votes.items()}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# --- blend ---

def _avg_notes_per_folder(config: MetisConfig) -> float:
    """average number of notes per folder in the vault."""
    folders = vault_folders(config)
    if not folders:
        return 0

    counts = []
    for f in folders:
        folder_path = config.vault_path / f
        count = len(list(folder_path.glob("*.md")))
        if count > 0:
            counts.append(count)

    return sum(counts) / len(counts) if counts else 0


def _blend_weights(config: MetisConfig) -> tuple[float, float]:
    """determine semantic vs KNN weights based on vault maturity.

    few notes → trust semantic. many notes → trust KNN.
    """
    avg = _avg_notes_per_folder(config)

    if avg < 5:
        return 0.8, 0.2
    elif avg < 15:
        return 0.5, 0.5
    else:
        return 0.2, 0.8


def suggest_folder(
    note_embedding: list[float],
    config: MetisConfig,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """suggest folders for a note using blended classifier.

    returns list of (folder, confidence) tuples, sorted by confidence.
    """
    # get both signals
    folder_embs = get_folder_embeddings(config)
    sem_scores = semantic_scores(note_embedding, folder_embs)
    knn_scores_dict = knn_scores(note_embedding, config)

    if not sem_scores:
        return []

    # blend
    w_sem, w_knn = _blend_weights(config)

    all_folders = set(list(sem_scores.keys()) + list(knn_scores_dict.keys()))
    blended = {}

    for f in all_folders:
        sem = sem_scores.get(f, 0)
        knn = knn_scores_dict.get(f, 0)
        blended[f] = round(w_sem * sem + w_knn * knn, 4)

    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# --- feedback ---

def record_feedback(note_path: str, suggested_folder: str, actual_folder: str, config: MetisConfig) -> None:
    """record whether the suggestion was accepted or overridden."""
    data = _load_categorization(config)
    data.setdefault("feedback", []).append({
        "note": note_path,
        "suggested": suggested_folder,
        "actual": actual_folder,
        "accepted": suggested_folder == actual_folder,
    })
    _save_categorization(data, config)
