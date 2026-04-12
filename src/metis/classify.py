"""auto-categorization — suggests which folder a note belongs in.

two-signal blended classifier:
  1. semantic matching: compare note embedding against folder identity embeddings
  2. KNN voting: find nearest existing notes, count their folders

the blend shifts from semantic (early, few notes) to KNN (later, many notes)
as your vault grows.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

from metis.config import MetisConfig, CONFIG_DIR
from metis.index.embed import embed_texts
from metis.index.store import get_collection

CATEGORIZATION_PATH = CONFIG_DIR / "categorization.json"


# --- data storage ---

def _load_categorization() -> dict:
    if CATEGORIZATION_PATH.exists():
        return json.loads(CATEGORIZATION_PATH.read_text())
    return {"folder_descriptions": {}, "folder_embeddings": {}, "feedback": []}


def _save_categorization(data: dict) -> None:
    CATEGORIZATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATEGORIZATION_PATH.write_text(json.dumps(data, indent=2))


# --- folder descriptions ---

def _get_vault_folders(config: MetisConfig) -> list[str]:
    """list all folders in the vault (relative paths)."""
    vault = config.vault_path
    return sorted(
        str(p.relative_to(vault))
        for p in vault.rglob("*")
        if p.is_dir() and not p.name.startswith(".")
    )


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
        text = md.read_text(encoding="utf-8")
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
    data = _load_categorization()
    cached = data.get("folder_embeddings", {})
    descriptions = data.get("folder_descriptions", {})

    folders = _get_vault_folders(config)
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
    _save_categorization(data)

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

def knn_scores(note_embedding: list[float], config: MetisConfig, k: int = 5) -> dict[str, float]:
    """find k nearest notes in chromadb, count their folders.

    returns {folder: vote_fraction} sorted by votes descending.
    """
    collection = get_collection(config)
    if collection.count() == 0:
        return {}

    results = collection.query(
        query_embeddings=[note_embedding],
        n_results=min(k, collection.count()),
    )

    if not results["metadatas"] or not results["metadatas"][0]:
        return {}

    # count folder votes
    vault = config.vault_path
    folder_votes: dict[str, int] = {}

    for meta in results["metadatas"][0]:
        file_path = meta.get("file_path", "")
        try:
            rel = Path(file_path).relative_to(vault)
            folder = str(rel.parent)
            if folder == ".":
                folder = "metis-ingested"
            folder_votes[folder] = folder_votes.get(folder, 0) + 1
        except ValueError:
            continue

    total = sum(folder_votes.values())
    if total == 0:
        return {}

    scores = {f: round(v / total, 4) for f, v in folder_votes.items()}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# --- blend ---

def _avg_notes_per_folder(config: MetisConfig) -> float:
    """average number of notes per folder in the vault."""
    folders = _get_vault_folders(config)
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

def record_feedback(note_path: str, suggested_folder: str, actual_folder: str) -> None:
    """record whether the suggestion was accepted or overridden."""
    data = _load_categorization()
    data.setdefault("feedback", []).append({
        "note": note_path,
        "suggested": suggested_folder,
        "actual": actual_folder,
        "accepted": suggested_folder == actual_folder,
    })
    _save_categorization(data)
