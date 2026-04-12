"""vault health analysis using DBSCAN clustering and KNN misplacement detection."""

import yaml
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from metis.config import MetisConfig
from metis.index.store import get_collection


# --- data extraction ---

def _extract_vault_data(config: MetisConfig) -> tuple[list[str], list[list[float]], list[str]]:
    """extract file paths, embeddings, and folder labels from chromadb.

    returns (file_paths, embeddings, folders)
    uses first chunk per file as representative embedding.
    """
    collection = get_collection(config)
    if collection.count() == 0:
        return [], [], []

    all_data = collection.get(include=["metadatas", "embeddings"])

    vault = config.vault_path
    seen_files: dict[str, list[float]] = {}

    for i, meta in enumerate(all_data["metadatas"]):
        fp = meta.get("file_path", "")
        if fp and fp not in seen_files:
            seen_files[fp] = all_data["embeddings"][i]

    file_paths = list(seen_files.keys())
    embeddings = [seen_files[fp] for fp in file_paths]

    folders = []
    for fp in file_paths:
        try:
            rel = Path(fp).relative_to(vault)
            folder = str(rel.parent)
            if folder == ".":
                folder = "(root)"
            folders.append(folder)
        except ValueError:
            folders.append("(unknown)")

    return file_paths, embeddings, folders


# --- cluster labeling ---

def _label_cluster(file_paths: list[str]) -> str:
    """derive a topic label from frontmatter tags of cluster members.

    fallback: tags -> summaries -> note titles.
    """
    tag_counts: Counter = Counter()
    summaries = []

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            fm = yaml.safe_load(parts[1])
            if not fm:
                continue
            if isinstance(fm.get("tags"), list):
                tag_counts.update(str(t) for t in fm["tags"])
            summary = fm.get("summary", "")
            if summary and isinstance(summary, str):
                summaries.append(summary[:50])
        except Exception:
            continue

    if tag_counts:
        return ", ".join(tag for tag, _ in tag_counts.most_common(3))

    if summaries:
        return summaries[0]

    names = [Path(fp).stem[:25] for fp in file_paths[:3]]
    return " / ".join(names)


def _suggest_folder_name(label: str) -> str:
    """turn a cluster label into a suggested folder name.

    takes the top tag and makes it filesystem-friendly.
    """
    top_tag = label.split(",")[0].strip()
    return top_tag.replace(" ", "-").lower()


# --- DBSCAN clustering ---

@dataclass
class ClusterInfo:
    cluster_id: int
    label: str
    folder_name: str
    members: list[tuple[str, str]]  # (file_path, folder)
    size: int


@dataclass
class FolderHealth:
    folder: str
    total: int
    status: str  # tight, mixed, scattered
    topics: list[ClusterInfo]  # clusters this folder spans


@dataclass
class Misplaced:
    file_path: str
    current_folder: str
    suggested_folder: str
    neighbor_count: int  # how many of 5 neighbors are in suggested folder


@dataclass
class HealthReport:
    folders: list[FolderHealth]
    misplaced: list[Misplaced]
    unique: list[tuple[str, str]]  # (file_path, folder) — true outliers
    split_folders: list[str]  # folders that span 2+ topics
    n_notes: int
    n_clusters: int


def run_health(config: MetisConfig) -> HealthReport:
    """full vault health analysis."""
    file_paths, embeddings, folders = _extract_vault_data(config)

    if len(file_paths) < 2:
        return HealthReport([], [], [], [], len(file_paths), 0)

    emb_array = np.array(embeddings)
    distance_matrix = cosine_distances(emb_array)

    # --- DBSCAN for topic detection ---
    flat_distances = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    eps = float(np.percentile(flat_distances, 25))
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(distance_matrix)

    # organize clusters
    clusters: dict[int, list[int]] = defaultdict(list)  # cluster_id -> indices
    outlier_indices: list[int] = []

    for i, label in enumerate(labels):
        if label == -1:
            outlier_indices.append(i)
        else:
            clusters[label].append(i)

    # label each cluster
    cluster_infos: dict[int, ClusterInfo] = {}
    for cid, indices in clusters.items():
        fps = [file_paths[i] for i in indices]
        members = [(file_paths[i], folders[i]) for i in indices]
        label = _label_cluster(fps)
        cluster_infos[cid] = ClusterInfo(
            cluster_id=cid,
            label=label,
            folder_name=_suggest_folder_name(label),
            members=members,
            size=len(indices),
        )

    # --- folder health ---
    folder_notes: dict[str, list[int]] = defaultdict(list)
    for i, folder in enumerate(folders):
        folder_notes[folder].append(i)

    folder_healths = []
    split_folders = []

    for folder in sorted(folder_notes.keys()):
        indices = folder_notes[folder]
        total = len(indices)

        # which clusters do this folder's notes appear in?
        folder_cluster_counts: Counter = Counter()
        for i in indices:
            if labels[i] != -1:
                folder_cluster_counts[labels[i]] += 1

        topics = [cluster_infos[cid] for cid in folder_cluster_counts if cid in cluster_infos]

        # determine status
        n_significant = len([c for c, n in folder_cluster_counts.items() if n >= 2])
        if total == 1:
            status = "—"
        elif n_significant <= 1:
            status = "tight"
        elif n_significant == 2:
            status = "mixed"
            split_folders.append(folder)
        else:
            status = "scattered"
            split_folders.append(folder)

        folder_healths.append(FolderHealth(
            folder=folder,
            total=total,
            status=status,
            topics=topics,
        ))

    # --- KNN misplacement detection ---
    misplaced = []
    k = 5

    for i in range(len(file_paths)):
        my_folder = folders[i]
        # skip single-note folders
        if len(folder_notes[my_folder]) < 2:
            continue

        distances = distance_matrix[i]
        nearest = np.argsort(distances)[1:k+1]  # skip self (index 0)

        neighbor_folders = [folders[j] for j in nearest]
        folder_counts = Counter(neighbor_folders)

        # if majority of neighbors are in a different folder
        for other_folder, count in folder_counts.most_common(1):
            if other_folder != my_folder and count >= 4:
                misplaced.append(Misplaced(
                    file_path=file_paths[i],
                    current_folder=my_folder,
                    suggested_folder=other_folder,
                    neighbor_count=count,
                ))

    # --- unique notes (DBSCAN outliers in multi-note folders) ---
    unique = []
    for i in outlier_indices:
        folder = folders[i]
        if len(folder_notes[folder]) >= 2:
            unique.append((file_paths[i], folder))

    return HealthReport(
        folders=folder_healths,
        misplaced=misplaced,
        unique=unique,
        split_folders=split_folders,
        n_notes=len(file_paths),
        n_clusters=len(clusters),
    )


def analyze_split(folder: str, config: MetisConfig) -> list[ClusterInfo] | None:
    """analyze a specific folder for potential splitting.

    runs KMeans(k=2) on just this folder's embeddings, independent of global clustering.
    returns two ClusterInfo groups, or None if folder is too small.
    """
    from sklearn.cluster import KMeans

    file_paths, embeddings, folders = _extract_vault_data(config)

    # filter to just this folder
    indices = [i for i, f in enumerate(folders) if f == folder]
    if len(indices) < 4:
        return None

    folder_fps = [file_paths[i] for i in indices]
    folder_embs = np.array([embeddings[i] for i in indices])

    # run KMeans with k=2
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(folder_embs)

    # organize into two groups
    groups: dict[int, list[str]] = defaultdict(list)
    for fp, label in zip(folder_fps, labels):
        groups[label].append(fp)

    result = []
    for group_id, fps in groups.items():
        label = _label_cluster(fps)
        result.append(ClusterInfo(
            cluster_id=group_id,
            label=label,
            folder_name=_suggest_folder_name(label),
            members=[(fp, folder) for fp in fps],
            size=len(fps),
        ))

    return result
