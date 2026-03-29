"""semantic search across the vault."""

from dataclasses import dataclass
from pathlib import Path

from metis.config import MetisConfig
from metis.index.embed import embed_texts
from metis.index.store import get_collection


@dataclass
class SearchResult:
    text: str
    file_path: str
    chunk_index: int
    score: float


def search_vault(
    query: str,
    config: MetisConfig,
    limit: int = 5,
    note_path: str | None = None,
) -> list[SearchResult]:
    """embed query and find nearest chunks in chromadb.

    if note_path is given, only search within that note's chunks.
    """
    collection = get_collection(config)

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query], config)[0]

    where_filter = {"file_path": note_path} if note_path else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, collection.count()),
        where=where_filter,
    )

    search_results = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i] if results["distances"] else 0
        score = 1 - distance  # cosine distance → similarity

        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        text = results["documents"][0][i] if results["documents"] else ""

        search_results.append(SearchResult(
            text=text,
            file_path=metadata.get("file_path", ""),
            chunk_index=metadata.get("chunk_index", 0),
            score=round(score, 3),
        ))

    return search_results
