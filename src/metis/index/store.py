"""ChromaDB vector store operations."""

from pathlib import Path

import chromadb

from metis.client import get_embedding_model
from metis.config import MetisConfig
from metis.index.embed import embed_texts

COLLECTION_NAME = "metis_vault"

# the model every pre-stamp index was built with; used when a collection carries no stamp.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingModelMismatch(Exception):
    """the configured embedding model differs from the one that built the index."""

    def __init__(self, indexed: str, configured: str):
        self.indexed = indexed
        self.configured = configured
        super().__init__(
            f"index built with '{indexed}', config says '{configured}'. "
            "run 'metis reindex' to rebuild the index with the new model."
        )


def get_collection(config: MetisConfig) -> chromadb.Collection:
    config.chromadb_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chromadb_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": get_embedding_model(config)},
    )


def indexed_embedding_model(collection: chromadb.Collection) -> str:
    """the embedding model stamped on the index, defaulting to the historical model if unstamped."""
    return (collection.metadata or {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)


def check_embedding_model(config: MetisConfig) -> None:
    """raise EmbeddingModelMismatch if config's embedding model differs from the index's stamp.

    a no-op on an empty index: nothing is committed yet, so the first write sets the stamp.
    """
    collection = get_collection(config)
    if collection.count() == 0:
        return
    indexed = indexed_embedding_model(collection)
    configured = get_embedding_model(config)
    if indexed != configured:
        raise EmbeddingModelMismatch(indexed, configured)


def store_chunks(
    chunks: list[str],
    file_path: Path,
    config: MetisConfig,
) -> int:
    """embed and store chunks in ChromaDB. returns number of chunks stored."""
    if not chunks:
        return 0

    collection = get_collection(config)
    embeddings = embed_texts(chunks, config)

    file_key = str(file_path)
    ids = [f"{file_key}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"file_path": file_key, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def store_chunks_with_embeddings(
    chunks: list[str],
    embeddings: list[list[float]],
    file_path: Path,
    config: MetisConfig,
) -> int:
    """store pre-computed chunks and embeddings in ChromaDB."""
    if not chunks:
        return 0

    collection = get_collection(config)

    file_key = str(file_path)
    ids = [f"{file_key}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"file_path": file_key, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)
