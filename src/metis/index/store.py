"""ChromaDB vector store operations."""

from pathlib import Path

import chromadb

from metis.config import MetisConfig
from metis.index.embed import embed_texts

COLLECTION_NAME = "metis_vault"


def _get_collection(config: MetisConfig) -> chromadb.Collection:
    config.chromadb_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chromadb_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    chunks: list[str],
    file_path: Path,
    config: MetisConfig,
) -> int:
    """embed and store chunks in ChromaDB. returns number of chunks stored."""
    if not chunks:
        return 0

    collection = _get_collection(config)
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

    collection = _get_collection(config)

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
