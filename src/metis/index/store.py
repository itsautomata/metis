"""ChromaDB vector store operations."""

import re
from pathlib import Path

import chromadb
from chromadb.errors import InvalidArgumentError

from metis.client import get_embedding_model
from metis.config import MetisConfig
from metis.index.embed import embed_texts

COLLECTION_NAME = "metis_vault"

# the model every pre-stamp index was built with; used when a collection carries no stamp.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingModelMismatch(Exception):
    """the configured embedding model differs from the one that built the index."""

    def __init__(self, indexed: str, configured: str, detail: str | None = None):
        self.indexed = indexed
        self.configured = configured
        reason = detail or f"index built with '{indexed}', config says '{configured}'."
        super().__init__(
            f"{reason} run 'metis reindex' to rebuild the index with the new model."
        )


def get_collection(config: MetisConfig) -> chromadb.Collection:
    config.chromadb_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chromadb_path))
    # the embedding model is stamped at the first write (store_*), not here: a read command
    # run before the first ingest would otherwise stamp an empty index with the current model
    # and hide a later model switch.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def indexed_embedding_model(collection: chromadb.Collection) -> str:
    """the embedding model stamped on the index, defaulting to the historical model if unstamped."""
    return (collection.metadata or {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)


def _stamp_embedding_model(collection: chromadb.Collection, config: MetisConfig) -> None:
    """record the embedding model on the collection's first write.

    chromadb only writes metadata at create time, so the stamp is set here (once real vectors
    have landed) rather than in get_collection, keeping it truthful about what is indexed.
    """
    if not (collection.metadata or {}).get("embedding_model"):
        collection.modify(metadata={"embedding_model": get_embedding_model(config)})


def _canonical_embedding_model(name: str) -> str:
    """drop an 'openai/' vendor prefix so the same model reads equal across providers.

    _adapt_embedding_model prefixes 'openai/' when routing through OpenRouter; the bare and
    prefixed ids name the identical model and identical vectors, so they must not read as a mismatch.
    """
    return name.removeprefix("openai/") if name else name


def _safe_upsert(collection, config, *, ids, embeddings, documents, metadatas) -> None:
    """upsert, translating chromadb's fixed-dimension lock into an actionable reindex hint.

    chromadb pins the vector width on the first insert; a model whose id still matches the stamp
    but whose backend now emits a different width (a gateway repointed behind the same id) would
    otherwise raise a raw InvalidArgumentError mid-write.
    """
    try:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    except InvalidArgumentError as e:
        dims = re.search(r"dimension of (\d+), got (\d+)", str(e))
        if not dims:
            raise
        configured = get_embedding_model(config)
        raise EmbeddingModelMismatch(
            indexed_embedding_model(collection),
            configured,
            detail=(f"the index stores {dims.group(1)}-dim vectors but the configured model "
                    f"'{configured}' produced {dims.group(2)}-dim vectors."),
        ) from e


def check_embedding_model(config: MetisConfig) -> None:
    """raise EmbeddingModelMismatch if config's embedding model differs from the index's stamp.

    a no-op on an empty index: nothing is committed yet, so the first write sets the stamp.
    """
    collection = get_collection(config)
    if collection.count() == 0:
        return
    indexed = indexed_embedding_model(collection)
    configured = get_embedding_model(config)
    if _canonical_embedding_model(indexed) != _canonical_embedding_model(configured):
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

    _safe_upsert(collection, config, ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    _stamp_embedding_model(collection, config)

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
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"embedding count {len(embeddings)} != chunk count {len(chunks)}; "
            "refusing to store a misaligned index."
        )

    collection = get_collection(config)

    file_key = str(file_path)
    ids = [f"{file_key}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"file_path": file_key, "chunk_index": i}
        for i in range(len(chunks))
    ]

    _safe_upsert(collection, config, ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    _stamp_embedding_model(collection, config)

    return len(chunks)
