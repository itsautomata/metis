"""a vector-width mismatch at query time (read path) must surface as EmbeddingModelMismatch."""

from pathlib import Path

import pytest

from metis.config import MetisConfig, OpenAIConfig
from metis.index import store


def _cfg(model, tmp_path):
    return MetisConfig(openai=OpenAIConfig(embedding_model=model), chromadb_path=tmp_path / "cdb")


def test_query_width_mismatch_becomes_embedding_model_mismatch(tmp_path):
    """the write path already translates this; the read path (query_collection) must match it."""
    cfg = _cfg("gateway-model", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), cfg)  # locks width 2
    collection = store.get_collection(cfg)

    with pytest.raises(store.EmbeddingModelMismatch):
        store.query_collection(collection, cfg, query_embeddings=[[0.1, 0.2, 0.3]], n_results=1)


def test_query_passes_through_on_matching_width(tmp_path):
    """a same-width query returns results (no false mismatch, no regression)."""
    cfg = _cfg("gateway-model", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), cfg)
    collection = store.get_collection(cfg)

    results = store.query_collection(collection, cfg, query_embeddings=[[0.1, 0.2]], n_results=1)
    assert results["ids"][0]  # a hit came back
