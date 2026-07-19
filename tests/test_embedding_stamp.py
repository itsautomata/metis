"""the index's embedding-model stamp must name the model that actually wrote its vectors."""

from pathlib import Path

import pytest

from metis.config import MetisConfig, OpenAIConfig
from metis.index import store


def _cfg(model, tmp_path):
    return MetisConfig(openai=OpenAIConfig(embedding_model=model), chromadb_path=tmp_path / "cdb")


def test_stamp_reflects_the_writing_model_not_creation_time(tmp_path):
    """a read command creates the empty index under model-a; a later write under model-b
    must stamp model-b (the model that produced the vectors), not model-a."""
    store.get_collection(_cfg("model-a", tmp_path))  # empty collection created under model-a
    store.store_chunks_with_embeddings(["chunk"], [[0.1, 0.2]], Path("note.md"), _cfg("model-b", tmp_path))

    col = store.get_collection(_cfg("model-b", tmp_path))
    assert store.indexed_embedding_model(col) == "model-b"


def test_check_flags_mismatch_after_a_model_switch(tmp_path):
    """with model-b vectors in the index, checking under a model-a config must raise, not pass."""
    store.get_collection(_cfg("model-a", tmp_path))  # empty, created under model-a
    store.store_chunks_with_embeddings(["chunk"], [[0.1, 0.2]], Path("note.md"), _cfg("model-b", tmp_path))

    with pytest.raises(store.EmbeddingModelMismatch):
        store.check_embedding_model(_cfg("model-a", tmp_path))


def test_check_passes_when_model_matches_the_vectors(tmp_path):
    """the same model that wrote the vectors is not a mismatch."""
    store.store_chunks_with_embeddings(["chunk"], [[0.1, 0.2]], Path("note.md"), _cfg("model-x", tmp_path))
    store.check_embedding_model(_cfg("model-x", tmp_path))  # must not raise
