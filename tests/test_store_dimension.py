"""a vector-width mismatch at upsert must surface as an actionable reindex error, not raw chromadb."""

from pathlib import Path

import pytest

from metis.config import MetisConfig, OpenAIConfig
from metis.index import store


def _cfg(model, tmp_path):
    return MetisConfig(openai=OpenAIConfig(embedding_model=model), chromadb_path=tmp_path / "cdb")


def test_dimension_change_becomes_embedding_model_mismatch(tmp_path):
    """the same model id producing a different vector width (a repointed gateway) locks the
    chromadb dimension; the second write must raise EmbeddingModelMismatch, not InvalidArgumentError."""
    cfg = _cfg("gateway-model", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), cfg)  # locks width 2

    with pytest.raises(store.EmbeddingModelMismatch):
        store.store_chunks_with_embeddings(["b"], [[0.1, 0.2, 0.3]], Path("b.md"), cfg)  # width 3


def test_mismatch_message_names_both_dimensions(tmp_path):
    """the error explains the width clash so the user knows why a reindex is needed."""
    cfg = _cfg("gateway-model", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), cfg)

    with pytest.raises(store.EmbeddingModelMismatch) as exc:
        store.store_chunks_with_embeddings(["b"], [[0.1, 0.2, 0.3]], Path("b.md"), cfg)
    msg = str(exc.value)
    assert "2-dim" in msg and "3-dim" in msg
    assert "reindex" in msg
