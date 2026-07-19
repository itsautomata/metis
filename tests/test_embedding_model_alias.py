"""the same embedding model via OpenRouter must not read as a mismatch against a plain-OpenAI index."""

from pathlib import Path

from metis.config import MetisConfig, OpenAIConfig
from metis.index import store


def _cfg(model, base_url, tmp_path):
    return MetisConfig(
        openai=OpenAIConfig(embedding_model=model, base_url=base_url),
        chromadb_path=tmp_path / "cdb",
    )


def test_openrouter_prefix_is_not_a_mismatch(tmp_path):
    """an index stamped 'text-embedding-3-small' (built on OpenAI) must pass the guard when the
    config later routes the identical model through OpenRouter as 'openai/text-embedding-3-small'."""
    plain = _cfg("text-embedding-3-small", "", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), plain)

    via_openrouter = _cfg("text-embedding-3-small", "https://openrouter.ai/api/v1", tmp_path)
    # get_embedding_model adapts this to 'openai/text-embedding-3-small'; must not raise
    store.check_embedding_model(via_openrouter)


def test_a_genuinely_different_model_still_mismatches(tmp_path):
    """canonicalization only strips the vendor prefix; a real model change must still raise."""
    import pytest

    plain = _cfg("text-embedding-3-small", "", tmp_path)
    store.store_chunks_with_embeddings(["a"], [[0.1, 0.2]], Path("a.md"), plain)

    other = _cfg("text-embedding-3-large", "", tmp_path)
    with pytest.raises(store.EmbeddingModelMismatch):
        store.check_embedding_model(other)
