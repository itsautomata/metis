"""tests for the embedding-model stamp and reindex guard."""

import chromadb
import pytest

from metis.config import MetisConfig, OpenAIConfig
from metis.index import store, sync


def _cfg(tmp_path, model):
    return MetisConfig(
        vault_path=tmp_path / "vault",
        chromadb_path=tmp_path / "cdb",
        openai=OpenAIConfig(embedding_model=model),
    )


def test_empty_index_no_mismatch(tmp_path):
    store.check_embedding_model(_cfg(tmp_path, "text-embedding-3-small"))  # no raise


def test_stamp_persists_and_mismatch_raises(tmp_path):
    cfg = _cfg(tmp_path, "text-embedding-3-small")
    coll = store.get_collection(cfg)
    coll.add(ids=["x::chunk_0"], embeddings=[[0.1, 0.2, 0.3]], documents=["hi"],
             metadatas=[{"file_path": "x", "chunk_index": 0}])

    assert store.indexed_embedding_model(coll) == "text-embedding-3-small"
    store.check_embedding_model(cfg)  # same model, no raise
    with pytest.raises(store.EmbeddingModelMismatch):
        store.check_embedding_model(_cfg(tmp_path, "text-embedding-3-large"))


def test_unstamped_collection_treated_as_default(tmp_path):
    (tmp_path / "cdb").mkdir()
    client = chromadb.PersistentClient(path=str(tmp_path / "cdb"))
    coll = client.get_or_create_collection(name=store.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    assert store.indexed_embedding_model(coll) == "text-embedding-3-small"


def test_reindex_restamps_and_reembeds(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# a\n\ncontent")
    monkeypatch.setattr(store, "embed_texts", lambda texts, config: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", tmp_path / "sync_state.json")
    monkeypatch.setattr("metis.classify.CATEGORIZATION_PATH", tmp_path / "cat.json")

    sync.sync_vault(_cfg(tmp_path, "text-embedding-3-small"))
    cfg_large = _cfg(tmp_path, "text-embedding-3-large")
    with pytest.raises(store.EmbeddingModelMismatch):
        store.check_embedding_model(cfg_large)

    sync.reindex_vault(cfg_large)

    store.check_embedding_model(cfg_large)  # resolved
    assert store.indexed_embedding_model(store.get_collection(cfg_large)) == "text-embedding-3-large"
