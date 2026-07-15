"""embeddings must stay aligned 1:1 with their chunks, in input order."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from metis.client import ProviderError
from metis.config import MetisConfig


class _FakeEmbeddings:
    def __init__(self, responder):
        self._responder = responder

    def create(self, model, input):
        return self._responder(input)


class _FakeClient:
    def __init__(self, responder):
        self.embeddings = _FakeEmbeddings(responder)


def _patch_client(monkeypatch, responder):
    from metis.index import embed
    monkeypatch.setattr(embed, "get_embedding_client", lambda c: _FakeClient(responder))
    monkeypatch.setattr(embed, "get_embedding_model", lambda c: "m")


def test_embed_texts_realigns_out_of_order_response(monkeypatch):
    """a gateway that returns vectors out of order must not misalign the index."""
    from metis.index import embed

    def responder(inp):
        # correct .index tags, but returned in REVERSED order
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(inp))]
        return SimpleNamespace(data=list(reversed(data)))

    _patch_client(monkeypatch, responder)
    out = embed.embed_texts(["a", "b", "c"], MetisConfig())
    assert out == [[0.0], [1.0], [2.0]]  # realigned to input order despite reversed data


def test_embed_texts_rejects_short_batch(monkeypatch):
    """fewer vectors than inputs is a clear ProviderError, not a silent/misaligned store."""
    from metis.index import embed

    def responder(inp):
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(inp) - 1)]
        return SimpleNamespace(data=data)

    _patch_client(monkeypatch, responder)
    with pytest.raises(ProviderError):
        embed.embed_texts(["a", "b", "c"], MetisConfig())


def test_store_with_embeddings_rejects_length_mismatch():
    """the store refuses misaligned inputs before touching chromadb."""
    from metis.index.store import store_chunks_with_embeddings

    with pytest.raises(ValueError):
        store_chunks_with_embeddings(["c1", "c2"], [[0.1]], Path("note.md"), MetisConfig())
