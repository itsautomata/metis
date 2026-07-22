"""tests for the OpenAI-compatible client factory."""

from unittest.mock import patch

import pytest

from metis.client import (
    ProviderError,
    get_client,
    get_embedding_client,
    get_embedding_model,
)
from metis.config import EmbeddingConfig, MetisConfig, OpenAIConfig


def test_base_url_threads_into_client(monkeypatch):
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "sk-test")
    with patch("metis.client.OpenAI") as mock_openai:
        get_client(MetisConfig(openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1")))
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "sk-test"


def test_empty_base_url_passes_none(monkeypatch):
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "sk-test")
    with patch("metis.client.OpenAI") as mock_openai:
        get_client(MetisConfig(openai=OpenAIConfig(base_url="")))
        assert mock_openai.call_args.kwargs["base_url"] is None


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "")
    with pytest.raises(ProviderError, match="secret set provider-key"):
        get_client(MetisConfig())


def test_embedding_client_falls_back_to_chat_without_override(monkeypatch):
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "sk-chat")
    with patch("metis.client.OpenAI") as mock_openai:
        get_embedding_client(MetisConfig(openai=OpenAIConfig(base_url="https://chat")))
        assert mock_openai.call_args.kwargs["base_url"] == "https://chat"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-chat"


def test_embedding_client_uses_override(monkeypatch):
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "sk-chat")
    monkeypatch.setattr("metis.client.get_embedding_key", lambda: "sk-embed")
    with patch("metis.client.OpenAI") as mock_openai:
        cfg = MetisConfig(
            openai=OpenAIConfig(base_url="https://chat"),
            embedding=EmbeddingConfig(base_url="https://embed"),
        )
        get_embedding_client(cfg)
        assert mock_openai.call_args.kwargs["base_url"] == "https://embed"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-embed"


def test_embedding_client_falls_back_to_openai_key(monkeypatch):
    monkeypatch.setattr("metis.client.get_embedding_key", lambda: "")
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "sk-chat")
    with patch("metis.client.OpenAI") as mock_openai:
        get_embedding_client(MetisConfig(embedding=EmbeddingConfig(base_url="https://embed")))
        assert mock_openai.call_args.kwargs["api_key"] == "sk-chat"
        assert mock_openai.call_args.kwargs["base_url"] == "https://embed"


def test_embedding_client_raises_when_no_key_anywhere(monkeypatch):
    monkeypatch.setattr("metis.client.get_embedding_key", lambda: "")
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "")
    with pytest.raises(ProviderError, match="no API key for embeddings"):
        get_embedding_client(MetisConfig(embedding=EmbeddingConfig(base_url="https://embed")))


def test_embedding_model_honors_override():
    shared = MetisConfig(openai=OpenAIConfig(embedding_model="text-embedding-3-small"))
    assert get_embedding_model(shared) == "text-embedding-3-small"

    split = MetisConfig(
        openai=OpenAIConfig(embedding_model="text-embedding-3-small"),
        embedding=EmbeddingConfig(base_url="https://embed", model="bge-large"),
    )
    assert get_embedding_model(split) == "bge-large"


def test_embedding_model_auto_prefixes_bare_name_on_openrouter():
    cfg = MetisConfig(openai=OpenAIConfig(
        base_url="https://openrouter.ai/api/v1", embedding_model="text-embedding-3-small"))
    assert get_embedding_model(cfg) == "openai/text-embedding-3-small"


def test_embedding_model_leaves_prefixed_and_plain_openai_untouched():
    # already vendor-prefixed on openrouter: unchanged
    pref = MetisConfig(openai=OpenAIConfig(
        base_url="https://openrouter.ai/api/v1", embedding_model="openai/text-embedding-3-large"))
    assert get_embedding_model(pref) == "openai/text-embedding-3-large"
    # plain openai (no base_url): unchanged
    plain = MetisConfig(openai=OpenAIConfig(embedding_model="text-embedding-3-small"))
    assert get_embedding_model(plain) == "text-embedding-3-small"
