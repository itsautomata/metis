"""tests for the OpenAI-compatible client factory."""

from unittest.mock import patch

import pytest

from metis.client import get_client, get_embedding_client, get_embedding_model
from metis.config import EmbeddingConfig, MetisConfig, OpenAIConfig


def test_base_url_threads_into_client(monkeypatch):
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "sk-test")
    with patch("metis.client.OpenAI") as mock_openai:
        get_client(MetisConfig(openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1")))
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "sk-test"


def test_empty_base_url_passes_none(monkeypatch):
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "sk-test")
    with patch("metis.client.OpenAI") as mock_openai:
        get_client(MetisConfig(openai=OpenAIConfig(base_url="")))
        assert mock_openai.call_args.kwargs["base_url"] is None


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "")
    with pytest.raises(ValueError, match="secret set openai-key"):
        get_client(MetisConfig())


def test_embedding_client_falls_back_to_chat_without_override(monkeypatch):
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "sk-chat")
    with patch("metis.client.OpenAI") as mock_openai:
        get_embedding_client(MetisConfig(openai=OpenAIConfig(base_url="https://chat")))
        assert mock_openai.call_args.kwargs["base_url"] == "https://chat"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-chat"


def test_embedding_client_uses_override(monkeypatch):
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "sk-chat")
    monkeypatch.setattr("metis.client.get_embedding_key", lambda cfg: "sk-embed")
    with patch("metis.client.OpenAI") as mock_openai:
        cfg = MetisConfig(
            openai=OpenAIConfig(base_url="https://chat"),
            embedding=EmbeddingConfig(base_url="https://embed"),
        )
        get_embedding_client(cfg)
        assert mock_openai.call_args.kwargs["base_url"] == "https://embed"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-embed"


def test_embedding_client_falls_back_to_openai_key(monkeypatch):
    monkeypatch.setattr("metis.client.get_embedding_key", lambda cfg: "")
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "sk-chat")
    with patch("metis.client.OpenAI") as mock_openai:
        get_embedding_client(MetisConfig(embedding=EmbeddingConfig(base_url="https://embed")))
        assert mock_openai.call_args.kwargs["api_key"] == "sk-chat"
        assert mock_openai.call_args.kwargs["base_url"] == "https://embed"


def test_embedding_client_raises_when_no_key_anywhere(monkeypatch):
    monkeypatch.setattr("metis.client.get_embedding_key", lambda cfg: "")
    monkeypatch.setattr("metis.client.get_openai_key", lambda cfg: "")
    with pytest.raises(ValueError, match="no API key for embeddings"):
        get_embedding_client(MetisConfig(embedding=EmbeddingConfig(base_url="https://embed")))


def test_embedding_model_honors_override():
    shared = MetisConfig(openai=OpenAIConfig(embedding_model="text-embedding-3-small"))
    assert get_embedding_model(shared) == "text-embedding-3-small"

    split = MetisConfig(
        openai=OpenAIConfig(embedding_model="text-embedding-3-small"),
        embedding=EmbeddingConfig(base_url="https://embed", model="bge-large"),
    )
    assert get_embedding_model(split) == "bge-large"
