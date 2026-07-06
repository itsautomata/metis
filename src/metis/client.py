"""shared client factory for any OpenAI-compatible provider."""

from openai import OpenAI

from metis.config import MetisConfig
from metis.secrets import get_embedding_key, get_provider_key


class ProviderError(RuntimeError):
    """a provider/model call failed, wrapped with a message the user can act on."""


def provider_of(base_url: str) -> str:
    """identify the provider from its base_url: 'openai' | 'openrouter' | 'custom'."""
    if not base_url:
        return "openai"
    if "openrouter.ai" in base_url:
        return "openrouter"
    return "custom"


def get_client(config: MetisConfig) -> OpenAI:
    """the chat/summary client.

    base_url points at any compatible provider (OpenAI, OpenRouter, Ollama, ...);
    empty base_url uses OpenAI's default. the key comes from the keychain (or the
    METIS_PROVIDER_KEY env override), never the config file.
    """
    api_key = get_provider_key()
    if not api_key:
        raise ValueError("API key not set. run 'metis secret set provider-key' or set METIS_PROVIDER_KEY")
    return OpenAI(api_key=api_key, base_url=config.openai.base_url or None)


def get_embedding_client(config: MetisConfig) -> OpenAI:
    """the embedding client: the same as the chat client unless an embedding override is set.

    when the override is active, uses the embedding key if set, otherwise falls back to
    the openai key (handy when both providers share a key, or the embedder is keyless).
    """
    emb = config.embedding
    if not emb.base_url:
        return get_client(config)
    api_key = get_embedding_key() or get_provider_key()
    if not api_key:
        raise ValueError(
            "no API key for embeddings. run 'metis secret set embedding-key' or 'metis secret set provider-key'"
        )
    return OpenAI(api_key=api_key, base_url=emb.base_url)


def get_chat_model(config: MetisConfig) -> str:
    return config.openai.chat_model


def _adapt_embedding_model(model: str, base_url: str) -> str:
    """OpenRouter needs vendor-prefixed embedding ids; prefix a bare openai name."""
    if provider_of(base_url) == "openrouter" and model.startswith("text-embedding-") and "/" not in model:
        return "openai/" + model
    return model


def get_embedding_model(config: MetisConfig) -> str:
    emb = config.embedding
    if emb.base_url:
        model, base_url = (emb.model or config.openai.embedding_model), emb.base_url
    else:
        model, base_url = config.openai.embedding_model, config.openai.base_url
    return _adapt_embedding_model(model, base_url)
