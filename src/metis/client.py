"""shared client factory for any OpenAI-compatible provider."""

from openai import OpenAI

from metis.config import MetisConfig
from metis.secrets import get_embedding_key, get_openai_key


def get_client(config: MetisConfig) -> OpenAI:
    """the chat/summary client.

    base_url points at any compatible provider (OpenAI, OpenRouter, Ollama, ...);
    empty base_url uses OpenAI's default. secret lookup: keychain, env var, config file.
    """
    api_key = get_openai_key(config.openai.api_key)
    if not api_key:
        raise ValueError("API key not set. run 'metis secret set openai-key' or set METIS_OPENAI_KEY")
    return OpenAI(api_key=api_key, base_url=config.openai.base_url or None)


def get_embedding_client(config: MetisConfig) -> OpenAI:
    """the embedding client: the same as the chat client unless an embedding override is set.

    when the override is active, uses the embedding key if set, otherwise falls back to
    the openai key (handy when both providers share a key, or the embedder is keyless).
    """
    emb = config.embedding
    if not emb.base_url:
        return get_client(config)
    api_key = get_embedding_key(emb.api_key) or get_openai_key(config.openai.api_key)
    if not api_key:
        raise ValueError(
            "no API key for embeddings. run 'metis secret set embedding-key' or 'metis secret set openai-key'"
        )
    return OpenAI(api_key=api_key, base_url=emb.base_url)


def get_chat_model(config: MetisConfig) -> str:
    return config.openai.chat_model


def get_embedding_model(config: MetisConfig) -> str:
    emb = config.embedding
    if emb.base_url and emb.model:
        return emb.model
    return config.openai.embedding_model
