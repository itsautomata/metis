"""embedding generation via OpenAI."""

import openai

from metis.client import ProviderError, get_embedding_client, get_embedding_model
from metis.config import MetisConfig

BATCH_SIZE = 40


def _embedding_error(model: str, config: MetisConfig, err: Exception) -> str:
    msg = f"embedding model '{model}' failed: {err}"
    if config.openai.base_url or config.embedding.base_url:
        msg += ". on a gateway, embedding ids are often vendor-prefixed, e.g. openai/text-embedding-3-small"
    return msg


def embed_texts(texts: list[str], config: MetisConfig) -> list[list[float]]:
    """embed a list of texts in batches. returns list of vectors."""
    if not texts:
        return []

    client = get_embedding_client(config)
    model = get_embedding_model(config)

    # filter out empty strings (openai rejects them)
    texts = [t if t.strip() else " " for t in texts]

    all_embeddings = []
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
    except openai.OpenAIError as e:
        raise ProviderError(_embedding_error(model, config, e)) from e

    return all_embeddings
