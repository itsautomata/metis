"""embedding generation via OpenAI."""

from metis.client import get_client, get_embedding_model
from metis.config import MetisConfig


def embed_texts(texts: list[str], config: MetisConfig) -> list[list[float]]:
    """embed a list of texts. returns list of vectors."""
    client = get_client(config)

    response = client.embeddings.create(
        model=get_embedding_model(config),
        input=texts,
    )

    return [item.embedding for item in response.data]
