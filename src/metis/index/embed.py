"""embedding generation via OpenAI."""

from metis.client import get_client, get_embedding_model
from metis.config import MetisConfig

BATCH_SIZE = 40


def embed_texts(texts: list[str], config: MetisConfig) -> list[list[float]]:
    """embed a list of texts in batches. returns list of vectors."""
    if not texts:
        return []

    client = get_client(config)
    model = get_embedding_model(config)

    # filter out empty strings (openai rejects them)
    texts = [t if t.strip() else " " for t in texts]

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings
