"""shared OpenAI client factory. supports both openai and azure providers."""

import os

from openai import AzureOpenAI, OpenAI

from metis.config import MetisConfig


def get_client(config: MetisConfig) -> OpenAI:
    """create the right client based on provider config.

    returns OpenAI or AzureOpenAI — same interface, different backend.
    """
    if config.provider == "azure":
        api_key = config.azure.api_key or os.environ.get("METIS_AZURE_KEY", "")
        if not api_key:
            raise ValueError("azure API key not set. edit ~/.metis/config.yaml or set METIS_AZURE_KEY")
        if not config.azure.endpoint:
            raise ValueError("azure endpoint not set. edit ~/.metis/config.yaml")
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-10-21",
            azure_endpoint=config.azure.endpoint,
        )

    # default: regular openai
    api_key = config.openai.api_key or os.environ.get("METIS_OPENAI_KEY", "")
    if not api_key:
        raise ValueError("openai API key not set. edit ~/.metis/config.yaml or set METIS_OPENAI_KEY")
    return OpenAI(api_key=api_key)


def get_chat_model(config: MetisConfig) -> str:
    """return the chat model name for the active provider."""
    if config.provider == "azure":
        return config.azure.chat_model
    return config.openai.chat_model


def get_embedding_model(config: MetisConfig) -> str:
    """return the embedding model name for the active provider."""
    if config.provider == "azure":
        return config.azure.embedding_model
    return config.openai.embedding_model
