"""shared OpenAI client factory. supports both openai and azure providers."""

from openai import AzureOpenAI, OpenAI

from metis.config import MetisConfig
from metis.secrets import get_openai_key, get_azure_key


def get_client(config: MetisConfig) -> OpenAI:
    """create the right client based on provider config.

    returns OpenAI or AzureOpenAI — same interface, different backend.
    secret lookup: keychain → env var → config file.
    """
    if config.provider == "azure":
        api_key = get_azure_key(config.azure.api_key)
        if not api_key:
            raise ValueError("azure API key not set. run 'metis config set azure-key' or set METIS_AZURE_KEY")
        if not config.azure.endpoint:
            raise ValueError("azure endpoint not set. edit ~/.metis/config.yaml")
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-10-21",
            azure_endpoint=config.azure.endpoint,
        )

    # default: regular openai
    api_key = get_openai_key(config.openai.api_key)
    if not api_key:
        raise ValueError("openai API key not set. run 'metis config set openai-key' or set METIS_OPENAI_KEY")
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
