"""secure secret storage via OS keychain."""

import os

import keyring

SERVICE = "metis"

# key names in the keychain
PROVIDER_KEY = "provider-key"
EMBEDDING_KEY = "embedding-api-key"
X_BEARER = "x-bearer-token"


class KeychainError(RuntimeError):
    """the OS keychain is unavailable, so a secret could not be written or removed."""


def get_secret(name: str, fallback_env: str = "") -> str:
    """retrieve a secret. checks the keychain, then the env var. empty string if neither.

    secrets never live in the config file: the keychain is the home, env is the override.
    """
    try:
        value = keyring.get_password(SERVICE, name)
        if value:
            return value
    except Exception:
        pass

    if fallback_env:
        value = os.environ.get(fallback_env, "")
        if value:
            return value

    return ""


def set_secret(name: str, value: str) -> None:
    """store a secret in the OS keychain."""
    try:
        keyring.set_password(SERVICE, name, value)
    except keyring.errors.KeyringError as e:
        raise KeychainError(
            "no usable OS keychain backend, so the secret was not saved. "
            "set the matching METIS_*_KEY environment variable instead."
        ) from e


def delete_secret(name: str) -> None:
    """remove a secret from the OS keychain."""
    try:
        keyring.delete_password(SERVICE, name)
    except keyring.errors.KeyringError:
        pass


def get_provider_key() -> str:
    return get_secret(PROVIDER_KEY, fallback_env="METIS_PROVIDER_KEY")


def get_embedding_key() -> str:
    return get_secret(EMBEDDING_KEY, fallback_env="METIS_EMBEDDING_KEY")


def get_x_bearer() -> str:
    return get_secret(X_BEARER, fallback_env="METIS_X_BEARER")
