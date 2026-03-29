"""secure secret storage via OS keychain."""

import os

import keyring

SERVICE = "metis"

# key names in the keychain
OPENAI_KEY = "openai-api-key"
AZURE_KEY = "azure-api-key"
X_BEARER = "x-bearer-token"


def get_secret(name: str, fallback_env: str = "", fallback_config: str = "") -> str:
    """retrieve a secret. checks: keychain → env var → config file value.

    returns empty string if not found anywhere.
    """
    # 1. keychain (most secure)
    try:
        value = keyring.get_password(SERVICE, name)
        if value:
            return value
    except Exception:
        pass

    # 2. environment variable
    if fallback_env:
        value = os.environ.get(fallback_env, "")
        if value:
            return value

    # 3. config file value (least secure, last resort)
    return fallback_config


def set_secret(name: str, value: str) -> None:
    """store a secret in the OS keychain."""
    keyring.set_password(SERVICE, name, value)


def delete_secret(name: str) -> None:
    """remove a secret from the OS keychain."""
    try:
        keyring.delete_password(SERVICE, name)
    except keyring.errors.PasswordDeleteError:
        pass


def get_openai_key(config_value: str = "") -> str:
    return get_secret(OPENAI_KEY, fallback_env="METIS_OPENAI_KEY", fallback_config=config_value)


def get_azure_key(config_value: str = "") -> str:
    return get_secret(AZURE_KEY, fallback_env="METIS_AZURE_KEY", fallback_config=config_value)


def get_x_bearer(config_value: str = "") -> str:
    return get_secret(X_BEARER, fallback_env="METIS_X_BEARER", fallback_config=config_value)
