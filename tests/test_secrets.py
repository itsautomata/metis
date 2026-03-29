"""tests for secret management."""

import os
from unittest.mock import patch, MagicMock

from metis.secrets import get_secret, get_openai_key, get_azure_key, get_x_bearer


# --- fallback chain ---

@patch("metis.secrets.keyring")
def test_get_secret_from_keychain(mock_keyring):
    """keychain is checked first."""
    mock_keyring.get_password.return_value = "keychain-value"
    result = get_secret("test-key", fallback_env="TEST_ENV", fallback_config="config-value")
    assert result == "keychain-value"


@patch("metis.secrets.keyring")
def test_get_secret_falls_back_to_env(mock_keyring, monkeypatch):
    """if keychain is empty, env var is checked."""
    mock_keyring.get_password.return_value = None
    monkeypatch.setenv("TEST_ENV", "env-value")
    result = get_secret("test-key", fallback_env="TEST_ENV", fallback_config="config-value")
    assert result == "env-value"


@patch("metis.secrets.keyring")
def test_get_secret_falls_back_to_config(mock_keyring, monkeypatch):
    """if keychain and env are empty, config value is used."""
    mock_keyring.get_password.return_value = None
    monkeypatch.delenv("TEST_ENV", raising=False)
    result = get_secret("test-key", fallback_env="TEST_ENV", fallback_config="config-value")
    assert result == "config-value"


@patch("metis.secrets.keyring")
def test_get_secret_returns_empty_if_nothing(mock_keyring, monkeypatch):
    """if nothing is set anywhere, returns empty string."""
    mock_keyring.get_password.return_value = None
    result = get_secret("test-key")
    assert result == ""


@patch("metis.secrets.keyring")
def test_get_secret_keychain_error_falls_through(mock_keyring, monkeypatch):
    """if keychain throws, falls through to env."""
    mock_keyring.get_password.side_effect = Exception("keychain locked")
    monkeypatch.setenv("TEST_ENV", "env-value")
    result = get_secret("test-key", fallback_env="TEST_ENV")
    assert result == "env-value"


# --- convenience functions ---

@patch("metis.secrets.get_secret")
def test_get_openai_key(mock_get):
    mock_get.return_value = "sk-test"
    result = get_openai_key("config-fallback")
    mock_get.assert_called_once_with("openai-api-key", fallback_env="METIS_OPENAI_KEY", fallback_config="config-fallback")
    assert result == "sk-test"


@patch("metis.secrets.get_secret")
def test_get_azure_key(mock_get):
    mock_get.return_value = "azure-test"
    result = get_azure_key("config-fallback")
    mock_get.assert_called_once_with("azure-api-key", fallback_env="METIS_AZURE_KEY", fallback_config="config-fallback")
    assert result == "azure-test"


@patch("metis.secrets.get_secret")
def test_get_x_bearer(mock_get):
    mock_get.return_value = "bearer-test"
    result = get_x_bearer("config-fallback")
    mock_get.assert_called_once_with("x-bearer-token", fallback_env="METIS_X_BEARER", fallback_config="config-fallback")
    assert result == "bearer-test"
