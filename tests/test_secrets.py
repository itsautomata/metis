"""tests for secret management."""

from unittest.mock import patch

from metis.secrets import get_provider_key, get_secret, get_x_bearer

# --- fallback chain: keychain then env, never a config file ---

@patch("metis.secrets.keyring")
def test_get_secret_from_keychain(mock_keyring):
    """keychain is checked first."""
    mock_keyring.get_password.return_value = "keychain-value"
    result = get_secret("test-key", fallback_env="TEST_ENV")
    assert result == "keychain-value"


@patch("metis.secrets.keyring")
def test_get_secret_falls_back_to_env(mock_keyring, monkeypatch):
    """if keychain is empty, env var is checked."""
    mock_keyring.get_password.return_value = None
    monkeypatch.setenv("TEST_ENV", "env-value")
    result = get_secret("test-key", fallback_env="TEST_ENV")
    assert result == "env-value"


@patch("metis.secrets.keyring")
def test_get_secret_returns_empty_if_nothing(mock_keyring, monkeypatch):
    """if neither keychain nor env is set, returns empty string."""
    mock_keyring.get_password.return_value = None
    monkeypatch.delenv("TEST_ENV", raising=False)
    result = get_secret("test-key", fallback_env="TEST_ENV")
    assert result == ""


@patch("metis.secrets.keyring")
def test_get_secret_keychain_error_falls_through(mock_keyring, monkeypatch):
    """if keychain throws, falls through to env."""
    mock_keyring.get_password.side_effect = Exception("keychain locked")
    monkeypatch.setenv("TEST_ENV", "env-value")
    result = get_secret("test-key", fallback_env="TEST_ENV")
    assert result == "env-value"


# --- convenience functions: no args, keychain then env ---

@patch("metis.secrets.get_secret")
def test_get_provider_key(mock_get):
    mock_get.return_value = "sk-test"
    result = get_provider_key()
    mock_get.assert_called_once_with("provider-key", fallback_env="METIS_PROVIDER_KEY")
    assert result == "sk-test"


@patch("metis.secrets.get_secret")
def test_get_x_bearer(mock_get):
    mock_get.return_value = "bearer-test"
    result = get_x_bearer()
    mock_get.assert_called_once_with("x-bearer-token", fallback_env="METIS_X_BEARER")
    assert result == "bearer-test"
