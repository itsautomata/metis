"""secret writes must fail with a clear error on a host with no keychain, not a raw traceback."""

import keyring.errors
import pytest

from metis import secrets


def test_set_secret_raises_keychain_error_without_backend(monkeypatch):
    """no usable keyring backend must surface KeychainError, so the CLI can print guidance
    instead of falsely reporting the secret as saved."""
    def _no_backend(*args, **kwargs):
        raise keyring.errors.NoKeyringError("no backend")

    monkeypatch.setattr(secrets.keyring, "set_password", _no_backend)
    with pytest.raises(secrets.KeychainError):
        secrets.set_secret("provider-key", "sk-test")


def test_delete_secret_noops_without_backend(monkeypatch):
    """delete degrades to a no-op on a dead backend, mirroring get_secret's graceful read."""
    def _no_backend(*args, **kwargs):
        raise keyring.errors.NoKeyringError("no backend")

    monkeypatch.setattr(secrets.keyring, "delete_password", _no_backend)
    secrets.delete_secret("provider-key")  # must not raise
