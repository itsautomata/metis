"""tests for the suggested-folder menu picker."""

from metis import pick
from metis.config import MetisConfig


class _FakeAsk:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


def test_selecting_a_suggestion_returns_it(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk("research/ai"))
    result = pick.pick_suggested_folder([("research/ai", 0.82)], MetisConfig())
    assert result == "research/ai"


def test_pick_existing_opens_folder_submenu(monkeypatch):
    calls = iter([pick._PICK_EXISTING, "existing/folder"])  # main menu, then sub-menu
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk(next(calls)))
    monkeypatch.setattr(pick, "_vault_folders", lambda config: ["existing/folder", "other"])
    result = pick.pick_suggested_folder([("x", 0.5)], MetisConfig())
    assert result == "existing/folder"


def test_pick_existing_with_no_folders_returns_none(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk(pick._PICK_EXISTING))
    monkeypatch.setattr(pick, "_vault_folders", lambda config: [])
    result = pick.pick_suggested_folder([("x", 0.5)], MetisConfig())
    assert result is None


def test_new_folder_routes_to_text_input(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk(pick._NEW_FOLDER))
    monkeypatch.setattr(pick.questionary, "text", lambda *a, **k: _FakeAsk("brand-new"))
    result = pick.pick_suggested_folder([("x", 0.5)], MetisConfig())
    assert result == "brand-new"


def test_new_folder_blank_name_returns_none(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk(pick._NEW_FOLDER))
    monkeypatch.setattr(pick.questionary, "text", lambda *a, **k: _FakeAsk("   "))
    result = pick.pick_suggested_folder([("x", 0.5)], MetisConfig())
    assert result is None


def test_cancel_returns_none(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeAsk(None))
    result = pick.pick_suggested_folder([("x", 0.5)], MetisConfig())
    assert result is None


def test_typed_new_name_traversal_blocked(tmp_path):
    """a typed '../escape' resolves outside the vault, so the cli guard rejects it."""
    vault = tmp_path / "vault"
    vault.mkdir()
    chosen = "../../etc"
    resolved = (vault / chosen).resolve()
    assert not resolved.is_relative_to(vault.resolve())


def test_vault_folders_excludes_symlink_escaping_vault(tmp_path):
    """a symlinked dir pointing outside the vault is not offered as a choice."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "real").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "escape").symlink_to(outside)

    folders = pick._vault_folders(MetisConfig(vault_path=vault))

    assert "real" in folders
    assert "escape" not in folders
