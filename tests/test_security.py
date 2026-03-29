"""tests for security — path traversal guards."""

from pathlib import Path


def test_folder_traversal_blocked(tmp_path):
    """--folder with ../ should not resolve outside vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    folder = "../../etc"
    resolved = (vault / folder).resolve()
    assert not resolved.is_relative_to(vault.resolve())


def test_folder_inside_vault_allowed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    folder = "research/ai"
    resolved = (vault / folder).resolve()
    assert resolved.is_relative_to(vault.resolve())


def test_note_traversal_blocked(tmp_path):
    """--note with ../ should not resolve outside vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = "../../etc/passwd.md"
    note_p = vault / note
    assert not note_p.resolve().is_relative_to(vault.resolve())


def test_note_inside_vault_allowed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "notes").mkdir()
    note_file = vault / "notes" / "test.md"
    note_file.write_text("# test")
    assert note_file.resolve().is_relative_to(vault.resolve())


def test_note_dot_dot_with_md_suffix(tmp_path):
    """even with .md appended, traversal should be caught."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = Path("../../secret")
    if not note.suffix:
        note = note.with_suffix(".md")
    note_p = vault / note
    assert not note_p.resolve().is_relative_to(vault.resolve())
