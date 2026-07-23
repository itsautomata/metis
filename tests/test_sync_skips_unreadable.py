"""sync must skip a file that vanishes / is unreadable between the scan and the read, not crash."""

from metis.config import MetisConfig, OpenAIConfig
from metis.index import sync


class _EmptyCollection:
    def get(self, include=None):
        return {"metadatas": []}

    def count(self):
        return 0


def _cfg(vault, tmp_path):
    return MetisConfig(openai=OpenAIConfig(base_url=""), vault_path=vault, chromadb_path=tmp_path / "cdb")


def test_sync_skips_unreadable_file(monkeypatch, tmp_path):
    """a file that raises FileNotFoundError at hash time is counted skipped; the readable note indexes.

    modelled by forcing _file_hash to fail for one file (a broken symlink or a file deleted between
    the rglob scan and the read), which is platform-independent unlike creating a real broken symlink.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("a real note body", encoding="utf-8")
    (vault / "gone.md").write_text("removed before it is hashed", encoding="utf-8")

    real_hash = sync._file_hash

    def _hash(path):
        if path.name == "gone.md":
            raise FileNotFoundError(str(path))
        return real_hash(path)

    monkeypatch.setattr(sync, "_file_hash", _hash)
    # the skip logic is the unit under test, not chromadb: keep both off the real DB and provider
    monkeypatch.setattr(sync, "get_collection", lambda config: _EmptyCollection())
    monkeypatch.setattr(sync, "store_chunks", lambda chunks, path, config: len(chunks))

    report = sync.sync_vault(_cfg(vault, tmp_path))

    assert report.skipped == 1
    assert report.added == 1


def test_find_vault_files_skips_dot_dirs(tmp_path):
    """.obsidian and .trash markdown is not indexed, so deleted notes don't resurface in search."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "live.md").write_text("live")
    (vault / ".trash").mkdir()
    (vault / ".trash" / "deleted.md").write_text("deleted note")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "stray.md").write_text("plugin")
    (vault / ".hidden.md").write_text("dotfile")

    found = {p.name for p in sync._find_vault_files(_cfg(vault, tmp_path))}
    assert found == {"live.md"}


def test_sync_skips_permission_denied_file(monkeypatch, tmp_path):
    """a chmod 000 note (PermissionError) is skipped, not a whole-sync abort."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "good.md").write_text("a real note body", encoding="utf-8")
    (vault / "locked.md").write_text("secret", encoding="utf-8")

    real_hash = sync._file_hash

    def _hash(path):
        if path.name == "locked.md":
            raise PermissionError(str(path))
        return real_hash(path)

    monkeypatch.setattr(sync, "_file_hash", _hash)
    monkeypatch.setattr(sync, "get_collection", lambda config: _EmptyCollection())
    monkeypatch.setattr(sync, "store_chunks", lambda chunks, path, config: len(chunks))

    report = sync.sync_vault(_cfg(vault, tmp_path))

    assert report.skipped == 1
    assert report.added == 1
