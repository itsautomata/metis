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
