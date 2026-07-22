"""switching vaults must not cross-contaminate: disjoint dedup, sync state, and chromadb collection."""

from metis import config
from metis.config import MetisConfig
from metis.index import store, sync
from metis.ingest import write


def _cfg(vault, tmp_path):
    return MetisConfig(vault_path=vault, chromadb_path=tmp_path / "cdb")


def test_collection_name_is_per_vault_and_stable(tmp_path):
    a = _cfg(tmp_path / "A", tmp_path)
    b = _cfg(tmp_path / "B", tmp_path)
    assert store.collection_name(a) != store.collection_name(b)
    assert store.collection_name(a) == store.collection_name(_cfg(tmp_path / "A", tmp_path))


def test_dedup_does_not_cross_vaults(tmp_path):
    """a source registered in vault A is not reported as a duplicate in vault B."""
    va, vb = tmp_path / "A", tmp_path / "B"
    va.mkdir()
    vb.mkdir()
    cfg_a, cfg_b = _cfg(va, tmp_path), _cfg(vb, tmp_path)
    note = va / "note.md"
    note.write_text("x")
    write._register_source("https://example.com", note, cfg_a)

    assert write.check_duplicate("https://example.com", cfg_a) == note
    assert write.check_duplicate("https://example.com", cfg_b) is None


def test_sync_state_slices_are_disjoint(tmp_path):
    cfg_a = _cfg(tmp_path / "A", tmp_path)
    cfg_b = _cfg(tmp_path / "B", tmp_path)
    sync._save_sync_state({"a.md": "h"}, cfg_a)
    sync._save_sync_state({"b.md": "h"}, cfg_b)
    assert sync._load_sync_state(cfg_a) == {"a.md": "h"}
    assert sync._load_sync_state(cfg_b) == {"b.md": "h"}


def test_chromadb_collections_are_disjoint(tmp_path):
    """each vault's vectors live in its own collection inside the one shared db."""
    ca = store.get_collection(_cfg(tmp_path / "A", tmp_path))
    cb = store.get_collection(_cfg(tmp_path / "B", tmp_path))
    assert ca.name != cb.name
    ca.add(ids=["x::0"], embeddings=[[0.1, 0.2]], documents=["a doc"], metadatas=[{"file_path": "x"}])
    assert ca.count() == 1
    assert cb.count() == 0


def test_legacy_flat_state_adopted_by_owning_vault(tmp_path):
    """pre-per-vault flat sidecars are adopted for the vault whose notes they track."""
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "a.md"
    note.write_text("note")  # check_duplicate cleans entries whose file no longer exists
    config.write_json(config.SYNC_STATE_PATH, {str(note): "h"})
    config.write_json(config.SOURCES_INDEX_PATH, {"https://x": str(note)})
    cfg = _cfg(vault, tmp_path)

    assert sync._load_sync_state(cfg) == {str(note): "h"}
    assert write.check_duplicate("https://x", cfg) == note


def test_legacy_flat_state_parked_then_reclaimed(tmp_path):
    """a vault that does not own the legacy state starts empty; the owner still reclaims it later."""
    owner = tmp_path / "owner"
    other = tmp_path / "other"
    owner.mkdir()
    other.mkdir()
    config.write_json(config.SYNC_STATE_PATH, {str(owner / "a.md"): "h"})  # belongs to `owner`

    # `other` does not adopt owner's state, and the parking does not destroy it
    assert sync._load_sync_state(_cfg(other, tmp_path)) == {}
    # `owner`, configured later, reclaims the parked legacy
    assert sync._load_sync_state(_cfg(owner, tmp_path)) == {str(owner / "a.md"): "h"}
