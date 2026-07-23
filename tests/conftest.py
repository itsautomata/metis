"""keep the whole suite off the real ~/.metis: redirect every sidecar to a per-test tmp dir.

Without this, any test that drives the ingest/sync commands writes to the operator's real state
(sync_state.json, canary.json, categorization.json). Individual tests may still override these.
"""

import pytest


@pytest.fixture(autouse=True)
def _deterministic_console(monkeypatch):
    """CLI tests assert on rendered output. Pin the console to a wide, un-highlighted render so a
    narrow or non-tty runner cannot wrap, truncate, or number-highlight the asserted text.
    """
    from rich.console import Console

    from metis import cli
    from metis.ui import THEME

    monkeypatch.setattr(cli, "console", Console(width=200, highlight=False, theme=THEME))


@pytest.fixture(autouse=True)
def _isolate_metis_state(tmp_path, monkeypatch):
    from metis import config
    from metis.index import canary

    state = tmp_path / "_metis_state"
    state.mkdir(exist_ok=True)
    # the vault-scoped sidecars now live on config (content-keyed by vault); the canary is global
    monkeypatch.setattr(config, "SOURCES_INDEX_PATH", state / "sources.json")
    monkeypatch.setattr(config, "SYNC_STATE_PATH", state / "sync_state.json")
    monkeypatch.setattr(config, "CATEGORIZATION_PATH", state / "categorization.json")
    monkeypatch.setattr(canary, "CANARY_PATH", state / "canary.json")


@pytest.fixture(autouse=True)
def _release_chromadb_systems():
    """chromadb caches a System (holding open sqlite handles) per client path and never releases it.
    Across a long suite that piles up open files until SQLite cannot open another database once the
    process reaches its open-file limit (256 by default on macOS, commonly 1024 on Linux, lower in
    some containers). Clearing the cache alone only partly frees them; the rust bindings release
    their descriptors on garbage collection, so collect after clearing.
    """
    yield
    import gc

    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    gc.collect()
