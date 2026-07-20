"""keep the whole suite off the real ~/.metis: redirect every sidecar to a per-test tmp dir.

Without this, any test that drives the ingest/sync commands writes to the operator's real state
(sync_state.json, canary.json, categorization.json). Individual tests may still override these.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_metis_state(tmp_path, monkeypatch):
    from metis import classify
    from metis.index import canary, sync

    state = tmp_path / "_metis_state"
    state.mkdir(exist_ok=True)
    monkeypatch.setattr(sync, "SYNC_STATE_PATH", state / "sync_state.json")
    monkeypatch.setattr(canary, "CANARY_PATH", state / "canary.json")
    monkeypatch.setattr(classify, "CATEGORIZATION_PATH", state / "categorization.json")
