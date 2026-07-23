"""choosing the 'skip' menu item must return None, not the string 'skip' (a questionary Choice gotcha)."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from metis import pick


class _FakeSelect:
    """stands in for questionary.select; .ask() returns the last choice's value (the 'skip' item)."""
    def __init__(self, choices):
        self._choices = choices

    def ask(self):
        return self._choices[-1].value


@pytest.fixture
def select_picks_skip(monkeypatch):
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _FakeSelect(k["choices"]))


def test_search_pick_skip_maps_to_none(select_picks_skip):
    rows = [SimpleNamespace(file_path="/v/a.md", score=0.9, text="body")]
    assert pick.pick_search_result(rows, SimpleNamespace(vault_path=Path("/v"))) is None


def test_wikipedia_pick_skip_maps_to_none(select_picks_skip):
    assert pick.pick_wikipedia([("Stoicism", "preview")]) is None


def test_search_pick_real_result_still_returns_its_path(monkeypatch):
    # first choice selected -> the real file_path, unaffected by the skip fix
    monkeypatch.setattr(pick.questionary, "select", lambda *a, **k: _Select0(k["choices"]))
    rows = [SimpleNamespace(file_path="/v/a.md", score=0.9, text="body")]
    assert pick.pick_search_result(rows, SimpleNamespace(vault_path=Path("/v"))) == "/v/a.md"


class _Select0:
    def __init__(self, choices):
        self._choices = choices

    def ask(self):
        return self._choices[0].value


def test_accessible_search_pick_skip_maps_to_none(monkeypatch):
    monkeypatch.setenv("METIS_ACCESSIBLE", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("2\n"))  # 1 result + skip -> skip is option 2
    rows = [SimpleNamespace(file_path="/v/a.md", score=0.9, text="body")]
    assert pick.pick_search_result(rows, SimpleNamespace(vault_path=Path("/v"))) is None
