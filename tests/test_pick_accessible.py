"""J28: METIS_ACCESSIBLE swaps arrow-key widgets for numbered/typed prompts (screen-reader path)."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from metis import pick


@pytest.fixture
def accessible(monkeypatch):
    monkeypatch.setenv("METIS_ACCESSIBLE", "1")


def _stdin(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_numbered_select_returns_chosen_value(accessible, monkeypatch):
    _stdin(monkeypatch, "2\n")
    rows = [
        SimpleNamespace(file_path="/v/a.md", score=0.9, text="aaa"),
        SimpleNamespace(file_path="/v/b.md", score=0.8, text="bbb"),
    ]
    got = pick.pick_search_result(rows, SimpleNamespace(vault_path=Path("/v")))
    assert got == "/v/b.md"


def test_numbered_select_blank_cancels(accessible, monkeypatch):
    _stdin(monkeypatch, "\n")
    got = pick.pick_secret(["provider-key", "embedding-key"])
    assert got is None


def test_numbered_select_out_of_range_cancels(accessible, monkeypatch):
    _stdin(monkeypatch, "9\n")
    got = pick.pick_secret(["provider-key", "embedding-key"])
    assert got is None


def test_typed_autocomplete_unique_substring(accessible, monkeypatch):
    _stdin(monkeypatch, "beta\n")
    assert pick._typed_choice("folder", ["alpha", "beta-notes", "gamma"]) == "beta-notes"


def test_typed_autocomplete_no_match_is_none(accessible, monkeypatch):
    _stdin(monkeypatch, "zzz\n")
    assert pick._typed_choice("folder", ["alpha", "beta", "gamma"]) is None


def test_typed_autocomplete_ambiguous_is_none(accessible, monkeypatch):
    _stdin(monkeypatch, "note\n")
    assert pick._typed_choice("note", ["daily-note", "meeting-note"]) is None


def test_pick_from_numbered_returns_value(accessible, monkeypatch):
    _stdin(monkeypatch, "2\n")
    assert pick.pick_from("provider", [("openai", ""), ("openrouter", "url")]) == "url"


def test_confirm_menu_numbered_yes(accessible, monkeypatch):
    _stdin(monkeypatch, "1\n")
    assert pick.confirm_menu("go?", default=False) is True


def test_confirm_menu_blank_uses_default(accessible, monkeypatch):
    _stdin(monkeypatch, "\n")
    assert pick.confirm_menu("go?", default=True) is True
