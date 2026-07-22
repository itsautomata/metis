"""Ctrl-D (EOFError) at an interactive prompt must cancel (return None), like Ctrl-C already does."""

from metis import pick


class _EofQuestion:
    def ask(self):
        raise EOFError


class _ValueQuestion:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


def test_ask_treats_eof_as_cancel():
    """a prompt that raises EOFError (Ctrl-D on an empty buffer) returns None, not a traceback."""
    assert pick._ask(_EofQuestion()) is None


def test_ask_returns_value_normally():
    """a normal answer passes straight through (no regression)."""
    assert pick._ask(_ValueQuestion("folder-a")) == "folder-a"
