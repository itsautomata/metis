"""tests for connection explanation"""

from metis.config import MetisConfig
from metis.link import Connection, explain_connection


class _Msg:
    content = None


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


class _Completions:
    def create(self, *a, **k):
        return _Resp()


class _Chat:
    completions = _Completions()


class _FakeClient:
    chat = _Chat()


def test_explain_connection_survives_null_content(monkeypatch):
    """a provider returning message.content=None must yield "" instead of crashing."""
    monkeypatch.setattr("metis.link.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.link.get_chat_model", lambda c: "m")
    conn = Connection(source="a.md", target="b.md", score=0.9, source_preview="A", target_preview="B")
    assert explain_connection(conn, MetisConfig()) == ""
