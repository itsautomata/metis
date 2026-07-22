"""an empty `choices` array from the provider must not raise IndexError at any LLM call site."""

from types import SimpleNamespace

from metis.config import MetisConfig


class _Resp:
    choices = []


class _Completions:
    def create(self, *a, **k):
        return _Resp()


class _Chat:
    completions = _Completions()


class _FakeClient:
    chat = _Chat()


def test_chat_ask_survives_empty_choices(monkeypatch):
    """chat gives a plain message, not an IndexError, when the model returns zero choices."""
    from metis.chat import ask

    monkeypatch.setattr("metis.chat.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.chat.get_chat_model", lambda c: "m")
    monkeypatch.setattr("metis.chat._build_context", lambda results: "CTX")
    monkeypatch.setattr(
        "metis.chat.search_vault",
        lambda q, cfg, limit=5, note_path=None: [SimpleNamespace(file_path="a.md", score=0.9)],
    )

    answer, _sources, _conf = ask("q", MetisConfig())
    assert isinstance(answer, str) and answer  # a message, not a crash


def test_link_explain_survives_empty_choices(monkeypatch):
    """explain_connection returns "" instead of crashing on zero choices."""
    from metis.link import Connection, explain_connection

    monkeypatch.setattr("metis.link.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.link.get_chat_model", lambda c: "m")
    conn = Connection(source="a.md", target="b.md", score=0.9, source_preview="A", target_preview="B")
    assert explain_connection(conn, MetisConfig()) == ""


def test_expand_keywords_survives_empty_choices(monkeypatch):
    """extract_search_keywords returns "" instead of crashing on zero choices."""
    from metis.expand import extract_search_keywords

    monkeypatch.setattr("metis.expand.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.expand.get_chat_model", lambda c: "m")
    assert extract_search_keywords("q", MetisConfig()) == ""
