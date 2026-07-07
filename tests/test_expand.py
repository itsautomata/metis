"""tests for search-keyword extraction"""

from metis.config import MetisConfig
from metis.expand import extract_search_keywords


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


def test_extract_keywords_survives_null_content(monkeypatch):
    """a provider returning message.content=None must yield "" instead of crashing."""
    monkeypatch.setattr("metis.expand.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.expand.get_chat_model", lambda c: "m")
    assert extract_search_keywords("some question", MetisConfig()) == ""
