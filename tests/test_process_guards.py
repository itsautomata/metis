"""summarize_and_tag must not crash or corrupt on a malformed LLM response."""

from types import SimpleNamespace

from metis.config import MetisConfig
from metis.ingest.process import _sanitize_key_points, _sanitize_tags, summarize_and_tag


def _patch_llm(monkeypatch, *, content=None, choices=None):
    from metis.ingest import process

    def create(**kwargs):
        if choices is not None:
            return SimpleNamespace(choices=choices)
        msg = SimpleNamespace(message=SimpleNamespace(content=content))
        return SimpleNamespace(choices=[msg])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(process, "get_client", lambda c: client)
    monkeypatch.setattr(process, "get_chat_model", lambda c: "m")


def test_empty_choices_skips_metadata(monkeypatch):
    """a provider returning no choices skips summary/tags instead of raising IndexError."""
    _patch_llm(monkeypatch, choices=[])
    assert summarize_and_tag("some longer body text", MetisConfig()) == ("", [], [])


def test_non_object_json_skips_metadata(monkeypatch):
    """valid-but-non-object JSON skips instead of AttributeError on parsed.get()."""
    _patch_llm(monkeypatch, content='["a", "b", "c"]')
    assert summarize_and_tag("some longer body text", MetisConfig()) == ("", [], [])


def test_scalar_tags_do_not_char_explode(monkeypatch):
    """a string tags field must become [], not ['p','y','t','h','o','n']."""
    _patch_llm(monkeypatch, content='{"summary": "s", "key_points": ["a", "b"], "tags": "python"}')
    _, _, tags = summarize_and_tag("some longer body text", MetisConfig())
    assert tags == []


def test_sanitizers_reject_non_lists():
    """the sanitizers coerce a scalar to [] rather than iterate its characters/raise."""
    assert _sanitize_tags("python") == []
    assert _sanitize_tags(123) == []          # not a TypeError
    assert _sanitize_key_points("x") == []
    assert _sanitize_key_points(42) == []     # not a TypeError
