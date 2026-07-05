"""tests for LLM-output sanitization."""

from metis.ingest.process import (
    _sanitize_key_points,
    _sanitize_summary,
    _sanitize_tags,
)

# --- legitimate content survives ---

def test_summary_with_ignore_preserved():
    s = "the paper argues we should not ignore small effects."
    assert _sanitize_summary(s, "x" * 500) == s


def test_summary_with_you_are_preserved():
    s = "the thesis is that you are shaped by your habits."
    assert _sanitize_summary(s, "x" * 500) == s


def test_key_points_with_trigger_words_preserved():
    points = ["don't ignore edge cases", "you are the average of your inputs"]
    assert _sanitize_key_points(points) == points


def test_tag_with_trigger_word_preserved():
    assert "override" in _sanitize_tags(["override", "polymorphism"])


# --- structural guards still hold ---

def test_summary_capped_to_original_length():
    assert _sanitize_summary("x" * 100, "yy") == "xx"


def test_summary_non_string_rejected():
    assert _sanitize_summary(None, "text") == ""
    assert _sanitize_summary(123, "text") == ""


def test_key_points_drop_overlong():
    assert _sanitize_key_points(["short", "x" * 400]) == ["short"]


def test_key_points_drop_non_string():
    assert _sanitize_key_points(["ok", 42, None]) == ["ok"]


def test_tags_lowercased_and_stripped():
    assert _sanitize_tags(["  AI  ", "ML"]) == ["ai", "ml"]


def test_tags_drop_overlong():
    assert _sanitize_tags(["x" * 40, "ok"]) == ["ok"]


def test_tags_drop_multiword_without_hyphen():
    assert _sanitize_tags(["machine learning", "deep-learning"]) == ["deep-learning"]


def test_tags_drop_non_string():
    assert _sanitize_tags(["ok", 5, None]) == ["ok"]
