"""tests for process helpers."""

from metis.ingest.process import _strip_code_fence


def test_strip_plain_json():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_strip_json_fence():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_bare_fence():
    assert _strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_none():
    assert _strip_code_fence(None) == ""
