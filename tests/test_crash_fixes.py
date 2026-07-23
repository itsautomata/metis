"""regression tests for crashes and robustness fix pass."""

import io
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


def test_search_limit_zero_is_a_clean_error(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    result = runner.invoke(app, ["search", "x", "-n", "0"])
    assert result.exit_code == 2  # click IntRange rejection, not an uncaught TypeError
    assert not (result.exception and type(result.exception).__name__ == "TypeError")


def test_strip_code_fence_non_string_returns_empty():
    from metis.ingest.process import _strip_code_fence
    assert _strip_code_fence(["a", "b"]) == ""
    assert _strip_code_fence(None) == ""


def test_numbered_choice_superscript_digit_no_crash(monkeypatch):
    from metis import pick
    monkeypatch.setattr("sys.stdin", io.StringIO("²\n"))  # superscript two: isdigit true, int() raises
    assert pick._numbered_choice("p", [("a", "a")]) is None


def test_oembed_http_error_becomes_value_error(monkeypatch):
    from metis.ingest import extract
    monkeypatch.setattr(extract.httpx, "get", _raise(httpx.ConnectError("down")))
    with pytest.raises(ValueError):
        extract._extract_via_oembed("https://x.com/j/status/20")


def test_lang_pick_eof_returns_default(monkeypatch):
    from metis.ingest import extract
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    avail = [SimpleNamespace(language="English", language_code="en", is_generated=False)]
    assert extract._interactive_lang_pick(avail) is avail[0]


def test_ingest_external_dedup_skips_reprocessing(monkeypatch, tmp_path):
    import metis.expand as X
    cached = tmp_path / "cached.md"
    cached.write_text("cached body")
    monkeypatch.setattr(X, "check_duplicate", lambda url, config: cached)
    monkeypatch.setattr(X, "extract", _raise(AssertionError("re-extracted a duplicate")))
    file_path, _ = X.ingest_external(SimpleNamespace(url="u", source_type="wikipedia"), MetisConfig())
    assert file_path == cached


def _wire_expand(monkeypatch, keyword_fn, ingest_fn):
    import metis.chat as CH
    import metis.cli as C
    import metis.expand as E
    import metis.pick as P
    from metis.expand import ExternalResult
    monkeypatch.setattr(C, "_confirm", lambda *a, **k: True)
    monkeypatch.setattr(E, "extract_search_keywords", keyword_fn)
    monkeypatch.setattr(E, "search_wikipedia", lambda kw: [ExternalResult("Stoicism", "p", "u", "wikipedia")])
    monkeypatch.setattr(P, "pick_wikipedia", lambda ch: "Stoicism")
    monkeypatch.setattr(E, "ingest_external", ingest_fn)
    return C, CH


def test_expand_keyword_failure_saves_original(monkeypatch):
    from metis.client import ProviderError
    saved = []
    C, _ = _wire_expand(monkeypatch, _raise(ProviderError("503")), lambda b, c: (Path("/w/S.md"), "t"))
    monkeypatch.setattr(C, "_maybe_save_qa", lambda *a, **k: saved.append(1))
    C._offer_expand("q", "orig answer", MetisConfig(), "/n.md", True)
    assert saved  # the original answer is not silently dropped on a keyword-extraction failure


def test_expand_ingest_failure_saves_original(monkeypatch):
    from metis.client import ProviderError
    saved = []
    C, _ = _wire_expand(monkeypatch, lambda q, c: "Stoicism", _raise(ProviderError("embed down")))
    monkeypatch.setattr(C, "_maybe_save_qa", lambda *a, **k: saved.append(1))
    C._offer_expand("q", "orig answer", MetisConfig(), "/n.md", True)
    assert saved  # a non-ValueError ingest failure still falls back


def test_expand_reanswers_scoped_to_new_article(monkeypatch):
    cap = {}
    C, CH = _wire_expand(monkeypatch, lambda q, c: "Stoicism", lambda b, c: (Path("/w/Stoicism.md"), "t"))
    monkeypatch.setattr(CH, "ask", lambda q, c, note_path=None: (cap.update(np=note_path) or ("re", [], 0.9)))
    monkeypatch.setattr(C, "_maybe_save_qa", lambda *a, **k: None)
    C._offer_expand("q", "orig", MetisConfig(), "/vault/mynote.md", False)
    assert cap["np"] == "/w/Stoicism.md"  # the re-answer sees the ingested article, not the original scope
