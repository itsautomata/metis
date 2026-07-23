"""dynamic text (queries, LLM answers, note previews) is markup-escaped, so brackets never crash rich."""

from types import SimpleNamespace

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig

runner = CliRunner()


def test_search_bracketed_query_and_preview_do_not_crash(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    row = SimpleNamespace(file_path="/v/n.md", score=0.9, text="run rm [/tmp/cache] then read [docs](x)")
    monkeypatch.setattr("metis.search.search_vault", lambda q, c, limit=5: [row])
    monkeypatch.setattr("metis.pick.pick_search_result", lambda results, config: None)

    result = runner.invoke(app, ["search", "read [/etc/passwd]"])

    assert result.exit_code == 0, result.output
    assert "[/etc/passwd]" in result.output   # the query echoes verbatim
    assert "[/tmp/cache]" in result.output     # the note preview shows verbatim
    assert "[docs]" in result.output           # a markdown link label is not eaten


def test_chat_answer_with_brackets_does_not_crash(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr("metis.chat.ask", lambda q, config, note_path=None: ("run rm [/tmp/x]; see [d](u)", [], 0.9))

    result = runner.invoke(app, ["chat", "what is in [/dev/null]?"])

    assert result.exit_code == 0, result.output
    assert "[/dev/null]" in result.output   # the question echoes verbatim
    assert "[/tmp/x]" in result.output       # the answer shows verbatim
