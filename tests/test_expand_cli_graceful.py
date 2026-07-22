"""a failed wikipedia ingest during chat --expand must fall back gracefully, not crash mid-chat."""

from metis.cli import _offer_expand
from metis.config import MetisConfig
from metis.expand import ExternalResult


def test_offer_expand_survives_ingest_failure(monkeypatch):
    """when ingest_external raises ValueError, _offer_expand returns without re-answering or crashing."""
    monkeypatch.setattr("metis.cli.typer.confirm", lambda *a, **k: True)
    monkeypatch.setattr("metis.expand.extract_search_keywords", lambda q, c: "kw")
    monkeypatch.setattr(
        "metis.expand.search_wikipedia",
        lambda kw: [ExternalResult(title="T", preview="p", url="https://en.wikipedia.org/wiki/T", source_type="wikipedia")],
    )
    monkeypatch.setattr("metis.pick.pick_wikipedia", lambda choices: "T")

    def _boom(result, config):
        raise ValueError("could not extract text from: https://en.wikipedia.org/wiki/T")

    monkeypatch.setattr("metis.expand.ingest_external", _boom)

    reached = {"ask": False}

    def _ask(*a, **k):
        reached["ask"] = True
        return ("expanded", [], 1.0)

    monkeypatch.setattr("metis.chat.ask", _ask)

    # note_path=None keeps the save path silent; the call must simply return, not raise
    _offer_expand("question", "original answer", MetisConfig(), None, False)
    assert reached["ask"] is False  # the failed ingest returned before the re-answer step
