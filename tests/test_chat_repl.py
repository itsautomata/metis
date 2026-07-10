"""tests for the interactive chat repl + ask() conversation memory."""

from types import SimpleNamespace

from metis.config import MetisConfig


class _Msg:
    content = "an answer"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


class _Completions:
    def create(self, model, messages, temperature):
        _Completions.seen = messages
        return _Resp()


class _Chat:
    completions = _Completions()


class _FakeClient:
    chat = _Chat()


def test_ask_threads_history_into_messages(monkeypatch):
    from metis.chat import ask

    monkeypatch.setattr("metis.chat.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.chat.get_chat_model", lambda c: "m")
    monkeypatch.setattr("metis.chat._build_context", lambda results: "CONTEXT")
    monkeypatch.setattr(
        "metis.chat.search_vault",
        lambda q, cfg, limit=5, note_path=None: [SimpleNamespace(file_path="a.md", score=0.9)],
    )

    history = [{"role": "user", "content": "prior q"}, {"role": "assistant", "content": "prior a"}]
    answer, sources, _ = ask("current q", MetisConfig(), history=history)

    contents = [m["content"] for m in _Completions.seen]
    assert _Completions.seen[0]["role"] == "system"
    assert "prior q" in contents and "prior a" in contents
    assert contents.index("prior q") < contents.index("current q")  # history before the new question
    assert answer == "an answer"


def test_ask_without_history_is_unchanged(monkeypatch):
    from metis.chat import ask

    monkeypatch.setattr("metis.chat.get_client", lambda c: _FakeClient())
    monkeypatch.setattr("metis.chat.get_chat_model", lambda c: "m")
    monkeypatch.setattr("metis.chat._build_context", lambda results: "CONTEXT")
    monkeypatch.setattr(
        "metis.chat.search_vault",
        lambda q, cfg, limit=5, note_path=None: [SimpleNamespace(file_path="a.md", score=0.9)],
    )

    ask("only q", MetisConfig())

    roles = [m["role"] for m in _Completions.seen]
    assert roles == ["system", "user", "user"]  # system, question, context -- no extra turns


def test_chat_repl_runs_a_turn_then_exits(monkeypatch):
    from metis import cli

    inputs = iter(["hi", None])  # one question, then cancel / EOF

    class _Prompt:
        def ask(self):
            return next(inputs)

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Prompt())
    monkeypatch.setattr("metis.chat.ask", lambda q, config, note_path=None, history=None: ("answer", ["s.md"], 0.9))

    cli._chat_repl(MetisConfig(), None, False)  # runs one turn, then exits cleanly (no exception)


def test_chat_repl_survives_a_failed_turn(monkeypatch):
    """a ProviderError on one turn must not kill the whole session."""
    from metis import cli
    from metis.client import ProviderError

    inputs = iter(["q1", "q2", None])

    class _Prompt:
        def ask(self):
            return next(inputs)

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Prompt())

    calls = []

    def _ask(q, config, note_path=None, history=None):
        calls.append(q)
        if q == "q1":
            raise ProviderError("rate limited")
        return ("ok", [], 0.9)

    monkeypatch.setattr("metis.chat.ask", _ask)

    cli._chat_repl(MetisConfig(), None, False)  # must not raise
    assert calls == ["q1", "q2"]  # survived q1's failure, processed q2


def test_chat_repl_menu_exits(monkeypatch):
    """bare enter opens the arrow menu; choosing exit ends the loop without calling ask."""
    from metis import cli

    texts = iter([""])  # a bare enter opens the menu

    class _Text:
        def ask(self):
            return next(texts)

    class _Select:
        def ask(self):
            return "exit"

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Text())
    monkeypatch.setattr("questionary.select", lambda *a, **k: _Select())

    called = []
    monkeypatch.setattr("metis.chat.ask", lambda *a, **k: called.append(1) or ("x", [], 0.9))

    cli._chat_repl(MetisConfig(), None, False)
    assert called == []  # the menu path is not a question; exit breaks cleanly


def test_chat_repl_menu_cancel_keeps_chatting(monkeypatch):
    """ctrl-c / cancel on the menu (a None result) returns to chatting, not quit."""
    from metis import cli

    texts = iter(["", "after", None])  # menu (cancel), then a real question, then quit

    class _Text:
        def ask(self):
            return next(texts)

    class _Select:
        def ask(self):
            return None  # ctrl-c / cancel on the menu

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Text())
    monkeypatch.setattr("questionary.select", lambda *a, **k: _Select())

    calls = []
    monkeypatch.setattr(
        "metis.chat.ask",
        lambda q, config, note_path=None, history=None: calls.append(q) or ("ok", [], 0.9),
    )

    cli._chat_repl(MetisConfig(), None, False)
    assert calls == ["after"]  # survived the menu-cancel and kept chatting


def test_chat_repl_menu_keep_chatting(monkeypatch):
    """the 'keep chatting' menu item returns to the prompt without saving or exiting."""
    from metis import cli

    texts = iter(["", "after", None])  # menu (keep chatting), then a question, then quit

    class _Text:
        def ask(self):
            return next(texts)

    class _Select:
        def ask(self):
            return "chat"  # the explicit "keep chatting" choice

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Text())
    monkeypatch.setattr("questionary.select", lambda *a, **k: _Select())

    calls = []
    monkeypatch.setattr(
        "metis.chat.ask",
        lambda q, config, note_path=None, history=None: calls.append(q) or ("ok", [], 0.9),
    )

    cli._chat_repl(MetisConfig(), None, False)
    assert calls == ["after"]  # kept chatting, then answered the next question


def test_chat_repl_menu_saves_last_answer(monkeypatch, tmp_path):
    """after a turn, the menu's save writes the last Q&A into the target note."""
    from metis import cli

    note = tmp_path / "n.md"
    note.write_text("# n\n\n## Content\nbody\n", encoding="utf-8")

    texts = iter(["what is x?", "", ""])  # question, then two bare-enters (a menu each)
    selects = iter(["save", "exit"])  # first menu saves, second exits

    class _Text:
        def ask(self):
            return next(texts)

    class _Select:
        def ask(self):
            return next(selects)

    monkeypatch.setattr("questionary.text", lambda *a, **k: _Text())
    monkeypatch.setattr("questionary.select", lambda *a, **k: _Select())
    monkeypatch.setattr(
        "metis.chat.ask",
        lambda q, config, note_path=None, history=None: ("x is a letter", ["n.md"], 0.9),
    )

    cli._chat_repl(MetisConfig(), str(note), False)

    saved = note.read_text(encoding="utf-8")
    assert "x is a letter" in saved  # the answer was saved on demand
    assert "## Q&A" in saved
