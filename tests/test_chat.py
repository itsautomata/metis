"""tests for chat Q&A saving."""

from metis.chat import format_qa_entry, save_qa_to_note


def test_format_qa_entry():
    entry = format_qa_entry("what is X?", "X is a thing.")
    assert "**what is X?**" in entry
    assert "*(metis," in entry
    assert "X is a thing." in entry


def test_save_qa_creates_section_before_transcript(tmp_path):
    note = tmp_path / "test.md"
    note.write_text(
        "# Title\n\n"
        "## Summary\n\nsome summary\n\n"
        "## Transcript\n\nthe transcript text"
    )

    save_qa_to_note(str(note), "what is X?", "X is a thing.")

    text = note.read_text()
    qa_pos = text.index("## Q&A")
    transcript_pos = text.index("## Transcript")
    assert qa_pos < transcript_pos
    assert "**what is X?**" in text


def test_save_qa_creates_section_before_content(tmp_path):
    note = tmp_path / "test.md"
    note.write_text(
        "# Title\n\n"
        "## Summary\n\nsome summary\n\n"
        "## Content\n\nthe article text"
    )

    save_qa_to_note(str(note), "what is Y?", "Y is a thing.")

    text = note.read_text()
    qa_pos = text.index("## Q&A")
    content_pos = text.index("## Content")
    assert qa_pos < content_pos


def test_save_qa_appends_to_existing_section(tmp_path):
    note = tmp_path / "test.md"
    note.write_text(
        "# Title\n\n"
        "## Q&A\n\n"
        "**first question?** *(metis, 2026-03-27)*\nfirst answer.\n\n"
        "## Transcript\n\ntext"
    )

    save_qa_to_note(str(note), "second question?", "second answer.")

    text = note.read_text()
    assert "first question?" in text
    assert "second question?" in text
    # Q&A section should still be before Transcript
    qa_pos = text.index("## Q&A")
    transcript_pos = text.index("## Transcript")
    assert qa_pos < transcript_pos


def test_save_qa_no_transcript_appends_at_end(tmp_path):
    note = tmp_path / "test.md"
    note.write_text("# Title\n\n## Summary\n\nsome summary")

    save_qa_to_note(str(note), "question?", "answer.")

    text = note.read_text()
    assert "## Q&A" in text
    assert "**question?**" in text


def test_save_qa_with_blockquotes_in_answer(tmp_path):
    """answer with markdown formatting should not corrupt the note."""
    note = tmp_path / "test.md"
    note.write_text("# Title\n\n## Transcript\n\ntext")

    answer = '> "this is a quote"\n\nthe explanation here.'
    save_qa_to_note(str(note), "what?", answer)

    text = note.read_text()
    assert '> "this is a quote"' in text
    assert "## Transcript" in text


def test_save_qa_empty_answer(tmp_path):
    note = tmp_path / "test.md"
    note.write_text("# Title\n\n## Content\n\ntext")

    save_qa_to_note(str(note), "question?", "")

    text = note.read_text()
    assert "**question?**" in text
    assert "## Content" in text


# --- query simplification ---

from metis.chat import _simplify_query


def test_simplify_query_strips_stop_words():
    result = _simplify_query("what does he say about nash equilibrium?")
    assert "what" not in result
    assert "does" not in result
    assert "nash" in result
    assert "equilibrium" in result


def test_simplify_query_keeps_keywords():
    result = _simplify_query("multi-agent coordination in dynamic environments")
    assert "multi-agent" in result or "agent" in result
    assert "coordination" in result
    assert "dynamic" in result


def test_simplify_query_limits_to_5():
    result = _simplify_query("transformer attention mechanism self supervised learning neural network architecture")
    words = result.split()
    assert len(words) <= 5


def test_simplify_query_empty():
    result = _simplify_query("")
    assert result == ""


def test_simplify_query_all_stop_words():
    result = _simplify_query("what is the a an")
    # falls back to original since no keywords remain
    assert result == "what is the a an"
