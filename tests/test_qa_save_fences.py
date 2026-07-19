"""save_qa_to_note must ignore section markers that live inside code fences."""

from metis.chat import save_qa_to_note


def test_qa_save_ignores_content_marker_inside_a_fence(tmp_path):
    """a '## Content' inside a code block must not be used as the insert point (would corrupt it)."""
    note = tmp_path / "n.md"
    note.write_text(
        "# title\n\n"
        "```markdown\n"
        "## Content\n"
        "how metis formats a note\n"
        "```\n\n"
        "## Content\n\n"
        "the actual body\n",
        encoding="utf-8",
    )
    save_qa_to_note(str(note), "what is x?", "x is a thing.")
    out = note.read_text(encoding="utf-8")

    # the code fence is intact (nothing was spliced into it)
    assert "```markdown\n## Content\nhow metis formats a note\n```" in out
    assert "## Q&A" in out
    # the Q&A landed after the fence and before the real ## Content
    assert out.index("```") < out.index("## Q&A") < out.rindex("## Content")


def test_qa_save_normal_note_inserts_before_content(tmp_path):
    """no fences: Q&A still inserts before ## Content (no regression)."""
    note = tmp_path / "n.md"
    note.write_text("# title\n\n## Content\n\nbody\n", encoding="utf-8")
    save_qa_to_note(str(note), "q?", "a.")
    out = note.read_text(encoding="utf-8")

    assert "## Q&A" in out
    assert out.index("## Q&A") < out.index("## Content")
