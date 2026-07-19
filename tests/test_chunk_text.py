"""chunk_text must not emit empty chunks or leave a space-free blob oversized."""

from metis.ingest.process import chunk_text


def test_space_free_blob_is_char_sliced():
    """a single giant token (no spaces) is sliced into bounded chunks, none empty, none oversized."""
    blob = "x" * 4000  # one word, no whitespace, well over max_chars*2
    chunks = chunk_text(blob, max_chars=1500)

    assert "" not in chunks                        # no spurious empty chunk
    assert all(len(c) <= 1500 for c in chunks)     # nothing left oversized
    assert "".join(chunks) == blob                 # content preserved, nothing dropped


def test_oversize_run_inside_normal_text_leaves_no_empty_chunk():
    """a giant token between normal paragraphs splits cleanly, no empty chunk."""
    text = "intro paragraph.\n\n" + "y" * 3500 + "\n\ntail paragraph."
    chunks = chunk_text(text, max_chars=1500)

    assert "" not in chunks
    assert all(len(c) <= 1500 * 2 for c in chunks)


def test_short_text_stays_one_chunk():
    """short text is untouched (no regression)."""
    assert chunk_text("a short note", max_chars=1500) == ["a short note"]
