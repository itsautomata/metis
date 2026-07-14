"""tests for tolerant note reading (non-UTF-8 vault files)."""

from metis.textio import read_note_text


def test_utf8_roundtrip(tmp_path):
    p = tmp_path / "n.md"
    content = "# Héllo\n\ncafé “quote”"
    p.write_text(content, encoding="utf-8")
    assert read_note_text(p) == content


def test_cp1252_content_preserved(tmp_path):
    """a Word/Notes note saved as CP-1252 is read with its characters intact, not dropped."""
    p = tmp_path / "n.md"
    original = "It’s naïve, a café “quote” and an en dash – end."
    p.write_bytes(original.encode("cp1252"))

    out = read_note_text(p)
    assert out == original          # lossless: nothing dropped or mangled
    assert "�" not in out      # and no replacement-char garbage


def test_undecodable_bytes_never_crash(tmp_path):
    """bytes undefined in CP-1252 still decode via the Latin-1 last resort without raising."""
    p = tmp_path / "n.md"
    p.write_bytes(b"\x81\x8d body ok")  # 0x81/0x8d are undefined in CP-1252
    out = read_note_text(p)             # must not raise
    assert "body ok" in out


def test_accepts_str_path(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("plain", encoding="utf-8")
    assert read_note_text(str(p)) == "plain"
