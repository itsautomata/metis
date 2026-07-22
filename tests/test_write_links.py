"""write_links must not false-report on a no-op, nor splice into a code fence."""

from metis.config import MetisConfig
from metis.link import Connection, write_links


def _conn(note, target, score=0.9):
    return Connection(source=str(note), target=target, score=score, source_preview="", target_preview="")


def test_second_identical_write_reports_zero(tmp_path):
    """re-running with the same connections changes nothing, so it must report 0, not success."""
    note = tmp_path / "src.md"
    note.write_text("# src\n\nbody\n", encoding="utf-8")
    cfg = MetisConfig(vault_path=tmp_path)
    conns = [_conn(note, "/vault/target.md")]

    assert write_links(conns, cfg) == 1
    assert write_links(conns, cfg) == 0                     # no false "wrote 1" on a no-op
    assert note.read_text().count("## Connections") == 1    # and not duplicated


def test_ignores_content_marker_inside_a_fence(tmp_path):
    """a '## Content' inside a code fence must not be the insert point (would break the fence)."""
    note = tmp_path / "src.md"
    note.write_text(
        "# src\n\n```\n## Content\n```\n\n## Content\n\nreal body\n",
        encoding="utf-8",
    )
    write_links([_conn(note, "/v/t.md", 0.8)], MetisConfig(vault_path=tmp_path))
    out = note.read_text(encoding="utf-8")

    assert "```\n## Content\n```" in out                    # fence intact
    assert out.index("```") < out.index("## Connections") < out.rindex("## Content")
