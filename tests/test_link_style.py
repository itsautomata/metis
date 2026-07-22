"""link style is read from the vault's notes app, overridable via config, resolved at write time."""

import json

from metis import config
from metis.config import MetisConfig
from metis.link import (
    Connection,
    _get_existing_links,
    detect_link_style,
    resolve_link_style,
    write_links,
)


def _conn(source, target, score=0.9):
    return Connection(source=str(source), target=str(target), score=score, source_preview="", target_preview="")


# --- detection: the tool's own marker is definitive ---

def test_obsidian_default_is_wikilink(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    assert detect_link_style(tmp_path) == "wikilink"


def test_obsidian_use_markdown_links_is_markdown(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text(json.dumps({"useMarkdownLinks": True}), encoding="utf-8")
    assert detect_link_style(tmp_path) == "markdown"


def test_obsidian_without_app_json_is_wikilink(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    assert detect_link_style(tmp_path) == "wikilink"


def test_corrupt_app_json_falls_back_to_wikilink(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text("{ truncated", encoding="utf-8")
    assert detect_link_style(tmp_path) == "wikilink"


def test_logseq_is_wikilink(tmp_path):
    (tmp_path / "logseq").mkdir()
    (tmp_path / "logseq" / "config.edn").write_text(";; logseq", encoding="utf-8")
    assert detect_link_style(tmp_path) == "wikilink"


def test_dendron_is_wikilink(tmp_path):
    (tmp_path / "dendron.yml").write_text("version: 5", encoding="utf-8")
    assert detect_link_style(tmp_path) == "wikilink"


def test_foam_is_wikilink(tmp_path):
    (tmp_path / ".foam").mkdir()
    assert detect_link_style(tmp_path) == "wikilink"


def test_plain_folder_defaults_to_markdown(tmp_path):
    (tmp_path / "note.md").write_text("# note", encoding="utf-8")
    assert detect_link_style(tmp_path) == "markdown"


# --- override wins over detection ---

def test_config_link_style_overrides_detection(tmp_path):
    (tmp_path / ".obsidian").mkdir()  # detection alone would say wikilink
    assert resolve_link_style(MetisConfig(vault_path=tmp_path, link_style="markdown")) == "markdown"


def test_unset_link_style_auto_detects(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    assert resolve_link_style(MetisConfig(vault_path=tmp_path)) == "wikilink"


def test_config_reads_link_style(monkeypatch, tmp_path):
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("link_style: markdown\n", encoding="utf-8")
    assert config.load_config().link_style == "markdown"


def test_config_invalid_link_style_falls_back_to_auto(monkeypatch, tmp_path):
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("link_style: bogus\n", encoding="utf-8")
    assert config.load_config().link_style == ""


# --- writing in the resolved style ---

def test_write_wikilink_style(tmp_path):
    src = tmp_path / "src.md"
    src.write_text("# src\n\nbody\n", encoding="utf-8")
    write_links([_conn(src, tmp_path / "target.md")], MetisConfig(vault_path=tmp_path, link_style="wikilink"))
    assert "[[target]]" in src.read_text()


def test_write_markdown_style(tmp_path):
    src = tmp_path / "src.md"
    src.write_text("# src\n\nbody\n", encoding="utf-8")
    write_links([_conn(src, tmp_path / "target.md")], MetisConfig(vault_path=tmp_path, link_style="markdown"))
    out = src.read_text()
    assert "[target](target.md)" in out
    assert "[[target]]" not in out


def test_existing_markdown_link_is_seen_for_dedup(tmp_path):
    """dedup must recognize a prior markdown-style link, so a style change does not re-add it."""
    note = tmp_path / "n.md"
    note.write_text("see [target](target.md) and [[Other]]\n", encoding="utf-8")
    names = _get_existing_links(note)
    assert "target" in names and "Other" in names


def test_ambiguous_stem_qualifies_wikilink(tmp_path):
    """two notes share a stem: the wikilink is path-qualified so obsidian links the intended one."""
    (tmp_path / "work").mkdir()
    (tmp_path / "personal").mkdir()
    (tmp_path / "work" / "review.md").write_text("w")
    (tmp_path / "personal" / "review.md").write_text("p")
    src = tmp_path / "src.md"
    src.write_text("# src\n", encoding="utf-8")

    write_links([_conn(src, str(tmp_path / "personal" / "review.md"))],
                MetisConfig(vault_path=tmp_path, link_style="wikilink"))
    out = src.read_text()
    assert "[[personal/review]]" in out   # qualified
    assert "[[review]]" not in out        # not the ambiguous bare form


def test_unique_stem_stays_bare_wikilink(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "unique.md").write_text("u")
    src = tmp_path / "src.md"
    src.write_text("# src\n", encoding="utf-8")

    write_links([_conn(src, str(tmp_path / "notes" / "unique.md"))],
                MetisConfig(vault_path=tmp_path, link_style="wikilink"))
    assert "[[unique]]" in src.read_text()   # unambiguous -> bare


def test_wikilink_alias_and_heading_seen_for_dedup(tmp_path):
    """aliased / heading / path-qualified wikilinks reduce to the note name for dedup."""
    note = tmp_path / "n.md"
    note.write_text("[[review|last quarter]] and [[projects/plan#Q3]]\n", encoding="utf-8")
    names = _get_existing_links(note)
    assert "review" in names   # alias stripped
    assert "plan" in names     # path + heading reduced to the stem


def test_detect_tolerates_non_utf8_app_json(tmp_path):
    """a non-UTF-8 app.json falls back to the obsidian default instead of crashing."""
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_bytes(b"\xff\xfe not utf-8")
    assert detect_link_style(tmp_path) == "wikilink"


def test_markdown_link_percent_encodes_spaces(tmp_path):
    """markdown links percent-encode the destination (obsidian's form), not angle brackets."""
    (tmp_path / "my notes").mkdir()
    (tmp_path / "my notes" / "a note.md").write_text("x")
    src = tmp_path / "src.md"
    src.write_text("# src\n", encoding="utf-8")

    write_links([_conn(src, str(tmp_path / "my notes" / "a note.md"))],
                MetisConfig(vault_path=tmp_path, link_style="markdown"))
    connections = src.read_text().split("## Connections", 1)[1]
    assert "%20" in connections          # spaces encoded
    assert "<" not in connections        # not the angle-bracket form
    assert "[a note](" in connections    # link text is the bare note name
