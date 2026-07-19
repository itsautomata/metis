"""write processed content as markdown to the obsidian vault."""

import json
import re
from datetime import date
from pathlib import Path

import yaml

from metis.config import CONFIG_DIR, MetisConfig
from metis.ingest.process import ProcessedContent

SOURCES_INDEX_PATH = CONFIG_DIR / "sources.json"


def _load_sources_index() -> dict[str, str]:
    """load the source->path dedup index, tolerating a missing or corrupt file."""
    if not SOURCES_INDEX_PATH.exists():
        return {}
    try:
        data = json.loads(SOURCES_INDEX_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_sources_index(index: dict[str, str]) -> None:
    """persist the dedup index atomically so an interrupted write can't corrupt it."""
    SOURCES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SOURCES_INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(SOURCES_INDEX_PATH)


def check_duplicate(source_link: str) -> Path | None:
    """check if source was already ingested. returns existing path or None."""
    index = _load_sources_index()
    existing = index.get(source_link)
    if existing and Path(existing).exists():
        return Path(existing)
    # clean stale entry
    if existing:
        del index[source_link]
        _save_sources_index(index)
    return None


def _register_source(source_link: str, file_path: Path) -> None:
    """add source to the dedup index."""
    index = _load_sources_index()
    index[source_link] = str(file_path)
    _save_sources_index(index)


def slugify(title: str) -> str:
    """turn a title into a filename-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    # emoji/symbol-only or empty titles strip to "" -> a hidden ".md" dotfile; fall back.
    return slug.strip("-")[:80] or "note"


def _demote_headings(md: str, by: int = 2) -> str:
    """shift markdown ATX headings down `by` levels (capped at h6) so a body nests under the note's sections.

    fenced code blocks are skipped so a `#` comment line inside them is not mistaken for a heading.
    """
    out, in_fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = re.match(r"(#{1,6})(\s)", line)
            if m:
                line = "#" * min(len(m.group(1)) + by, 6) + line[m.end(1):]
        out.append(line)
    return "\n".join(out)


def build_markdown(
    title: str,
    text: str,
    source_link: str,
    source_type: str,
    processed: ProcessedContent,
    extra: dict | None = None,
) -> str:
    """build a complete markdown note with frontmatter."""
    key_points_str = "\n".join(f"- {kp}" for kp in processed.key_points)

    fm = {
        "source": source_link,
        "ingested": date.today(),
        "tags": processed.tags,
        "summary": processed.summary,
        "type": source_type,
    }
    if extra:
        for k, v in extra.items():
            if v:
                fm[k] = v
    dumped = yaml.safe_dump(fm, default_flow_style=None, sort_keys=False, allow_unicode=True, width=1000)
    frontmatter = f"---\n{dumped}---"

    # content section label varies by type
    content_label = "Transcript" if source_type == "youtube" else "Content"

    body = (
        f"# {title}\n"
        f"\n"
        f"> [source]({source_link})\n"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"{processed.summary}\n"
        f"\n"
        f"## Key Points\n"
        f"\n"
        f"{key_points_str}\n"
        f"\n"
        f"## {content_label}\n"
        f"\n"
        f"{_demote_headings(text)}"
    )

    return f"{frontmatter}\n\n{body}\n"


def write_to_vault(
    title: str,
    text: str,
    source_link: str,
    source_type: str,
    processed: ProcessedContent,
    config: MetisConfig,
    extra: dict | None = None,
) -> Path:
    """write the markdown note to the vault. returns the file path."""
    output_dir = config.vault_path / config.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(title)
    file_path = output_dir / f"{slug}.md"

    # avoid overwriting — append number if exists
    counter = 1
    while file_path.exists():
        file_path = output_dir / f"{slug}-{counter}.md"
        counter += 1

    markdown = build_markdown(title, text, source_link, source_type, processed, extra=extra)
    file_path.write_text(markdown, encoding="utf-8")
    _register_source(source_link, file_path)

    return file_path


def write_link_only(url: str, config: MetisConfig) -> Path:
    """write a link-only note via the standard write path."""
    import httpx

    title = url.split("/")[-1] if "/" in url else url

    # try to get a title via oembed for youtube
    if "youtube.com" in url or "youtu.be" in url:
        try:
            resp = httpx.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            title = resp.json().get("title", title)
        except Exception:
            pass

    empty = ProcessedContent(
        summary="link only — no transcript available at time of ingest.",
        key_points=[],
        tags=[],
        chunks=[],
    )

    return write_to_vault(title, "", url, "link", empty, config)
