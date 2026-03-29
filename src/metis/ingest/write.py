"""write processed content as markdown to the obsidian vault."""

import json
import re
from datetime import date
from pathlib import Path

from metis.config import MetisConfig, CONFIG_DIR
from metis.ingest.process import ProcessedContent

SOURCES_INDEX_PATH = CONFIG_DIR / "sources.json"


def _load_sources_index() -> dict[str, str]:
    if SOURCES_INDEX_PATH.exists():
        return json.loads(SOURCES_INDEX_PATH.read_text())
    return {}


def _save_sources_index(index: dict[str, str]) -> None:
    SOURCES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_INDEX_PATH.write_text(json.dumps(index, indent=2))


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
    return slug.strip("-")[:80]


def build_markdown(
    title: str,
    text: str,
    source_link: str,
    source_type: str,
    processed: ProcessedContent,
    extra: dict | None = None,
) -> str:
    """build a complete markdown note with frontmatter."""
    tags_str = ", ".join(processed.tags)
    key_points_str = "\n".join(f"- {kp}" for kp in processed.key_points)
    today = date.today().isoformat()

    # build frontmatter lines
    fm_lines = [
        "---",
        f"source: \"{source_link}\"",
        f"ingested: {today}",
        f"tags: [{tags_str}]",
        f"summary: \"{processed.summary}\"",
        f"type: {source_type}",
    ]
    if extra:
        for k, v in extra.items():
            if v:
                fm_lines.append(f"{k}: \"{v}\"")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines)

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
        f"{text}"
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
