"""write processed content as markdown to the obsidian vault."""

import re
from datetime import date
from pathlib import Path

from metis.config import MetisConfig
from metis.ingest.process import ProcessedContent


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

    return file_path


def write_link_only(url: str, config: MetisConfig) -> Path:
    """write a minimal link-only note (no summary, no embedding)."""
    today = date.today().isoformat()
    title = url.split("/")[-1] if "/" in url else url

    # try to get a title via oembed for youtube
    if "youtube.com" in url or "youtu.be" in url:
        try:
            import httpx
            resp = httpx.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            title = resp.json().get("title", title)
        except Exception:
            pass

    slug = slugify(title)
    output_dir = config.vault_path / config.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{slug}.md"

    counter = 1
    while file_path.exists():
        file_path = output_dir / f"{slug}-{counter}.md"
        counter += 1

    markdown = (
        f"---\n"
        f"source: \"{url}\"\n"
        f"ingested: {today}\n"
        f"tags: []\n"
        f"type: link\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"> [source]({url})\n\n"
        f"*link only — no transcript available at time of ingest.*\n"
    )

    file_path.write_text(markdown, encoding="utf-8")
    return file_path
