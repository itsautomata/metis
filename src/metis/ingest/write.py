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
) -> str:
    """build a complete markdown note with frontmatter."""
    tags_str = ", ".join(processed.tags)
    key_points_str = "\n".join(f"- {kp}" for kp in processed.key_points)
    today = date.today().isoformat()

    frontmatter = (
        f"---\n"
        f"source: \"{source_link}\"\n"
        f"ingested: {today}\n"
        f"tags: [{tags_str}]\n"
        f"summary: \"{processed.summary}\"\n"
        f"type: {source_type}\n"
        f"---"
    )

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
        f"## Content\n"
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

    markdown = build_markdown(title, text, source_link, source_type, processed)
    file_path.write_text(markdown, encoding="utf-8")

    return file_path
