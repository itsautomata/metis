"""text extraction from different source types."""

import re
from pathlib import Path

import fitz
import httpx
import trafilatura


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def extract_from_pdf(path: Path) -> tuple[str, str]:
    """extract text from PDF. returns (title, text)."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    text = "\n\n".join(pages).strip()
    title = path.stem.replace("-", " ").replace("_", " ")
    return title, text


def extract_from_url(url: str) -> tuple[str, str]:
    """extract article text from URL. returns (title, text)."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"could not fetch URL: {url}")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text:
        raise ValueError(f"could not extract text from: {url}")

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else _title_from_url(url)

    return title, text


def extract_from_markdown(path: Path) -> tuple[str, str]:
    """read markdown file. returns (title, text)."""
    text = path.read_text(encoding="utf-8")

    title_match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem.replace("-", " ").replace("_", " ")

    return title, text


def extract(source: str) -> tuple[str, str, str, str]:
    """extract text from any source.

    returns (title, text, source_type, source_link)
    - source_link is the URL or file:// URI for linking back
    """
    if is_url(source):
        title, text = extract_from_url(source)
        return title, text, "url", source

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"file not found: {source}")

    source_link = path.as_uri()

    if path.suffix.lower() == ".pdf":
        title, text = extract_from_pdf(path)
        return title, text, "pdf", source_link

    if path.suffix.lower() in (".md", ".markdown", ".txt"):
        title, text = extract_from_markdown(path)
        return title, text, "markdown", source_link

    raise ValueError(f"unsupported file type: {path.suffix}")


def _title_from_url(url: str) -> str:
    """derive a title from URL path."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else parsed.netloc
    return path.replace("-", " ").replace("_", " ").replace(".html", "").replace(".htm", "")
