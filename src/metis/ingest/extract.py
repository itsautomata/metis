"""text extraction from different source types."""

import re
from pathlib import Path

import fitz
import httpx
import trafilatura


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def is_arxiv(url: str) -> bool:
    """check if URL is an arxiv paper."""
    return bool(re.match(r"https?://(www\.)?arxiv\.org/(abs|pdf)/", url))


def _arxiv_to_pdf_url(url: str) -> str:
    """convert any arxiv URL to its PDF download URL."""
    # extract the paper ID (e.g. 2401.12345 or 2401.12345v2)
    match = re.search(r"arxiv\.org/(abs|pdf)/([0-9.]+(?:v\d+)?)", url)
    if not match:
        raise ValueError(f"could not parse arxiv URL: {url}")
    paper_id = match.group(2)
    return f"https://arxiv.org/pdf/{paper_id}"


def _arxiv_abs_url(url: str) -> str:
    """get the abstract page URL for linking back."""
    match = re.search(r"arxiv\.org/(abs|pdf)/([0-9.]+(?:v\d+)?)", url)
    if not match:
        return url
    paper_id = match.group(2)
    return f"https://arxiv.org/abs/{paper_id}"


def extract_from_arxiv(url: str) -> tuple[str, str]:
    """fetch and extract text from an arxiv paper. returns (title, text)."""
    import tempfile

    pdf_url = _arxiv_to_pdf_url(url)

    # download the PDF
    response = httpx.get(pdf_url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    # write to temp file and extract with pymupdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    try:
        title, text = extract_from_pdf(tmp_path)
    finally:
        # delete temp PDF
        tmp_path.unlink()

    # try to get a better title from the arxiv abstract page
    try:
        abs_url = _arxiv_abs_url(url)
        downloaded = trafilatura.fetch_url(abs_url)
        if downloaded:
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata and metadata.title:
                title = metadata.title
    except Exception:
        pass  # keep the PDF-derived title

    return title, text


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
        if is_arxiv(source):
            title, text = extract_from_arxiv(source)
            return title, text, "arxiv", _arxiv_abs_url(source)

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
