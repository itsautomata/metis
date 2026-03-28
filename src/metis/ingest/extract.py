"""text extraction from different source types."""

import re
from pathlib import Path

import fitz
import httpx
import trafilatura


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def is_youtube(url: str) -> bool:
    """check if URL is a youtube video."""
    return bool(re.match(r"https?://(www\.)?(youtube\.com/watch|youtu\.be/)", url))


def _youtube_video_id(url: str) -> str:
    """extract video ID from youtube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"could not parse youtube URL: {url}")
    return match.group(1)


def extract_from_youtube(
    url: str,
    lang: str | None = None,
    pick_lang: bool = False,
) -> tuple[str, str, str | None]:
    """extract transcript from youtube video.

    returns (title, text, channel) or raises if no transcript.
    lang: specific language code (e.g. "fr", "en")
    pick_lang: if True, show interactive language picker
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    ytt = YouTubeTranscriptApi()
    video_id = _youtube_video_id(url)

    # get available transcripts
    try:
        transcript_list = ytt.list(video_id)
        available = list(transcript_list)
    except Exception:
        raise NoTranscriptError(url)

    if not available:
        raise NoTranscriptError(url)

    # pick language
    if pick_lang:
        transcript = _interactive_lang_pick(available)
    elif lang:
        try:
            transcript = transcript_list.find_transcript([lang])
        except Exception:
            lang_names = [f"{t.language} ({t.language_code})" for t in available]
            raise ValueError(
                f"language '{lang}' not available. available: {', '.join(lang_names)}"
            )
    else:
        # default: english, then first available
        try:
            transcript = transcript_list.find_transcript(["en"])
        except Exception:
            transcript = available[0]

    entries = transcript.fetch()
    text = "\n".join(
        entry.text if hasattr(entry, "text") else entry.get("text", str(entry))
        for entry in entries
    )

    # get video title via oembed (no API key needed)
    title, channel = _youtube_metadata(url)

    return title, text, channel


def _interactive_lang_pick(available: list) -> object:
    """show language menu and let user pick."""
    from rich.console import Console
    console = Console()

    console.print("\n[bold]available transcripts:[/bold]")
    for i, t in enumerate(available, 1):
        auto = " (auto-generated)" if t.is_generated else ""
        console.print(f"  {i}. {t.language} ({t.language_code}){auto}")

    while True:
        choice = input(f"\npick language [1]: ").strip()
        if not choice:
            return available[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
        except ValueError:
            pass
        console.print("[red]invalid choice[/red]")


def _youtube_metadata(url: str) -> tuple[str, str | None]:
    """get video title and channel via oembed."""
    try:
        response = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("title", "untitled"), data.get("author_name")
    except Exception:
        return _title_from_url(url), None


class NoTranscriptError(Exception):
    """raised when a youtube video has no transcript."""
    def __init__(self, url: str):
        self.url = url
        super().__init__(f"no transcript available for: {url}")


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

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem.replace("-", " ").replace("_", " ")

    return title, text


def extract(
    source: str,
    lang: str | None = None,
    pick_lang: bool = False,
) -> tuple[str, str, str, str, dict | None]:
    """extract text from any source.

    returns (title, text, source_type, source_link, extra_metadata)
    - source_link is the URL or file:// URI for linking back
    - extra_metadata is optional dict (e.g. channel for youtube)
    """
    if is_url(source):
        if is_youtube(source):
            title, text, channel = extract_from_youtube(source, lang=lang, pick_lang=pick_lang)
            extra = {"channel": channel} if channel else None
            return title, text, "youtube", source, extra

        if is_arxiv(source):
            title, text = extract_from_arxiv(source)
            return title, text, "arxiv", _arxiv_abs_url(source), None

        title, text = extract_from_url(source)
        return title, text, "url", source, None

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"file not found: {source}")

    source_link = path.as_uri()

    if path.suffix.lower() == ".pdf":
        title, text = extract_from_pdf(path)
        return title, text, "pdf", source_link, None

    if path.suffix.lower() in (".md", ".markdown", ".txt"):
        title, text = extract_from_markdown(path)
        return title, text, "markdown", source_link, None

    raise ValueError(f"unsupported file type: {path.suffix}")


def _title_from_url(url: str) -> str:
    """derive a title from URL path."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else parsed.netloc
    return path.replace("-", " ").replace("_", " ").replace(".html", "").replace(".htm", "")
