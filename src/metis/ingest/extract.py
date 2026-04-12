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


def is_xtweet(url: str) -> bool:
    """check if URL is an X/Twitter post."""
    return bool(re.match(r"https?://(www\.)?(twitter\.com|x\.com)/\w+/status/\d+", url))


def _extract_tweet_id(url: str) -> str:
    """extract tweet ID from X/Twitter URL."""
    match = re.search(r"/status/(\d+)", url)
    if not match:
        raise ValueError(f"could not parse tweet URL: {url}")
    return match.group(1)


def _extract_via_x_api(url: str, bearer_token: str) -> tuple[str, str, str | None]:
    """extract tweet/thread/article via X API v2. returns (title, text, author)."""
    tweet_id = _extract_tweet_id(url)

    headers = {"Authorization": f"Bearer {bearer_token}"}

    response = httpx.get(
        f"https://api.x.com/2/tweets/{tweet_id}",
        headers=headers,
        params={
            "expansions": "author_id,referenced_tweets.id",
            "tweet.fields": "text,conversation_id,created_at,entities,note_tweet,article",
            "user.fields": "username,name",
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    tweet_data = data.get("data", {})
    text = tweet_data.get("text", "")
    conversation_id = tweet_data.get("conversation_id", "")

    # get author from includes
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    author_id = tweet_data.get("author_id", "")
    author_info = users.get(author_id, {})
    author = author_info.get("username", "unknown")
    author_name = author_info.get("name", author)

    # check for X article — full text is in article.plain_text
    article = tweet_data.get("article")
    if article:
        article_text = article.get("plain_text", "")
        article_title = article.get("title", f"@{author_name}")
        if article_text:
            return article_title, article_text, author

    # if this is part of a thread, fetch the full conversation
    if conversation_id and conversation_id != tweet_id:
        thread_text = _fetch_thread(conversation_id, bearer_token)
        if thread_text:
            text = thread_text

    title = f"@{author_name}"
    return title, text, author


def _fetch_thread(conversation_id: str, bearer_token: str) -> str | None:
    """fetch all tweets in a conversation thread."""
    headers = {"Authorization": f"Bearer {bearer_token}"}

    response = httpx.get(
        "https://api.x.com/2/tweets/search/recent",
        headers=headers,
        params={
            "query": f"conversation_id:{conversation_id}",
            "tweet.fields": "text,created_at",
            "max_results": 100,
        },
        timeout=15,
    )

    if response.status_code != 200:
        return None

    data = response.json()
    tweets = data.get("data", [])
    if not tweets:
        return None

    # sort by creation time and join
    tweets.sort(key=lambda t: t.get("created_at", ""))
    return "\n\n".join(t.get("text", "") for t in tweets)


def _extract_via_oembed(url: str) -> tuple[str, str, str | None]:
    """extract tweet text via oembed (no auth). returns (title, text, author)."""
    response = httpx.get(
        "https://publish.twitter.com/oembed",
        params={"url": url, "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    author = data.get("author_name", "unknown")
    # html field contains the tweet in a blockquote — strip tags
    html = data.get("html", "")
    text = re.sub(r"<[^>]+>", "", html).strip()
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()

    if not text:
        raise ValueError(f"could not extract text from tweet: {url}")

    title = f"@{author}"
    return title, text, author


def extract_from_xtweet(url: str, bearer_token: str = "") -> tuple[str, str, str | None]:
    """extract tweet text. uses X API if token provided, oembed otherwise.

    returns (title, text, author).
    """
    if bearer_token:
        try:
            return _extract_via_x_api(url, bearer_token)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[yellow]X API failed ({e}), falling back to oembed[/yellow]")

    return _extract_via_oembed(url)


def is_pdf_url(url: str) -> bool:
    """check if URL points directly to a PDF."""
    from urllib.parse import urlparse
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def extract_from_pdf_url(url: str) -> tuple[str, str]:
    """download PDF from URL, extract text, delete temp file. returns (title, text)."""
    import tempfile

    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()

    # verify we actually got a PDF, not an HTML redirect
    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type and not response.content[:5] == b"%PDF-":
        raise ValueError(f"URL does not serve a PDF (got {content_type}): {url}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    try:
        _, text = extract_from_pdf(tmp_path)
    finally:
        tmp_path.unlink()

    # derive title: first meaningful line of text, fallback to URL
    first_lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    title = first_lines[0][:100] if first_lines else _title_from_url(url)

    return title, text


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


def _extract_distill(html: str) -> tuple[str, str] | None:
    """extract from Distill.js format (used by transformer-circuits.pub, distill.pub, etc.).

    returns (title, text) or None if not a Distill paper.
    """
    if "<d-article>" not in html:
        return None

    # title from d-front-matter JSON
    title = "untitled"
    fm_match = re.search(r"<d-front-matter>.*?<script[^>]*>(.*?)</script>", html, re.DOTALL)
    if fm_match:
        try:
            import json
            fm = json.loads(fm_match.group(1))
            title = fm.get("title", title)
        except Exception:
            pass

    # fallback: <title> tag
    if title == "untitled":
        title_match = re.search(r"<title>(.*?)</title>", html)
        if title_match:
            title = title_match.group(1).strip()

    # article body from <d-article>
    article_match = re.search(r"<d-article>(.*?)</d-article>", html, re.DOTALL)
    if not article_match:
        return None

    article_html = article_match.group(1)

    # strip tags but preserve structure
    # convert block elements to newlines first
    text = re.sub(r"<(p|h[1-6]|li|dt|dd|figcaption)[^>]*>", "\n", article_html)
    text = re.sub(r"</(p|h[1-6]|li|dt|dd|figcaption)>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<d-math[^>]*>.*?</d-math>", "[math]", text, flags=re.DOTALL)
    text = re.sub(r"<d-figure[^>]*>.*?</d-figure>", "\n[figure]\n", text, flags=re.DOTALL)
    text = re.sub(r"<d-footnote[^>]*>(.*?)</d-footnote>", r" [\1]", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    # strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # clean up whitespace
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = text.strip()

    if not text:
        return None

    return title, text


def _extract_with_httpx(url: str) -> tuple[str, str] | None:
    """last-resort extraction: fetch with httpx, strip HTML tags.

    returns (title, text) or None.
    """
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30,
                           headers={"User-Agent": "metis/0.1.0"})
        response.raise_for_status()
    except Exception:
        return None

    html = response.text

    # try distill format first
    distill = _extract_distill(html)
    if distill:
        return distill

    # generic: get title + strip body
    title_match = re.search(r"<title>(.*?)</title>", html)
    title = title_match.group(1).strip() if title_match else _title_from_url(url)

    # remove script, style, nav, header, footer
    for tag in ["script", "style", "nav", "header", "footer", "aside"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL)

    # strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 50:
        return None

    return title, text


def extract_from_url(url: str) -> tuple[str, str]:
    """extract article text from URL. returns (title, text).

    fallback chain: trafilatura → distill.js extractor → httpx + tag stripping.
    """
    # 1. trafilatura (best for standard articles)
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if text:
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else _title_from_url(url)
            return title, text

    # 2. httpx fallback (handles distill.js + large pages trafilatura can't fetch)
    result = _extract_with_httpx(url)
    if result:
        return result

    raise ValueError(f"could not extract text from: {url}")


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
    x_bearer_token: str = "",
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

        if is_xtweet(source):
            title, text, author = extract_from_xtweet(source, bearer_token=x_bearer_token)
            extra = {"author": author} if author else None
            return title, text, "tweet", source, extra

        if is_arxiv(source):
            title, text = extract_from_arxiv(source)
            return title, text, "arxiv", _arxiv_abs_url(source), None

        if is_pdf_url(source):
            title, text = extract_from_pdf_url(source)
            return title, text, "pdf", source, None

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
