"""summarization, tagging, and chunking via OpenAI."""

import json
from dataclasses import dataclass

from metis.client import get_client, get_chat_model
from metis.config import MetisConfig


@dataclass
class ProcessedContent:
    summary: str
    key_points: list[str]
    tags: list[str]
    chunks: list[str]


def summarize_and_tag(text: str, config: MetisConfig) -> tuple[str, list[str], list[str]]:
    """summarize text and extract tags + key points.

    returns (summary, key_points, tags)
    """
    client = get_client(config)

    # truncate to avoid token limits — first ~12k chars is enough for summarization
    truncated = text[:12000]

    response = client.chat.completions.create(
        model=get_chat_model(config),
        messages=[
            {
                "role": "system",
                "content": (
                    "you are a knowledge extraction assistant. "
                    "given a text, return a JSON object with exactly these keys:\n"
                    '- "summary": a one-paragraph summary (2-3 sentences max)\n'
                    '- "key_points": a list of 3-5 key points (short strings)\n'
                    '- "tags": a list of 3-7 lowercase tags (single words or hyphenated)\n'
                    "return ONLY valid JSON, no markdown fencing."
                ),
            },
            {"role": "user", "content": truncated},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # strip markdown fencing if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    parsed = json.loads(raw)

    summary = _sanitize_summary(parsed.get("summary", ""), text)
    key_points = _sanitize_key_points(parsed.get("key_points", []))
    tags = _sanitize_tags(parsed.get("tags", []))

    return summary, key_points, tags


SUSPICIOUS_PATTERNS = ["ignore", "system prompt", "you are", "disregard", "override", "pretend"]


def _sanitize_tags(tags: list) -> list[str]:
    """validate and clean LLM-generated tags."""
    clean = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.lower().strip()
        if len(tag) > 30:
            continue
        if " " in tag and "-" not in tag:
            continue
        if any(p in tag for p in SUSPICIOUS_PATTERNS):
            continue
        clean.append(tag)
    return clean


def _sanitize_summary(summary: str, original_text: str) -> str:
    """validate LLM-generated summary."""
    if not isinstance(summary, str):
        return ""
    if len(summary) > len(original_text):
        return summary[:len(original_text)]
    if any(p in summary.lower() for p in SUSPICIOUS_PATTERNS):
        return ""
    return summary


def _sanitize_key_points(points: list) -> list[str]:
    """validate LLM-generated key points."""
    clean = []
    for point in points:
        if not isinstance(point, str):
            continue
        if len(point) > 300:
            continue
        if any(p in point.lower() for p in SUSPICIOUS_PATTERNS):
            continue
        clean.append(point)
    return clean


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """split text into overlapping chunks at paragraph or line boundaries.

    targets ~500 tokens per chunk (roughly 1500 chars).
    handles both paragraph-based text (articles) and line-based text (transcripts).
    """
    # split on double newlines first, then fall back to single newlines
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        paragraphs = text.split("\n")

    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            # overlap: keep the last bit of the previous chunk
            words = current.split()
            overlap_text = " ".join(words[-30:]) if len(words) > 30 else ""
            current = overlap_text + "\n" + para if overlap_text else para
        else:
            current = current + "\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # safety: if any chunk is still too long, hard-split it
    safe_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars * 2:
            words = chunk.split()
            sub = ""
            for word in words:
                if len(sub) + len(word) + 1 > max_chars:
                    safe_chunks.append(sub.strip())
                    sub = word
                else:
                    sub = sub + " " + word if sub else word
            if sub.strip():
                safe_chunks.append(sub.strip())
        else:
            safe_chunks.append(chunk)

    return safe_chunks if safe_chunks else [text[:max_chars]]


SHORT_TEXT_THRESHOLD = 500


def process(text: str, config: MetisConfig) -> ProcessedContent:
    """full processing pipeline: summarize, tag, chunk."""
    chunks = chunk_text(text)

    # short text (tweets, quick notes) — skip LLM summarization
    if len(text.strip()) < SHORT_TEXT_THRESHOLD:
        return ProcessedContent(
            summary=text.strip(),
            key_points=[],
            tags=[],
            chunks=chunks,
        )

    summary, key_points, tags = summarize_and_tag(text, config)

    return ProcessedContent(
        summary=summary,
        key_points=key_points,
        tags=tags,
        chunks=chunks,
    )
