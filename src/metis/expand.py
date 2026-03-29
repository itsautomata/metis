"""external source search — wikipedia."""

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from metis.client import get_client, get_chat_model
from metis.config import MetisConfig
from metis.ingest.extract import extract
from metis.ingest.process import process
from metis.ingest.write import write_to_vault
from metis.index.store import store_chunks


@dataclass
class ExternalResult:
    title: str
    preview: str
    url: str
    source_type: str


def extract_search_keywords(question: str, config: MetisConfig) -> str:
    """turn a natural language question into optimized wikipedia search terms."""
    client = get_client(config)
    model = get_chat_model(config)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "extract 1-3 topic names from the user's question "
                    "optimized for searching wikipedia articles. "
                    "use the most specific encyclopedic topic name possible. "
                    "return ONLY the topic names separated by spaces. no punctuation, no explanation."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def search_wikipedia(query: str, max_results: int = 5) -> list[ExternalResult]:
    """search wikipedia for articles. free API, no key needed."""
    response = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        },
        headers={"User-Agent": "metis/0.1.0 (CLI second brain)"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()

        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        results.append(ExternalResult(
            title=title,
            preview=clean_snippet,
            url=url,
            source_type="wikipedia",
        ))

    return results


def ingest_external(result: ExternalResult, config: MetisConfig) -> tuple[Path, str]:
    """ingest an external result into the vault. returns (file_path, full_text).

    auto-organizes by source type: wikipedia/.
    """
    original_folder = config.output_folder
    config.output_folder = result.source_type

    try:
        title, text, source_type, source_link, extra = extract(result.url)

        processed = process(text, config)

        file_path = write_to_vault(
            title, text, source_link, source_type, processed, config, extra=extra,
        )

        store_chunks(processed.chunks, file_path, config)

        return file_path, text
    finally:
        config.output_folder = original_folder
