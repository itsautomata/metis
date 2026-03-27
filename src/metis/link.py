"""connection discovery between vault notes."""

import re
from dataclasses import dataclass
from pathlib import Path

from metis.config import MetisConfig
from metis.index.embed import embed_texts
from metis.index.store import _get_collection


@dataclass
class Connection:
    source: str
    target: str
    score: float
    source_preview: str
    target_preview: str


def _get_existing_links(file_path: Path) -> set[str]:
    """find all [[wikilinks]] already in a file."""
    if not file_path.exists():
        return set()
    text = file_path.read_text(encoding="utf-8")
    return set(re.findall(r"\[\[(.+?)\]\]", text))


def _note_name(file_path: str) -> str:
    """extract note name from path (filename without extension)."""
    return Path(file_path).stem


def find_connections(
    config: MetisConfig,
    note_path: str | None = None,
    limit: int = 5,
    min_score: float = 0.7,
) -> list[Connection]:
    """find related notes that aren't already linked.

    if note_path is given, find connections for that note.
    if None, find connections across all notes.
    """
    collection = _get_collection(config)

    if collection.count() == 0:
        return []

    # get all unique file paths in the index
    all_data = collection.get(include=["metadatas", "documents"])
    file_chunks: dict[str, str] = {}

    for i, meta in enumerate(all_data["metadatas"]):
        fp = meta.get("file_path", "")
        if fp and fp not in file_chunks:
            # use first chunk as representative text
            file_chunks[fp] = all_data["documents"][i]

    if note_path:
        # find connections for a specific note
        source_paths = [note_path]
    else:
        source_paths = list(file_chunks.keys())

    connections = []

    for source_fp in source_paths:
        if source_fp not in file_chunks:
            continue

        source_text = file_chunks[source_fp]
        existing_links = _get_existing_links(Path(source_fp))

        # embed and search
        query_embedding = embed_texts([source_text], config)[0]
        n_results = min(limit + len(existing_links) + 1, collection.count())

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        seen_targets = set()

        for i in range(len(results["ids"][0])):
            target_fp = results["metadatas"][0][i].get("file_path", "")
            target_name = _note_name(target_fp)

            # skip self, already linked, already seen
            if target_fp == source_fp:
                continue
            if target_name in existing_links:
                continue
            if target_fp in seen_targets:
                continue

            distance = results["distances"][0][i] if results["distances"] else 1
            score = round(1 - distance, 3)

            if score < min_score:
                continue

            seen_targets.add(target_fp)
            connections.append(Connection(
                source=source_fp,
                target=target_fp,
                score=score,
                source_preview=source_text[:100],
                target_preview=results["documents"][0][i][:100],
            ))

            if len([c for c in connections if c.source == source_fp]) >= limit:
                break

    connections.sort(key=lambda c: c.score, reverse=True)
    return connections


def write_links(connections: list[Connection]) -> int:
    """write [[wikilinks]] into source notes. returns count of links written."""
    written = 0

    # group by source
    by_source: dict[str, list[Connection]] = {}
    for c in connections:
        by_source.setdefault(c.source, []).append(c)

    for source_fp, conns in by_source.items():
        path = Path(source_fp)
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        links_section = "\n\n## Connections\n\n"
        for c in conns:
            target_name = _note_name(c.target)
            links_section += f"- [[{target_name}]] [{c.score}]\n"

        # append or replace existing connections section
        if "## Connections" in text:
            text = re.sub(
                r"\n\n## Connections\n\n.*",
                links_section,
                text,
                flags=re.DOTALL,
            )
        else:
            text += links_section

        path.write_text(text, encoding="utf-8")
        written += len(conns)

    return written
