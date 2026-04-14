"""sync local vault index to azure AI search."""

import yaml
import re
from pathlib import Path

from metis.config import MetisConfig
from metis.index.store import get_collection
from metis.cloud.search import create_index, push_documents, delete_documents


def _extract_frontmatter(file_path: str) -> tuple[str, str]:
    """extract title and tags from note frontmatter.

    returns (title, tags_string)
    """
    path = Path(file_path)
    if not path.exists():
        return "", ""

    text = path.read_text(encoding="utf-8")

    # title from first heading
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem

    # tags from frontmatter
    tags = ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1])
                if fm and isinstance(fm.get("tags"), list):
                    tags = ", ".join(str(t) for t in fm["tags"])
            except Exception:
                pass

    return title, tags


def sync_to_cloud(config: MetisConfig, endpoint: str, key: str) -> dict:
    """push all local chromadb data to azure AI search.

    returns {created: int, uploaded: int, errors: int}
    """
    # create or update the index schema
    create_index(endpoint, key)

    # get all data from chromadb
    collection = get_collection(config)
    if collection.count() == 0:
        return {"created": 0, "uploaded": 0, "errors": 0}

    all_data = collection.get(include=["metadatas", "embeddings", "documents"])
    vault = config.vault_path

    # build documents for azure AI search
    documents = []
    for i in range(len(all_data["ids"])):
        meta = all_data["metadatas"][i]
        file_path = meta.get("file_path", "")
        chunk_index = meta.get("chunk_index", 0)

        # derive folder
        try:
            rel = Path(file_path).relative_to(vault)
            folder = str(rel.parent)
            if folder == ".":
                folder = "(root)"
        except ValueError:
            folder = "(unknown)"

        # get title and tags from the note file
        title, tags = _extract_frontmatter(file_path)

        # azure search ID: letters, numbers, dashes, underscores only, no leading underscore
        import base64
        doc_id = base64.urlsafe_b64encode(all_data["ids"][i].encode()).decode().rstrip("=")

        documents.append({
            "id": doc_id,
            "file_path": file_path,
            "folder": folder,
            "chunk_index": chunk_index,
            "text": all_data["documents"][i],
            "title": title,
            "tags": tags,
            "embedding": list(float(x) for x in all_data["embeddings"][i]),
        })

    # upload in batches of 100
    total_uploaded = 0
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        uploaded = push_documents(endpoint, key, batch)
        total_uploaded += uploaded

    return {
        "created": len(documents),
        "uploaded": total_uploaded,
        "errors": len(documents) - total_uploaded,
    }
