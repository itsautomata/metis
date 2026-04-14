"""azure AI search: create index, push documents, query."""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SimpleField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

INDEX_NAME = "metis-vault"
VECTOR_DIMENSIONS = 1536


def _get_index_client(endpoint: str, key: str) -> SearchIndexClient:
    return SearchIndexClient(endpoint, AzureKeyCredential(key))


def _get_search_client(endpoint: str, key: str) -> SearchClient:
    return SearchClient(endpoint, INDEX_NAME, AzureKeyCredential(key))


def create_index(endpoint: str, key: str) -> None:
    """create the metis vault index in azure AI search."""
    client = _get_index_client(endpoint, key)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="folder", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="tags", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="metis-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="metis-hnsw")],
        profiles=[VectorSearchProfile(name="metis-vector-profile", algorithm_configuration_name="metis-hnsw")],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    client.create_or_update_index(index)


def push_documents(endpoint: str, key: str, documents: list[dict]) -> int:
    """upload documents to the index. returns count uploaded.

    each document: {id, file_path, folder, chunk_index, text, title, tags, embedding}
    """
    client = _get_search_client(endpoint, key)
    result = client.upload_documents(documents)
    succeeded = sum(1 for r in result if r.succeeded)
    return succeeded


def delete_documents(endpoint: str, key: str, ids: list[str]) -> int:
    """delete documents by ID. returns count deleted."""
    client = _get_search_client(endpoint, key)
    docs = [{"id": doc_id} for doc_id in ids]
    result = client.delete_documents(docs)
    return sum(1 for r in result if r.succeeded)


def search_cloud(
    endpoint: str,
    key: str,
    query: str,
    embedding: list[float],
    limit: int = 5,
    folder: str | None = None,
) -> list[dict]:
    """hybrid search: vector + keyword.

    returns list of {file_path, text, score, chunk_index, folder, title, tags}
    """
    client = _get_search_client(endpoint, key)

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=limit,
        fields="embedding",
    )

    filter_expr = f"folder eq '{folder}'" if folder else None

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=limit,
        filter=filter_expr,
    )

    hits = []
    for r in results:
        hits.append({
            "file_path": r["file_path"],
            "text": r["text"],
            "score": r["@search.score"],
            "chunk_index": r["chunk_index"],
            "folder": r["folder"],
            "title": r.get("title", ""),
            "tags": r.get("tags", ""),
        })

    return hits
