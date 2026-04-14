"""azure blob storage: upload pending notes for local sync."""

from metis.secrets import get_secret


def _get_connection_string() -> str:
    return get_secret("azure-storage-connection")


def upload_pending_note(name: str, content: str) -> None:
    """upload a markdown note to the pending/ container for local sync."""
    from azure.storage.blob import BlobServiceClient

    conn = _get_connection_string()
    if not conn:
        return

    blob_service = BlobServiceClient.from_connection_string(conn)
    container = blob_service.get_container_client("pending")
    container.upload_blob(name, content, overwrite=True)


def list_pending() -> list[str]:
    """list all pending notes waiting for local sync."""
    from azure.storage.blob import BlobServiceClient

    conn = _get_connection_string()
    if not conn:
        return []

    blob_service = BlobServiceClient.from_connection_string(conn)
    container = blob_service.get_container_client("pending")
    return [blob.name for blob in container.list_blobs()]


def download_pending(name: str) -> str:
    """download a pending note's content."""
    from azure.storage.blob import BlobServiceClient

    conn = _get_connection_string()
    blob_service = BlobServiceClient.from_connection_string(conn)
    container = blob_service.get_container_client("pending")
    blob = container.download_blob(name)
    return blob.readall().decode("utf-8")


def delete_pending(name: str) -> None:
    """delete a pending note after local sync."""
    from azure.storage.blob import BlobServiceClient

    conn = _get_connection_string()
    blob_service = BlobServiceClient.from_connection_string(conn)
    container = blob_service.get_container_client("pending")
    container.delete_blob(name)
