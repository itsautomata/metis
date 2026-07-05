"""tests for KNN folder voting."""

from pathlib import Path

from metis.classify import _tally_folder_votes

VAULT = Path("/vault")


def test_multiple_chunks_from_one_note_count_once():
    ordered = [
        "/vault/papers/big.md",
        "/vault/papers/big.md",
        "/vault/papers/big.md",
        "/vault/papers/big.md",
        "/vault/notes/small.md",
    ]
    assert _tally_folder_votes(ordered, VAULT, k=5) == {"papers": 1, "notes": 1}


def test_stops_at_k_distinct_notes():
    ordered = [f"/vault/f{i}/n.md" for i in range(10)]
    assert sum(_tally_folder_votes(ordered, VAULT, k=3).values()) == 3


def test_root_note_labeled_metis_ingested():
    assert _tally_folder_votes(["/vault/loose.md"], VAULT, k=5) == {"metis-ingested": 1}


def test_paths_outside_vault_skipped():
    assert _tally_folder_votes(["/elsewhere/x.md", "/vault/a/n.md"], VAULT, k=5) == {"a": 1}


def test_empty_paths_skipped():
    assert _tally_folder_votes(["", "/vault/a/n.md", ""], VAULT, k=5) == {"a": 1}


def test_nearest_chunk_wins_then_duplicates_ignored():
    ordered = ["/vault/a/n.md", "/vault/a/n.md", "/vault/b/m.md"]
    assert _tally_folder_votes(ordered, VAULT, k=5) == {"a": 1, "b": 1}
