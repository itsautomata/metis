"""metis CLI — second brain that pairs with Obsidian."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from metis.config import load_config, init_config

app = typer.Typer(
    name="metis",
    help="CLI second brain that pairs with Obsidian.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    source: str = typer.Argument(help="file path or URL to ingest"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="vault subfolder to save in"),
):
    """save, summarize, tag, embed, and find links for a file or URL."""
    from metis.ingest.extract import extract
    from metis.ingest.process import process
    from metis.ingest.write import write_to_vault
    from metis.index.store import store_chunks

    config = load_config()
    if folder:
        config.output_folder = folder

    console.print(f"[bold]ingesting:[/bold] {source}")

    # 1. extract
    console.print("[dim]extracting text...[/dim]")
    title, text, source_type, source_link = extract(source)
    console.print(f"  title: {title}")
    console.print(f"  type:  {source_type}")
    console.print(f"  chars: {len(text)}")

    # 2. summarize + tag + chunk
    console.print(f"[dim]processing with {config.provider}...[/dim]")
    processed = process(text, config)
    console.print(f"  tags:   {', '.join(processed.tags)}")
    console.print(f"  chunks: {len(processed.chunks)}")

    # 3. write to vault
    console.print("[dim]writing to vault...[/dim]")
    file_path = write_to_vault(title, text, source_link, source_type, processed, config)
    console.print(f"  saved: {file_path}")

    # 4. embed + store
    console.print("[dim]embedding and indexing...[/dim]")
    n = store_chunks(processed.chunks, file_path, config)
    console.print(f"  indexed: {n} chunks")

    # done
    console.print(f"\n[bold green]done.[/bold green]")
    console.print(f"  note:   {file_path}")
    console.print(f"  source: {source_link}")


@app.command()
def search(
    query: str = typer.Argument(help="what to search for"),
    limit: int = typer.Option(5, "--limit", "-n", help="number of results"),
):
    """semantic search across your vault."""
    from metis.search import search_vault

    config = load_config()
    console.print(f"[bold]searching:[/bold] {query}\n")

    results = search_vault(query, config, limit=limit)

    if not results:
        console.print("[yellow]no results. ingest some content first.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        path = Path(r.file_path).name
        preview = r.text[:150].replace("\n", " ").strip()
        console.print(f"[bold]{i}.[/bold] [{r.score}] [cyan]{path}[/cyan]")
        console.print(f"   {preview}...")
        console.print()


@app.command()
def chat(
    question: str = typer.Argument(help="question to ask your vault"),
):
    """RAG agent loop over your knowledge base."""
    from metis.chat import ask

    config = load_config()
    console.print(f"[bold]asking:[/bold] {question}\n")

    answer, sources = ask(question, config)

    console.print(answer)
    console.print()

    if sources:
        console.print("[dim]sources:[/dim]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [dim]- {name}[/dim]")


@app.command()
def link(
    note: Optional[str] = typer.Argument(None, help="note to find connections for (all notes if omitted)"),
    write: bool = typer.Option(False, "--write", "-w", help="write wikilinks into notes"),
    min_score: float = typer.Option(0.7, "--min-score", help="minimum similarity score"),
):
    """surface connections between notes."""
    from metis.link import find_connections, write_links

    config = load_config()
    target = note or "all notes"
    console.print(f"[bold]linking:[/bold] {target}\n")

    connections = find_connections(config, note_path=note, min_score=min_score)

    if not connections:
        console.print("[yellow]no connections found above threshold.[/yellow]")
        return

    for c in connections:
        source_name = Path(c.source).stem
        target_name = Path(c.target).stem
        console.print(f"  [cyan]{source_name}[/cyan] → [cyan]{target_name}[/cyan] [{c.score}]")

    console.print(f"\n{len(connections)} connections found.")

    if write:
        n = write_links(connections)
        console.print(f"[bold green]{n} wikilinks written.[/bold green]")
    else:
        console.print("[dim]use --write to add [[wikilinks]] to your notes.[/dim]")


@app.command()
def sync():
    """re-index vault to catch manual edits."""
    from metis.index.sync import sync_vault

    config = load_config()
    console.print(f"[bold]syncing:[/bold] {config.vault_path}\n")

    report = sync_vault(config)

    console.print(f"  added:     {report.added} files")
    console.print(f"  updated:   {report.updated} files")
    console.print(f"  deleted:   {report.deleted} files")
    console.print(f"  unchanged: {report.unchanged} files")
    console.print()
    console.print(f"[bold green]vault indexed.[/bold green] {report.total_files} files.")


@app.command()
def init():
    """initialize metis config and directories."""
    config_path = init_config()
    config = load_config()

    config.vault_path.mkdir(parents=True, exist_ok=True)
    config.chromadb_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]metis initialized.[/bold green]")
    console.print(f"  config: {config_path}")
    console.print(f"  vault:  {config.vault_path}")
    console.print(f"  db:     {config.chromadb_path}")
    console.print("\n[dim]edit ~/.metis/config.yaml to set your vault path and azure credentials.[/dim]")
