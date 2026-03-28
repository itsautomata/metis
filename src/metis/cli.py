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


def _complete_vault_folders(incomplete: str) -> list[str]:
    """autocomplete vault subfolder paths."""
    try:
        config = load_config()
        vault = config.vault_path
        if not vault.exists():
            return []

        # list directories in vault
        prefix = Path(incomplete) if incomplete else Path()
        search_dir = vault / prefix.parent if incomplete else vault

        if not search_dir.exists():
            return []

        results = []
        for p in search_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                rel = str(p.relative_to(vault))
                if rel.startswith(incomplete):
                    results.append(rel)
        return results
    except Exception:
        return []


def _complete_vault_notes(incomplete: str) -> list[str]:
    """autocomplete vault note paths (.md files)."""
    try:
        config = load_config()
        vault = config.vault_path
        if not vault.exists():
            return []

        results = []
        for p in vault.rglob("*.md"):
            rel = str(p.relative_to(vault))
            if rel.startswith(incomplete):
                results.append(rel)
        return results
    except Exception:
        return []


@app.command()
def ingest(
    source: str = typer.Argument(help="file path or URL to ingest"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="vault subfolder to save in", autocompletion=_complete_vault_folders),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="transcript language code (youtube)"),
    pick_lang: bool = typer.Option(False, "--pick-lang", help="interactively pick transcript language (youtube)"),
):
    """save, summarize, tag, embed, and find links for a file or URL."""
    from metis.ingest.extract import extract, NoTranscriptError
    from metis.ingest.process import process
    from metis.ingest.write import write_to_vault, write_link_only
    from metis.index.store import store_chunks

    config = load_config()
    if folder:
        config.output_folder = folder

    console.print(f"[bold]ingesting:[/bold] {source}")

    # 1. extract
    console.print("[dim]extracting text...[/dim]")
    try:
        title, text, source_type, source_link, extra = extract(
            source, lang=lang, pick_lang=pick_lang,
            x_bearer_token=config.x_api.bearer_token,
        )
    except NoTranscriptError:
        console.print("[yellow]no transcript found.[/yellow]")
        save = typer.confirm("save link anyway?")
        if save:
            file_path = write_link_only(source, config)
            console.print(f"[bold green]link saved.[/bold green]")
            console.print(f"  note: {file_path}")
        return

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
    file_path = write_to_vault(title, text, source_link, source_type, processed, config, extra=extra)
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
    note: Optional[str] = typer.Option(None, "--note", help="scope to a specific note", autocompletion=_complete_vault_notes),
    save: bool = typer.Option(False, "--save", "-s", help="save Q&A to the note"),
):
    """RAG agent loop over your knowledge base."""
    from metis.chat import ask, save_qa_to_note, LOW_CONFIDENCE_THRESHOLD

    config = load_config()

    # resolve note path
    note_path = None
    if note:
        note_p = Path(note).expanduser()
        if not note_p.is_absolute():
            note_p = config.vault_path / note
        note_path = str(note_p)
        if not note_p.exists():
            console.print(f"[red]note not found: {note_path}[/red]")
            return

    console.print(f"[bold]asking:[/bold] {question}\n")

    answer, sources, confidence, clean_question = ask(question, config, note_path=note_path)

    # show reformulated query if different
    if clean_question.lower().strip("?. ") != question.lower().strip("?. "):
        console.print(f"[dim]query: {clean_question}[/dim]\n")

    console.print(answer)
    console.print()

    if sources:
        console.print("[dim]sources:[/dim]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [dim]- {name}[/dim]")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        console.print(f"\n[yellow]low confidence ({confidence:.2f}) — consider verifying against the source.[/yellow]")

    # save Q&A to note
    if save and note_path:
        if typer.confirm("\nsave to note?"):
            save_qa_to_note(note_path, clean_question, answer)
            console.print("[bold green]Q&A saved.[/bold green]")
    elif save and not note_path:
        console.print("[yellow]--save requires --note to specify which note to save to.[/yellow]")


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
    console.print("\n[dim]edit ~/.metis/config.yaml to set your vault path and api keys.[/dim]")
