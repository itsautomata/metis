"""metis CLI: second brain that pairs with obsidian."""

import functools
from pathlib import Path
from typing import Optional

import typer
import typer.rich_utils
from rich.console import Console

from metis.config import init_config, load_config

# align typer's auto-generated help accent with metis's magenta
typer.rich_utils.STYLE_OPTION = "bold magenta"
typer.rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = "bold magenta"

app = typer.Typer(
    name="metis",
    help="CLI second brain that pairs with Obsidian.",
    no_args_is_help=True,
)
console = Console()


def _provider_guard(fn):
    """turn a provider/model failure into a clean message instead of a traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import openai

        from metis.client import ProviderError
        try:
            return fn(*args, **kwargs)
        except (ProviderError, openai.OpenAIError) as e:
            console.print(f"[red]✗ {e}[/red]")
            if isinstance(e, openai.AuthenticationError) or "401" in str(e):
                console.print("[dim]a 401 usually means a wrong or missing key. run 'metis models' to check the key and provider.[/dim]")
            else:
                console.print("[dim]check the model id, base_url, and key (metis config / metis secret).[/dim]")
            raise typer.Exit(1)
    return wrapper


def _key_provider(key: str) -> str:
    """guess the provider from a key prefix (no secret revealed)."""
    if not key:
        return "none"
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("sk-"):
        return "openai"
    return "unknown"


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


def _ensure_index_model(config) -> bool:
    """False (and prints guidance) if the index was built with a different embedding model."""
    from metis.index.store import EmbeddingModelMismatch, check_embedding_model
    try:
        check_embedding_model(config)
        return True
    except EmbeddingModelMismatch as e:
        console.print(f"[red]✗ {e}[/red]")
        return False


@app.command()
@_provider_guard
def ingest(
    sources: list[str] = typer.Argument(help="file paths or URLs to ingest"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="vault subfolder to save in", autocompletion=_complete_vault_folders),
    pick_folder_flag: bool = typer.Option(False, "--pick-folder", help="interactively pick vault folder"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="transcript language code (youtube)"),
    pick_lang: bool = typer.Option(False, "--pick-lang", help="interactively pick transcript language (youtube)"),
):
    """save, summarize, tag, embed, and find links for files or URLs."""
    from metis.client import ProviderError
    from metis.index.embed import embed_texts
    from metis.index.store import EmbeddingModelMismatch, store_chunks_with_embeddings
    from metis.index.sync import mark_file_synced
    from metis.ingest.extract import NoTranscriptError, extract
    from metis.ingest.process import process
    from metis.ingest.write import check_duplicate, write_link_only, write_to_vault

    config = load_config()
    if not _ensure_index_model(config):
        return

    if pick_folder_flag and not folder:
        from metis.pick import pick_folder
        folder = pick_folder(config)
        if not folder:
            console.print("[yellow]! folder pick cancelled, nothing ingested.[/yellow]")
            return

    if folder:
        resolved = (config.vault_path / folder).resolve()
        if not resolved.is_relative_to(config.vault_path.resolve()):
            console.print(f"[red]✗ folder must be inside the vault: {folder}[/red]")
            return
        config.output_folder = folder

    default_folder = config.output_folder

    for i, source in enumerate(sources):
        if len(sources) > 1:
            console.print(f"\n[bold]({i+1}/{len(sources)})[/bold]")

        console.print(f"[bold]ingesting:[/bold] {source}")

        # 1. extract
        try:
            from contextlib import nullcontext

            from metis.secrets import get_x_bearer
            # --pick-lang opens an interactive prompt; a spinner would fight it
            spinner = nullcontext() if pick_lang else console.status("extracting text...")
            with spinner:
                title, text, source_type, source_link, extra = extract(
                    source, lang=lang, pick_lang=pick_lang,
                    x_bearer_token=get_x_bearer(),
                )
        except NoTranscriptError:
            console.print("[yellow]! no transcript found.[/yellow]")
            save = typer.confirm("save link anyway?")
            if save:
                file_path = write_link_only(source, config)
                console.print("[bold green]✓ link saved.[/bold green]")
                console.print(f"  note: {file_path}")
            continue
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]✗ {e}[/red]")
            continue

        console.print(f"  title: {title}")
        console.print(f"  type:  {source_type}")
        console.print(f"  chars: {len(text):,}")

        # check for duplicate, keyed on the canonical source_link that write_to_vault registers
        existing = check_duplicate(source_link)
        replace_path: Optional[Path] = None
        if existing:
            console.print(f"[yellow]! already ingested:[/yellow] {existing.name}")
            if not typer.confirm("update?"):
                continue
            # defer removing the old note's vectors until embedding succeeds (below), so a
            # failed embed leaves the existing note fully intact.
            replace_path = existing

        # 2. summarize + tag + chunk
        with console.status(f"processing with {config.openai.chat_model}..."):
            processed = process(text, config)
        console.print(f"  tags:   {', '.join(processed.tags)}")
        console.print(f"  chunks: {len(processed.chunks)}")

        # 3. embed first — if this fails, vault stays clean (nothing written yet)
        try:
            with console.status("embedding and indexing..."):
                embeddings = embed_texts(processed.chunks, config)
        except ProviderError:
            raise  # a model/provider config error: the guard reports it once and aborts
        except Exception as e:
            console.print(f"[red]✗ embedding failed: {e}[/red]")
            console.print("[yellow]! note was NOT saved. vault is unchanged.[/yellow]")
            continue

        # 4. suggest folder if none specified (only for first source in batch, or each)
        if not folder and not pick_folder_flag:
            config.output_folder = default_folder
            from metis.classify import record_feedback, suggest_folder
            suggestions = suggest_folder(embeddings[0], config)
            if suggestions:
                top_folder = suggestions[0][0]
                from metis.pick import pick_suggested_folder
                chosen = pick_suggested_folder(suggestions, config)
                if chosen:
                    resolved = (config.vault_path / chosen).resolve()
                    if resolved.is_relative_to(config.vault_path.resolve()):
                        config.output_folder = chosen
                        record_feedback(source, top_folder, chosen)
                    else:
                        console.print(f"[red]✗ folder must be inside the vault: {chosen}. using default.[/red]")

        # 5. write to vault — only after embedding succeeds.
        if replace_path:
            from metis.index.sync import _remove_file_from_index
            _remove_file_from_index(str(replace_path), config)
            if replace_path.exists():
                replace_path.unlink()
        console.print("[dim]writing to vault...[/dim]")
        file_path = write_to_vault(title, text, source_link, source_type, processed, config, extra=extra)
        console.print(f"  saved: {file_path}")

        # 6. store vectors with pre-computed embeddings
        try:
            n = store_chunks_with_embeddings(processed.chunks, embeddings, file_path, config)
        except EmbeddingModelMismatch as e:
            console.print(f"[red]✗ {e}[/red]")
            continue
        console.print(f"  indexed: {n} chunks")

        # record the note in sync state so a later `metis sync` won't re-embed it
        mark_file_synced(file_path)

        console.print(f"[bold green]✓ done.[/bold green] {file_path.name}")


@app.command()
@_provider_guard
def search(
    query: str = typer.Argument(help="what to search for"),
    limit: int = typer.Option(5, "--limit", "-n", help="number of results"),
):
    """semantic search across your vault."""
    from metis.search import search_vault

    config = load_config()
    if not _ensure_index_model(config):
        return
    console.print(f"[bold]searching:[/bold] {query}\n")

    with console.status("searching..."):
        results = search_vault(query, config, limit=limit)

    if not results:
        console.print("[yellow]! no results. ingest some content first.[/yellow]")
        return

    # deduplicate by note (keep best chunk per file)
    seen = {}
    for r in results:
        if r.file_path not in seen:
            seen[r.file_path] = r
    deduped = list(seen.values())

    for i, r in enumerate(deduped, 1):
        path = Path(r.file_path).name
        preview = r.text[:150].replace("\n", " ").strip()
        if preview.startswith("---"):
            parts = preview.split("---", 2)
            preview = parts[2].strip()[:150] if len(parts) > 2 else preview
        console.print(f"[bold]{i}.[/bold] [{r.score}] [magenta]{path}[/magenta]")
        console.print(f"   {preview}...")
        console.print()

    # interactive: pick a result to chat about
    from metis.pick import pick_search_result
    selected = pick_search_result(results, config)
    if selected:
        console.print(f"\n[bold]opening chat for:[/bold] {Path(selected).stem}\n")
        from metis.chat import ask
        with console.status("thinking..."):
            answer, sources, confidence = ask(query, config, note_path=selected)
        console.print(answer)
        console.print()
        if sources:
            console.print("[dim]sources:[/dim]")
            for s in sources:
                console.print(f"  [dim]- {Path(s).name}[/dim]")


def _chat_repl(config, note_path: Optional[str], save: bool) -> None:
    """interactive multi-turn chat loop over the vault; each turn remembers the prior ones."""
    import questionary
    from questionary import Choice

    from metis.chat import ask, save_qa_to_note
    from metis.client import ProviderError
    from metis.pick import STYLE

    scope = Path(note_path).name if note_path else "the vault"
    console.print(
        f"[dim]chatting with {scope}. ask anything. "
        "/save keeps the last answer, /exit quits, /menu for options.[/dim]\n"
    )

    history: list[dict] = []
    last: tuple[str, str] | None = None

    def _save_last() -> None:
        if not last:
            console.print("[yellow]nothing to save yet.[/yellow]\n")
            return
        target = note_path
        if not target:
            name = questionary.text("save to which note?", style=STYLE).ask()
            if not name or not name.strip():
                return
            path = (config.vault_path / Path(name.strip()).with_suffix(".md")).resolve()
            if not path.is_relative_to(config.vault_path.resolve()):
                console.print("[red]✗ note must be inside the vault.[/red]\n")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"# {name.strip()}\n", encoding="utf-8")
            target = str(path)
        save_qa_to_note(target, last[0], last[1])
        console.print("[green]✓ saved.[/green]\n")

    def _menu() -> Optional[str]:
        return questionary.select(
            "menu:",
            choices=[
                Choice("keep chatting", "chat"),
                Choice("save the last answer", "save"),
                Choice("exit", "exit"),
            ],
            style=STYLE,
        ).ask()

    while True:
        q = questionary.text("you:", style=STYLE).ask()
        if q is None:
            break
        q = q.strip()
        if q in ("/exit", "/quit", "/q"):
            break
        if q == "/save":
            _save_last()
            continue
        if q == "" or q == "/menu":
            action = _menu()
            if action == "exit":
                break
            if action == "save":
                _save_last()
            continue  # keep chatting -- also ctrl-c / cancel, which returns None

        try:
            with console.status("thinking..."):
                answer, sources, _ = ask(q, config, note_path=note_path, history=history)
        except ProviderError as e:
            console.print(f"[red]✗ {e}[/red]\n")
            continue
        except Exception as e:
            console.print(f"[red]✗ chat turn failed: {e}[/red]\n")
            continue

        console.print(f"\n[magenta]metis[/magenta] {answer}\n")
        if sources:
            console.print(f"[dim]sources: {', '.join(Path(s).name for s in sources)}[/dim]\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        history = history[-10:]
        last = (q, answer)
        if note_path and save:
            save_qa_to_note(note_path, q, answer)

    console.print("[dim]bye.[/dim]")


@app.command()
@_provider_guard
def chat(
    question: Optional[str] = typer.Argument(None, help="question to ask your vault (omit for an interactive chat loop)"),
    note: Optional[str] = typer.Option(None, "--note", help="scope to a specific note", autocompletion=_complete_vault_notes),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    save: bool = typer.Option(False, "--save", "-s", help="save Q&A to the note"),
    expand: bool = typer.Option(False, "--expand", "-e", help="always offer external source search"),
):
    """RAG agent loop over your knowledge base."""
    from metis.chat import LOW_CONFIDENCE_THRESHOLD, ask

    config = load_config()
    if not _ensure_index_model(config):
        return

    if pick and not note:
        from metis.pick import pick_note
        note = pick_note(config)
        if not note:
            return

    # resolve note path
    note_path = None
    if note:
        note_p = Path(note).expanduser()
        if not note_p.suffix:
            note_p = note_p.with_suffix(".md")
        if not note_p.is_absolute():
            note_p = config.vault_path / note_p
        resolved = note_p.resolve()
        if not resolved.is_relative_to(config.vault_path.resolve()):
            console.print(f"[red]✗ note must be inside the vault: {note}[/red]")
            return
        if not note_p.exists():
            console.print(f"[red]✗ note not found: {note}[/red]")
            return
        # match the exact file_path the index stores (vault_path + clean relative), so a
        # `..` or symlinked --note path still hits the stored chunks instead of silently missing.
        note_path = str(config.vault_path / resolved.relative_to(config.vault_path.resolve()))

    if question is None:
        _chat_repl(config, note_path, save)
        return

    console.print(f"[bold]asking:[/bold] {question}\n")

    with console.status("thinking..."):
        answer, sources, confidence = ask(question, config, note_path=note_path)

    console.print(answer)
    console.print()

    if sources:
        console.print("[dim]sources:[/dim]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [dim]- {name}[/dim]")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        console.print(f"\n[yellow]low confidence ({confidence:.2f})[/yellow]")

    # when an expansion will be offered, defer the save so the final answer replaces (not
    # duplicates) this one; _offer_expand then owns the single save for this question.
    will_expand = expand or confidence < LOW_CONFIDENCE_THRESHOLD
    if not will_expand:
        _maybe_save_qa(note_path, question, answer, save)
    else:
        _offer_expand(question, answer, config, note_path, save)


def _maybe_save_qa(note_path, question, answer, save, *, expanded_from=None):
    """save the Q&A when --save is set, else offer it; keeps one entry per question."""
    if not note_path:
        return
    from metis.chat import save_qa_to_note
    if save or typer.confirm("\nsave to note?"):
        save_qa_to_note(note_path, question, answer, expanded_from=expanded_from)
        console.print("[bold green]✓ Q&A saved.[/bold green]")


def _offer_expand(question: str, answer: str, config, note_path: str | None, save: bool):
    """offer wikipedia expansion, then save exactly one Q&A entry.

    the caller defers its save so this owns it: the expanded answer is saved on success, and the
    original answer is kept as a fallback whenever the expansion does not complete.
    """
    from metis.chat import ask
    from metis.expand import extract_search_keywords, ingest_external, search_wikipedia

    console.print()
    if not typer.confirm("expand via wikipedia?"):
        _maybe_save_qa(note_path, question, answer, save)
        return

    console.print("[dim]extracting search keywords...[/dim]")
    keywords = extract_search_keywords(question, config)
    console.print(f"  keywords: {keywords}")

    try:
        console.print("[dim]searching wikipedia...[/dim]")
        results = search_wikipedia(keywords)
    except Exception as e:
        err = str(e)
        if "429" in err:
            console.print("[yellow]! rate limited — wait a minute and try again.[/yellow]")
        elif "timeout" in err.lower() or "ReadTimeout" in err:
            console.print("[yellow]! search timed out — try again later.[/yellow]")
        else:
            console.print(f"[red]✗ search failed: {err}[/red]")
        _maybe_save_qa(note_path, question, answer, save)
        return

    if not results:
        console.print("[yellow]! no results found.[/yellow]")
        _maybe_save_qa(note_path, question, answer, save)
        return

    # interactive picker for wikipedia results
    from metis.pick import pick_wikipedia
    wiki_choices = [(r.title, r.preview) for r in results]
    picked_title = pick_wikipedia(wiki_choices)

    if not picked_title:
        _maybe_save_qa(note_path, question, answer, save)
        return

    best = next(r for r in results if r.title == picked_title)

    # ingest and re-answer
    console.print("[dim]ingesting...[/dim]")
    file_path, _ = ingest_external(best, config)
    console.print(f"  saved: {file_path}")

    console.print("[dim]re-answering with new source...[/dim]\n")
    answer, sources, confidence = ask(question, config, note_path=note_path)

    console.print(answer)
    console.print()

    if sources:
        console.print("[dim]sources:[/dim]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [dim]- {name}[/dim]")

    # save the expanded answer — the single Q&A entry for this question
    note_name = Path(file_path).stem
    _maybe_save_qa(note_path, question, answer, save, expanded_from=(best.source_type, note_name))


@app.command()
@_provider_guard
def link(
    note: Optional[str] = typer.Argument(None, help="note to find connections for (all notes if omitted)"),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    write: bool = typer.Option(False, "--write", "-w", help="write wikilinks into notes"),
    min_score: float = typer.Option(0.7, "--min-score", help="minimum similarity score"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="explain why notes are connected"),
):
    """surface connections between notes."""
    from metis.link import explain_connection, find_connections, write_links

    config = load_config()
    if not _ensure_index_model(config):
        return

    if pick and not note:
        from metis.pick import pick_note
        note = pick_note(config)
        if not note:
            return

    # resolve note to full path for chromadb matching
    note_path = None
    if note:
        note_p = Path(note).expanduser()
        if not note_p.suffix:
            note_p = note_p.with_suffix(".md")
        if not note_p.is_absolute():
            note_p = config.vault_path / note_p
        note_path = str(note_p)

    target = note or "all notes"
    console.print(f"[bold]linking:[/bold] {target}\n")

    with console.status("finding connections..."):
        connections = find_connections(config, note_path=note_path, min_score=min_score)

    if not connections:
        console.print("[yellow]! no connections found above threshold.[/yellow]")
        return

    for c in connections:
        source_rel = str(Path(c.source).relative_to(config.vault_path)) if config.vault_path in Path(c.source).parents else Path(c.source).name
        target_rel = str(Path(c.target).relative_to(config.vault_path)) if config.vault_path in Path(c.target).parents else Path(c.target).name
        # remove .md for cleaner display
        source_rel = str(source_rel).removesuffix(".md")
        target_rel = str(target_rel).removesuffix(".md")
        console.print(f"  [magenta]{source_rel}[/magenta] → [magenta]{target_rel}[/magenta] [{c.score}]")
        if verbose:
            reason = explain_connection(c, config)
            console.print(f"    [dim]{reason}[/dim]")

    console.print(f"\n{len(connections)} connections found.")

    if write:
        n = write_links(connections)
        console.print(f"[bold green]✓ {n} wikilinks written.[/bold green]")
    else:
        console.print("[dim]use --write to add [[wikilinks]] to your notes.[/dim]")


@app.command()
@_provider_guard
def sync(
    force: bool = typer.Option(False, "--force", help="sync even if the vault resolves to zero files (removes all indexed notes)"),
):
    """re-index vault to catch manual edits."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from metis.index.store import EmbeddingModelMismatch
    from metis.index.sync import EmptyVaultError, sync_vault

    config = load_config()
    if not _ensure_index_model(config):
        return
    console.print(f"[bold]syncing:[/bold] {config.vault_path}\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("scanning...", total=None)

            def _on_progress(done: int, total: int, name: str) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"[dim]{name[:44]}[/dim]" if name else "[dim]finishing...[/dim]",
                )

            report = sync_vault(config, on_progress=_on_progress, force=force)
    except EmptyVaultError as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("[dim]if you really emptied the vault, re-run with --force (or 'metis reindex' to rebuild).[/dim]")
        raise typer.Exit(1)
    except EmbeddingModelMismatch as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    console.print(f"  added:     {report.added} files")
    console.print(f"  updated:   {report.updated} files")
    console.print(f"  deleted:   {report.deleted} files")
    console.print(f"  unchanged: {report.unchanged} files")
    console.print()
    console.print(f"[bold green]✓ vault indexed.[/bold green] {report.total_files} files.")


@app.command()
@_provider_guard
def reindex():
    """rebuild the whole index from scratch (use after changing the embedding model)."""
    from metis.index.sync import reindex_vault

    config = load_config()
    model = config.openai.embedding_model
    console.print(f"[bold]reindexing:[/bold] {config.vault_path}")
    console.print(f"  embedding model: [magenta]{model}[/magenta]\n")
    if not typer.confirm(f"re-embed every note with {model}? (costs one embedding call per chunk)"):
        return

    with console.status("re-embedding the whole vault..."):
        report = reindex_vault(config)

    console.print(f"  reindexed: {report.total_files} files, {report.total_chunks} chunks")
    console.print("[bold green]✓ index rebuilt.[/bold green]")


@app.command()
def init():
    """initialize metis config and directories."""
    config_path = init_config()
    config = load_config()

    config.vault_path.mkdir(parents=True, exist_ok=True)
    config.chromadb_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]✓ metis initialized.[/bold green]")
    console.print(f"  config: {config_path}")
    console.print(f"  vault:  {config.vault_path}")
    console.print(f"  db:     {config.chromadb_path}")
    console.print("\n[dim]edit ~/.metis/config.yaml to set your vault path.[/dim]")
    console.print("[dim]run 'metis secret set <name>' to store api keys securely.[/dim]")


def _complete_config_keys(incomplete: str) -> list[str]:
    return [k for k in ["vault", "folder"] if k.startswith(incomplete)]


@app.command(name="config")
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="setting: vault, folder", autocompletion=_complete_config_keys),
    value: Optional[str] = typer.Argument(None, help="new value"),
):
    """view or change metis settings."""
    import yaml

    from metis.config import CONFIG_PATH, init_config

    init_config()

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    config_keys = {
        "vault": "vault_path",
        "folder": "output_folder",
    }

    # no args: show current settings
    if not key:
        config = load_config()
        console.print(f"  vault:    {config.vault_path}")
        console.print(f"  folder:   {config.output_folder}")
        console.print(f"  base_url: {config.openai.base_url or 'default (openai)'}")
        console.print(f"\n[dim]{CONFIG_PATH}[/dim]")
        return

    if key not in config_keys:
        console.print(f"[red]✗ unknown setting: {key}. options: {', '.join(config_keys.keys())}[/red]")
        return

    # no value: show current
    if not value:
        yaml_key = config_keys[key]
        console.print(f"  {key}: {raw.get(yaml_key, 'not set')}")
        return

    # set value
    yaml_key = config_keys[key]
    raw[yaml_key] = value
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    console.print(f"[bold green]✓ {key} set to: {value}[/bold green]")


def _keychain_key() -> str:
    """the provider key from the keychain, or "" when the keyring backend is unavailable."""
    import keyring

    from metis.secrets import PROVIDER_KEY, SERVICE
    try:
        return keyring.get_password(SERVICE, PROVIDER_KEY) or ""
    except Exception:
        return ""


@app.command()
def models():
    """show the chat and embedding models in use, the resolved key, and whether the index matches."""
    import os

    from metis.client import get_chat_model, get_embedding_model, provider_of
    from metis.index.store import get_collection, indexed_embedding_model
    from metis.secrets import get_provider_key

    config = load_config()
    chat_endpoint = config.openai.base_url or "openai (default)"
    embed_endpoint = config.embedding.base_url or config.openai.base_url or "openai (default)"
    embed_tag = " (shared with chat)" if not config.embedding.base_url else " (separate)"
    resolved_model = get_embedding_model(config)
    raw_model = (config.embedding.model or config.openai.embedding_model) if config.embedding.base_url else config.openai.embedding_model
    adapted = " [dim](adapted for openrouter)[/dim]" if resolved_model != raw_model else ""

    console.print("[bold]chat[/bold]")
    console.print(f"  model:    [magenta]{get_chat_model(config)}[/magenta]")
    console.print(f"  provider: {chat_endpoint}")
    console.print()
    console.print("[bold]embedding[/bold]")
    console.print(f"  model:    [magenta]{resolved_model}[/magenta]{adapted}")
    console.print(f"  provider: {embed_endpoint}{embed_tag}")

    collection = get_collection(config)
    if collection.count() == 0:
        console.print("  index:    [dim](empty)[/dim]")
    else:
        stamped = indexed_embedding_model(collection)
        if stamped == resolved_model:
            console.print(f"  index:    {stamped} [green]✓[/green]")
        else:
            console.print(f"  index:    {stamped} [red]✗ config says {resolved_model}, run 'metis reindex'[/red]")

    # key: source + provider guess + conflict/mismatch warnings (never prints the key)
    console.print()
    console.print("[bold]key[/bold]")
    kc = _keychain_key()
    env = os.environ.get("METIS_PROVIDER_KEY", "") or ""
    resolved_key = get_provider_key()
    if not resolved_key:
        console.print("  [red]✗ no provider-key set. run 'metis secret set provider-key'[/red]")
    else:
        source = "keychain" if kc else "env"
        key_prov = _key_provider(resolved_key)
        console.print(f"  source:   {source} (looks like {key_prov})")
        base_prov = provider_of(config.openai.base_url)
        if key_prov in ("openai", "openrouter") and base_prov in ("openai", "openrouter") and key_prov != base_prov:
            console.print(f"  [red]⚠ base_url is {base_prov} but the key looks like {key_prov}: likely the wrong key[/red]")
        if len({v for v in (kc, env) if v}) > 1:
            console.print("  [yellow]⚠ different keys in keychain and env; keychain wins. clear one to avoid confusion.[/yellow]")


@app.command()
def doctor():
    """validate the setup offline and print a ✓/✗ checklist. exits non-zero if anything is off."""
    import os

    from metis.client import get_chat_model, get_embedding_model, provider_of
    from metis.index.store import get_collection, indexed_embedding_model
    from metis.secrets import get_provider_key

    config = load_config()
    ok = True

    def check(passed: bool, label: str, detail: str, fix: str = "") -> None:
        nonlocal ok
        if passed:
            console.print(f"  [green]✓[/green] {label:<10}{detail}")
        else:
            ok = False
            tail = f" [dim]{fix}[/dim]" if fix else ""
            console.print(f"  [red]✗[/red] {label:<10}{detail}{tail}")

    base_prov = provider_of(config.openai.base_url)
    resolved_key = get_provider_key()
    source = "keychain" if _keychain_key() else "env" if os.environ.get("METIS_PROVIDER_KEY") else "none"
    key_prov = _key_provider(resolved_key)
    if not resolved_key:
        check(False, "key", "not set", "run 'metis secret set provider-key'")
    elif key_prov in ("openai", "openrouter") and base_prov in ("openai", "openrouter") and key_prov != base_prov:
        check(False, "key", f"{source}, looks like {key_prov} but base_url is {base_prov}", "likely the wrong key")
    else:
        check(True, "key", f"{source} ({key_prov})")

    check(True, "chat", f"{get_chat_model(config)} via {base_prov}")

    resolved_model = get_embedding_model(config)
    raw_model = (config.embedding.model or config.openai.embedding_model) if config.embedding.base_url else config.openai.embedding_model
    adapted = " (adapted for openrouter)" if resolved_model != raw_model else ""
    embed_prov = provider_of(config.embedding.base_url or config.openai.base_url)
    check(True, "embedding", f"{resolved_model} via {embed_prov}{adapted}")

    collection = get_collection(config)
    if collection.count() == 0:
        check(True, "index", "empty (nothing embedded yet)")
    else:
        stamped = indexed_embedding_model(collection)
        matches = stamped == resolved_model
        check(matches, "index", f"built with {stamped}", "" if matches else "run 'metis reindex'")

    console.print()
    if ok:
        console.print("[bold green]✓ metis is ready.[/bold green]")
    else:
        console.print("[bold red]✗ setup has issues. fix the marked lines above.[/bold red]")
        raise typer.Exit(1)


@app.command()
def secret(
    action: str = typer.Argument(help="'set', 'delete', or 'list'"),
    name: Optional[str] = typer.Argument(None, help="key name: provider-key, embedding-key, x-token"),
):
    """manage api keys in the OS keychain."""
    from metis.secrets import (
        EMBEDDING_KEY,
        PROVIDER_KEY,
        X_BEARER,
        KeychainError,
        delete_secret,
        set_secret,
    )

    key_map = {
        "provider-key": PROVIDER_KEY,
        "embedding-key": EMBEDDING_KEY,
        "x-token": X_BEARER,
    }

    if action == "list":
        from metis.secrets import get_embedding_key, get_provider_key, get_x_bearer
        resolved = {
            "provider-key": get_provider_key(),
            "embedding-key": get_embedding_key(),
            "x-token": get_x_bearer(),
        }
        for display_name in key_map:
            status = "[green]set[/green]" if resolved[display_name] else "[dim]not set[/dim]"
            console.print(f"  {display_name}: {status}")
        return

    # interactive picker if no name given
    if not name:
        from metis.pick import pick_secret
        name = pick_secret(list(key_map.keys()))
        if not name:
            return

    if name not in key_map:
        console.print(f"[red]✗ unknown key: {name}. options: {', '.join(key_map.keys())}[/red]")
        return

    keychain_name = key_map[name]

    if action == "set":
        import getpass
        value = getpass.getpass(f"enter {name}: ")
        if not value:
            console.print("[yellow]! empty value, nothing saved.[/yellow]")
            return
        try:
            set_secret(keychain_name, value)
        except KeychainError as e:
            console.print(f"[red]✗ {e}[/red]")
            return
        console.print(f"[bold green]✓ {name} saved to keychain.[/bold green]")

    elif action == "delete":
        delete_secret(keychain_name)
        console.print(f"[bold green]✓ {name} removed from keychain.[/bold green]")

    else:
        console.print(f"[red]✗ unknown action: {action}. use 'set', 'delete', or 'list'.[/red]")


@app.command()
def folders(
    edit: bool = typer.Option(False, "--edit", "-e", help="open folder descriptions in editor"),
):
    """list vault folders with their descriptions, or edit them."""
    import os
    import subprocess
    import tempfile

    from metis.classify import (
        _auto_describe_folder,
        _load_categorization,
        _save_categorization,
        get_folder_embeddings,
    )
    from metis.config import vault_folders

    config = load_config()
    folders = vault_folders(config)

    if not folders:
        console.print("[yellow]! no folders in vault.[/yellow]")
        return

    data = _load_categorization()
    descriptions = data.get("folder_descriptions", {})

    # ensure all folders have descriptions
    for f in folders:
        if f not in descriptions:
            descriptions[f] = _auto_describe_folder(f, config)
    data["folder_descriptions"] = descriptions
    _save_categorization(data)

    if edit:
        # write descriptions to temp file, open in editor
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, prefix="metis-folders-") as tmp:
            tmp.write("# folder descriptions\n")
            tmp.write("# edit descriptions below. one per line: folder: description\n")
            tmp.write("# save and close to apply.\n\n")
            for f in folders:
                tmp.write(f"{f}: {descriptions.get(f, '')}\n")
            tmp_path = tmp.name

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        subprocess.call([editor, tmp_path])

        # read back changes
        updated = {}
        with open(tmp_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ": " in line:
                    folder, desc = line.split(": ", 1)
                    folder = folder.strip()
                    if folder in folders:
                        updated[folder] = desc.strip()

        os.unlink(tmp_path)

        if not updated:
            console.print("[yellow]! no changes detected.[/yellow]")
            return

        # apply changes and re-embed updated folders
        changed = []
        for f, desc in updated.items():
            if descriptions.get(f) != desc:
                descriptions[f] = desc
                # clear cached embedding so it gets recomputed
                if f in data.get("folder_embeddings", {}):
                    del data["folder_embeddings"][f]
                changed.append(f)

        data["folder_descriptions"] = descriptions
        _save_categorization(data)

        if changed:
            console.print(f"[dim]re-embedding {len(changed)} updated folders...[/dim]")
            get_folder_embeddings(config)
            console.print(f"[bold green]✓ {len(changed)} folder descriptions updated.[/bold green]")
            for f in changed:
                console.print(f"  [magenta]{f}[/magenta]")
        else:
            console.print("[yellow]! no changes detected.[/yellow]")

    else:
        # list mode
        for f in folders:
            note_count = len(list((config.vault_path / f).glob("*.md")))
            desc = descriptions.get(f, "")
            console.print(f"[bold magenta]{f}[/bold magenta] ({note_count} notes)")
            console.print(f"  [dim]{desc}[/dim]")
            console.print()


@app.command()
def health(
    misplaced: bool = typer.Option(False, "--misplaced", help="show notes that might belong in a different folder"),
    split: Optional[str] = typer.Option(None, "--split", help="show split suggestion for a specific folder"),
    unique: bool = typer.Option(False, "--unique", help="show notes that don't cluster with anything"),
):
    """vault health checkup: folder alignment, misplaced notes, split suggestions."""
    from metis.health import run_health

    config = load_config()
    if not _ensure_index_model(config):
        return
    vault = config.vault_path

    def _short(fp: str) -> str:
        try:
            return str(Path(fp).relative_to(vault)).removesuffix(".md")
        except ValueError:
            return Path(fp).stem

    console.print("[bold]checking vault health...[/bold]\n")
    report = run_health(config)

    if report.n_notes < 2:
        console.print("[yellow]! not enough notes to analyze.[/yellow]")
        return

    # --- flag: --split <folder> ---
    if split:
        from metis.health import analyze_split
        groups = analyze_split(split, config)
        if groups is None:
            console.print(f"[yellow]! {split}/ has too few notes to split (need 4+).[/yellow]")
            return
        console.print(f"[bold]{split}/[/bold] could split into:\n")
        for group in groups:
            console.print(f"  [magenta]{split}/{group.folder_name}/[/magenta] ({group.size} notes)")
            console.print(f"  topics: [dim]{group.label}[/dim]")
            for fp, _ in group.members[:5]:
                console.print(f"    {_short(fp)}")
            if group.size > 5:
                console.print(f"    [dim]...and {group.size - 5} more[/dim]")
            console.print()
        return

    # --- flag: --misplaced ---
    if misplaced:
        if not report.misplaced:
            console.print("[green]✓ no misplaced notes found. everything looks right.[/green]")
            return
        console.print(f"[bold]{len(report.misplaced)} potentially misplaced notes:[/bold]\n")
        from collections import defaultdict as _dd
        by_dest: dict[str, list] = _dd(list)
        for m in report.misplaced:
            by_dest[m.suggested_folder].append(m)
        for dest, items in sorted(by_dest.items()):
            console.print(f"  move to [magenta]{dest}/[/magenta]:")
            for m in items:
                console.print(f"    {_short(m.file_path)} ({m.neighbor_count}/5)")
            console.print()
        return

    # --- flag: --unique ---
    if unique:
        if not report.unique:
            console.print("[green]✓ no isolated notes found.[/green]")
            return
        console.print(f"[bold]{len(report.unique)} unique notes:[/bold]\n")
        for fp, folder in report.unique:
            console.print(f"  [dim]{_short(fp)}[/dim]")
        return

    # --- default: folder health overview ---
    for fh in report.folders:
        if fh.status == "—":
            label = "[dim]—[/dim]"
        elif fh.status == "tight":
            label = "[green]tight[/green]"
        elif fh.status == "mixed":
            label = "[yellow]mixed[/yellow]"
        else:
            label = "[red]scattered[/red]"

        if len(fh.topics) >= 2:
            topic_names = " + ".join(f"[{t.label.split(',')[0].strip()}]" for t in fh.topics)
            console.print(f"  {label}  [magenta]{fh.folder}/[/magenta] ({fh.total} notes)")
            console.print(f"           spans: {topic_names}")
        else:
            console.print(f"  {label}  [magenta]{fh.folder}/[/magenta] ({fh.total} notes)")
    console.print()

    # hints for next steps
    hints = []
    if report.misplaced:
        hints.append(f"{len(report.misplaced)} notes might be misplaced. run: [bold]metis health --misplaced[/bold]")
    if report.split_folders:
        for f in report.split_folders:
            hints.append(f"{f}/ could split. run: [bold]metis health --split {f}[/bold]")
    if report.unique:
        hints.append(f"{len(report.unique)} unique notes. run: [bold]metis health --unique[/bold]")

    if hints:
        console.print()
        for h in hints:
            console.print(f"  {h}")
