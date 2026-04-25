"""metis CLI: second brain that pairs with obsidian."""

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
    sources: list[str] = typer.Argument(help="file paths or URLs to ingest"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="vault subfolder to save in", autocompletion=_complete_vault_folders),
    pick_folder_flag: bool = typer.Option(False, "--pick-folder", help="interactively pick vault folder"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="transcript language code (youtube)"),
    pick_lang: bool = typer.Option(False, "--pick-lang", help="interactively pick transcript language (youtube)"),
):
    """save, summarize, tag, embed, and find links for files or URLs."""
    from metis.ingest.extract import extract, NoTranscriptError
    from metis.ingest.process import process
    from metis.ingest.write import write_to_vault, write_link_only, check_duplicate
    from metis.index.embed import embed_texts
    from metis.index.store import store_chunks_with_embeddings

    config = load_config()

    if pick_folder_flag and not folder:
        from metis.pick import pick_folder
        folder = pick_folder(config)

    if folder:
        resolved = (config.vault_path / folder).resolve()
        if not resolved.is_relative_to(config.vault_path.resolve()):
            console.print(f"[red]folder must be inside the vault: {folder}[/red]")
            return
        config.output_folder = folder

    for i, source in enumerate(sources):
        if len(sources) > 1:
            console.print(f"\n[bold]({i+1}/{len(sources)})[/bold]")

        # check for duplicate
        existing = check_duplicate(source)
        if existing:
            console.print(f"[yellow]already ingested:[/yellow] {existing.name}")
            if not typer.confirm("update?"):
                continue
            from metis.index.sync import _remove_file_from_index
            _remove_file_from_index(str(existing), config)
            existing.unlink()

        console.print(f"[bold]ingesting:[/bold] {source}")

        # 1. extract
        console.print("[dim]extracting text...[/dim]")
        try:
            from metis.secrets import get_x_bearer
            title, text, source_type, source_link, extra = extract(
                source, lang=lang, pick_lang=pick_lang,
                x_bearer_token=get_x_bearer(config.x_api.bearer_token),
            )
        except NoTranscriptError:
            console.print("[yellow]no transcript found.[/yellow]")
            save = typer.confirm("save link anyway?")
            if save:
                file_path = write_link_only(source, config)
                console.print(f"[bold green]link saved.[/bold green]")
                console.print(f"  note: {file_path}")
            continue
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            continue

        console.print(f"  title: {title}")
        console.print(f"  type:  {source_type}")
        console.print(f"  chars: {len(text)}")

        # 2. summarize + tag + chunk
        console.print(f"[dim]processing with {config.provider}...[/dim]")
        processed = process(text, config)
        console.print(f"  tags:   {', '.join(processed.tags)}")
        console.print(f"  chunks: {len(processed.chunks)}")

        # 3. embed first — if this fails, vault stays clean
        console.print("[dim]embedding and indexing...[/dim]")
        try:
            embeddings = embed_texts(processed.chunks, config)
        except Exception as e:
            console.print(f"[red]embedding failed: {e}[/red]")
            console.print("[yellow]note was NOT saved. vault is unchanged.[/yellow]")
            continue

        # 4. suggest folder if none specified (only for first source in batch, or each)
        if not folder and not pick_folder_flag:
            from metis.classify import suggest_folder, record_feedback
            suggestions = suggest_folder(embeddings[0], config)
            if suggestions:
                top_folder, top_score = suggestions[0]
                console.print(f"\n[bold]suggested folder:[/bold] [cyan]{top_folder}[/cyan] ({top_score:.2f})")
                if len(suggestions) > 1:
                    others = ", ".join(f"{f} ({s:.2f})" for f, s in suggestions[1:])
                    console.print(f"  [dim]also: {others}[/dim]")

                choice = input(f"\naccept? [Y/n/other]: ").strip()
                if choice == "" or choice.lower() == "y":
                    config.output_folder = top_folder
                    record_feedback(source, top_folder, top_folder)
                elif choice.lower() == "n":
                    from metis.pick import pick_folder
                    picked = pick_folder(config)
                    if picked:
                        config.output_folder = picked
                        record_feedback(source, top_folder, picked)
                else:
                    config.output_folder = choice
                    record_feedback(source, top_folder, choice)

        # 5. write to vault — only after embedding succeeds
        console.print("[dim]writing to vault...[/dim]")
        file_path = write_to_vault(title, text, source_link, source_type, processed, config, extra=extra)
        console.print(f"  saved: {file_path}")

        # 6. store vectors with pre-computed embeddings
        n = store_chunks_with_embeddings(processed.chunks, embeddings, file_path, config)
        console.print(f"  indexed: {n} chunks")

        console.print(f"[bold green]done.[/bold green] {file_path.name}")


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
        console.print(f"[bold]{i}.[/bold] [{r.score}] [cyan]{path}[/cyan]")
        console.print(f"   {preview}...")
        console.print()

    # interactive: pick a result to chat about
    from metis.pick import pick_search_result
    selected = pick_search_result(results, config)
    if selected:
        console.print(f"\n[bold]opening chat for:[/bold] {Path(selected).stem}\n")
        from metis.chat import ask
        answer, sources, confidence = ask(query, config, note_path=selected)
        console.print(answer)
        console.print()
        if sources:
            console.print("[dim]sources:[/dim]")
            for s in sources:
                console.print(f"  [dim]- {Path(s).name}[/dim]")


@app.command()
def chat(
    question: str = typer.Argument(help="question to ask your vault"),
    note: Optional[str] = typer.Option(None, "--note", help="scope to a specific note", autocompletion=_complete_vault_notes),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    save: bool = typer.Option(False, "--save", "-s", help="save Q&A to the note"),
    expand: bool = typer.Option(False, "--expand", "-e", help="always offer external source search"),
):
    """RAG agent loop over your knowledge base."""
    from metis.chat import ask, save_qa_to_note, LOW_CONFIDENCE_THRESHOLD

    config = load_config()

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
        if not note_p.resolve().is_relative_to(config.vault_path.resolve()):
            console.print(f"[red]note must be inside the vault: {note}[/red]")
            return
        note_path = str(note_p)
        if not note_p.exists():
            console.print(f"[red]note not found: {note_path}[/red]")
            return

    console.print(f"[bold]asking:[/bold] {question}\n")

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

    # save Q&A to note — always offer when --note is used
    if note_path:
        if save or typer.confirm("\nsave to note?"):
            save_qa_to_note(note_path, question, answer)
            console.print("[bold green]Q&A saved.[/bold green]")

    # offer external expansion — on low confidence or --expand flag
    if expand or confidence < LOW_CONFIDENCE_THRESHOLD:
        _offer_expand(question, config, note_path, save)


def _offer_expand(question: str, config, note_path: str | None, save: bool):
    """offer wikipedia search after a chat answer."""
    from metis.expand import search_wikipedia, ingest_external, extract_search_keywords
    from metis.chat import ask, save_qa_to_note

    console.print()
    choice = input("expand via wikipedia? [y/N]: ").strip().lower()

    if choice != "y":
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
            console.print("[yellow]rate limited — wait a minute and try again.[/yellow]")
        elif "timeout" in err.lower() or "ReadTimeout" in err:
            console.print("[yellow]search timed out — try again later.[/yellow]")
        else:
            console.print(f"[red]search failed: {err}[/red]")
        return

    if not results:
        console.print("[yellow]no results found.[/yellow]")
        return

    # interactive picker for wikipedia results
    from metis.pick import pick_wikipedia
    wiki_choices = [(r.title, r.preview) for r in results]
    picked_title = pick_wikipedia(wiki_choices)

    if not picked_title:
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

    # save Q&A to note if requested
    if save and note_path:
        if typer.confirm("\nsave to note?"):
            note_name = Path(file_path).stem
            expanded_from = (best.source_type, note_name)
            save_qa_to_note(note_path, question, answer, expanded_from=expanded_from)
            console.print("[bold green]Q&A saved.[/bold green]")


@app.command()
def link(
    note: Optional[str] = typer.Argument(None, help="note to find connections for (all notes if omitted)"),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    write: bool = typer.Option(False, "--write", "-w", help="write wikilinks into notes"),
    min_score: float = typer.Option(0.7, "--min-score", help="minimum similarity score"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="explain why notes are connected"),
):
    """surface connections between notes."""
    from metis.link import find_connections, write_links, explain_connection

    config = load_config()

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

    connections = find_connections(config, note_path=note_path, min_score=min_score)

    if not connections:
        console.print("[yellow]no connections found above threshold.[/yellow]")
        return

    for c in connections:
        source_rel = str(Path(c.source).relative_to(config.vault_path)) if config.vault_path in Path(c.source).parents else Path(c.source).name
        target_rel = str(Path(c.target).relative_to(config.vault_path)) if config.vault_path in Path(c.target).parents else Path(c.target).name
        # remove .md for cleaner display
        source_rel = str(source_rel).removesuffix(".md")
        target_rel = str(target_rel).removesuffix(".md")
        console.print(f"  [cyan]{source_rel}[/cyan] → [cyan]{target_rel}[/cyan] [{c.score}]")
        if verbose:
            reason = explain_connection(c, config)
            console.print(f"    [dim]{reason}[/dim]")

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
    console.print("\n[dim]edit ~/.metis/config.yaml to set your vault path.[/dim]")
    console.print("[dim]run 'metis secret set <name>' to store api keys securely.[/dim]")


def _complete_config_keys(incomplete: str) -> list[str]:
    return [k for k in ["vault", "folder", "provider"] if k.startswith(incomplete)]


@app.command(name="config")
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="setting: vault, folder, provider", autocompletion=_complete_config_keys),
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
        "provider": "provider",
    }

    # no args: show current settings
    if not key:
        config = load_config()
        console.print(f"  vault:    {config.vault_path}")
        console.print(f"  folder:   {config.output_folder}")
        console.print(f"  provider: {config.provider}")
        console.print(f"\n[dim]{CONFIG_PATH}[/dim]")
        return

    if key not in config_keys:
        console.print(f"[red]unknown setting: {key}. options: {', '.join(config_keys.keys())}[/red]")
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

    console.print(f"[bold green]{key} set to: {value}[/bold green]")


@app.command()
def secret(
    action: str = typer.Argument(help="'set', 'delete', or 'list'"),
    name: Optional[str] = typer.Argument(None, help="key name: openai-key, azure-key, x-token"),
):
    """manage api keys in the OS keychain."""
    from metis.secrets import set_secret, delete_secret, OPENAI_KEY, AZURE_KEY, X_BEARER

    key_map = {
        "openai-key": OPENAI_KEY,
        "azure-key": AZURE_KEY,
        "x-token": X_BEARER,
        "azure-search-endpoint": "azure-search-endpoint",
        "azure-search-key": "azure-search-key",
        "discord-token": "discord-token",
        "azure-storage-connection": "azure-storage-connection",
    }

    if action == "list":
        from metis.secrets import get_secret
        for display_name, keychain_name in key_map.items():
            value = get_secret(keychain_name)
            status = "[green]set[/green]" if value else "[dim]not set[/dim]"
            console.print(f"  {display_name}: {status}")
        return

    # interactive picker if no name given
    if not name:
        from metis.pick import pick_secret
        name = pick_secret(list(key_map.keys()))
        if not name:
            return

    if name not in key_map:
        console.print(f"[red]unknown key: {name}. options: {', '.join(key_map.keys())}[/red]")
        return

    keychain_name = key_map[name]

    if action == "set":
        import getpass
        value = getpass.getpass(f"enter {name}: ")
        if not value:
            console.print("[yellow]empty value, nothing saved.[/yellow]")
            return
        set_secret(keychain_name, value)
        console.print(f"[bold green]{name} saved to keychain.[/bold green]")

    elif action == "delete":
        delete_secret(keychain_name)
        console.print(f"[bold green]{name} removed from keychain.[/bold green]")

    else:
        console.print(f"[red]unknown action: {action}. use 'set', 'delete', or 'list'.[/red]")


@app.command()
def folders(
    edit: bool = typer.Option(False, "--edit", "-e", help="open folder descriptions in editor"),
):
    """list vault folders with their descriptions, or edit them."""
    import os
    import subprocess
    import tempfile
    from metis.classify import (
        _get_vault_folders, _auto_describe_folder, _load_categorization, _save_categorization,
        get_folder_embeddings,
    )

    config = load_config()
    vault_folders = _get_vault_folders(config)

    if not vault_folders:
        console.print("[yellow]no folders in vault.[/yellow]")
        return

    data = _load_categorization()
    descriptions = data.get("folder_descriptions", {})

    # ensure all folders have descriptions
    for f in vault_folders:
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
            for f in vault_folders:
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
                    if folder in vault_folders:
                        updated[folder] = desc.strip()

        os.unlink(tmp_path)

        if not updated:
            console.print("[yellow]no changes detected.[/yellow]")
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
            console.print(f"[bold green]{len(changed)} folder descriptions updated.[/bold green]")
            for f in changed:
                console.print(f"  [cyan]{f}[/cyan]")
        else:
            console.print("[yellow]no changes detected.[/yellow]")

    else:
        # list mode
        for f in vault_folders:
            note_count = len(list((config.vault_path / f).glob("*.md")))
            desc = descriptions.get(f, "")
            console.print(f"[bold cyan]{f}[/bold cyan] ({note_count} notes)")
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
    vault = config.vault_path

    def _short(fp: str) -> str:
        try:
            return str(Path(fp).relative_to(vault)).removesuffix(".md")
        except ValueError:
            return Path(fp).stem

    console.print("[bold]checking vault health...[/bold]\n")
    report = run_health(config)

    if report.n_notes < 2:
        console.print("[yellow]not enough notes to analyze.[/yellow]")
        return

    # --- flag: --split <folder> ---
    if split:
        from metis.health import analyze_split
        groups = analyze_split(split, config)
        if groups is None:
            console.print(f"[yellow]{split}/ has too few notes to split (need 4+).[/yellow]")
            return
        console.print(f"[bold]{split}/[/bold] could split into:\n")
        for group in groups:
            console.print(f"  [cyan]{split}/{group.folder_name}/[/cyan] ({group.size} notes)")
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
            console.print("[green]no misplaced notes found. everything looks right.[/green]")
            return
        console.print(f"[bold]{len(report.misplaced)} potentially misplaced notes:[/bold]\n")
        from collections import defaultdict as _dd
        by_dest: dict[str, list] = _dd(list)
        for m in report.misplaced:
            by_dest[m.suggested_folder].append(m)
        for dest, items in sorted(by_dest.items()):
            console.print(f"  move to [cyan]{dest}/[/cyan]:")
            for m in items:
                console.print(f"    {_short(m.file_path)} ({m.neighbor_count}/5)")
            console.print()
        return

    # --- flag: --unique ---
    if unique:
        if not report.unique:
            console.print("[green]no isolated notes found.[/green]")
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
            console.print(f"  {label}  [cyan]{fh.folder}/[/cyan] ({fh.total} notes)")
            console.print(f"           spans: {topic_names}")
        else:
            console.print(f"  {label}  [cyan]{fh.folder}/[/cyan] ({fh.total} notes)")
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
