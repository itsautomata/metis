"""metis CLI: second brain for your markdown notes."""

import functools
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
import typer.rich_utils
from rich.markup import escape

from metis.config import init_config, load_config
from metis.ui import console, err_console, show_wordmark

# align typer's auto-generated help accent with the metis accent
typer.rich_utils.STYLE_OPTION = "bold #e0a458"
typer.rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = "bold #e0a458"

app = typer.Typer(
    name="metis",
    help="terminal second brain: ingest anything, search and chat over your markdown files.",
    no_args_is_help=False,
)

# global flags, set by the app callback below
_OPTS = {"yes": False, "no_input": False, "debug": False}


class SecretAction(str, Enum):
    set = "set"
    delete = "delete"
    list = "list"


class SecretName(str, Enum):
    provider_key = "provider-key"
    embedding_key = "embedding-key"
    x_token = "x-token"


class ConfigKey(str, Enum):
    vault = "vault"
    folder = "folder"
    link_style = "link-style"


def _version_callback(value: bool) -> None:
    if value:
        from metis import __version__
        show_wordmark()
        console.print(f"metis {__version__}")
        raise typer.Exit()


def _splash() -> None:
    """the warm no-args landing: the wordmark and the commands that start the loop."""
    from rich.table import Table

    show_wordmark()
    table = Table(box=None, show_header=False, padding=(0, 3, 0, 2))
    table.add_column(style="accent", no_wrap=True)
    table.add_column(style="muted")
    table.add_row("metis init", "set up (guided)")
    table.add_row("metis ingest <url or file>", "save, summarize, tag, link")
    table.add_row("metis search \"<query>\"", "semantic search")
    table.add_row("metis chat \"<question>\"", "ask your vault")
    console.print(table)
    console.print("[muted]metis --help for every command.[/muted]")


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, hidden=True, help="show the version and exit"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="assume yes to every prompt (non-interactive)"),
    no_input: bool = typer.Option(False, "--no-input", help="never prompt: decline optional prompts, fail on required ones"),
    debug: bool = typer.Option(False, "--debug", hidden=True, help="show the full traceback on an unexpected error"),
) -> None:
    _OPTS["yes"], _OPTS["no_input"], _OPTS["debug"] = yes, no_input, debug
    if ctx.invoked_subcommand is None:
        _splash()
        raise typer.Exit()


def _confirm(prompt: str, *, default: bool = False, require_tty: bool = False) -> bool:
    """confirm on a TTY, honoring --yes/--no-input. off a TTY: decline an optional prompt, or (for a
    gating/destructive one) fail naming the flag instead of hanging on a cursor."""
    if _OPTS["yes"]:
        return True
    if _OPTS["no_input"] or not sys.stdin.isatty():
        if require_tty:
            err_console.print(f"[err]✗ this needs confirmation ({prompt.strip()}); re-run with --yes.[/err]")
            raise typer.Exit(1)
        return default
    return typer.confirm(prompt, default=default)


def _provider_guard(fn):
    """turn a provider/model failure into a clean message instead of a traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import openai

        from metis.client import ProviderError
        from metis.index.store import EmbeddingModelMismatch
        try:
            return fn(*args, **kwargs)
        except EmbeddingModelMismatch as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]")
            raise typer.Exit(1)
        except (ProviderError, openai.OpenAIError) as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]")
            if isinstance(e, openai.AuthenticationError) or "401" in str(e):
                err_console.print("[muted]a 401 usually means a wrong or missing key. run 'metis models' to check the key and provider.[/muted]")
            else:
                err_console.print("[muted]check the model id, base_url, and key (metis config / metis secret).[/muted]")
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
        err_console.print(f"[err]✗ {escape(str(e))}[/err]")
        return False


@app.command(rich_help_panel="USE")
@_provider_guard
def ingest(
    sources: list[str] = typer.Argument(help="file paths or URLs to ingest"),
    folder: Optional[str] = typer.Option(None, "--folder", "-f", help="vault subfolder to save in", autocompletion=_complete_vault_folders),
    pick_folder_flag: bool = typer.Option(False, "--pick-folder", help="interactively pick vault folder"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l", help="transcript language code (youtube)"),
    pick_lang: bool = typer.Option(False, "--pick-lang", help="interactively pick transcript language (youtube)"),
):
    """save, summarize, tag, embed, and find links for files or URLs"""
    from metis.client import ProviderError
    from metis.index.canary import ensure_baseline
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
            err_console.print("[warn]! folder pick cancelled, nothing ingested.[/warn]")
            return

    if folder:
        resolved = (config.vault_path / folder).resolve()
        if not resolved.is_relative_to(config.vault_path.resolve()):
            err_console.print(f"[err]✗ folder must be inside the vault: {escape(folder)}[/err]")
            return
        config.output_folder = folder

    default_folder = config.output_folder

    for i, source in enumerate(sources):
        if len(sources) > 1:
            console.print(f"\n[bold]({i+1}/{len(sources)})[/bold]")

        console.print(f"[bold]ingesting:[/bold] {escape(str(source))}")

        # 1. extract
        try:
            from contextlib import nullcontext

            from metis.secrets import get_x_bearer
            # --pick-lang opens an interactive prompt; a spinner would fight it
            spinner = nullcontext() if pick_lang else err_console.status("extracting text...")
            with spinner:
                title, text, source_type, source_link, extra = extract(
                    source, lang=lang, pick_lang=pick_lang,
                    x_bearer_token=get_x_bearer(),
                )
        except NoTranscriptError:
            err_console.print("[warn]! no transcript found.[/warn]")
            save = _confirm("save link anyway?")
            if save:
                file_path = write_link_only(source, config)
                console.print("[success]✓ link saved.[/success]")
                console.print(f"  note: {escape(str(file_path))}")
            continue
        except (FileNotFoundError, ValueError) as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]")
            continue

        console.print(f"  title: {escape(title)}")
        console.print(f"  type:  {source_type}")
        console.print(f"  chars: {len(text):,}")

        # check for duplicate, keyed on the canonical source_link that write_to_vault registers
        existing = check_duplicate(source_link, config)
        replace_path: Optional[Path] = None
        if existing:
            err_console.print(f"[warn]! already ingested:[/warn] {escape(existing.name)}")
            if not _confirm("update?"):
                continue
            # defer removing the old note's vectors until embedding succeeds (below), so a
            # failed embed leaves the existing note fully intact.
            replace_path = existing

        # 2. summarize + tag + chunk
        with err_console.status(f"processing with {config.openai.chat_model}..."):
            processed = process(text, config)
        console.print(f"  tags:   {', '.join(processed.tags)}")
        console.print(f"  chunks: {len(processed.chunks)}")

        # 3. embed first — if this fails, vault stays clean (nothing written yet)
        try:
            with err_console.status("embedding and indexing..."):
                embeddings = embed_texts(processed.chunks, config)
        except ProviderError:
            raise  # a model/provider config error: the guard reports it once and aborts
        except Exception as e:
            err_console.print(f"[err]✗ embedding failed: {escape(str(e))}[/err]")
            err_console.print("[warn]! note was NOT saved. vault is unchanged.[/warn]")
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
                        record_feedback(source, top_folder, chosen, config)
                    else:
                        err_console.print(f"[err]✗ folder must be inside the vault: {escape(chosen)}. using default.[/err]")

        # 5. write to vault — only after embedding succeeds.
        if replace_path:
            from metis.index.sync import _remove_file_from_index
            _remove_file_from_index(str(replace_path), config)
            if replace_path.exists():
                replace_path.unlink()
        console.print("[muted]writing to vault...[/muted]")
        file_path = write_to_vault(title, text, source_link, source_type, processed, config, extra=extra)
        console.print(f"  saved: {escape(str(file_path))}")

        # 6. store vectors with pre-computed embeddings
        try:
            n = store_chunks_with_embeddings(processed.chunks, embeddings, file_path, config)
        except EmbeddingModelMismatch as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]")
            continue
        console.print(f"  indexed: {n} chunks")

        # record the note in sync state so a later `metis sync` won't re-embed it
        mark_file_synced(file_path, config)
        # baseline the drift canary once the first vectors for this model have landed
        ensure_baseline(config)

        console.print(f"[success]✓ done.[/success] {escape(file_path.name)}")


@app.command(rich_help_panel="USE")
@_provider_guard
def search(
    query: str = typer.Argument(help="what to search for"),
    limit: int = typer.Option(5, "--limit", "-n", min=1, help="number of results"),
    json_out: bool = typer.Option(False, "--json", help="emit results as JSON (non-interactive)"),
):
    """semantic search across your vault"""
    from metis.search import search_vault

    config = load_config()
    if not _ensure_index_model(config):
        return
    if not json_out:
        console.print(f"[bold]searching:[/bold] {escape(query)}\n")

    with err_console.status("searching..."):
        results = search_vault(query, config, limit=limit)

    if not results:
        if json_out:
            console.print_json(data=[])
        else:
            err_console.print("[warn]! no results. ingest some content first.[/warn]")
        return

    # deduplicate by note (keep best chunk per file)
    seen = {}
    for r in results:
        if r.file_path not in seen:
            seen[r.file_path] = r
    deduped = list(seen.values())

    def _preview(r) -> str:
        preview = r.text[:150].replace("\n", " ").strip()
        if preview.startswith("---"):
            parts = preview.split("---", 2)
            preview = parts[2].strip()[:150] if len(parts) > 2 else preview
        return preview

    if json_out:
        console.print_json(data=[
            {"rank": i, "score": r.score, "note": Path(r.file_path).name,
             "path": r.file_path, "preview": _preview(r)}
            for i, r in enumerate(deduped, 1)
        ])
        return

    for i, r in enumerate(deduped, 1):
        path = Path(r.file_path).name
        console.print(f"[bold]{i}.[/bold] [{r.score}] [accent]{path}[/accent]")
        console.print(f"   {escape(_preview(r))}...")
        console.print()

    # interactive: pick a result to chat about
    from metis.pick import pick_search_result
    selected = pick_search_result(results, config)
    if selected:
        console.print(f"\n[bold]opening chat for:[/bold] {escape(Path(selected).stem)}\n")
        from metis.chat import ask
        with err_console.status("thinking..."):
            answer, sources, confidence = ask(query, config, note_path=selected)
        console.print(escape(answer))
        console.print()
        if sources:
            console.print("[muted]sources:[/muted]")
            for s in sources:
                console.print(f"  [muted]- {escape(Path(s).name)}[/muted]")


def _chat_repl(config, note_path: Optional[str], save: bool) -> None:
    """interactive multi-turn chat loop over the vault; each turn remembers the prior ones."""
    import questionary
    from questionary import Choice

    from metis.chat import ask, save_qa_to_note
    from metis.client import ProviderError
    from metis.pick import STYLE, _ask

    scope = Path(note_path).name if note_path else "the vault"
    console.print(
        f"[muted]chatting with {scope}. ask anything. "
        "/save keeps the last answer, /exit quits, /menu for options.[/muted]\n"
    )

    history: list[dict] = []
    last: tuple[str, str] | None = None

    def _save_last() -> None:
        if not last:
            console.print("[warn]nothing to save yet.[/warn]\n")
            return
        target = note_path
        if not target:
            name = _ask(questionary.text("save to which note?", style=STYLE))
            if not name or not name.strip():
                return
            path = (config.vault_path / Path(name.strip()).with_suffix(".md")).resolve()
            if not path.is_relative_to(config.vault_path.resolve()):
                err_console.print("[err]✗ note must be inside the vault.[/err]\n")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"# {name.strip()}\n", encoding="utf-8")
            target = str(path)
        save_qa_to_note(target, last[0], last[1])
        console.print("[ok]✓ saved.[/ok]\n")

    def _menu() -> Optional[str]:
        return _ask(questionary.select(
            "menu:",
            choices=[
                Choice("keep chatting", "chat"),
                Choice("save the last answer", "save"),
                Choice("exit", "exit"),
            ],
            style=STYLE,
        ))

    while True:
        q = _ask(questionary.text("you:", style=STYLE))
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
            with err_console.status("thinking..."):
                answer, sources, _ = ask(q, config, note_path=note_path, history=history)
        except ProviderError as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]\n")
            continue
        except Exception as e:
            err_console.print(f"[err]✗ chat turn failed: {escape(str(e))}[/err]\n")
            continue

        console.print(f"\n[accent]metis[/accent] {escape(answer)}\n")
        if sources:
            console.print(f"[muted]sources: {', '.join(Path(s).name for s in sources)}[/muted]\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        history = history[-10:]
        last = (q, answer)
        if note_path and save:
            save_qa_to_note(note_path, q, answer)

    console.print("[muted]bye.[/muted]")


@app.command(rich_help_panel="USE")
@_provider_guard
def chat(
    question: Optional[str] = typer.Argument(None, help="question to ask your vault (omit for an interactive chat loop)"),
    note: Optional[str] = typer.Option(None, "--note", help="scope to a specific note", autocompletion=_complete_vault_notes),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    save: bool = typer.Option(False, "--save", "-s", help="save Q&A to the note"),
    expand: bool = typer.Option(False, "--expand", "-e", help="always offer external source search"),
):
    """ask your vault a question; answers cite the notes they draw on"""
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
            err_console.print(f"[err]✗ note must be inside the vault: {escape(note)}[/err]")
            return
        if not note_p.exists():
            err_console.print(f"[err]✗ note not found: {escape(note)}[/err]")
            return
        # match the exact file_path the index stores (vault_path + clean relative), so a
        # `..` or symlinked --note path still hits the stored chunks instead of silently missing.
        note_path = str(config.vault_path / resolved.relative_to(config.vault_path.resolve()))

    if question is None:
        _chat_repl(config, note_path, save)
        return

    console.print(f"[bold]asking:[/bold] {escape(question)}\n")

    with err_console.status("thinking..."):
        answer, sources, confidence = ask(question, config, note_path=note_path)

    console.print(escape(answer))
    console.print()

    if sources:
        console.print("[muted]sources:[/muted]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [muted]- {escape(name)}[/muted]")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        console.print(f"\n[warn]low confidence ({confidence:.2f})[/warn]")

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
    if save or _confirm("save to note?"):
        save_qa_to_note(note_path, question, answer, expanded_from=expanded_from)
        console.print("[success]✓ Q&A saved.[/success]")


def _offer_expand(question: str, answer: str, config, note_path: str | None, save: bool):
    """offer wikipedia expansion, then save exactly one Q&A entry.

    the caller defers its save so this owns it: the expanded answer is saved on success, and the
    original answer is kept as a fallback whenever the expansion does not complete.
    """
    from metis.chat import ask
    from metis.expand import extract_search_keywords, ingest_external, search_wikipedia

    console.print()
    if not _confirm("expand via wikipedia?"):
        _maybe_save_qa(note_path, question, answer, save)
        return

    try:
        console.print("[muted]extracting search keywords...[/muted]")
        keywords = extract_search_keywords(question, config)
        console.print(f"  keywords: {escape(keywords)}")
        console.print("[muted]searching wikipedia...[/muted]")
        results = search_wikipedia(keywords)
    except Exception as e:
        err = str(e)
        if "429" in err:
            err_console.print("[warn]! rate limited, wait a minute and try again.[/warn]")
        elif "timeout" in err.lower() or "ReadTimeout" in err:
            err_console.print("[warn]! search timed out, try again later.[/warn]")
        else:
            err_console.print(f"[err]✗ expansion failed: {escape(str(err))}[/err]")
        _maybe_save_qa(note_path, question, answer, save)
        return

    if not results:
        err_console.print("[warn]! no results found.[/warn]")
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

    # ingest and re-answer, scoped to the freshly ingested article
    console.print("[muted]ingesting...[/muted]")
    try:
        file_path, _ = ingest_external(best, config)
        console.print(f"  saved: {escape(str(file_path))}")
        console.print("[muted]re-answering with new source...[/muted]\n")
        answer, sources, confidence = ask(question, config, note_path=str(file_path))
    except Exception as e:
        err_console.print(f"[err]✗ could not expand with that article: {escape(str(e))}[/err]")
        _maybe_save_qa(note_path, question, answer, save)
        return

    console.print(escape(answer))
    console.print()

    if sources:
        console.print("[muted]sources:[/muted]")
        for s in sources:
            name = Path(s).name
            console.print(f"  [muted]- {escape(name)}[/muted]")

    # save the expanded answer — the single Q&A entry for this question
    note_name = Path(file_path).stem
    _maybe_save_qa(note_path, question, answer, save, expanded_from=(best.source_type, note_name))


@app.command(rich_help_panel="USE")
@_provider_guard
def link(
    note: Optional[str] = typer.Argument(None, help="note to find connections for (all notes if omitted)"),
    pick: bool = typer.Option(False, "--pick", "-p", help="interactively pick a note"),
    write: bool = typer.Option(False, "--write", "-w", help="write wikilinks into notes"),
    min_score: float = typer.Option(0.7, "--min-score", help="minimum similarity score"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="explain why notes are connected"),
):
    """surface connections between notes"""
    from metis.link import (
        explain_connection,
        find_connections,
        resolve_link_style,
        write_links,
    )

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
    console.print(f"[bold]linking:[/bold] {escape(str(target))}\n")

    with err_console.status("finding connections..."):
        connections = find_connections(config, note_path=note_path, min_score=min_score)

    if not connections:
        err_console.print("[warn]! no connections found above threshold.[/warn]")
        return

    for c in connections:
        source_rel = str(Path(c.source).relative_to(config.vault_path)) if config.vault_path in Path(c.source).parents else Path(c.source).name
        target_rel = str(Path(c.target).relative_to(config.vault_path)) if config.vault_path in Path(c.target).parents else Path(c.target).name
        # remove .md for cleaner display
        source_rel = str(source_rel).removesuffix(".md")
        target_rel = str(target_rel).removesuffix(".md")
        console.print(f"  [accent]{escape(source_rel)}[/accent] → [accent]{escape(target_rel)}[/accent] [{c.score}]")
        if verbose:
            reason = explain_connection(c, config)
            console.print(f"    [muted]{escape(reason)}[/muted]")

    console.print(f"\n{len(connections)} connections found.")

    style = resolve_link_style(config)
    if write:
        n = write_links(connections, config)
        label = "wikilinks" if style == "wikilink" else "markdown links"
        console.print(f"[success]✓ {n} {label} written.[/success]")
    else:
        example = "[[wikilinks]]" if style == "wikilink" else "[markdown](links)"
        console.print(f"[muted]use --write to add {example} to your notes.[/muted]")


@app.command(rich_help_panel="MAINTAIN")
@_provider_guard
def sync(
    force: bool = typer.Option(False, "--force", help="sync even if the vault resolves to zero files (removes all indexed notes)"),
):
    """re-index vault to catch manual edits"""
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
    console.print(f"[bold]syncing:[/bold] {escape(str(config.vault_path))}\n")

    from metis.index.canary import check_drift
    with err_console.status("checking embedding drift..."):
        _drift = check_drift(config)
    if _drift.status == "drift":
        console.print("[warn]⚠ embedding output has drifted since the index was built; new chunks will land in a different space. consider 'metis reindex' first.[/warn]\n")
    elif _drift.status == "variance":
        console.print("[warn]⚠ provider returns unstable embeddings (non-deterministic routing); pin the provider/quantization, reindex will not fix this.[/warn]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task = progress.add_task("scanning...", total=None)

            def _on_progress(done: int, total: int, name: str) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=done,
                    description=f"[muted]{name[:44]}[/muted]" if name else "[muted]finishing...[/muted]",
                )

            report = sync_vault(config, on_progress=_on_progress, force=force)
    except EmptyVaultError as e:
        err_console.print(f"[err]✗ {escape(str(e))}[/err]")
        console.print("[muted]if you really emptied the vault, re-run with --force (or 'metis reindex' to rebuild).[/muted]")
        raise typer.Exit(1)
    except EmbeddingModelMismatch as e:
        err_console.print(f"[err]✗ {escape(str(e))}[/err]")
        raise typer.Exit(1)

    console.print(f"  added:     {report.added} files")
    console.print(f"  updated:   {report.updated} files")
    console.print(f"  deleted:   {report.deleted} files")
    console.print(f"  unchanged: {report.unchanged} files")
    if report.skipped:
        console.print(f"  [warn]skipped:   {report.skipped} unreadable files[/warn]")
    console.print()
    console.print(f"[success]✓ vault indexed.[/success] {report.total_files} files.")


@app.command(rich_help_panel="MAINTAIN")
@_provider_guard
def reindex(
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would be re-embedded without calling the provider"),
):
    """rebuild the whole index from scratch (use after changing the embedding model)"""
    from metis.index.sync import reindex_vault

    config = load_config()
    model = config.openai.embedding_model
    console.print(f"[bold]reindexing:[/bold] {escape(str(config.vault_path))}")
    console.print(f"  embedding model: [accent]{escape(model)}[/accent]\n")

    if dry_run:
        from metis.index.sync import _find_vault_files
        n = sum(1 for _ in _find_vault_files(config))
        console.print(f"  would re-embed [bold]{n}[/bold] notes with [accent]{escape(model)}[/accent] (one embedding call per chunk).")
        console.print("[muted]run without --dry-run to execute.[/muted]")
        return

    if not _confirm(f"re-embed every note with {model}? (costs one embedding call per chunk)", require_tty=True):
        return

    with err_console.status("re-embedding the whole vault..."):
        report = reindex_vault(config)

    console.print(f"  reindexed: {report.total_files} files, {report.total_chunks} chunks")
    console.print("[success]✓ index rebuilt.[/success]")


def _interactive() -> bool:
    """a real terminal and no non-interactive flag: the guided wizard may prompt."""
    return sys.stdin.isatty() and not _OPTS["no_input"] and not _OPTS["yes"]


# provider presets: a menu pick fills base_url
_PROVIDERS = [
    ("openai", ""),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("ollama / local", "http://localhost:11434/v1"),
]
_CUSTOM_BASE_URL = object()


def _wizard_base_url(current: str) -> str:
    """provider menu returning a base_url; the last option accepts a custom endpoint."""
    from metis import pick

    options = [
        (f"{name}  (default)" if not url else f"{name}  [{url}]", url)
        for name, url in _PROVIDERS
    ]
    options.append(("custom base url", _CUSTOM_BASE_URL))
    picked = pick.pick_from("provider:", options, default=current)
    if picked is _CUSTOM_BASE_URL:
        return typer.prompt("base url", default=current or "").strip()
    if picked is None:
        return current
    return picked


def _store_key(label: str, keychain_name: str) -> None:
    """prompt for a secret and store it; a blank entry skips. a keychain failure is reported, not fatal."""
    from metis.secrets import KeychainError, set_secret

    value = typer.prompt(f"{label} (blank = set later)", default="", hide_input=True).strip()
    if not value:
        return
    try:
        set_secret(keychain_name, value)
        console.print(f"[ok]✓ {label} saved to the keychain.[/ok]")
    except KeychainError as e:
        err_console.print(f"[err]✗ {escape(str(e))}[/err]")


def _init_wizard(config) -> None:
    """a-to-z guided setup on a terminal: vault, provider, key, models, folder, links, optional extras.
    every prompt shows a default (enter accepts); optional fields take a blank line to stay unset.
    """
    import yaml

    from metis.config import CONFIG_PATH
    from metis.secrets import EMBEDDING_KEY, PROVIDER_KEY, X_BEARER, get_provider_key

    show_wordmark()
    console.print("[bold]metis setup[/bold] [muted](enter accepts the default, blank leaves it unset)[/muted]\n")

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}
    raw["openai"] = raw.get("openai") or {}

    vault_path = Path(typer.prompt("vault path", default=str(config.vault_path)).strip()).expanduser()
    raw["vault_path"] = str(vault_path)

    raw["openai"]["base_url"] = _wizard_base_url(config.openai.base_url)

    if not get_provider_key():
        _store_key("provider api key", PROVIDER_KEY)

    raw["openai"]["chat_model"] = typer.prompt("chat model", default=config.openai.chat_model).strip()
    raw["openai"]["embedding_model"] = typer.prompt("embedding model", default=config.openai.embedding_model).strip()

    raw["output_folder"] = typer.prompt("save ingested notes to", default=config.output_folder).strip()

    from metis import pick

    link = pick.pick_from("link style:", [
        ("auto (detect from the vault's notes app)", "auto"),
        ("wikilink  [[note]]", "wikilink"),
        ("markdown  [note](note.md)", "markdown"),
    ], default=config.link_style or "auto")
    if link in ("wikilink", "markdown"):
        raw["link_style"] = link
    else:
        raw.pop("link_style", None)

    if pick.confirm_menu("advanced options (separate embedding endpoint, extra keys)?", default=False):
        embed_url = typer.prompt("embedding base url (blank = share with chat)", default="").strip()
        if embed_url:
            embed_model = typer.prompt("embedding model on that endpoint", default=raw["openai"]["embedding_model"]).strip()
            raw["embedding"] = {"base_url": embed_url, "model": embed_model}
            _store_key("embedding api key", EMBEDDING_KEY)
        _store_key("x / twitter token", X_BEARER)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    console.print()


@app.command(rich_help_panel="SETUP")
def init():
    """set up metis, guided.

    creates the config and directories, and on a terminal runs a wizard for the vault, provider,
    key, and models. with --no-input or --yes it writes defaults for CI.
    """
    config_path = init_config()
    config = load_config()

    interactive = _interactive()
    if interactive:
        _init_wizard(config)
        config = load_config()

    for label, path in (("vault", config.vault_path), ("index", config.chromadb_path)):
        if path.exists() and not path.is_dir():
            err_console.print(f"[err]✗ the {label} path is a file, not a directory: {escape(str(path))}[/err]")
            raise typer.Exit(1)
    config.vault_path.mkdir(parents=True, exist_ok=True)
    config.chromadb_path.mkdir(parents=True, exist_ok=True)

    from rich.table import Table
    console.print("[success]✓ metis initialized[/success]")
    paths = Table(box=None, show_header=False, padding=(0, 2, 0, 2))
    paths.add_column(style="muted", no_wrap=True)
    paths.add_column(overflow="fold")
    paths.add_row("config", escape(str(config_path)))
    paths.add_row("vault", escape(str(config.vault_path)))
    paths.add_row("index", escape(str(config.chromadb_path)))
    console.print(paths)

    if not interactive:
        console.print("\n[muted]set the vault path: metis config vault <path>[/muted]")
        console.print("[muted]store an api key:   metis secret set provider-key[/muted]")
        return

    console.print()
    from metis import pick
    if pick.confirm_menu("run doctor to validate the setup now?", default=True):
        console.print()
        try:
            doctor(json_out=False)
        except typer.Exit:
            pass
    console.print("\n[muted]next: metis ingest <url or file>[/muted]")


@app.command(name="config", rich_help_panel="SETUP")
def config_cmd(
    key: Optional[ConfigKey] = typer.Argument(None, help="setting to change"),
    value: Optional[str] = typer.Argument(None, help="new value"),
):
    """view or change metis settings"""
    key = key.value if key is not None else None
    import yaml

    from metis.config import CONFIG_PATH, init_config

    init_config()

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    config_keys = {
        "vault": "vault_path",
        "folder": "output_folder",
        "link-style": "link_style",
    }

    # no args: show current settings
    if not key:
        from metis.link import resolve_link_style

        config = load_config()
        console.print(f"  vault:    {escape(str(config.vault_path))}")
        console.print(f"  folder:   {escape(config.output_folder)}")
        console.print(f"  base_url: {config.openai.base_url or 'default (openai)'}")
        console.print(f"  links:    {resolve_link_style(config)}{' (forced)' if config.link_style else ' (auto)'}")
        console.print(f"\n[muted]{escape(str(CONFIG_PATH))}[/muted]")
        return

    if key not in config_keys:
        err_console.print(f"[err]✗ unknown setting: {escape(key)}. options: {', '.join(config_keys.keys())}[/err]")
        return

    # no value: show current
    if not value:
        yaml_key = config_keys[key]
        console.print(f"  {key}: {raw.get(yaml_key, 'not set')}")
        return

    # set value
    yaml_key = config_keys[key]
    if key == "link-style":
        if value not in ("wikilink", "markdown", "auto"):
            err_console.print(f"[err]✗ link-style must be wikilink, markdown, or auto (got: {escape(value)})[/err]")
            return
        if value == "auto":
            raw.pop(yaml_key, None)  # clear so it auto-detects from the vault's notes app
        else:
            raw[yaml_key] = value
    else:
        raw[yaml_key] = value
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    console.print(f"[success]✓ {escape(key)} set to: {escape(value)}[/success]")


def _keychain_key() -> str:
    """the provider key from the keychain, or "" when the keyring backend is unavailable."""
    import keyring

    from metis.secrets import PROVIDER_KEY, SERVICE
    try:
        return keyring.get_password(SERVICE, PROVIDER_KEY) or ""
    except Exception:
        return ""


@app.command(rich_help_panel="INSPECT")
def models():
    """show the chat and embedding models in use, the resolved key, and whether the index matches"""
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
    adapted = " [muted](adapted for openrouter)[/muted]" if resolved_model != raw_model else ""

    console.print("[bold]chat[/bold]")
    console.print(f"  model:    [accent]{escape(get_chat_model(config))}[/accent]")
    console.print(f"  provider: {escape(chat_endpoint)}")
    console.print()
    console.print("[bold]embedding[/bold]")
    console.print(f"  model:    [accent]{escape(resolved_model)}[/accent]{adapted}")
    console.print(f"  provider: {escape(embed_endpoint)}{embed_tag}")

    collection = get_collection(config)
    if collection.count() == 0:
        console.print("  index:    [muted](empty)[/muted]")
    else:
        stamped = indexed_embedding_model(collection)
        if stamped == resolved_model:
            console.print(f"  index:    {escape(stamped)} [ok]✓[/ok]")
        else:
            console.print(f"  index:    {escape(stamped)} [err]✗ config says {escape(resolved_model)}, run 'metis reindex'[/err]")

    # key: source + provider guess + conflict/mismatch warnings (never prints the key)
    console.print()
    console.print("[bold]key[/bold]")
    kc = _keychain_key()
    env = os.environ.get("METIS_PROVIDER_KEY", "") or ""
    resolved_key = get_provider_key()
    if not resolved_key:
        console.print("  [err]✗ no provider-key set. run 'metis secret set provider-key'[/err]")
    else:
        source = "keychain" if kc else "env"
        key_prov = _key_provider(resolved_key)
        console.print(f"  source:   {source} (looks like {key_prov})")
        base_prov = provider_of(config.openai.base_url)
        if key_prov in ("openai", "openrouter") and base_prov in ("openai", "openrouter") and key_prov != base_prov:
            console.print(f"  [err]⚠ base_url is {base_prov} but the key looks like {key_prov}: likely the wrong key[/err]")
        if len({v for v in (kc, env) if v}) > 1:
            console.print("  [warn]⚠ different keys in keychain and env; keychain wins. clear one to avoid confusion.[/warn]")


@app.command(rich_help_panel="INSPECT")
def doctor(
    json_out: bool = typer.Option(False, "--json", help="emit the checklist as JSON"),
):
    """validate the setup and print a ✓/✗ checklist; exits non-zero if anything is off

    includes a live embedding-drift check (re-embeds a canary); it degrades to a neutral
    result when the provider is unreachable, so the rest of the checklist still works offline.
    """
    import os

    from metis.client import get_chat_model, get_embedding_model, provider_of
    from metis.index.store import get_collection, indexed_embedding_model
    from metis.secrets import get_provider_key

    config = load_config()
    ok = True
    checks: list[dict] = []

    def check(passed: bool, label: str, detail: str, fix: str = "") -> None:
        nonlocal ok
        checks.append({"check": label, "passed": passed, "detail": detail, "fix": fix})
        if not passed:
            ok = False
        if json_out:
            return
        if passed:
            console.print(f"  [ok]✓[/ok] {label:<10}{escape(detail)}")
        else:
            tail = f" [muted]{escape(fix)}[/muted]" if fix else ""
            console.print(f"  [err]✗[/err] {label:<10}{escape(detail)}{tail}")

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

    if collection.count() > 0:
        from metis.index.canary import check_drift
        with err_console.status("checking embedding drift..."):
            verdict = check_drift(config)
        if verdict.status == "drift":
            check(False, "drift", "embedding model changed since the index was built", "run 'metis reindex'")
        elif verdict.status == "variance":
            check(False, "drift", "provider returns unstable embeddings (non-deterministic routing)", "pin the provider/quantization; reindex will not fix this")
        elif verdict.status == "unavailable":
            check(False, "drift", verdict.detail, "check your connection or key")
        else:  # stable | not_baselined
            check(True, "drift", verdict.detail)

    if json_out:
        console.print_json(data={"ok": ok, "checks": checks})
        if not ok:
            raise typer.Exit(1)
        return

    console.print()
    if ok:
        console.print("[success]✓ metis is ready.[/success]")
    else:
        console.print("[danger]✗ setup has issues. fix the marked lines above.[/danger]")
        raise typer.Exit(1)


@app.command(rich_help_panel="SETUP")
def secret(
    action: SecretAction = typer.Argument(help="set, delete, or list"),
    name: Optional[SecretName] = typer.Argument(None, help="which key"),
):
    """manage api keys in the OS keychain"""
    action = action.value
    name = name.value if name is not None else None
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
            status = "[ok]set[/ok]" if resolved[display_name] else "[muted]not set[/muted]"
            console.print(f"  {display_name}: {status}")
        return

    # interactive picker if no name given
    if not name:
        from metis.pick import pick_secret
        name = pick_secret(list(key_map.keys()))
        if not name:
            return

    if name not in key_map:
        err_console.print(f"[err]✗ unknown key: {name}. options: {', '.join(key_map.keys())}[/err]")
        return

    keychain_name = key_map[name]

    if action == "set":
        import getpass
        value = getpass.getpass(f"enter {name}: ")
        if not value:
            err_console.print("[warn]! empty value, nothing saved.[/warn]")
            return
        try:
            set_secret(keychain_name, value)
        except KeychainError as e:
            err_console.print(f"[err]✗ {escape(str(e))}[/err]")
            return
        console.print(f"[success]✓ {name} saved to keychain.[/success]")

    elif action == "delete":
        delete_secret(keychain_name)
        console.print(f"[success]✓ {name} removed from keychain.[/success]")

    else:
        err_console.print(f"[err]✗ unknown action: {action}. use 'set', 'delete', or 'list'.[/err]")


@app.command(rich_help_panel="INSPECT")
@_provider_guard
def folders(
    edit: bool = typer.Option(False, "--edit", "-e", help="open folder descriptions in editor"),
):
    """list vault folders with their descriptions, or edit them"""
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
        err_console.print("[warn]! no folders in vault.[/warn]")
        return

    data = _load_categorization(config)
    descriptions = data.get("folder_descriptions", {})

    # ensure all folders have descriptions
    for f in folders:
        if f not in descriptions:
            descriptions[f] = _auto_describe_folder(f, config)
    data["folder_descriptions"] = descriptions
    _save_categorization(data, config)

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
            err_console.print("[warn]! no changes detected.[/warn]")
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
        _save_categorization(data, config)

        if changed:
            console.print(f"[muted]re-embedding {len(changed)} updated folders...[/muted]")
            get_folder_embeddings(config)
            console.print(f"[success]✓ {len(changed)} folder descriptions updated.[/success]")
            for f in changed:
                console.print(f"  [accent]{escape(f)}[/accent]")
        else:
            err_console.print("[warn]! no changes detected.[/warn]")

    else:
        # list mode
        for f in folders:
            note_count = len(list((config.vault_path / f).glob("*.md")))
            desc = descriptions.get(f, "")
            console.print(f"[heading]{escape(f)}[/heading] ({note_count} notes)")
            console.print(f"  [muted]{escape(desc)}[/muted]")
            console.print()


@app.command(rich_help_panel="INSPECT")
def health(
    misplaced: bool = typer.Option(False, "--misplaced", help="show notes that might belong in a different folder"),
    split: Optional[str] = typer.Option(None, "--split", help="show split suggestion for a specific folder"),
    unique: bool = typer.Option(False, "--unique", help="show notes that don't cluster with anything"),
):
    """vault health checkup: folder alignment, misplaced notes, split suggestions"""
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
    with err_console.status("analyzing notes..."):
        report = run_health(config)

    if report.n_notes < 2:
        err_console.print("[warn]! not enough notes to analyze.[/warn]")
        return

    # --- flag: --split <folder> ---
    if split:
        from metis.health import analyze_split
        groups = analyze_split(split, config)
        if groups is None:
            err_console.print(f"[warn]! {escape(split)}/ has too few notes to split (need 4+).[/warn]")
            return
        console.print(f"[bold]{escape(split)}/[/bold] could split into:\n")
        for group in groups:
            console.print(f"  [accent]{escape(split)}/{escape(group.folder_name)}/[/accent] ({group.size} notes)")
            console.print(f"  topics: [muted]{escape(group.label)}[/muted]")
            for fp, _ in group.members[:5]:
                console.print(f"    {escape(_short(fp))}")
            if group.size > 5:
                console.print(f"    [muted]...and {group.size - 5} more[/muted]")
            console.print()
        return

    # --- flag: --misplaced ---
    if misplaced:
        if not report.misplaced:
            console.print("[ok]✓ no misplaced notes found. everything looks right.[/ok]")
            return
        console.print(f"[bold]{len(report.misplaced)} potentially misplaced notes:[/bold]\n")
        from collections import defaultdict as _dd
        by_dest: dict[str, list] = _dd(list)
        for m in report.misplaced:
            by_dest[m.suggested_folder].append(m)
        for dest, items in sorted(by_dest.items()):
            console.print(f"  move to [accent]{escape(dest)}/[/accent]:")
            for m in items:
                console.print(f"    {escape(_short(m.file_path))} ({m.neighbor_count}/5)")
            console.print()
        return

    # --- flag: --unique ---
    if unique:
        if not report.unique:
            console.print("[ok]✓ no isolated notes found.[/ok]")
            return
        console.print(f"[bold]{len(report.unique)} unique notes:[/bold]\n")
        for fp, folder in report.unique:
            console.print(f"  [muted]{escape(_short(fp))}[/muted]")
        return

    # --- default: folder health overview ---
    for fh in report.folders:
        if fh.status == "—":
            label = "[muted]—[/muted]"
        elif fh.status == "tight":
            label = "[ok]tight[/ok]"
        elif fh.status == "mixed":
            label = "[warn]mixed[/warn]"
        else:
            label = "[err]scattered[/err]"

        if len(fh.topics) >= 2:
            topic_names = " + ".join(f"\\[{escape(t.label.split(',')[0].strip())}]" for t in fh.topics)
            console.print(f"  {label}  [accent]{escape(fh.folder)}/[/accent] ({fh.total} notes)")
            console.print(f"           spans: {topic_names}")
        else:
            console.print(f"  {label}  [accent]{escape(fh.folder)}/[/accent] ({fh.total} notes)")
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


def main() -> None:
    """entry point that turns an unexpected crash into a clean stderr message (honors --debug)."""
    try:
        app()
    except typer.Exit:
        raise
    except Exception as exc:
        if _OPTS["debug"]:
            err_console.print_exception(show_locals=False)
        else:
            err_console.print("[err]✗ something went wrong (this looks like a bug in metis).[/err]")
            err_console.print(f"  [muted]{escape(str(exc))}[/muted]")
            err_console.print("[muted]re-run the same command with --debug to see the full traceback.[/muted]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
