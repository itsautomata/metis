"""interactive pickers for vault files and folders."""

from pathlib import Path

import questionary
from questionary import Style

from metis.config import MetisConfig, vault_folders

STYLE = Style([
    ("qmark", "fg:magenta bold"),
    ("question", "bold"),
    ("answer", "fg:magenta"),
    ("pointer", "fg:magenta bold"),
    ("highlighted", "fg:magenta bold"),
    ("selected", "reverse"),
    ("instruction", "dim"),
    ("completion-menu.completion.current", "reverse"),
    ("completion-menu.meta.completion", "dim"),
    ("completion-menu.meta.completion.current", "reverse dim"),
    ("scrollbar.button", "bg:magenta"),
])


def _ask(question):
    """run a prompt, treating Ctrl-D (EOFError) as a cancel, the way questionary already treats Ctrl-C."""
    try:
        return question.ask()
    except EOFError:
        return None


def pick_note(config: MetisConfig) -> str | None:
    """interactive note picker. returns relative path or None if cancelled."""
    vault = config.vault_path
    if not vault.exists():
        return None

    notes = sorted(
        str(p.relative_to(vault)).removesuffix(".md")
        for p in vault.rglob("*.md")
        if not p.name.startswith(".")
    )

    if not notes:
        return None

    choice = _ask(questionary.autocomplete(
        "note:",
        choices=notes,
        match_middle=True,
        style=STYLE,
    ))

    return choice


def pick_folder(config: MetisConfig) -> str | None:
    """folder picker with type-to-filter. returns relative path or None if cancelled."""
    folders = vault_folders(config)
    if not folders:
        return None

    return _ask(questionary.autocomplete(
        "folder:",
        choices=folders,
        match_middle=True,
        style=STYLE,
    ))


_PICK_EXISTING = object()
_NEW_FOLDER = object()


def pick_suggested_folder(suggestions: list[tuple[str, float]], config: MetisConfig) -> str | None:
    """menu of ranked folder suggestions plus routes to an existing or new folder.

    returns the chosen folder (relative path), or None if cancelled.
    """
    choices = [
        questionary.Choice(title=folder, value=folder)
        for folder, _score in suggestions
    ]
    choices.append(questionary.Choice(title="pick an existing folder…", value=_PICK_EXISTING))
    choices.append(questionary.Choice(title="type a new folder name…", value=_NEW_FOLDER))

    choice = _ask(questionary.select(
        "folder:",
        choices=choices,
        style=STYLE,
    ))

    if choice is _PICK_EXISTING:
        folders = vault_folders(config)
        if not folders:
            return None
        return _ask(questionary.select(
            "existing folder:",
            choices=folders,
            style=STYLE,
        ))

    if choice is _NEW_FOLDER:
        name = _ask(questionary.text("new folder name:", style=STYLE))
        return name.strip() if name and name.strip() else None

    return choice


def pick_search_result(results: list, config: MetisConfig) -> str | None:
    """interactive search result picker. returns file_path or None if cancelled."""
    if not results:
        return None

    vault = config.vault_path
    choices = []
    for r in results:
        try:
            rel = str(Path(r.file_path).relative_to(vault)).removesuffix(".md")
        except ValueError:
            rel = Path(r.file_path).stem
        preview = r.text[:80].replace("\n", " ").strip()
        choices.append(questionary.Choice(
            title=f"[{r.score}] {rel}  {preview}",
            value=r.file_path,
        ))

    choices.append(questionary.Choice(title="skip", value=None))

    choice = _ask(questionary.select(
        "results:",
        choices=choices,
        style=STYLE,
    ))

    return choice


def pick_secret(key_names: list[str]) -> str | None:
    """interactive secret key picker."""
    choice = _ask(questionary.select(
        "key:",
        choices=key_names,
        style=STYLE,
    ))

    return choice


def pick_wikipedia(results: list[tuple[str, str]]) -> str | None:
    """interactive wikipedia result picker. takes list of (title, summary). returns title or None."""
    if not results:
        return None

    choices = []
    for title, summary in results:
        preview = summary[:80].replace("\n", " ").strip()
        choices.append(questionary.Choice(
            title=f"{title}  {preview}",
            value=title,
        ))

    choices.append(questionary.Choice(title="skip", value=None))

    choice = _ask(questionary.select(
        "article:",
        choices=choices,
        style=STYLE,
    ))

    return choice
