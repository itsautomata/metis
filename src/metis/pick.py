"""interactive pickers for vault files and folders."""

from pathlib import Path

import questionary
from questionary import Style

from metis.config import MetisConfig

STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:cyan"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "reverse"),
    ("instruction", "dim"),
    ("completion-menu.completion.current", "reverse"),
    ("completion-menu.meta.completion", "dim"),
    ("completion-menu.meta.completion.current", "reverse dim"),
    ("scrollbar.button", "bg:cyan"),
])


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

    choice = questionary.autocomplete(
        "note:",
        choices=notes,
        match_middle=True,
        style=STYLE,
    ).ask()

    return choice


def _vault_folders(config: MetisConfig) -> list[str]:
    """vault subfolders as sorted relative paths, excluding symlinks that escape the vault."""
    vault = config.vault_path
    if not vault.exists():
        return []
    vault_resolved = vault.resolve()
    return sorted(
        str(p.relative_to(vault))
        for p in vault.rglob("*")
        if p.is_dir()
        and not p.name.startswith(".")
        and p.resolve().is_relative_to(vault_resolved)
    )


def pick_folder(config: MetisConfig) -> str | None:
    """folder picker with type-to-filter. returns relative path or None if cancelled."""
    folders = _vault_folders(config)
    if not folders:
        return None

    return questionary.autocomplete(
        "folder:",
        choices=folders,
        match_middle=True,
        style=STYLE,
    ).ask()


_PICK_EXISTING = object()
_NEW_FOLDER = object()


def pick_suggested_folder(suggestions: list[tuple[str, float]], config: MetisConfig) -> str | None:
    """menu of ranked folder suggestions plus routes to an existing or new folder.

    returns the chosen folder (relative path), or None if cancelled.
    """
    choices = [
        questionary.Choice(title=f"{folder}  ({score:.2f})", value=folder)
        for folder, score in suggestions
    ]
    choices.append(questionary.Choice(title="pick an existing folder…", value=_PICK_EXISTING))
    choices.append(questionary.Choice(title="type a new folder name…", value=_NEW_FOLDER))

    choice = questionary.select(
        "save to which folder?",
        choices=choices,
        style=STYLE,
    ).ask()

    if choice is _PICK_EXISTING:
        folders = _vault_folders(config)
        if not folders:
            return None
        return questionary.select(
            "existing folder:",
            choices=folders,
            style=STYLE,
        ).ask()

    if choice is _NEW_FOLDER:
        name = questionary.text("new folder name:", style=STYLE).ask()
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

    choice = questionary.select(
        "results:",
        choices=choices,
        style=STYLE,
    ).ask()

    return choice


def pick_secret(key_names: list[str]) -> str | None:
    """interactive secret key picker."""
    choice = questionary.select(
        "which key?",
        choices=key_names,
        style=STYLE,
    ).ask()

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

    choice = questionary.select(
        "ingest which article?",
        choices=choices,
        style=STYLE,
    ).ask()

    return choice
