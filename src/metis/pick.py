"""interactive pickers for vault files and folders."""

from pathlib import Path

import questionary
from questionary import Style

from metis.config import MetisConfig

STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:cyan"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:cyan"),
    ("text", "fg:white"),
    ("instruction", "fg:white dim"),
    ("completion-menu", "bg:black fg:white"),
    ("completion-menu.completion", "bg:black fg:white"),
    ("completion-menu.completion.current", "bg:#333333 fg:cyan bold"),
    ("completion-menu.meta.completion", "bg:black fg:white dim"),
    ("completion-menu.meta.completion.current", "bg:#333333 fg:cyan"),
    ("scrollbar.background", "bg:#333333"),
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


def pick_folder(config: MetisConfig) -> str | None:
    """interactive folder picker. returns relative path or None if cancelled."""
    vault = config.vault_path
    if not vault.exists():
        return None

    folders = sorted(
        str(p.relative_to(vault))
        for p in vault.rglob("*")
        if p.is_dir() and not p.name.startswith(".")
    )

    if not folders:
        return None

    choice = questionary.autocomplete(
        "folder:",
        choices=folders,
        match_middle=True,
        style=STYLE,
    ).ask()

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
