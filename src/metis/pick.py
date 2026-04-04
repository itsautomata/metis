"""interactive pickers for vault files and folders."""

from pathlib import Path

import questionary

from metis.config import MetisConfig


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
    ).ask()

    return choice
