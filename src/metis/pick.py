"""interactive pickers for vault files and folders."""

import os
import sys
from pathlib import Path

import questionary
from questionary import Style

from metis.config import MetisConfig, vault_folders

# the selected row is dark text on an amber bar; `reverse` used to flip amber fg into an amber bg with
# the terminal's default (white) fg, which read as unreadable white-on-amber.
STYLE = Style([
    ("qmark", "fg:#e0a458 bold"),
    ("question", "bold"),
    ("answer", "fg:#e0a458"),
    ("pointer", "fg:#e0a458 bold"),
    ("highlighted", "fg:#1a1714 bg:#e0a458 bold"),
    ("selected", "fg:#1a1714 bg:#e0a458 bold"),
    ("instruction", "dim"),
    ("completion-menu.completion", "fg:#cdc5b8 bg:#2b2925"),
    ("completion-menu.completion.current", "fg:#1a1714 bg:#e0a458 bold"),
    ("completion-menu.meta.completion", "fg:#8a8272 bg:#2b2925"),
    ("completion-menu.meta.completion.current", "fg:#3a2f1a bg:#e0a458"),
    ("scrollbar.button", "bg:#e0a458"),
])


def _accessible() -> bool:
    """numbered/typed prompts instead of arrow-key widgets, for screen readers or no-arrow terminals."""
    return bool(os.environ.get("METIS_ACCESSIBLE") or os.environ.get("ACCESSIBLE"))


def _read_line(prompt: str) -> str | None:
    """read one line, prompting on stderr so stdout stays clean. None on Ctrl-D / Ctrl-C."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return None


def _numbered_choice(prompt: str, options: list):
    """render options as a numbered list on stderr and read a pick. options: list of (title, value)."""
    sys.stderr.write(prompt + "\n")
    for i, (title, _value) in enumerate(options, 1):
        sys.stderr.write(f"  {i}. {title}\n")
    raw = _read_line("number (blank to cancel): ")
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    if raw.isdecimal() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1][1]
    return None


def _typed_choice(prompt: str, choices: list[str]) -> str | None:
    """read a typed value validated against choices (exact, then a unique case-insensitive substring)."""
    raw = _read_line(f"{prompt} (type a name, blank to cancel): ")
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    if raw in choices:
        return raw
    matches = [c for c in choices if raw.lower() in c.lower()]
    return matches[0] if len(matches) == 1 else None


def _ask(question):
    """run a prompt, treating Ctrl-D (EOFError) as a cancel, the way questionary already treats Ctrl-C."""
    try:
        return question.ask()
    except EOFError:
        return None


def pick_from(prompt: str, options: list, default=None):
    """single-select arrow-key menu. options: list of (title, value); returns the chosen value or None.
    accessible mode renders a numbered list instead of an arrow-key widget."""
    if _accessible():
        return _numbered_choice(prompt, options)
    choices = [questionary.Choice(title=title, value=value) for title, value in options]
    kwargs = {"choices": choices, "style": STYLE}
    values = [value for _title, value in options]
    if default is not None and default in values:
        kwargs["default"] = default
    return _ask(questionary.select(prompt, **kwargs))


def confirm_menu(prompt: str, default: bool = True) -> bool:
    """a yes/no answered by navigating a menu, not by typing y/n. returns the default if cancelled."""
    result = pick_from(prompt, [("yes", True), ("no", False)], default=default)
    return default if result is None else result


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

    if _accessible():
        return _typed_choice("note", notes)

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

    if _accessible():
        return _typed_choice("folder", folders)

    return _ask(questionary.autocomplete(
        "folder:",
        choices=folders,
        match_middle=True,
        style=STYLE,
    ))


_PICK_EXISTING = object()
_NEW_FOLDER = object()
# questionary.Choice(value=None) falls back to the title, so "skip" needs a real sentinel
_SKIP = object()


def pick_suggested_folder(suggestions: list[tuple[str, float]], config: MetisConfig) -> str | None:
    """menu of ranked folder suggestions plus routes to an existing or new folder.

    returns the chosen folder (relative path), or None if cancelled.
    """
    if _accessible():
        options = [(folder, folder) for folder, _score in suggestions]
        options.append(("pick an existing folder…", _PICK_EXISTING))
        options.append(("type a new folder name…", _NEW_FOLDER))
        choice = _numbered_choice("folder:", options)
        if choice is _PICK_EXISTING:
            folders = vault_folders(config)
            return _typed_choice("existing folder", folders) if folders else None
        if choice is _NEW_FOLDER:
            name = _read_line("new folder name: ")
            return name.strip() if name and name.strip() else None
        return choice

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

    def _label(r) -> str:
        try:
            rel = str(Path(r.file_path).relative_to(vault)).removesuffix(".md")
        except ValueError:
            rel = Path(r.file_path).stem
        preview = r.text[:80].replace("\n", " ").strip()
        return f"[{r.score}] {rel}  {preview}"

    if _accessible():
        options = [(_label(r), r.file_path) for r in results]
        options.append(("skip", _SKIP))
        choice = _numbered_choice("results:", options)
        return None if choice is _SKIP else choice

    choices = [questionary.Choice(title=_label(r), value=r.file_path) for r in results]
    choices.append(questionary.Choice(title="skip", value=_SKIP))

    choice = _ask(questionary.select(
        "results:",
        choices=choices,
        style=STYLE,
    ))

    return None if choice is _SKIP else choice


def pick_secret(key_names: list[str]) -> str | None:
    """interactive secret key picker."""
    if _accessible():
        return _numbered_choice("key:", [(k, k) for k in key_names])

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

    def _label(title: str, summary: str) -> str:
        preview = summary[:80].replace("\n", " ").strip()
        return f"{title}  {preview}"

    if _accessible():
        options = [(_label(title, summary), title) for title, summary in results]
        options.append(("skip", _SKIP))
        choice = _numbered_choice("article:", options)
        return None if choice is _SKIP else choice

    choices = [
        questionary.Choice(title=_label(title, summary), value=title)
        for title, summary in results
    ]
    choices.append(questionary.Choice(title="skip", value=_SKIP))

    choice = _ask(questionary.select(
        "article:",
        choices=choices,
        style=STYLE,
    ))

    return None if choice is _SKIP else choice
