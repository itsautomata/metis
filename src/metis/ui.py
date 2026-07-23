"""shared console and the metis color theme, so the whole CLI speaks one palette."""

from rich.console import Console
from rich.theme import Theme

# the Ember palette
THEME = Theme({
    "accent": "#e0a458",
    "ok": "#9ec96a",
    "err": "#e0685f",
    "warn": "#d99a3a",
    "info": "#7bb0c9",
    "muted": "#8a8272",
    "heading": "bold #e0a458",
    # rich renders [bold ok] plain (no theme lookup in a compound tag), so bold+color are named styles
    "success": "bold #9ec96a",
    "danger": "bold #e0685f",
})

# highlight=False so only role tags color output; rich's highlighter would otherwise fight the palette
console = Console(theme=THEME, highlight=False)                   # data and results go to stdout
err_console = Console(stderr=True, theme=THEME, highlight=False)  # errors, warnings, hints, progress go to stderr

def show_wordmark(subtitle: str = "a terminal second brain") -> None:
    """print the compact METIS wordmark, only on an interactive terminal (never into a pipe or CI)."""
    if not console.is_terminal:
        return
    console.print()
    console.print("[heading]M E T I S[/heading]")
    console.print("[muted]▔▔▔▔▔▔▔▔▔[/muted]")
    if subtitle:
        console.print(f"[muted]{subtitle}[/muted]")
    console.print()
