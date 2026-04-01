"""PRSM CLI UI toolkit — monochromatic, geometric, minimal.

Shared UI components for all PRSM CLI commands. Uses Rich for rendering
and click for interactive prompts. No color except semantic status indicators.
Arrow-key selection menus via simple-term-menu with click.Choice fallback.
"""
import os
import random
import shutil
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .theme import THEME, ICONS
from .banner import PRSM_BANNER, PRSM_BANNER_COMPACT, TAGLINES, RULE

console = Console()

# ── Arrow-key menu support ───────────────────────────────────────────

_HAS_TERM_MENU = False
try:
    from simple_term_menu import TerminalMenu
    _HAS_TERM_MENU = True
except ImportError:
    pass


def _is_interactive_terminal() -> bool:
    """Check if stdin/stdout support interactive menus."""
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


# ── Banner & Headers ──────────────────────────────────────────────────


def show_banner(version: str = "0.1.0"):
    """Display the PRSM monochromatic banner."""
    width = shutil.get_terminal_size().columns
    if width < 60:
        banner_text = PRSM_BANNER_COMPACT
    else:
        banner_text = PRSM_BANNER

    styled = Text()
    styled.append(banner_text.strip(), style=THEME.primary)

    tagline = random.choice(TAGLINES)
    styled.append(f"\n\n  {tagline}", style=THEME.muted)
    styled.append(f"\n  v{version}", style=THEME.dim)

    console.print(
        Panel(
            styled,
            border_style=THEME.border,
            padding=(1, 2),
        )
    )


def section(title: str, icon: str = "bullet"):
    """Print a section header — clean, geometric."""
    ico = ICONS.get(icon, icon)
    console.print(f"\n  {ico} {title}", style=THEME.heading)
    console.print(f"    {'─' * len(title)}", style=THEME.rule)


def step_header(step: int, total: int, title: str):
    """Step progress: ◆ Step 1/7 ━━━━━━░░░░░ Title"""
    filled = int((step / total) * 20)
    bar = "━" * filled + "░" * (20 - filled)
    console.print(f"\n  ◆ Step {step}/{total}  {bar}", style=THEME.primary)
    console.print(f"    {title}", style=THEME.heading)
    console.print(f"    {'─' * len(title)}", style=THEME.rule)


# ── Status Messages ───────────────────────────────────────────────────


def success(msg: str):
    """Print a success message with green checkmark."""
    console.print(f"    {ICONS['success']} {msg}", style=THEME.success)


def warning(msg: str):
    """Print a warning message with yellow indicator."""
    console.print(f"    {ICONS['warning']} {msg}", style=THEME.warning)


def error(msg: str):
    """Print an error message with red cross."""
    console.print(f"    {ICONS['error']} {msg}", style=THEME.error)


def info(msg: str):
    """Print an informational message — dim, unobtrusive."""
    console.print(f"    {ICONS['info']} {msg}", style=THEME.muted)


def muted(msg: str):
    """Print muted/secondary text — dim italic."""
    console.print(f"      {msg}", style=THEME.dim)


# ── Interactive Prompts ───────────────────────────────────────────────


def prompt_text(message: str, default=None, **kwargs) -> str:
    """Clean text prompt — no color noise. Shows default hint."""
    if default is not None and default != "":
        display = f"    {ICONS['arrow']} {message} [{default}] (Enter for default)"
    else:
        display = f"    {ICONS['arrow']} {message}"
    return click.prompt(display, default=default, **kwargs)


def prompt_choice(message: str, choices: list, default: int = 0) -> str:
    """Selection menu with arrow-key navigation.

    Each choice is a dict with 'label', 'value', and optional 'hint'.
    Uses simple-term-menu for arrow-key selection when available,
    falls back to click.Choice for non-interactive terminals.
    """
    if _HAS_TERM_MENU and _is_interactive_terminal():
        return _prompt_choice_interactive(message, choices, default)
    else:
        return _prompt_choice_fallback(message, choices, default)


def _prompt_choice_interactive(message: str, choices: list, default: int = 0) -> str:
    """Arrow-key navigable selection menu using simple-term-menu."""
    console.print(f"\n    {message}", style=THEME.heading)
    console.print(f"    (↑/↓ arrows to move, Enter to select)", style=THEME.dim)
    console.print()

    # Build menu entries with hints
    entries = []
    for c in choices:
        label = c["label"]
        hint = c.get("hint", "")
        if hint:
            entries.append(f"  {label}  — {hint}")
        else:
            entries.append(f"  {label}")

    menu = TerminalMenu(
        entries,
        cursor_index=default,
        menu_cursor="  ► ",
        menu_cursor_style=("bold",),
        menu_highlight_style=("bold",),
    )

    idx = menu.show()
    if idx is None:
        # User pressed Escape/Ctrl-C — use default
        idx = default

    selected = choices[idx]
    console.print(f"    {ICONS['arrow']} Selected: {selected['label']}", style=THEME.primary)
    return selected["value"]


def _prompt_choice_fallback(message: str, choices: list, default: int = 0) -> str:
    """Fallback choice menu using click.Choice (for piped/non-interactive input)."""
    console.print(f"\n    {message}", style=THEME.heading)
    for i, c in enumerate(choices):
        marker = ICONS["check"] if i == default else ICONS["uncheck"]
        label = c["label"]
        hint = c.get("hint", "")
        if hint:
            console.print(f"      {marker} {label}", style=THEME.primary)
            console.print(f"        {hint}", style=THEME.dim)
        else:
            console.print(f"      {marker} {label}", style=THEME.primary)

    valid = [c["value"] for c in choices]
    return click.prompt(
        f"    {ICONS['arrow']} Choose",
        type=click.Choice(valid, case_sensitive=False),
        default=valid[default],
    )


def prompt_confirm(message: str, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    return click.confirm(f"    {ICONS['arrow']} {message}", default=default)


def prompt_number(
    message: str, default: int = 50, min_val: int = 0, max_val: int = 100
) -> int:
    """Numeric prompt with range validation and default hint."""
    return click.prompt(
        f"    {ICONS['arrow']} {message} [{default}] (Enter for default)",
        type=click.IntRange(min_val, max_val),
        default=default,
    )


# ── Panels & Visuals ─────────────────────────────────────────────────


def summary_panel(title: str, items: dict):
    """Display a summary panel — monochromatic with dim border."""
    table = Table.grid(padding=(0, 3))
    table.add_column(style=THEME.muted)
    table.add_column(style=THEME.primary)
    for k, v in items.items():
        table.add_row(k, str(v))
    console.print(
        Panel(
            table,
            title=f"  {ICONS['prism']} {title}  ",
            title_align="left",
            border_style=THEME.border,
            padding=(1, 2),
        )
    )


def resource_bar(label: str, value: int, max_val: int = 100, unit: str = "%"):
    """Visual resource bar — monochrome with semantic color only at threshold."""
    bar_width = 30
    filled = int((value / max_val) * bar_width)
    bar = ICONS["bar_full"] * filled + ICONS["bar_empty"] * (bar_width - filled)
    # Only use color when dangerously high
    if value >= 90:
        style = THEME.error
    elif value >= 75:
        style = THEME.warning
    else:
        style = THEME.body
    console.print(f"      {label:>12}  [{style}]{bar}[/]  {value}{unit}")


def progress_context(description: str = "Working..."):
    """Rich progress with minimal styling."""
    return Progress(
        SpinnerColumn("dots", style=THEME.muted),
        TextColumn(f"  {description}", style=THEME.body),
        TimeElapsedColumn(),
        console=console,
    )
