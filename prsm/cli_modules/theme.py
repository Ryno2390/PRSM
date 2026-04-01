"""PRSM CLI Theme — Black & white monochromatic design.

Art direction: geometric, isometric, sharp angles. No color palette.
Semantic colors (green/yellow/red) used ONLY for functional status indicators.
Everything else is white, dim white, bold white, or dim/italic for hierarchy.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PRSMTheme:
    """Monochromatic theme — geometric, minimal, sharp."""

    # Primary palette — black & white with grays
    primary: str = "bold white"
    secondary: str = "white"
    accent: str = "bold"

    # Text hierarchy
    heading: str = "bold white"
    subheading: str = "white"
    body: str = "default"
    muted: str = "dim"
    dim: str = "dim italic"

    # Semantic — color used ONLY for functional status
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    info: str = "dim"

    # Interactive elements
    prompt: str = "bold"
    selected: str = "bold white"
    unselected: str = "dim"

    # Borders & lines
    border: str = "dim white"
    rule: str = "dim"


THEME = PRSMTheme()

# Status indicators — minimal, geometric
ICONS = {
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
    "info": "→",
    "bullet": "▸",
    "arrow": "›",
    "check": "■",
    "uncheck": "□",
    "step": "◆",
    "network": "⬡",
    "compute": "⚡",
    "storage": "▪",
    "prism": "◇",
    "bar_full": "█",
    "bar_empty": "░",
    "line": "─",
    "corner": "┐",
}
