"""PRSM CLI Modules — UI toolkit, theming, and interactive components.

This package provides the shared visual identity and interactive elements
for all PRSM CLI commands. Monochromatic, geometric, minimal.
"""

from .theme import THEME, ICONS
from .ui import (
    console,
    show_banner,
    section,
    step_header,
    success,
    warning,
    error,
    info,
    muted,
    prompt_text,
    prompt_choice,
    prompt_confirm,
    prompt_number,
    summary_panel,
    resource_bar,
    progress_context,
)

__all__ = [
    "THEME",
    "ICONS",
    "console",
    "show_banner",
    "section",
    "step_header",
    "success",
    "warning",
    "error",
    "info",
    "muted",
    "prompt_text",
    "prompt_choice",
    "prompt_confirm",
    "prompt_number",
    "summary_panel",
    "resource_bar",
    "progress_context",
]
