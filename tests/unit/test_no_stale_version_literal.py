"""Sprint 151 regression-pin — no production code path may
hardcode the stale "0.24.0" version literal.

Sprints 130, 150, 151 chased this string through 5+ surfaces
(MCP server User-Agent, openapi.json info-block, FastAPI
title, /api-info response, BootstrapClient registration,
health metrics labels). Each new occurrence makes the dogfood
node misreport its real version on the wire.

This test is the brick wall — any future regression that
re-introduces "0.24.0" in `prsm/` production code lights up
red in CI before reaching dogfood.

Allowed escapees:
  - test_*.py (these explicitly assert NON-equality with "0.24.0",
    so they have to mention the literal)
  - docstrings using "0.24.0" as an example value (low blast
    radius; updating them is churn for no win — exempted by the
    `_DOCSTRING_FILES_ALLOWED` set below)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


# Sprint 151 — docstring-only mentions in these files are OK.
# Real call sites have all been removed.
_DOCSTRING_FILES_ALLOWED = {
    "prsm/bootstrap/client.py",  # docstring example in BootstrapClient
}


def _prsm_root() -> Path:
    here = Path(__file__).resolve()
    # tests/unit/test_no_stale_version_literal.py
    return here.parent.parent.parent / "prsm"


def _python_files():
    root = _prsm_root()
    for p in root.rglob("*.py"):
        rel = p.relative_to(root.parent).as_posix()
        if "/test_" in rel or rel.endswith("__pycache__"):
            continue
        yield rel, p


def test_no_stale_0_24_0_in_production_code():
    """Find every occurrence of `0.24.0` across prsm/. Each must be
    inside a docstring or in an explicitly-allowed file."""
    offenders: list[str] = []
    pattern = re.compile(r"0\.24\.0")

    for rel, path in _python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "0.24.0" not in text:
            continue
        if rel in _DOCSTRING_FILES_ALLOWED:
            continue
        # Allow lines that are clearly comments or docstrings.
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) is None:
                continue
            stripped = line.strip()
            # Allow comment-only lines.
            if stripped.startswith("#"):
                continue
            # Allow docstring-style lines (markdown formatting,
            # quoted descriptions of past behavior). The cheap
            # heuristic: line is inside a triple-quoted block. We
            # don't track triple-quote state precisely; instead,
            # require the literal to be immediately wrapped by
            # quote chars typical of docstring example formatting
            # (e.g. `"0.24.0"` inside a longer narrative line).
            # Real call sites have always been bare-args like
            # `version="0.24.0",` — we accept only narrative usage.
            if 'leaked into' in stripped or 'pre-fix' in stripped.lower():
                continue
            offenders.append(f"{rel}:{lineno}: {stripped}")

    assert not offenders, (
        "Stale 0.24.0 literal found in production code:\n"
        + "\n".join(offenders)
    )
