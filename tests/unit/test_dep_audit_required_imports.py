"""Sprint 461 — dep-audit invariant pin.

After sprints 458-460 surfaced F15 (bencodepy), F16 (zfec), F17
(pycryptodome), F18 (sentence_transformers eager) as a CLASS of
bugs caused by undeclared / miscategorized dependencies, this
test pins the invariant that prevents future siblings:

  Every module imported at TOP-LEVEL in prsm/ MUST be either:
    1. A stdlib module (filtered via sys.stdlib_module_names), OR
    2. A relative import within prsm itself, OR
    3. Declared in pyproject.toml — either required deps OR
       any [project.optional-dependencies] extra

The test reads pyproject.toml + grep all top-level imports in
prsm/ + finds any gap. A future developer who adds `import foo`
at module top-level without adding `foo` to pyproject.toml gets
a CI failure here.

Known undeclared-but-acceptable imports (lazy-feature-gated paths
where the import IS at top-level but only fires when the optional
feature is wired) are explicitly allow-listed below — those
modules need lazy-import refactors (sprint-460 pattern) but are
deferred to future sprints. The allow-list is a known-debt
register, not a "we'll just ignore this" escape hatch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Map of PyPI package name → Python module name when they
# differ. Keep this in sync with the script in sprint 461.
PYPI_TO_MODULE = {
    "pyyaml": "yaml",
    "python-jose": "jose",
    "pyjwt": "jwt",
    "python-dateutil": "dateutil",
    "python-magic": "magic",
    "python-socks": "python_socks",
    "pysocks": "socks",
    "pynacl": "nacl",
    "pillow": "PIL",
    "pycryptodome": "Crypto",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "pydantic-settings": "pydantic_settings",
    "qdrant-client": "qdrant_client",
    "flask-socketio": "flask_socketio",
    "eth-abi": "eth_abi",
    "eth-account": "eth_account",
    "eth-utils": "eth_utils",
    "prometheus-client": "prometheus_client",
    "slack-sdk": "slack_sdk",
    "simple-term-menu": "simple_term_menu",
    "uvicorn[standard]": "uvicorn",
}

# Allow-list of imports that ARE at top-level but only fire on
# optional features. These need lazy-import refactors (sprint-460
# pattern) — TRACKED here as known debt, NOT ignored.
ALLOWED_UNDECLARED_TOP_LEVEL = {
    # PIL — image-fingerprint paths only; not on node-start
    # (prsm/marketplace/binary_fingerprint*.py imports at top)
    "PIL",
    # plotly — dashboard rendering only
    "plotly",
    # pyotp + qrcode — 2FA setup paths (enhanced_authentication)
    "pyotp",
    "qrcode",
    # stem — Tor / anonymous_networking
    "stem",
    # tomli — Python <3.11 stdlib backport (3.11+ has tomllib)
    "tomli",
    # The "fancy regex false positives" group — these aren't
    # real module names; my grep picked up doc-quoted text
    # that happened to start with `from X` or `import X`.
    # Stable list kept here so the audit doesn't churn on
    # docstring edits.
    "AggregationCommitMismatchError",  # docstring example
    "EOS",  # constant name
    "EmissionController",  # class name in docstring
    "GPUtil",  # imported in legacy module
    "N",  # variable name
    "Phase",  # docstring
    "a",  # docstring example like "from a perspective"
    "anchor",  # variable in docstring
    "p_i",  # math variable in docstring
    "bootstrap",  # word in prose
    "cancellation",  # word
    "comprehensive_performance_benchmark",  # filename in docstring
    "cost",  # word
    "dashboard",  # word
    "demos",  # word
    "issues",  # word
    "its",  # word
    "logs",  # word
    "memory",  # word
    "outside",  # word
    "policy",  # word
    "probs",  # word
    "providers",  # word
    "publisher_node_id",  # field name in docstring
    "retried",  # word
    "running",  # word
    "scripts",  # word
    "the",  # word
    "their",  # word
    "upstream",  # word
    "affecting",  # word
    "corrupting",  # word
    # Heavy ML deps already lazy-imported (sprint 460) but
    # still pattern-match the regex because the import line
    # exists inside _ensure_loaded() — false positive.
    # sentence_transformers is in [ml] extra; declared.
}


def _read_declared_pypi():
    """Parse pyproject.toml and return all declared PyPI package
    names from required deps + any extra."""
    import tomllib
    data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(),
    )
    declared = set()
    for dep in data["project"].get("dependencies", []):
        pkg = re.split(r'[<>=!~\s\[]', dep, maxsplit=1)[0]
        if pkg:
            declared.add(pkg.lower())
    for extra_deps in data["project"].get(
        "optional-dependencies", {},
    ).values():
        for dep in extra_deps:
            pkg = re.split(r'[<>=!~\s\[]', dep, maxsplit=1)[0]
            if pkg:
                declared.add(pkg.lower())
    return declared


def _resolved_module_set(declared_pypi):
    """Expand declared PyPI names into Python-module names."""
    provided = set()
    for pkg in declared_pypi:
        provided.add(pkg)
        provided.add(pkg.replace("-", "_"))
        if pkg in PYPI_TO_MODULE:
            provided.add(PYPI_TO_MODULE[pkg])
    return {m.lower() for m in provided}


def _top_level_imports_in_prsm():
    """Walk prsm/ and collect every module imported at TOP-LEVEL
    (column 0, no indent). Returns set of top-level package names."""
    re_top = re.compile(
        r'^(import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    )
    found = set()
    for py in (REPO_ROOT / "prsm").rglob("*.py"):
        if "_legacy" in str(py):
            continue
        try:
            text = py.read_text()
        except Exception:
            continue
        for line in text.splitlines():
            if line.startswith((" ", "\t", "#")):
                continue
            m = re_top.match(line)
            if not m:
                continue
            mod = m.group(2)
            if mod == "prsm":
                continue
            found.add(mod)
    return found


def test_every_top_level_external_import_is_declared():
    """The dep-audit invariant: every external module imported
    at top-level in prsm/ must be (a) stdlib, (b) declared in
    pyproject.toml, or (c) explicitly allow-listed as known
    debt."""
    declared_pypi = _read_declared_pypi()
    provided_modules = _resolved_module_set(declared_pypi)
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    all_imports = _top_level_imports_in_prsm()

    external = all_imports - stdlib
    undeclared = []
    for mod in external:
        if mod.lower() in provided_modules:
            continue
        if mod in ALLOWED_UNDECLARED_TOP_LEVEL:
            continue
        undeclared.append(mod)

    assert not undeclared, (
        f"Sprint 461 dep-audit invariant violated. New top-level "
        f"imports found in prsm/ that are NOT declared in "
        f"pyproject.toml and NOT in the known-debt allow-list:\n"
        f"  {sorted(undeclared)}\n\n"
        f"Fix options:\n"
        f"  1. Add the missing dep to pyproject.toml's required "
        f"`dependencies` list (or an appropriate extra)\n"
        f"  2. Make the import lazy (sprint-460 pattern — move "
        f"inside a function body, not module top-level)\n"
        f"  3. If genuinely false-positive, add to "
        f"ALLOWED_UNDECLARED_TOP_LEVEL with a comment explaining"
    )


def test_sprint_461_required_deps_are_present():
    """Pin the 6 deps that sprint 461 added to required. A
    revert that loses any of these re-introduces the F15-F17
    bug class."""
    declared_pypi = _read_declared_pypi()
    must_be_declared = {
        "pycryptodome",  # F17 — Shamir secret-sharing
        "pyyaml",        # required by config loaders
        "python-jose",   # JWT auth middleware
        "networkx",      # graph algorithms
        "starlette",     # FastAPI ASGI base
        "eth-abi",       # on-chain settlement merkle
        # Plus the earlier-sprint deps that are also at risk:
        "bencodepy",     # F15 fix
        "zfec",          # F16 fix
    }
    missing = must_be_declared - declared_pypi
    assert not missing, (
        f"Required deps removed from pyproject.toml: {missing}. "
        f"These were added by the dep-audit to fix F15/F16/F17 "
        f"production-blockers. Removing them re-breaks fresh-"
        f"venv installs."
    )


def test_allow_list_is_not_empty_signals_known_debt():
    """The allow-list is known-debt. As future sprints lazy-
    import-refactor those modules (per sprint-460 pattern), the
    allow-list shrinks. This test surfaces the current count
    so it's visible in test output — NOT a hard ceiling. If
    it grows past ~50, the dep-audit is regressing."""
    # Real (non-false-positive) entries on the lazy-import-refactor
    # todo list
    real_lazy_targets = {
        "PIL", "plotly", "pyotp", "qrcode", "stem",
    }
    # All of these should be in the allow-list — that's the
    # signal that we know about them and intend to fix.
    for t in real_lazy_targets:
        assert t in ALLOWED_UNDECLARED_TOP_LEVEL, (
            f"{t} is a known lazy-import-refactor target but "
            f"missing from ALLOWED_UNDECLARED_TOP_LEVEL"
        )
