"""Sprint 460 — F18 fix: sentence_transformers lazy-import pin.

Pre-fix: SentenceTransformerEmbedder imported sentence_transformers
at module top-level. Every PRSM operator had to install ~2GB of
ML deps (torch + transformers + huggingface_hub) just to boot a
node, even if they never served forge queries. Sprint 458's
bootstrap1 deploy halted here — 2GB-RAM droplets risked OOM
during the install.

Post-fix: the import happens inside `_ensure_loaded()`, which only
runs on the first `encode()` call. Operators who never serve
queries never pay the ML-stack cost.

These pins defend the invariant via source-level + behavior
checks — a future refactor that re-adds the top-level import
would break operator UX for the ~80% of operators who don't use
the orchestrator.
"""
from __future__ import annotations

from pathlib import Path

EMBEDDER_PY = (
    Path(__file__).resolve().parents[2]
    / "prsm" / "compute" / "query_orchestrator"
    / "sentence_transformer_embedder.py"
)


def _read_source():
    return EMBEDDER_PY.read_text()


def test_module_top_level_does_not_import_sentence_transformers():
    """Source-level invariant: the top-level imports of the
    module MUST NOT include `from sentence_transformers
    import` outside a `TYPE_CHECKING` block. Pre-fix this
    was an unconditional top-level import.

    Any unconditional reference to sentence_transformers
    outside the `_ensure_loaded()` body or a TYPE_CHECKING
    guard re-introduces the F18 regression."""
    text = _read_source()
    # Find each line that imports sentence_transformers
    lines = text.splitlines()
    sentencet_imports = [
        (i, line) for i, line in enumerate(lines, 1)
        if "sentence_transformers" in line and "import" in line
    ]
    # At least one import must exist (lazy inside _ensure_loaded)
    assert sentencet_imports, (
        "sentence_transformers import disappeared entirely — "
        "the class needs it lazily inside _ensure_loaded()"
    )
    # Check that every actual import is INSIDE the lazy-load
    # function body or under TYPE_CHECKING. The lazy body is
    # indented; top-level imports start at column 0.
    for line_num, line in sentencet_imports:
        stripped = line.strip()
        if stripped.startswith("#") or "TYPE_CHECKING" in stripped:
            continue
        # Real import statement. Must be indented (inside function)
        assert line.startswith("    "), (
            f"sentence_transformers import at line {line_num} is "
            f"at column 0 (top-level) — must be moved inside "
            f"_ensure_loaded() body. Pre-fix this was the F18 bug."
        )


def test_class_instantiable_without_loading_model():
    """The class must be importable + instantiable without
    triggering sentence_transformers load. Constructor does
    NO ML work; only encode() / _ensure_loaded() does."""
    from prsm.compute.query_orchestrator.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )
    embedder = SentenceTransformerEmbedder()
    # Model attribute should be None until encode() is called
    assert embedder._model is None


def test_ensure_loaded_raises_clear_error_when_dep_missing():
    """When sentence_transformers is absent, _ensure_loaded()
    must raise ImportError with a clear pip-install
    instruction. Generic ModuleNotFoundError would leave
    operators guessing at which extra to install.

    Simulates the missing-dep case by patching the import
    statement inside _ensure_loaded directly."""
    import pytest
    from unittest.mock import patch
    from prsm.compute.query_orchestrator.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )
    embedder = SentenceTransformerEmbedder()

    # Patch the import statement inside _ensure_loaded by
    # making it raise ImportError on the from-import line.
    import builtins
    orig_import = builtins.__import__

    def _no_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith(
            "sentence_transformers.",
        ):
            raise ImportError(
                f"No module named '{name}' (simulated)",
            )
        return orig_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__",
                       side_effect=_no_sentence_transformers):
        with pytest.raises(ImportError) as exc_info:
            embedder._ensure_loaded()
        msg = str(exc_info.value).lower()
        # Must include actionable install command + reference
        # to the dep name
        assert "pip install" in msg
        assert "sentence" in msg


def test_orchestrator_package_imports_without_loading_ml_stack():
    """The broader invariant: importing
    `prsm.compute.query_orchestrator` package must NOT
    trigger sentence_transformers import. Pre-fix, the
    package's __init__ re-exported SentenceTransformerEmbedder
    from a module that did top-level `import
    sentence_transformers`, so the entire package was
    un-importable on operators without the ML stack."""
    # Just importing the package is the test. If this raises
    # ImportError("sentence_transformers"), F18 is regressed.
    import prsm.compute.query_orchestrator as qo  # noqa: F401
    assert hasattr(qo, "SentenceTransformerEmbedder")
