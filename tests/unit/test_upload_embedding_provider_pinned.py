"""Sprint 431 — F9 fix: upload-side embedding-provider parity.

The query orchestrator's embedder is pinned to
`sentence-transformers/all-MiniLM-L6-v2` (384-dim). Before this
sprint, the upload-side `_embedding_fn` was `RealEmbeddingAPI.
generate_embedding` with no preferred_provider, so it fell through
to whatever was first in the fallback chain — OpenAI ada-002
(1536-dim) when `OPENAI_API_KEY` was set.

Result: stored shard embeddings were 1536-dim, queries were 384-dim,
and `POST /compute/forge` failed with "shapes (384,) and (1536,) not
aligned" — the canonical query workflow broken whenever OpenAI was
configured.

Fix (node.py:1726ff): bind `preferred_provider="sentence_transformers"`
via functools.partial. Operator override via
`PRSM_UPLOAD_EMBEDDING_PROVIDER` env var (currently unsupported
because the orchestrator only has a sentence_transformers embedder).

These pins fire if the wiring regresses or if the env-override
behavior is silently changed.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_PY = REPO_ROOT / "prsm" / "node" / "node.py"


def _read_node():
    return NODE_PY.read_text()


def test_embedding_fn_pins_preferred_provider():
    """The upload-side `_embedding_fn` MUST bind a
    `preferred_provider` so it doesn't drift to OpenAI when
    OPENAI_API_KEY is set. functools.partial is the canonical
    binding mechanism here."""
    text = _read_node()
    assert "preferred_provider=_pref_provider" in text, (
        "node.py _embedding_fn must bind a preferred_provider "
        "(F9 dim-parity invariant)"
    )
    assert "functools.partial(" in text, (
        "node.py _embedding_fn must wrap generate_embedding "
        "with functools.partial to bind the preferred_provider "
        "argument"
    )


def test_default_provider_is_sentence_transformers():
    """The default preferred_provider must be
    `sentence_transformers` so the upload side matches the
    query-side embedder model out of the box. Default-install
    operators should NOT need to set an env var to get a
    working forge pipeline."""
    text = _read_node()
    # Look for the env-var default in the wiring block.
    pat = re.compile(
        r'PRSM_UPLOAD_EMBEDDING_PROVIDER\s*"\s*,\s*\n?\s*"sentence_transformers"',
        re.MULTILINE,
    )
    assert pat.search(text), (
        "default preferred_provider must be "
        "'sentence_transformers' so canonical install works "
        "without env-var configuration"
    )


def test_warning_when_openai_key_set_but_provider_overridden_to_st():
    """Operators who set OPENAI_API_KEY but didn't override
    the upload provider should see a log warning explaining
    the parity trade-off. Silent demotion would be worse than
    a warning."""
    text = _read_node()
    assert (
        "OPENAI_API_KEY" in text
        and "PRSM_UPLOAD_EMBEDDING_PROVIDER" in text
        and "parity" in text.lower()
    ), (
        "node.py must log when OPENAI_API_KEY is set + the "
        "upload-side falls back to sentence_transformers, so "
        "the trade-off is visible to operators"
    )


def test_dimension_parity_invariant_documented():
    """The F9 fix's rationale must remain in the comments so
    future editors don't silently re-enable OpenAI uploads
    and re-break the forge pipeline."""
    text = _read_node()
    assert "F9" in text or "384" in text and "1536" in text, (
        "the dimension-parity rationale (384 vs 1536 / F9) "
        "must stay attached to the wiring code so future "
        "editors see the constraint"
    )
