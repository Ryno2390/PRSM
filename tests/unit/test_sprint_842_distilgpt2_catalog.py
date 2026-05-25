"""Sprint 842 — distilgpt2 added to canonical model catalog.

Multi-host live re-attest 2026-05-25 (post-sp838 fleet deploy)
confirmed the relay carries real hw_profile end-to-end but
exposed the underlying capacity reality: the 2GB DO droplets
each advertise layer_capacity=1, so the allocator (which
excludes the requester as a settler-peer) sees total cap=2.
gpt2's 12 layers don't fit. Three remedies considered:

  1. Lambda A10 GPU peer ($1.29/hr + irreversible pubkey TX)
  2. Smaller model in catalog (distilgpt2, 6 layers; this sprint)
  3. PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE on droplets (sp686)

Option 2 ships now. distilgpt2 shares gpt2's architecture
(same hidden_dim, attention heads, vocab, head/intermediate
sizes) with half the transformer layers. Operators with 2GB
droplets can dispatch + receive signed receipts on distilgpt2
without provisioning GPU peers OR forcing layer-capacity
overrides. Larger fleets still use gpt2.

Pin tests:
- Catalog JSON parses
- distilgpt2 entry present
- distilgpt2 has 6 layers (half of gpt2's 12 — the key
  difference; sprint pivots on this number)
- distilgpt2 shares gpt2's arch dims (hidden, heads, vocab)
- Both entries valid against schema
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = REPO_ROOT / "config" / "parallax" / "model_catalog.json"


def _catalog():
    return json.loads(CATALOG_PATH.read_text())


def test_catalog_parses():
    d = _catalog()
    assert d["schema_version"] == "v1"
    assert "models" in d


def test_distilgpt2_entry_present():
    d = _catalog()
    assert "distilgpt2" in d["models"], (
        "sprint 842 added distilgpt2 to unblock multi-host "
        "inference on 2GB-droplet fleets; entry must persist"
    )


def test_distilgpt2_has_6_layers():
    """The whole point of the sprint — 6 layers vs gpt2's 12.
    Sprint pivots on this number. If it changes, the multi-host
    capacity math changes."""
    d = _catalog()["models"]["distilgpt2"]
    assert d["num_layers"] == 6


def test_distilgpt2_shares_gpt2_architecture():
    """distilgpt2 is gpt2 with half the layers — same hidden_dim,
    attention heads, head/intermediate sizes, vocab. The catalog
    entries should reflect that so downstream code (KV cache,
    embedding sizing, etc.) treats them uniformly."""
    models = _catalog()["models"]
    g, d = models["gpt2"], models["distilgpt2"]
    for field in (
        "hidden_dim", "num_attention_heads", "num_kv_heads",
        "vocab_size", "head_size", "intermediate_dim",
        "param_bytes_per_element", "mlx_param_bytes_per_element",
        "cache_bytes_per_element", "embedding_bytes_per_element",
    ):
        assert g[field] == d[field], (
            f"distilgpt2.{field} ({d[field]}) should match "
            f"gpt2.{field} ({g[field]}) — same architecture"
        )


def test_all_catalog_entries_have_required_fields():
    """Schema invariant: every entry must carry the fields
    sp559 (catalog schema hardening) defined as required."""
    required = {
        "model_name", "num_layers", "hidden_dim",
        "num_attention_heads", "num_kv_heads", "vocab_size",
        "head_size", "intermediate_dim",
        "param_bytes_per_element", "cache_bytes_per_element",
        "embedding_bytes_per_element",
    }
    for name, entry in _catalog()["models"].items():
        missing = required - set(entry.keys())
        assert not missing, (
            f"catalog entry {name!r} missing required fields: "
            f"{sorted(missing)}"
        )
