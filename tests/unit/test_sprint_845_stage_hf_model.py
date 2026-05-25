"""Sprint 845 — reusable HF model staging script.

Generalizes the one-off staging steps in sprint_625/626 into a
reusable scripts/stage_hf_model.py. Operators run it once per
model they want to serve; output populates the
FilesystemModelRegistry at PRSM_MODEL_REGISTRY_ROOT so
ChainExecutor's `registry.get(model_id)` resolves at dispatch
time.

Discovered the need during multi-host live test 2026-05-25:
sp842 added distilgpt2 to the canonical catalog + sp843
propagated overrides via the relay, but NYC operator returned
MODEL_NOT_FOUND on dispatch — the catalog is a CLIENT-side
manifest, the registry is an OPERATOR-side filesystem store.
Sprint 845 closes the gap.

Pin tests:
- _resolve_num_layers honors --layers CLI flag
- _resolve_num_layers reads canonical catalog
- _resolve_num_layers errors when neither CLI flag nor catalog entry
- _resolve_num_layers rejects < 1
- Catalog with malformed num_layers errors clearly
- Missing catalog path with no CLI flag errors clearly
- Argparse exposes the documented flags
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_catalog(tmp_path: Path, models: dict) -> Path:
    p = tmp_path / "model_catalog.json"
    p.write_text(json.dumps({
        "schema_version": "v1",
        "models": models,
    }))
    return p


def test_resolve_layers_cli_flag_wins(tmp_path: Path):
    from scripts.stage_hf_model import _resolve_num_layers
    catalog = _make_catalog(
        tmp_path, {"gpt2": {"num_layers": 12}},
    )
    # CLI flag (4) overrides catalog (12)
    assert _resolve_num_layers("gpt2", 4, catalog) == 4


def test_resolve_layers_reads_catalog(tmp_path: Path):
    from scripts.stage_hf_model import _resolve_num_layers
    catalog = _make_catalog(
        tmp_path,
        {"distilgpt2": {"num_layers": 6}},
    )
    assert _resolve_num_layers(
        "distilgpt2", None, catalog,
    ) == 6


def test_resolve_layers_errors_when_neither_set(tmp_path: Path):
    from scripts.stage_hf_model import _resolve_num_layers
    catalog = _make_catalog(tmp_path, {"gpt2": {"num_layers": 12}})
    with pytest.raises(ValueError) as exc:
        _resolve_num_layers("phi-2", None, catalog)
    assert "not in catalog" in str(exc.value)
    assert "phi-2" in str(exc.value)


def test_resolve_layers_rejects_zero():
    from scripts.stage_hf_model import _resolve_num_layers
    with pytest.raises(ValueError) as exc:
        _resolve_num_layers("anything", 0, Path("/dev/null"))
    assert ">=" in str(exc.value)


def test_resolve_layers_rejects_negative():
    from scripts.stage_hf_model import _resolve_num_layers
    with pytest.raises(ValueError):
        _resolve_num_layers("anything", -1, Path("/dev/null"))


def test_resolve_layers_catalog_bad_num_layers(tmp_path: Path):
    from scripts.stage_hf_model import _resolve_num_layers
    catalog = _make_catalog(
        tmp_path, {"bad": {"num_layers": "twelve"}},
    )
    with pytest.raises(ValueError) as exc:
        _resolve_num_layers("bad", None, catalog)
    assert "invalid num_layers" in str(exc.value)


def test_resolve_layers_missing_catalog(tmp_path: Path):
    from scripts.stage_hf_model import _resolve_num_layers
    with pytest.raises(ValueError) as exc:
        _resolve_num_layers(
            "anything", None, tmp_path / "nope.json",
        )
    assert "does not exist" in str(exc.value)


def test_resolve_layers_malformed_catalog(tmp_path: Path):
    """Catalog file exists but isn't valid JSON."""
    from scripts.stage_hf_model import _resolve_num_layers
    p = tmp_path / "bad.json"
    p.write_text("not-json")
    with pytest.raises(ValueError) as exc:
        _resolve_num_layers("any", None, p)
    assert "failed to read catalog" in str(exc.value)


def test_canonical_catalog_resolves_known_models():
    """The canonical project catalog should resolve gpt2 + distilgpt2
    cleanly via the resolver. Catches breaking changes to the catalog
    schema or to either sp559 (catalog hardening) or sp842 (distilgpt2
    entry)."""
    from scripts.stage_hf_model import (
        _resolve_num_layers, DEFAULT_CATALOG_PATH,
    )
    assert DEFAULT_CATALOG_PATH.exists()
    assert _resolve_num_layers(
        "gpt2", None, DEFAULT_CATALOG_PATH,
    ) == 12
    assert _resolve_num_layers(
        "distilgpt2", None, DEFAULT_CATALOG_PATH,
    ) == 6


def test_main_returns_1_on_bad_args(tmp_path: Path):
    """End-to-end smoke test of the script's error path. Pass a
    model_id that's not in the canonical catalog + no --layers."""
    from scripts.stage_hf_model import main
    code = main([
        "nonexistent-model-id",
        "--registry-root", str(tmp_path / "registry"),
        "--identity-file", str(tmp_path / "id.json"),
    ])
    assert code == 1


def test_main_help_works_without_prsm_env():
    """--help must work even when PRSM env is missing (so operators
    can debug from a cold shell). argparse exits with code 0 on
    --help; capture via SystemExit."""
    from scripts.stage_hf_model import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
