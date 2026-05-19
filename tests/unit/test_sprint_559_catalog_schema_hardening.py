"""Sprint 559 — catalog schema version + required-field validation.

Sprint 558's catalog loader did ``ModelInfo(**kw)`` per entry —
ModelInfo's ``__init__`` accepts ANY kwargs without validation, so
operator typos (``model_namee`` instead of ``model_name``, missing
``num_layers``, etc.) silently constructed a degenerate ModelInfo
that would later fail at scheduling time with cryptic errors.

Sprint 559 hardens the schema:

  - Top-level becomes ``{schema_version: "v1", models: {...}}``.
    Old sprint-558 format (top-level dict of models) is now
    rejected with a structured migration hint.
  - Required ModelInfo fields per entry: ``model_name``,
    ``num_layers``, ``hidden_dim``, ``num_attention_heads``,
    ``num_kv_heads``, ``vocab_size``, ``head_size``,
    ``intermediate_dim``. Each missing field logs a warning naming
    the model_id + field.
  - Unknown schema_version → log + None.

No new functionality, just stronger contracts at the front door.
Operator typos surface at daemon boot, not at first inference
request.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest


# ── happy path with new schema ────────────────────────────


def _new_schema_catalog():
    """Reference shape — every required field present."""
    return {
        "schema_version": "v1",
        "models": {
            "test-model": {
                "model_name": "test-model",
                "mlx_model_name": "test-model",
                "head_size": 64,
                "hidden_dim": 128,
                "intermediate_dim": 256,
                "num_attention_heads": 4,
                "num_kv_heads": 4,
                "vocab_size": 1000,
                "num_layers": 4,
            }
        },
    }


def _set_env(monkeypatch, tmp_path, catalog):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(catalog))
    monkeypatch.setenv(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", str(p),
    )
    monkeypatch.setenv("PRSM_PARALLAX_TRUST_STACK_KIND", "mock")
    monkeypatch.setenv(
        "PRSM_PARALLAX_GPU_POOL_KIND", "static-empty",
    )


def _stub_node():
    from prsm.node.identity import generate_node_identity
    n = MagicMock()
    n.identity = generate_node_identity("test-settler")
    return n


def test_new_schema_v1_constructs_executor(monkeypatch, tmp_path):
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )

    _set_env(monkeypatch, tmp_path, _new_schema_catalog())
    result = build_parallax_executor_or_none(_stub_node())
    assert isinstance(result, ParallaxScheduledExecutor)
    assert "test-model" in result.supported_models()


def test_multi_model_catalog_loads_all_entries(
    monkeypatch, tmp_path,
):
    """Operator can register multiple model_ids in one catalog file."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    catalog = _new_schema_catalog()
    catalog["models"]["model-b"] = {
        **catalog["models"]["test-model"],
        "model_name": "model-b",
        "num_layers": 8,
    }
    _set_env(monkeypatch, tmp_path, catalog)
    result = build_parallax_executor_or_none(_stub_node())
    assert result is not None
    supported = result.supported_models()
    assert "test-model" in supported
    assert "model-b" in supported


# ── rejection paths ───────────────────────────────────────


def test_rejects_legacy_sprint_558_top_level_dict_shape(
    monkeypatch, tmp_path, caplog,
):
    """The old format ``{"model-id": {...}}`` (sprint 558's shape)
    is no longer accepted. Operator gets a structured migration
    hint pointing at the new schema."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    legacy = {
        "test-model": _new_schema_catalog()["models"]["test-model"],
    }
    _set_env(monkeypatch, tmp_path, legacy)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "schema_version" in log_text
    # Migration hint mentions the new shape.
    assert "v1" in log_text or "models" in log_text


def test_rejects_missing_schema_version(
    monkeypatch, tmp_path, caplog,
):
    """Catalog has `models` but no `schema_version` → named warning."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    bad = {"models": _new_schema_catalog()["models"]}
    _set_env(monkeypatch, tmp_path, bad)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "schema_version" in log_text


def test_rejects_unknown_schema_version(
    monkeypatch, tmp_path, caplog,
):
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    bad = {
        "schema_version": "v99-from-the-future",
        "models": _new_schema_catalog()["models"],
    }
    _set_env(monkeypatch, tmp_path, bad)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "v99-from-the-future" in log_text


def test_rejects_missing_models_section(
    monkeypatch, tmp_path, caplog,
):
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    bad = {"schema_version": "v1"}  # no `models` key
    _set_env(monkeypatch, tmp_path, bad)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert "models" in log_text


@pytest.mark.parametrize(
    "missing_field",
    [
        "model_name", "num_layers", "hidden_dim",
        "num_attention_heads", "num_kv_heads", "vocab_size",
        "head_size", "intermediate_dim",
    ],
)
def test_rejects_entry_missing_required_field(
    monkeypatch, tmp_path, caplog, missing_field,
):
    """Each required ModelInfo field gets its own rejection path.
    Warning names BOTH the model_id and the missing field."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    catalog = _new_schema_catalog()
    catalog["models"]["test-model"].pop(missing_field)
    _set_env(monkeypatch, tmp_path, catalog)

    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records)
    assert missing_field in log_text, (
        f"warning must name missing field {missing_field!r}; "
        f"got: {log_text!r}"
    )
    assert "test-model" in log_text


def test_rejects_models_section_not_dict(
    monkeypatch, tmp_path, caplog,
):
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )

    bad = {"schema_version": "v1", "models": ["not", "a", "dict"]}
    _set_env(monkeypatch, tmp_path, bad)
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert result is None
    log_text = " ".join(r.message for r in caplog.records).lower()
    assert "models" in log_text


def test_empty_models_section_logs_warning_but_constructs(
    monkeypatch, tmp_path, caplog,
):
    """``{schema_version:v1, models:{}}`` is well-formed but useless.
    Log a warning so operators notice; still return a constructed
    executor (consistent with sprint 558 behavior — an empty
    catalog isn't fatal, it just means no models advertised)."""
    from prsm.node.inference_wiring import (
        build_parallax_executor_or_none,
    )
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )

    _set_env(monkeypatch, tmp_path, {
        "schema_version": "v1", "models": {},
    })
    with caplog.at_level(logging.WARNING):
        result = build_parallax_executor_or_none(_stub_node())
    assert isinstance(result, ParallaxScheduledExecutor)
    assert result.supported_models() == []
    log_text = " ".join(r.message for r in caplog.records).lower()
    assert "empty" in log_text or "no model" in log_text
