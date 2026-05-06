"""
PRSM-PROV-1 Item 6 T6.2 — ThresholdResolver tests.

Covers:
  - YAML load + parse round-trip from the canonical project path
  - Resolution by (kind, model_id, content_type_hint) priority order
  - Floor enforcement (multiplier cannot push below floor)
  - duplicate >= derivative invariant maintained after multiplier
  - All malformed-YAML failure modes raise ThresholdResolverError
  - Real YAML structure validates against project's
    `prsm/data/dedup_thresholds.yaml`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prsm.data.dedup import (
    DEFAULT_THRESHOLDS_PATH,
    EffectiveThresholds,
    ThresholdResolver,
    ThresholdResolverError,
)


# ---- load + smoke ---------------------------------------------------


def test_default_yaml_path_exists():
    assert DEFAULT_THRESHOLDS_PATH.exists(), (
        f"missing canonical dedup_thresholds.yaml at "
        f"{DEFAULT_THRESHOLDS_PATH}"
    )


def test_load_default_yaml_succeeds():
    resolver = ThresholdResolver.from_default_path()
    # Sanity: the canonical `text-vector` default must resolve.
    thr = resolver.resolve("text-vector")
    assert isinstance(thr, EffectiveThresholds)
    assert 0.0 < thr.derivative <= thr.duplicate <= 1.0


def test_load_via_yaml_path(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "defaults:\n"
        "  text-vector:\n"
        "    derivative: 0.92\n"
        "    duplicate: 0.99\n",
        encoding="utf-8",
    )
    resolver = ThresholdResolver.from_yaml_path(p)
    thr = resolver.resolve("text-vector")
    assert thr.derivative == pytest.approx(0.92)
    assert thr.duplicate == pytest.approx(0.99)
    assert thr.resolved_key == "text-vector"
    assert thr.hint_applied is None


# ---- resolution priority --------------------------------------------


def test_model_specific_key_beats_bare_kind():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
            "text-vector/model-A": {"derivative": 0.85, "duplicate": 0.97},
        },
    }
    resolver = ThresholdResolver(payload)

    bare = resolver.resolve("text-vector")
    assert bare.derivative == pytest.approx(0.92)
    assert bare.resolved_key == "text-vector"

    specific = resolver.resolve("text-vector", model_id="model-A")
    assert specific.derivative == pytest.approx(0.85)
    assert specific.resolved_key == "text-vector/model-A"


def test_unknown_model_id_falls_back_to_bare_kind():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve("text-vector", model_id="unconfigured-model")
    assert thr.derivative == pytest.approx(0.92)
    assert thr.resolved_key == "text-vector"


def test_unknown_kind_raises():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
        },
    }
    resolver = ThresholdResolver(payload)
    with pytest.raises(ThresholdResolverError, match="no threshold entry"):
        resolver.resolve("audio-chromaprint")


# ---- hint multipliers + floors -------------------------------------


def test_hint_multiplier_tightens_threshold():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.90, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "scientific_abstract": {
                "text-vector": {"derivative": 1.05, "duplicate": 1.00},
            },
        },
        "floors": {
            "text-vector": {"derivative": 0.80, "duplicate": 0.95},
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve(
        "text-vector", content_type_hint="scientific_abstract",
    )
    assert thr.derivative == pytest.approx(0.90 * 1.05)
    assert thr.duplicate == pytest.approx(0.99)
    assert thr.hint_applied == "scientific_abstract"


def test_hint_multiplier_clamped_above_one():
    """Multiplier that would push above 1.0 must be clamped."""
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.95, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "tighten": {
                "text-vector": {"derivative": 1.20, "duplicate": 1.10},
            },
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve("text-vector", content_type_hint="tighten")
    # 0.95 * 1.20 = 1.14 → clamped to 1.0
    assert thr.derivative == pytest.approx(1.0)
    assert thr.duplicate == pytest.approx(1.0)


def test_floor_enforced_against_loosen_multiplier():
    """A loosen-multiplier that drops below floor MUST be clamped."""
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.90, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "loosen": {
                "text-vector": {"derivative": 0.50, "duplicate": 1.00},
            },
        },
        "floors": {
            "text-vector": {"derivative": 0.85, "duplicate": 0.97},
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve("text-vector", content_type_hint="loosen")
    # 0.90 * 0.50 = 0.45, but floor is 0.85
    assert thr.derivative == pytest.approx(0.85)
    assert thr.duplicate == pytest.approx(0.99)


def test_no_hint_no_multiplier_applied():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "code": {
                "text-vector": {"derivative": 1.05, "duplicate": 1.00},
            },
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve("text-vector")  # no hint
    assert thr.derivative == pytest.approx(0.92)
    assert thr.hint_applied is None


def test_unknown_hint_silently_ignored():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "code": {
                "text-vector": {"derivative": 1.05, "duplicate": 1.00},
            },
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve(
        "text-vector", content_type_hint="not-a-real-hint",
    )
    # Hint not configured → no multiplier, no error.
    assert thr.derivative == pytest.approx(0.92)
    assert thr.hint_applied is None


def test_hint_with_unknown_kind_silently_ignored():
    """A hint configured under one kind doesn't accidentally apply to
    a different kind."""
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
            "image-phash": {"derivative": 0.81, "duplicate": 0.94},
        },
        "content_type_multipliers": {
            "scientific_abstract": {
                "text-vector": {"derivative": 1.05, "duplicate": 1.00},
                # No image-phash entry here.
            },
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve(
        "image-phash", content_type_hint="scientific_abstract",
    )
    # No image-phash entry under scientific_abstract → no multiplier.
    assert thr.derivative == pytest.approx(0.81)
    assert thr.hint_applied is None


# ---- invariants ----------------------------------------------------


def test_duplicate_geq_derivative_after_multiplier():
    """If a multiplier inverts the relationship, push duplicate up to
    derivative — duplicate is the stricter tier and must remain so."""
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.95, "duplicate": 0.99},
        },
        "content_type_multipliers": {
            "weird": {
                # Tighten derivative but loosen duplicate — should
                # never invert the order.
                "text-vector": {"derivative": 1.04, "duplicate": 0.95},
            },
        },
    }
    resolver = ThresholdResolver(payload)
    thr = resolver.resolve("text-vector", content_type_hint="weird")
    # derivative: 0.95 * 1.04 = 0.988
    # duplicate would be 0.99 * 0.95 = 0.9405, < derivative.
    # Must be pushed up to derivative.
    assert thr.duplicate >= thr.derivative


def test_effective_thresholds_dataclass_validates():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        EffectiveThresholds(
            derivative=1.5, duplicate=1.6,
            resolved_key="x", hint_applied=None,
        )
    with pytest.raises(ValueError, match=r"duplicate.*>= derivative"):
        EffectiveThresholds(
            derivative=0.9, duplicate=0.5,
            resolved_key="x", hint_applied=None,
        )


# ---- malformed YAML -------------------------------------------------


def test_missing_defaults_section_raises():
    with pytest.raises(ThresholdResolverError, match="defaults"):
        ThresholdResolver({})


def test_defaults_not_a_mapping_raises():
    with pytest.raises(ThresholdResolverError, match="must be a mapping"):
        ThresholdResolver({"defaults": "not a dict"})


def test_default_entry_not_a_mapping_raises():
    payload = {"defaults": {"text-vector": "not a dict"}}
    with pytest.raises(ThresholdResolverError, match="must be a mapping"):
        ThresholdResolver(payload)


def test_default_entry_missing_tier_raises():
    payload = {
        "defaults": {
            "text-vector": {"derivative": 0.92},  # missing duplicate
        },
    }
    with pytest.raises(ThresholdResolverError, match="duplicate"):
        ThresholdResolver(payload)


def test_default_entry_non_numeric_tier_raises():
    payload = {
        "defaults": {
            "text-vector": {"derivative": "high", "duplicate": 0.99},
        },
    }
    with pytest.raises(ThresholdResolverError, match="numeric"):
        ThresholdResolver(payload)


def test_empty_kind_raises_at_resolve_time():
    resolver = ThresholdResolver({
        "defaults": {
            "text-vector": {"derivative": 0.92, "duplicate": 0.99},
        },
    })
    with pytest.raises(ThresholdResolverError, match="non-empty"):
        resolver.resolve("")


def test_yaml_file_unreadable_raises(tmp_path):
    nonexistent = tmp_path / "nope.yaml"
    with pytest.raises(ThresholdResolverError, match="cannot read"):
        ThresholdResolver.from_yaml_path(nonexistent)


def test_yaml_file_with_top_level_list_raises(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ThresholdResolverError, match="must be a mapping"):
        ThresholdResolver.from_yaml_path(p)


def test_yaml_syntactically_invalid_raises(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(": not valid yaml :\n", encoding="utf-8")
    with pytest.raises(ThresholdResolverError, match="failed to parse"):
        ThresholdResolver.from_yaml_path(p)


# ---- canonical YAML coverage ---------------------------------------


def test_canonical_yaml_resolves_known_kinds():
    """Smoke against the project's canonical thresholds file —
    every kind we ship resolves cleanly + non-trivially."""
    resolver = ThresholdResolver.from_default_path()
    for kind in (
        "text-vector",
        "image-phash",
        "audio-chromaprint",
        "video-multihash",
    ):
        thr = resolver.resolve(kind)
        assert 0.0 < thr.derivative <= thr.duplicate <= 1.0, (
            f"kind {kind!r} has invalid thresholds: {thr}"
        )


def test_canonical_yaml_resolves_known_models():
    resolver = ThresholdResolver.from_default_path()
    for kind, model_id in (
        ("text-vector", "openai/text-embedding-ada-002"),
        ("text-vector", "openai/text-embedding-3-small"),
        ("text-vector", "sentence-transformers/all-MiniLM-L6-v2"),
    ):
        thr = resolver.resolve(kind, model_id=model_id)
        assert thr.resolved_key == f"{kind}/{model_id}"
        assert 0.0 < thr.derivative <= thr.duplicate <= 1.0


def test_canonical_yaml_scientific_abstract_hint_tightens():
    resolver = ThresholdResolver.from_default_path()
    base = resolver.resolve(
        "text-vector",
        model_id="openai/text-embedding-ada-002",
    )
    hinted = resolver.resolve(
        "text-vector",
        model_id="openai/text-embedding-ada-002",
        content_type_hint="scientific_abstract",
    )
    assert hinted.derivative > base.derivative, (
        "scientific_abstract hint must tighten the derivative threshold"
    )
    assert hinted.hint_applied == "scientific_abstract"
