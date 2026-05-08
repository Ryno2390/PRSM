"""
PRSM-PROV-1 Item 6 T6.2 — ThresholdResolver.

Loads ``prsm/data/dedup_thresholds.yaml`` and resolves the effective
``(derivative, duplicate)`` thresholds for a given fingerprint kind,
with optional content-type-hint multipliers.

Usage::

    resolver = ThresholdResolver.from_default_path()
    thr = resolver.resolve(
        fingerprint_kind="text-vector",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        content_type_hint="scientific_abstract",
    )
    if similarity >= thr.derivative:
        ...

Key resolution: ``fingerprint_kind`` plus optional ``model_id`` form
the lookup key as ``"<kind>"`` or ``"<kind>/<model_id>"``. The most
specific match wins; if no model-specific entry exists, the bare
``<kind>`` default is used.

Hint multipliers are advisory and bounded by the per-kind floor
defined in YAML. A malicious uploader cannot abuse a hint to pull
the threshold arbitrarily low — see plan §5.2 (3) for the security
rationale.

Calibration status: thresholds in YAML are the conservative starting
points from plan §5.2.1. T6.4 (gated on 30+ days of testnet upload
traffic) replaces these with empirically-tuned values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# The on-disk YAML lives next to this package by repo convention.
DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "dedup_thresholds.yaml"
)


class ThresholdResolverError(Exception):
    """Base for ThresholdResolver failures (load/parse/resolve)."""


@dataclass(frozen=True)
class EffectiveThresholds:
    """The resolved per-(kind, hint) thresholds.

    Both fields are similarity scores in [0, 1] (higher = more similar).
    For image-phash, callers convert ``1 - hamming_bits/64`` before
    comparing.
    """

    derivative: float
    duplicate: float
    # T6.5 — disputed-band lower bound. Hits in
    # ``[arbitration_floor, derivative)`` route to the human-review
    # arbitration queue rather than auto-attributing.
    arbitration_floor: float
    # The lookup key used to resolve these (e.g. "text-vector/openai/...").
    # Surfaced for telemetry / debugging — not part of the dedup decision.
    resolved_key: str
    # Whether a content-type-hint multiplier was applied.
    hint_applied: Optional[str]

    def __post_init__(self) -> None:
        if not (0.0 <= self.derivative <= 1.0):
            raise ValueError(
                f"derivative must be in [0,1], got {self.derivative!r}"
            )
        if not (0.0 <= self.duplicate <= 1.0):
            raise ValueError(
                f"duplicate must be in [0,1], got {self.duplicate!r}"
            )
        if self.duplicate < self.derivative:
            raise ValueError(
                f"duplicate ({self.duplicate}) must be >= derivative "
                f"({self.derivative}) — duplicate is the stricter tier"
            )
        if not (0.0 <= self.arbitration_floor <= 1.0):
            raise ValueError(
                f"arbitration_floor must be in [0,1], got "
                f"{self.arbitration_floor!r}"
            )
        if self.arbitration_floor > self.derivative:
            raise ValueError(
                f"arbitration_floor ({self.arbitration_floor}) must be "
                f"<= derivative ({self.derivative}) — the disputed band "
                f"sits below auto-attribution"
            )


class ThresholdResolver:
    """Loads + caches the dedup-thresholds YAML and resolves
    (kind, hint) → ``EffectiveThresholds``.

    Construction reads the YAML once. Subsequent ``resolve()`` calls
    are O(1) dict lookups + a clamp.

    Thread-safety: the resolver is read-only after construction. Safe
    to share across asyncio tasks and (CPython) threads. Not safe to
    mutate at runtime — reload by reconstructing.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._defaults = self._coerce_section(
            payload, "defaults",
            require=True,
        )
        self._multipliers = self._coerce_section(
            payload, "content_type_multipliers",
            require=False,
        )
        self._floors = self._coerce_section(
            payload, "floors",
            require=False,
        )
        # Validate every defaults entry has both derivative + duplicate.
        for key, entry in self._defaults.items():
            if not isinstance(entry, dict):
                raise ThresholdResolverError(
                    f"defaults[{key!r}] must be a mapping, "
                    f"got {type(entry).__name__}"
                )
            for tier in ("derivative", "duplicate"):
                if tier not in entry:
                    raise ThresholdResolverError(
                        f"defaults[{key!r}] missing required tier "
                        f"{tier!r}"
                    )
                v = entry[tier]
                if not isinstance(v, (int, float)):
                    raise ThresholdResolverError(
                        f"defaults[{key!r}].{tier} must be numeric, "
                        f"got {type(v).__name__}"
                    )

    @staticmethod
    def _coerce_section(
        payload: Dict[str, Any], name: str, *, require: bool,
    ) -> Dict[str, Any]:
        if name not in payload:
            if require:
                raise ThresholdResolverError(
                    f"thresholds YAML missing required section {name!r}"
                )
            return {}
        section = payload[name]
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise ThresholdResolverError(
                f"section {name!r} must be a mapping, got "
                f"{type(section).__name__}"
            )
        return section

    @classmethod
    def from_yaml_path(cls, path: Path) -> "ThresholdResolver":
        """Load and parse a thresholds YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ThresholdResolverError(
                "PyYAML is required to load dedup thresholds; "
                "install with `pip install pyyaml`"
            ) from exc
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ThresholdResolverError(
                f"cannot read thresholds YAML at {path}: {exc}"
            ) from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ThresholdResolverError(
                f"thresholds YAML at {path} failed to parse: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ThresholdResolverError(
                f"thresholds YAML at {path} must be a mapping at top "
                f"level, got {type(payload).__name__}"
            )
        return cls(payload)

    @classmethod
    def from_default_path(cls) -> "ThresholdResolver":
        """Load from the project's canonical
        ``prsm/data/dedup_thresholds.yaml``."""
        return cls.from_yaml_path(DEFAULT_THRESHOLDS_PATH)

    # -- resolution ------------------------------------------------------

    def resolve(
        self,
        fingerprint_kind: str,
        *,
        model_id: Optional[str] = None,
        content_type_hint: Optional[str] = None,
    ) -> EffectiveThresholds:
        """Look up the effective thresholds for a (kind, model_id, hint)
        triple.

        Args:
            fingerprint_kind: One of ``"text-vector"``, ``"image-phash"``,
                ``"audio-chromaprint"``, ``"video-multihash"``.
                ``"structural"`` is not handled — that kind is exact-match
                only and has no derivative tier.
            model_id: For ``"text-vector"``, the embedding model
                identifier (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
                Ignored for non-text kinds.
            content_type_hint: Advisory hint from upload metadata
                (e.g. ``"scientific_abstract"``, ``"code"``, ``"prose"``).
                Falls back to no multiplier if the hint isn't configured.

        Returns:
            ``EffectiveThresholds`` with the resolved derivative + duplicate
            thresholds, and the resolution metadata (key + hint applied).

        Raises:
            ThresholdResolverError when no entry is found for the kind.
        """
        if not isinstance(fingerprint_kind, str) or not fingerprint_kind:
            raise ThresholdResolverError(
                f"fingerprint_kind must be a non-empty string, got "
                f"{fingerprint_kind!r}"
            )

        # Most-specific lookup first.
        candidate_keys = []
        if model_id:
            candidate_keys.append(f"{fingerprint_kind}/{model_id}")
        candidate_keys.append(fingerprint_kind)

        resolved_key = None
        base_entry = None
        for key in candidate_keys:
            if key in self._defaults:
                resolved_key = key
                base_entry = self._defaults[key]
                break
        if base_entry is None:
            raise ThresholdResolverError(
                f"no threshold entry for kind={fingerprint_kind!r} "
                f"(tried {candidate_keys})"
            )

        derivative = float(base_entry["derivative"])
        duplicate = float(base_entry["duplicate"])
        # T6.5 — explicit arbitration_floor wins; otherwise fall back
        # to derivative - 0.10 (clamped to >= 0.0). The 0.10 default
        # is the design-doc conservative starting point until T6.4
        # calibrates per-kind ROC values.
        if "arbitration_floor" in base_entry and isinstance(
            base_entry["arbitration_floor"], (int, float),
        ):
            arbitration_floor = float(base_entry["arbitration_floor"])
        else:
            arbitration_floor = max(0.0, derivative - 0.10)

        hint_applied: Optional[str] = None
        if content_type_hint:
            hint_section = self._multipliers.get(content_type_hint)
            if isinstance(hint_section, dict):
                # Multipliers have the same key resolution as defaults —
                # most-specific (kind/model_id) wins, fall back to bare
                # kind. If neither is present, hint is silently ignored.
                mult_entry = None
                for key in candidate_keys:
                    if key in hint_section:
                        mult_entry = hint_section[key]
                        break
                if isinstance(mult_entry, dict):
                    derivative = self._apply_multiplier(
                        derivative, mult_entry.get("derivative", 1.0),
                    )
                    duplicate = self._apply_multiplier(
                        duplicate, mult_entry.get("duplicate", 1.0),
                    )
                    # Same multiplier applies to arbitration_floor so a
                    # tightening hint also tightens the disputed band's
                    # lower bound. If hint omits arbitration_floor, mirror
                    # the derivative multiplier (T6.5 design §"hint
                    # propagation").
                    arb_mult = mult_entry.get(
                        "arbitration_floor",
                        mult_entry.get("derivative", 1.0),
                    )
                    arbitration_floor = self._apply_multiplier(
                        arbitration_floor, arb_mult,
                    )
                    hint_applied = content_type_hint

        # Apply per-kind floor.
        floor_entry = None
        for key in candidate_keys:
            if key in self._floors:
                floor_entry = self._floors[key]
                break
        if isinstance(floor_entry, dict):
            d_floor = floor_entry.get("derivative")
            if isinstance(d_floor, (int, float)):
                derivative = max(derivative, float(d_floor))
            dup_floor = floor_entry.get("duplicate")
            if isinstance(dup_floor, (int, float)):
                duplicate = max(duplicate, float(dup_floor))

        # Clamp to [0, 1] regardless — defends against pathological YAML.
        derivative = min(1.0, max(0.0, derivative))
        duplicate = min(1.0, max(0.0, duplicate))
        arbitration_floor = min(1.0, max(0.0, arbitration_floor))

        # Invariant: duplicate >= derivative (duplicate is stricter).
        # If a hint multiplier inverted them, push duplicate up.
        if duplicate < derivative:
            duplicate = derivative
        # Invariant: arbitration_floor <= derivative. If a hint pushed
        # the floor above derivative, clamp down — the disputed band
        # collapses (effectively no arbitration zone).
        if arbitration_floor > derivative:
            arbitration_floor = derivative

        return EffectiveThresholds(
            derivative=derivative,
            duplicate=duplicate,
            arbitration_floor=arbitration_floor,
            resolved_key=resolved_key or fingerprint_kind,
            hint_applied=hint_applied,
        )

    @staticmethod
    def _apply_multiplier(base: float, multiplier: Any) -> float:
        """Multiply ``base`` by ``multiplier`` if it's a sensible
        number; otherwise log + ignore."""
        if not isinstance(multiplier, (int, float)):
            logger.warning(
                f"non-numeric multiplier {multiplier!r}, ignoring"
            )
            return base
        return base * float(multiplier)
