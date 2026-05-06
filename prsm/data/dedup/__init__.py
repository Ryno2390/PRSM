"""
PRSM-PROV-1 Item 6 — per-fingerprint-kind dedup thresholds.

Public surface:
- ThresholdResolver: load `dedup_thresholds.yaml` and resolve the
  effective (derivative, duplicate) thresholds for a given
  fingerprint kind, optionally with a content-type hint.

Why this lives outside ``prsm/node/content_uploader.py``:
The hardcoded ``DERIVATIVE_THRESHOLD`` / ``DUPLICATE_THRESHOLD``
class constants on ``_SemanticIndex`` are usable as fallbacks
(no behavior change for callers that don't wire a resolver), but
the centralized config + per-kind dispatch belongs in a dedicated
module so future calibration (T6.4 — gated on 30+ days of testnet
upload traffic) can update YAML without touching the upload
critical path.
"""

from prsm.data.dedup.thresholds import (
    DEFAULT_THRESHOLDS_PATH,
    EffectiveThresholds,
    ThresholdResolver,
    ThresholdResolverError,
)

__all__ = [
    "DEFAULT_THRESHOLDS_PATH",
    "EffectiveThresholds",
    "ThresholdResolver",
    "ThresholdResolverError",
]
