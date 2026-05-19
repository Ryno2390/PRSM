"""Sprint 558 — production wiring for ParallaxScheduledExecutor.

First sprint in the arc taking /compute/inference from "mock only"
(sprint 438) to "real ParallaxScheduledExecutor". Sprint 546 wired
sprint-419's ActivationDPAware decorator into the chain-RPC
factory, but no production caller invokes that factory — /compute/
inference still falls through to MockInferenceExecutor or 503.

Sprint 558 closes the loop at the OPT-IN PATH:

  PRSM_INFERENCE_EXECUTOR=parallax  → node.py calls
      build_parallax_executor_or_none(node)

The builder reads three component-kind env vars:

  PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON file, top-level dict
      mapping model_id → ModelInfo kwargs.
  PRSM_PARALLAX_TRUST_STACK_KIND   — currently only "mock"; sprint
      560 adds "production" backed by real anchor + stake-lookup
      adapters.
  PRSM_PARALLAX_GPU_POOL_KIND      — currently only "static-empty"
      (no peers); sprint 561 adds "dht" backed by bootstrap-derived
      peer list.

Every missing/invalid piece logs a structured warning naming the
env var + returns None. The daemon stays bootable; /compute/
inference returns the existing 503 from sprint 438 with the
operator-readable detail. No new failure modes for existing
operators — this is a purely additive opt-in surface.

Future sprints fill in the "production" / "dht" kinds + the
chain_executor wiring (sprint 546's factory). Sprint 558 just
locks the contract so the component knobs are operationally
discoverable.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


_KNOWN_TRUST_STACK_KINDS = ("mock",)
_KNOWN_GPU_POOL_KINDS = ("static-empty",)


def _build_mock_trust_stack():
    """Permissive trust stack — passes every GPU through unchanged.
    Clearly labeled in env-var ("mock"); production operators MUST
    pass PRSM_PARALLAX_TRUST_STACK_KIND=production (sprint 560)."""
    from prsm.compute.parallax_scheduling.trust_adapter import (
        AnchorVerifyAdapter,
        ConsensusMismatchHook,
        StakeWeightedTrustAdapter,
        TierGateAdapter,
        TrustStack,
    )
    from prsm.compute.parallax_scheduling.prsm_request_router import (
        InMemoryProfileSource,
    )

    class _AcceptAllAnchor:
        """Test/dev anchor — claims every node_id is registered."""
        def lookup(self, node_id: str):
            return f"{node_id}.local"

    class _RecordingSubmitter:
        """Mismatch-hook submitter that no-ops in dev."""
        def __init__(self):
            self.submissions = []

        async def submit(self, *args, **kwargs):
            self.submissions.append((args, kwargs))

    class _ZeroStakeLookup:
        def get_stake(self, node_id: str) -> int:
            return 0

    return TrustStack(
        anchor_verify=AnchorVerifyAdapter(anchor=_AcceptAllAnchor()),
        tier_gate=TierGateAdapter(),
        profile_source=StakeWeightedTrustAdapter(
            inner=InMemoryProfileSource(snapshots={}),
            stake_lookup=_ZeroStakeLookup(),
        ),
        consensus_hook=ConsensusMismatchHook(
            submitter=_RecordingSubmitter(), sample_rate=0.0,
        ),
    )


def _build_static_empty_pool_provider():
    """Pool provider that returns an empty list. /compute/inference
    requests will fail Phase-1 allocation with a clear error; this
    is the explicit "no peers wired" path. Sprint 561 replaces with
    a DHT-backed provider."""
    def _provider():
        return []
    return _provider


class _StubChainExecutor:
    """Sprint 558 placeholder. The real chain executor comes from
    sprint 546's make_rpc_chain_executor factory in a future sprint.
    Today, no /compute/inference request gets this far because the
    static-empty pool fails Phase-1 first."""

    def execute_chain(self, *, request, chain, **kwargs):
        raise RuntimeError(
            "Sprint 558 stub chain executor — not callable until "
            "sprint 562+ wires make_rpc_chain_executor."
        )


def build_parallax_executor_or_none(node: Any) -> Optional[Any]:
    """Construct a ParallaxScheduledExecutor if every required
    operator-supplied component is present + valid.

    Returns ``None`` and logs a structured warning naming the
    missing piece otherwise. Daemon caller treats None the same
    as "no inference executor wired" (the existing default).

    Component contract (env vars):
      PRSM_PARALLAX_MODEL_CATALOG_FILE — JSON path
      PRSM_PARALLAX_TRUST_STACK_KIND   — "mock"
      PRSM_PARALLAX_GPU_POOL_KIND      — "static-empty"

    Each kind is a tagged-union opt-in. Sprint 558 ships the dev/
    mock kinds only; sprints 560/561 add production kinds.
    """
    catalog_path = os.environ.get(
        "PRSM_PARALLAX_MODEL_CATALOG_FILE", "",
    ).strip()
    trust_kind = os.environ.get(
        "PRSM_PARALLAX_TRUST_STACK_KIND", "",
    ).strip()
    pool_kind = os.environ.get(
        "PRSM_PARALLAX_GPU_POOL_KIND", "",
    ).strip()

    # No-opt-in case: silent return. The operator simply didn't set
    # any of the parallax env vars.
    if not (catalog_path or trust_kind or pool_kind):
        return None

    # ── model catalog ────────────────────────────────────
    if not catalog_path:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE not set. Provide a "
            "JSON file mapping model_id → ModelInfo kwargs."
        )
        return None
    cat_p = Path(catalog_path)
    if not cat_p.exists():
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_MODEL_CATALOG_FILE=%s does not exist "
            "on disk.", catalog_path,
        )
        return None
    try:
        raw = json.loads(cat_p.read_text())
    except json.JSONDecodeError as exc:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s failed JSON parse: %s",
            catalog_path, exc,
        )
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s top-level must be a dict; got %s",
            catalog_path, type(raw).__name__,
        )
        return None
    try:
        from prsm.compute.parallax_scheduling.model_info import (
            ModelInfo,
        )
        catalog = {
            mid: ModelInfo(**(kw or {})) for mid, kw in raw.items()
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: model "
            "catalog %s contains an entry that failed ModelInfo "
            "construction: %s", catalog_path, exc,
        )
        return None

    # ── trust stack ──────────────────────────────────────
    if trust_kind not in _KNOWN_TRUST_STACK_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_TRUST_STACK_KIND=%r unrecognized. "
            "Known: %s", trust_kind, _KNOWN_TRUST_STACK_KINDS,
        )
        return None
    if trust_kind == "mock":
        trust_stack = _build_mock_trust_stack()
    else:  # defensive — already gated above
        return None

    # ── gpu pool ─────────────────────────────────────────
    if pool_kind not in _KNOWN_GPU_POOL_KINDS:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "PRSM_PARALLAX_GPU_POOL_KIND=%r unrecognized. "
            "Known: %s", pool_kind, _KNOWN_GPU_POOL_KINDS,
        )
        return None
    if pool_kind == "static-empty":
        pool_provider = _build_static_empty_pool_provider()
    else:  # defensive
        return None

    # ── identity ─────────────────────────────────────────
    identity = getattr(node, "identity", None)
    if identity is None:
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: node has "
            "no .identity — daemon must initialize identity before "
            "this builder runs."
        )
        return None

    # ── construct ────────────────────────────────────────
    try:
        from prsm.compute.inference.parallax_executor import (
            ParallaxScheduledExecutor,
        )
        return ParallaxScheduledExecutor(
            gpu_pool_provider=pool_provider,
            trust_stack=trust_stack,
            model_catalog=catalog,
            chain_executor=_StubChainExecutor(),
            node_identity=identity,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sprint 558 ParallaxScheduledExecutor wiring: "
            "ParallaxScheduledExecutor(...) raised at construction: "
            "%s", exc,
        )
        return None
