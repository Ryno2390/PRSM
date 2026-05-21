"""Sprint 682 — DHT-backed GpuPoolProvider.

Translates the local hardware_profile (sprint 681) + DHT-propagated
peer hardware_profiles (sprint 680) into the
``Sequence[ParallaxGPU]`` that ``ParallaxScheduledExecutor`` needs.

Replaces sprint 558's static-empty provider when the operator opts
in via ``PRSM_PARALLAX_GPU_POOL_KIND=dht-backed``.

Sprint 683 layers on-chain stake reads on top of the
``stake_amount=0`` placeholder used here. Until then, Adapter C
(stake-weighted trust) effectively treats every peer as unstaked —
which is correct behavior pre-commissioning.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from prsm.compute.parallax_scheduling.prsm_types import (
    ParallaxGPU, TIER_ATTESTATION_NONE,
)

logger = logging.getLogger(__name__)


# Conservative default for memory bandwidth when the profiler
# doesn't surface a real number. 50 GB/s ≈ typical DDR4/DDR5 dual-
# channel desktop. Better than crashing on the dataclass's positive-
# only validator.
_DEFAULT_MEMORY_BANDWIDTH_GBPS = 50.0

# Layer capacity heuristic: ~2 GB per transformer layer for a 7B
# model in fp16. Operators can override later.
_BYTES_PER_LAYER_GB = 2.0


def _hw_dict_to_parallax_gpu(
    node_id: str, hw: Any, region: str,
    stake_reader: Optional[Any] = None,
) -> Optional[ParallaxGPU]:
    """Map a HardwareProfile.to_dict() shape → ParallaxGPU.

    Returns None when essential fields can't be derived (no tflops,
    no memory). Defensive against malicious/buggy peers — never
    raises."""
    if not isinstance(hw, dict):
        return None

    tflops_fp16 = float(hw.get("tflops_fp16", 0.0) or 0.0)
    if tflops_fp16 <= 0.0:
        # Estimate from fp32 (consumer GPUs ≈ 2× fp32 for fp16)
        tflops_fp32 = float(hw.get("tflops_fp32", 0.0) or 0.0)
        if tflops_fp32 > 0.0:
            tflops_fp16 = tflops_fp32 * 2.0
    if tflops_fp16 <= 0.0:
        return None

    gpu_vram_gb = float(hw.get("gpu_vram_gb", 0.0) or 0.0)
    ram_total_gb = float(hw.get("ram_total_gb", 0.0) or 0.0)
    memory_gb = gpu_vram_gb if gpu_vram_gb > 0 else ram_total_gb
    if memory_gb <= 0:
        return None

    # Sprint 695 — operator override for the advertised memory_gb.
    # The upstream Parallax allocator uses memory_gb as input to
    # its water-filling step that decides per-stage layer counts;
    # large memory → all layers fit on one GPU → 1-stage allocation
    # (live-attested in sprint 688 for gpt2 on 1.92GB droplets).
    # Operators wanting to force a multi-stage split for testing
    # (or to reserve memory for runtime overhead) can pin this to
    # a smaller value. Pure-additive: env unset → existing
    # heuristic preserved.
    _mem_override_raw = os.environ.get(
        "PRSM_PARALLAX_MEMORY_GB_OVERRIDE", "",
    ).strip()
    if _mem_override_raw:
        try:
            memory_gb = float(_mem_override_raw)
            if memory_gb <= 0:
                return None
        except ValueError:
            logger.debug(
                "PRSM_PARALLAX_MEMORY_GB_OVERRIDE=%r not float; "
                "falling back to advertised memory_gb=%s",
                _mem_override_raw, memory_gb,
            )

    memory_bandwidth_gbps = float(
        hw.get("memory_bandwidth_gbps", _DEFAULT_MEMORY_BANDWIDTH_GBPS)
        or _DEFAULT_MEMORY_BANDWIDTH_GBPS
    )
    if memory_bandwidth_gbps <= 0:
        memory_bandwidth_gbps = _DEFAULT_MEMORY_BANDWIDTH_GBPS

    # Sprint 686 — operator override for the layer_capacity
    # heuristic. The default 2GB-per-layer formula targets 7B-class
    # fp16 models; operators serving smaller models (gpt2, phi-2,
    # etc.) need a higher cap or Phase-1 allocation can't place
    # all layers. Set PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=<int>
    # to pin per-GPU capacity explicitly. Live-attest of sprint
    # 685's gpt2 (12 layers / ~27MB each) surfaced this.
    _override_raw = os.environ.get(
        "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE", "",
    ).strip()
    if _override_raw:
        try:
            layer_capacity = max(1, int(_override_raw))
        except ValueError:
            logger.debug(
                "PRSM_PARALLAX_LAYER_CAPACITY_OVERRIDE=%r not "
                "integer; falling back to memory heuristic",
                _override_raw,
            )
            layer_capacity = max(1, int(memory_gb / _BYTES_PER_LAYER_GB))
    else:
        layer_capacity = max(1, int(memory_gb / _BYTES_PER_LAYER_GB))

    gpu_api = hw.get("gpu_api", "") or ""
    device_map = {
        "cuda": "cuda", "rocm": "cuda", "metal": "mps", "": "cpu",
    }
    device = device_map.get(gpu_api, "cpu")

    # Sprint 683 — optionally resolve stake via on-chain reader
    stake_amount = 0
    operator_address = hw.get("operator_address", "") or ""
    if stake_reader is not None and operator_address:
        try:
            stake_amount = int(
                stake_reader.stake_amount_for(operator_address) or 0
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "stake_reader.stake_amount_for(%s) raised: %s — "
                "treating as unstaked", operator_address[:10], exc,
            )
            stake_amount = 0

    try:
        return ParallaxGPU(
            node_id=node_id,
            region=region,
            layer_capacity=layer_capacity,
            stake_amount=stake_amount,
            tier_attestation=TIER_ATTESTATION_NONE,
            tflops_fp16=tflops_fp16,
            memory_gb=memory_gb,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            gpu_name=hw.get("gpu_name", "") or "",
            device=device,
            num_gpus=1,
        )
    except ValueError as exc:
        logger.debug(
            "Skipping peer %s — ParallaxGPU rejected derived "
            "fields: %s", node_id[:16], exc,
        )
        return None


def build_dht_backed_pool_provider(
    node: Any,
    stake_reader: Optional[Any] = None,
) -> Callable[[], Sequence[ParallaxGPU]]:
    """Returns a callable suitable for the
    ``ParallaxScheduledExecutor.gpu_pool_provider`` slot.

    Each invocation reads the *current* state of the node's
    discovery layer + own hardware profile — so peers joining/
    leaving the DHT show up in the next allocation pass without
    restarting the daemon."""
    region = os.environ.get("PRSM_PARALLAX_REGION", "default") or "default"

    # Sprint 683 — lazy-construct an OnChainStakeReader when the
    # caller didn't supply one + PRSM_STAKE_BOND_ADDRESS is set.
    # Tests pass an explicit reader via the kwarg.
    if stake_reader is None and os.environ.get(
        "PRSM_STAKE_BOND_ADDRESS", "",
    ).strip():
        try:
            from prsm.node.onchain_stake_reader import OnChainStakeReader
            stake_reader = OnChainStakeReader()
        except Exception as exc:  # noqa: BLE001
            logger.debug("OnChainStakeReader construction failed: %s", exc)
            stake_reader = None

    def _provider() -> List[ParallaxGPU]:
        gpus: List[ParallaxGPU] = []
        own_id: Optional[str] = None
        try:
            identity = getattr(node, "identity", None)
            if identity is not None:
                own_id = getattr(identity, "node_id", None)
        except Exception:  # noqa: BLE001
            pass

        discovery = getattr(node, "discovery", None)
        if discovery is None:
            return gpus

        own_hw = getattr(discovery, "_local_hardware_profile", None)
        if own_id and own_hw is not None:
            gpu = _hw_dict_to_parallax_gpu(
                own_id, own_hw, region, stake_reader=stake_reader,
            )
            if gpu is not None:
                gpus.append(gpu)

        peers: Dict[str, Any] = (
            getattr(discovery, "known_peers", {}) or {}
        )
        for peer_id, info in peers.items():
            if peer_id == own_id:
                continue
            peer_hw = getattr(info, "hardware_profile", None)
            if peer_hw is None:
                continue
            gpu = _hw_dict_to_parallax_gpu(
                peer_id, peer_hw, region, stake_reader=stake_reader,
            )
            if gpu is not None:
                gpus.append(gpu)

        return gpus

    return _provider
