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

    memory_bandwidth_gbps = float(
        hw.get("memory_bandwidth_gbps", _DEFAULT_MEMORY_BANDWIDTH_GBPS)
        or _DEFAULT_MEMORY_BANDWIDTH_GBPS
    )
    if memory_bandwidth_gbps <= 0:
        memory_bandwidth_gbps = _DEFAULT_MEMORY_BANDWIDTH_GBPS

    layer_capacity = max(1, int(memory_gb / _BYTES_PER_LAYER_GB))

    gpu_api = hw.get("gpu_api", "") or ""
    device_map = {
        "cuda": "cuda", "rocm": "cuda", "metal": "mps", "": "cpu",
    }
    device = device_map.get(gpu_api, "cpu")

    try:
        return ParallaxGPU(
            node_id=node_id,
            region=region,
            layer_capacity=layer_capacity,
            stake_amount=0,
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
) -> Callable[[], Sequence[ParallaxGPU]]:
    """Returns a callable suitable for the
    ``ParallaxScheduledExecutor.gpu_pool_provider`` slot.

    Each invocation reads the *current* state of the node's
    discovery layer + own hardware profile — so peers joining/
    leaving the DHT show up in the next allocation pass without
    restarting the daemon."""
    region = os.environ.get("PRSM_PARALLAX_REGION", "default") or "default"

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
            gpu = _hw_dict_to_parallax_gpu(own_id, own_hw, region)
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
            gpu = _hw_dict_to_parallax_gpu(peer_id, peer_hw, region)
            if gpu is not None:
                gpus.append(gpu)

        return gpus

    return _provider
