"""Sprint 312 — pipeline partition spec.

Declares how a model's layers are split across stages and
which node runs each stage. Pure data + validation; says
nothing about how layers compute (that's the StageRunner
Protocol).

Validation invariants enforced:
  - total_layers > 0; covered exactly once by the union
    of stage ranges (no gaps, no overlaps)
  - Each stage's layers are a contiguous block; layer
    indices in [0, total_layers)
  - One node_id per stage; n_stages == len(node_ids)
  - No empty stages

The `partition_hash()` is the deterministic hash the
orchestrator binds into pipeline receipts, so verifiers
can detect substituted partitions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List


class PartitionValidationError(ValueError):
    """Raised when a PipelinePartition fails validation."""


@dataclass
class PipelinePartition:
    total_layers: int
    stage_layer_ranges: List[List[int]]
    stage_node_ids: List[str]

    @property
    def n_stages(self) -> int:
        return len(self.stage_layer_ranges)

    # ── Validation ───────────────────────────────────

    def validate(self) -> None:
        if self.total_layers < 1:
            raise PartitionValidationError(
                f"total_layers must be >= 1; got "
                f"{self.total_layers}"
            )
        if not self.stage_layer_ranges:
            raise PartitionValidationError(
                "at least one stage required"
            )
        if len(self.stage_node_ids) != self.n_stages:
            raise PartitionValidationError(
                f"stage_node_ids has {len(self.stage_node_ids)} "
                f"entries but partition has "
                f"{self.n_stages} stages — each stage "
                f"needs exactly one node_id"
            )

        seen: set[int] = set()
        for stage_idx, layers in enumerate(
            self.stage_layer_ranges,
        ):
            if not layers:
                raise PartitionValidationError(
                    f"stage {stage_idx} is empty; every "
                    f"stage must own at least one layer"
                )
            # Range bounds + non-duplicate within stage
            for layer in layers:
                if (
                    layer < 0
                    or layer >= self.total_layers
                ):
                    raise PartitionValidationError(
                        f"stage {stage_idx} layer {layer} "
                        f"out of range [0, "
                        f"{self.total_layers})"
                    )
                if layer in seen:
                    raise PartitionValidationError(
                        f"layer {layer} appears in "
                        f"multiple stages (overlap / "
                        f"duplicate)"
                    )
                seen.add(layer)
            # Contiguous block within stage
            sorted_layers = sorted(layers)
            for i in range(1, len(sorted_layers)):
                if sorted_layers[i] != sorted_layers[i - 1] + 1:
                    raise PartitionValidationError(
                        f"stage {stage_idx} layers "
                        f"{layers!r} are not a contiguous "
                        f"block"
                    )

        # Full coverage: union of stages = all layers
        if len(seen) != self.total_layers:
            missing = sorted(
                set(range(self.total_layers)) - seen,
            )
            raise PartitionValidationError(
                f"partition has gaps; missing layer(s): "
                f"{missing}"
            )

    # ── Serialization ────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_layers": int(self.total_layers),
            "stage_layer_ranges": [
                list(s) for s in self.stage_layer_ranges
            ],
            "stage_node_ids": list(self.stage_node_ids),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "PipelinePartition":
        return cls(
            total_layers=int(d["total_layers"]),
            stage_layer_ranges=[
                list(s)
                for s in (d.get("stage_layer_ranges") or [])
            ],
            stage_node_ids=list(
                d.get("stage_node_ids") or [],
            ),
        )

    # ── Hash ─────────────────────────────────────────

    def partition_hash(self) -> str:
        """SHA-256 of the canonical JSON form. The
        orchestrator binds this hash into the pipeline
        receipt; verifiers compare against the partition
        they expect to detect substitution attacks."""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def even_layer_partition(
    *,
    total_layers: int,
    node_ids: List[str],
) -> PipelinePartition:
    """Convenience: evenly distribute total_layers across
    len(node_ids) stages. First stage absorbs the
    remainder (so stage 0 may have one extra layer when
    total_layers % n_nodes != 0)."""
    if total_layers < 1:
        raise ValueError(
            f"total_layers must be >= 1; got {total_layers}"
        )
    n = len(node_ids)
    if n == 0:
        raise ValueError("node_ids must be non-empty")
    if n > total_layers:
        raise ValueError(
            f"more nodes ({n}) than layers "
            f"({total_layers}); reduce node pool or add "
            f"layers"
        )
    base = total_layers // n
    remainder = total_layers % n
    ranges: List[List[int]] = []
    cursor = 0
    for i in range(n):
        size = base + (1 if i < remainder else 0)
        ranges.append(list(range(cursor, cursor + size)))
        cursor += size
    return PipelinePartition(
        total_layers=total_layers,
        stage_layer_ranges=ranges,
        stage_node_ids=list(node_ids),
    )
