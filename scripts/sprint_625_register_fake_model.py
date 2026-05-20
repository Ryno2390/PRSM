"""Sprint 625 step B prereq — register a fake ShardedModel on a
local FilesystemModelRegistry, signed by Mac's NodeIdentity.

Output: /tmp/sprint625-registry/ directory containing the model's
manifest + shards + publisher.pubkey sidecar. Rsync to droplet's
/var/lib/prsm-registry/ to make it available there.

Model has 1 shard covering layer_range=(0,1) with 32 bytes of
zero tensor_data — enough for the IdentityLayerSliceRunner to
"forward through" without needing real weights.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODEL_ID = "sprint625-identity-test"
REGISTRY_ROOT = Path("/tmp/sprint625-registry")


def main() -> int:
    from prsm.node.config import NodeConfig
    from prsm.node.identity import load_node_identity
    from prsm.compute.model_registry.registry import FilesystemModelRegistry
    from prsm.compute.model_sharding.models import ShardedModel, ModelShard

    cfg = NodeConfig.load()
    mac = load_node_identity(cfg.identity_path)
    print(f"Publisher (Mac) node_id: {mac.node_id}")

    # Clean + recreate root
    if REGISTRY_ROOT.exists():
        shutil.rmtree(REGISTRY_ROOT)
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)

    registry = FilesystemModelRegistry(root=REGISTRY_ROOT)

    # Single fake shard
    shard = ModelShard(
        shard_id=f"{MODEL_ID}-shard-0",
        model_id=MODEL_ID,
        shard_index=0,
        total_shards=1,
        tensor_data=b"\x00" * 32,  # 32 zero bytes; identity runner ignores
        tensor_shape=(32,),
        layer_range=(0, 1),
        size_bytes=32,
    )
    model = ShardedModel(
        model_id=MODEL_ID,
        model_name="Sprint 625 identity-runner test",
        total_shards=1,
        shards=[shard],
    )

    manifest = registry.register(model, identity=mac)
    print(f"Registered: {manifest.model_id}")
    print(f"  publisher_node_id: {manifest.publisher_node_id}")
    print(f"  shards: {len(manifest.shards)}")
    print(f"  signature: {manifest.publisher_signature[:32]}...")

    # Verify the round-trip
    loaded = registry.get(MODEL_ID)
    print(f"\nRoundtrip verification: registry.get({MODEL_ID!r}) → ShardedModel("
          f"shards={len(loaded.shards)}, layer_range={loaded.shards[0].layer_range})")
    print(f"\nRegistry layout at {REGISTRY_ROOT}:")
    for p in sorted(REGISTRY_ROOT.rglob("*")):
        rel = p.relative_to(REGISTRY_ROOT)
        if p.is_file():
            print(f"  {rel}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
