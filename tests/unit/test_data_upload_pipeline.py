"""Tests for semantic sharding upload pipeline."""

import pytest
from click.testing import CliRunner


class TestSemanticShardUploadCLI:
    def test_semantic_shard_flag_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["storage", "upload", "--help"])
        assert result.exit_code == 0
        assert "--semantic-shard" in result.output


class TestSemanticShardManifestCreation:
    def test_create_manifest_from_data(self):
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        shards = [
            SemanticShard(
                shard_id=f"shard-{i}",
                parent_dataset="test-dataset",
                cid=f"QmShard{i}",
                centroid=[float(i) * 0.25],
                record_count=1000,
                size_bytes=1024 * 1024,
                keywords=["test"],
            )
            for i in range(4)
        ]
        manifest = SemanticShardManifest(
            dataset_id="test-dataset",
            total_records=4000,
            total_size_bytes=4 * 1024 * 1024,
            shards=shards,
        )
        assert manifest.dataset_id == "test-dataset"
        assert len(manifest.shards) == 4

        # Test serialization
        d = manifest.to_dict()
        restored = SemanticShardManifest.from_dict(d)
        assert restored.dataset_id == "test-dataset"
        assert len(restored.shards) == 4

    def test_find_relevant_shards(self):
        from prsm.data.shard_models import SemanticShard, SemanticShardManifest

        shards = [
            SemanticShard(shard_id="a", parent_dataset="ds", cid="QmA",
                         centroid=[1.0, 0.0], record_count=100, size_bytes=1024, keywords=["electric"]),
            SemanticShard(shard_id="b", parent_dataset="ds", cid="QmB",
                         centroid=[0.0, 1.0], record_count=100, size_bytes=1024, keywords=["gasoline"]),
        ]
        manifest = SemanticShardManifest(dataset_id="ds", total_records=200, total_size_bytes=2048, shards=shards)
        relevant = manifest.find_relevant_shards([0.9, 0.1], top_k=1)
        assert len(relevant) == 1
        shard, score = relevant[0]
        assert shard.shard_id == "a"


class TestDataListingIntegration:
    def test_listing_with_semantic_shards(self):
        from prsm.economy.pricing.data_listing import DataListingManager, DataListing
        from decimal import Decimal

        mgr = DataListingManager()
        listing = DataListing(
            dataset_id="nada-nc-2025",
            owner_id="data-provider-001",
            title="NADA NC Vehicle Registrations 2025",
            shard_count=12,
            total_size_bytes=50 * 1024 * 1024,
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.5"),
            requires_stake=Decimal("1000"),
        )
        lid = mgr.publish(listing)

        # Verify pricing for full access
        from prsm.economy.pricing.models import DataAccessFee
        fee = DataAccessFee(
            dataset_id="nada-nc-2025",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.5"),
            bulk_discount=Decimal("0.2"),
        )
        total = fee.total_for_shards(12)
        assert total > 0

        # Verify access control
        allowed, reason = mgr.check_access(lid, accessor_stake=Decimal("500"))
        assert not allowed
        allowed, reason = mgr.check_access(lid, accessor_stake=Decimal("1500"))
        assert allowed
