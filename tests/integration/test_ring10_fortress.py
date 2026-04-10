"""Ring 10 Smoke Test — security hardening verification."""

import pytest
import hashlib

from prsm.security import IntegrityVerifier, PrivacyBudgetTracker, PipelineAuditLog


class TestRing10Smoke:
    def test_integrity_verification_pipeline(self):
        verifier = IntegrityVerifier()
        data = b"model shard tensor data"
        checksum = verifier.compute_checksum(data)
        assert verifier.verify_shard(data, checksum)
        assert not verifier.verify_shard(b"tampered", checksum)

    def test_privacy_budget_lifecycle(self):
        tracker = PrivacyBudgetTracker(max_epsilon=20.0)
        assert tracker.record_spend(8.0, "inference-1", "nwtn-v1")
        assert tracker.record_spend(8.0, "inference-2", "nwtn-v1")
        assert not tracker.record_spend(8.0, "inference-3", "nwtn-v1")  # Would exceed
        report = tracker.get_audit_report()
        assert report["total_spent"] == 16.0
        assert report["remaining"] == 4.0

    def test_audit_log_chain_integrity(self):
        log = PipelineAuditLog()
        for i in range(5):
            log.record(f"model-{i}", 4, [{"node_id": f"n{j}"} for j in range(4)], 20)
        assert log.verify_chain()
        assert log.entry_count == 5
        report = log.entropy_report()
        assert report["chain_valid"] is True

    def test_all_rings_1_through_10_import(self):
        # Ring 1
        from prsm.compute.wasm import WASMRuntime, WasmtimeRuntime, HardwareProfiler, ComputeTier
        # Ring 2
        from prsm.compute.agents import AgentDispatcher, AgentExecutor, MobileAgent, AgentManifest
        # Ring 3
        from prsm.compute.swarm import SwarmCoordinator, SwarmJob, SwarmResult, MapReduceStrategy
        # Ring 4
        from prsm.economy.pricing import PricingEngine, CostQuote, ProsumerTier, DataAccessFee
        from prsm.economy.prosumer import ProsumerManager
        # Ring 5 AgentForge removed in v1.6.0 (legacy NWTN AGI framework pruned)
        # Ring 6 (hardening from Phase 1)
        from prsm.economy.ftns_onchain import estimate_gas_price, RPCFailover
        from prsm.node.settler_registry import verify_settler_signature
        # Ring 7
        from prsm.compute.tee import TEEType, TEECapability, DPConfig, PrivacyLevel, DPNoiseInjector, ConfidentialResult
        from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
        from prsm.compute.tee.confidential_executor import ConfidentialExecutor
        # Ring 8
        from prsm.compute.model_sharding import ModelSharder, PipelineRandomizer, ShardedModel, PipelineStakeTier
        from prsm.compute.model_sharding.executor import TensorParallelExecutor
        from prsm.compute.model_sharding.collision_detector import CollisionDetector
        # Ring 9
        from prsm.compute.nwtn.training import TrainingPipeline, TrainingConfig, ModelCard, DeploymentStatus
        from prsm.compute.nwtn.training.model_service import NWTNModelService
        # Ring 10
        from prsm.security import IntegrityVerifier, PrivacyBudgetTracker, PipelineAuditLog

        # Verify all are real classes/functions
        assert all(x is not None for x in [
            WASMRuntime, HardwareProfiler, ComputeTier,
            AgentDispatcher, AgentExecutor, MobileAgent,
            SwarmCoordinator, SwarmJob, MapReduceStrategy,
            PricingEngine, CostQuote, ProsumerTier, ProsumerManager,
            estimate_gas_price, RPCFailover, verify_settler_signature,
            TEEType, DPConfig, DPNoiseInjector, ConfidentialExecutor,
            ModelSharder, PipelineRandomizer, TensorParallelExecutor, CollisionDetector,
            TrainingPipeline, ModelCard, NWTNModelService,
            IntegrityVerifier, PrivacyBudgetTracker, PipelineAuditLog,
        ])

        # MCP tools check removed in v1.6.1 (Ring 5 AgentForge pruned in v1.6.0)
