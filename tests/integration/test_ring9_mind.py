"""Ring 9 Smoke Test — NWTN training pipeline + model service."""

import pytest
import json
import numpy as np
from prsm.compute.nwtn.training import TrainingPipeline, TrainingConfig, ModelCard, DeploymentStatus
from prsm.compute.nwtn.training.model_service import NWTNModelService


class TestRing9Smoke:
    def test_full_training_pipeline(self):
        pipeline = TrainingPipeline(TrainingConfig(min_corpus_size=2))
        traces = [
            {"query": f"q{i}", "decomposition": {"estimated_complexity": 0.5}, "plan": {"route": "swarm"}, "execution_result": {"status": "success"}, "execution_metrics": {"pcu": 1.0}}
            for i in range(5)
        ]
        count = pipeline.ingest_traces(traces)
        assert count == 5
        valid, errors = pipeline.validate_corpus()
        assert valid, f"Validation errors: {errors}"
        jsonl = pipeline.export_dataset()
        assert len(jsonl.strip().split("\n")) == 5
        card = pipeline.create_model_card("nwtn-v1", "NWTN", {"accuracy": 0.92})
        assert card.corpus_stats["total_traces"] == 5

    def test_model_deploy_lifecycle(self):
        service = NWTNModelService()
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama-3.1")
        service.register_model(card)
        weights = {"attn": np.random.randn(32, 16), "mlp": np.random.randn(16, 8)}
        deployment = service.deploy_model("nwtn-v1", weight_tensors=weights, n_shards=4)
        assert deployment["n_shards"] == 4
        assert card.status == DeploymentStatus.DEPLOYED
        status = service.get_model_status("nwtn-v1")
        assert status["status"] == "deployed"
        service.retire_model("nwtn-v1")
        assert card.status == DeploymentStatus.RETIRED

    def test_all_rings_1_through_9_import(self):
        # Ring 5 AgentForge was removed in v1.6.0 scope alignment — the legacy
        # NWTN AGI framework (agent_forge, orchestrator, meta_reasoning_engine,
        # etc.) is no longer part of PRSM. Reasoning belongs in third-party LLMs.
        from prsm.compute.wasm import WASMRuntime, HardwareProfiler
        from prsm.compute.agents import AgentDispatcher, AgentExecutor
        from prsm.compute.swarm import SwarmCoordinator
        from prsm.economy.pricing import PricingEngine
        from prsm.compute.tee import DPNoiseInjector, ConfidentialResult
        from prsm.compute.model_sharding import ModelSharder, PipelineRandomizer
        from prsm.compute.nwtn.training import TrainingPipeline, ModelCard
        from prsm.compute.nwtn.training.model_service import NWTNModelService
        assert all(x is not None for x in [
            WASMRuntime, AgentDispatcher, SwarmCoordinator, PricingEngine,
            DPNoiseInjector, ModelSharder, TrainingPipeline, NWTNModelService,
        ])
