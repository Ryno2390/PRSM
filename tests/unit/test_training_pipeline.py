"""Tests for NWTN training pipeline."""

import pytest
import json

from prsm.compute.nwtn.training.models import (
    TrainingConfig,
    TrainingCorpus,
    ModelCard,
    DeploymentStatus,
)
from prsm.compute.nwtn.training.pipeline import TrainingPipeline


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.base_model == "meta-llama/Llama-3.1-8B"
        assert config.epochs == 3
        assert config.min_corpus_size == 100

    def test_custom(self):
        config = TrainingConfig(base_model="custom/model", epochs=5, lora_rank=32)
        assert config.base_model == "custom/model"
        assert config.lora_rank == 32


class TestTrainingCorpus:
    def _make_trace(self, query="test", route="swarm"):
        return {
            "query": query,
            "decomposition": {"operations": ["filter"], "estimated_complexity": 0.5},
            "plan": {"route": route},
            "execution_result": {"status": "success"},
            "execution_metrics": {"pcu": 1.0},
        }

    def test_add_and_count(self):
        corpus = TrainingCorpus()
        corpus.add_trace(self._make_trace())
        corpus.add_trace(self._make_trace("q2"))
        assert corpus.total_traces == 2

    def test_validate_empty_fails(self):
        corpus = TrainingCorpus()
        valid, errors = corpus.validate()
        assert not valid
        assert "empty" in errors[0].lower()

    def test_validate_missing_fields(self):
        corpus = TrainingCorpus()
        corpus.add_trace({"query": "test"})  # Missing other fields
        valid, errors = corpus.validate()
        assert not valid

    def test_validate_complete_passes(self):
        corpus = TrainingCorpus()
        corpus.add_trace(self._make_trace())
        valid, errors = corpus.validate()
        assert valid

    def test_export_jsonl(self):
        corpus = TrainingCorpus()
        corpus.add_trace(self._make_trace("q1"))
        corpus.add_trace(self._make_trace("q2"))
        jsonl = corpus.export_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert "query" in parsed

    def test_stats(self):
        corpus = TrainingCorpus()
        corpus.add_trace(self._make_trace("q1", "swarm"))
        corpus.add_trace(self._make_trace("q2", "direct_llm"))
        corpus.add_trace(self._make_trace("q3", "swarm"))
        stats = corpus.stats()
        assert stats["total_traces"] == 3
        assert stats["route_distribution"]["swarm"] == 2
        assert stats["route_distribution"]["direct_llm"] == 1


class TestModelCard:
    def test_creation(self):
        card = ModelCard(
            model_id="nwtn-v1",
            model_name="NWTN",
            base_model="meta-llama/Llama-3.1-8B",
            metrics={"accuracy": 0.92},
        )
        assert card.status == DeploymentStatus.REGISTERED
        assert card.metrics["accuracy"] == 0.92

    def test_to_dict_roundtrip(self):
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama")
        d = card.to_dict()
        restored = ModelCard.from_dict(d)
        assert restored.model_id == "nwtn-v1"
        assert restored.status == DeploymentStatus.REGISTERED


class TestDeploymentStatus:
    def test_all_statuses(self):
        assert DeploymentStatus.REGISTERED == "registered"
        assert DeploymentStatus.SERVING == "serving"
        assert DeploymentStatus.RETIRED == "retired"


class TestTrainingPipeline:
    def _make_trace(self, query="test"):
        return {
            "query": query,
            "decomposition": {"operations": ["filter"], "estimated_complexity": 0.5},
            "plan": {"route": "swarm"},
            "execution_result": {"status": "success"},
            "execution_metrics": {"pcu": 1.0},
        }

    def test_ingest_traces(self):
        pipeline = TrainingPipeline()
        count = pipeline.ingest_traces([self._make_trace("q1"), self._make_trace("q2")])
        assert count == 2
        assert pipeline.corpus.total_traces == 2

    def test_validate_below_minimum(self):
        pipeline = TrainingPipeline(TrainingConfig(min_corpus_size=10))
        pipeline.ingest_traces([self._make_trace()])
        valid, errors = pipeline.validate_corpus()
        assert not valid
        assert any("below minimum" in e for e in errors)

    def test_export_dataset(self):
        pipeline = TrainingPipeline()
        pipeline.ingest_traces([self._make_trace()])
        jsonl = pipeline.export_dataset()
        assert len(jsonl) > 0
        parsed = json.loads(jsonl.split("\n")[0])
        assert parsed["query"] == "test"

    def test_create_model_card(self):
        pipeline = TrainingPipeline()
        pipeline.ingest_traces([self._make_trace()])
        card = pipeline.create_model_card("nwtn-v1", "NWTN", {"accuracy": 0.9})
        assert card.model_id == "nwtn-v1"
        assert card.corpus_stats["total_traces"] == 1
        assert card.metrics["accuracy"] == 0.9

    def test_get_corpus_stats(self):
        pipeline = TrainingPipeline()
        pipeline.ingest_traces([self._make_trace(), self._make_trace()])
        stats = pipeline.get_corpus_stats()
        assert stats["total_traces"] == 2


from prsm.compute.nwtn.training.model_service import NWTNModelService
import numpy as np


class TestNWTNModelService:
    def test_register_model(self):
        service = NWTNModelService()
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama")
        model_id = service.register_model(card)
        assert model_id == "nwtn-v1"
        assert card.status == DeploymentStatus.REGISTERED

    def test_deploy_model_with_sharding(self):
        service = NWTNModelService()
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama")
        service.register_model(card)
        weights = {"layer1": np.random.randn(16, 8)}
        deployment = service.deploy_model("nwtn-v1", weight_tensors=weights, n_shards=4)
        assert deployment["n_shards"] == 4
        assert len(deployment["shard_ids"]) == 4
        assert card.status == DeploymentStatus.DEPLOYED

    def test_get_model_status(self):
        service = NWTNModelService()
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama")
        service.register_model(card)
        status = service.get_model_status("nwtn-v1")
        assert status is not None
        assert status["model_name"] == "NWTN"

    def test_list_models(self):
        service = NWTNModelService()
        service.register_model(ModelCard(model_id="m1", model_name="M1", base_model="a"))
        service.register_model(ModelCard(model_id="m2", model_name="M2", base_model="b"))
        models = service.list_models()
        assert len(models) == 2

    def test_retire_model(self):
        service = NWTNModelService()
        card = ModelCard(model_id="nwtn-v1", model_name="NWTN", base_model="llama")
        service.register_model(card)
        assert service.retire_model("nwtn-v1")
        assert card.status == DeploymentStatus.RETIRED

    def test_get_nonexistent_model(self):
        service = NWTNModelService()
        assert service.get_model_status("nonexistent") is None

    def test_deploy_unregistered_fails(self):
        service = NWTNModelService()
        with pytest.raises(ValueError, match="not registered"):
            service.deploy_model("nonexistent")
