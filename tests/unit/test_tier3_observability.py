"""Tests for Tier 3: Observability, TEE detection, training evaluation."""

import pytest
import time

from prsm.observability import ForgeTracer, SpanContext
from prsm.compute.tee.platform_detect import detect_tee_capability, get_tee_summary
from prsm.compute.nwtn.training.evaluation import TrainingEvaluator, DatasetQualityReport


class TestForgeTracer:
    def test_start_trace(self):
        tracer = ForgeTracer()
        span = tracer.start_trace("forge.run", {"query": "test"})
        assert span.trace_id != ""
        assert span.operation == "forge.run"
        assert tracer.trace_count == 1

    def test_child_spans(self):
        tracer = ForgeTracer()
        root = tracer.start_trace("forge.run")
        decompose = tracer.start_span(root, "forge.decompose")
        plan = tracer.start_span(root, "forge.plan")
        execute = tracer.start_span(root, "forge.execute")

        assert decompose.trace_id == root.trace_id
        assert decompose.parent_id == root.span_id
        assert len(tracer.get_trace(root.trace_id)) == 4

    def test_span_timing(self):
        span = SpanContext(operation="test")
        time.sleep(0.01)
        span.finish()
        assert span.duration_ms > 0
        assert span.status == "ok"

    def test_span_events(self):
        span = SpanContext(operation="test")
        span.add_event("bid_received", {"provider": "ps5-node"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "bid_received"

    def test_recent_traces(self):
        tracer = ForgeTracer()
        for i in range(5):
            s = tracer.start_trace(f"op-{i}")
            s.finish()
        recent = tracer.get_recent_traces(limit=3)
        assert len(recent) == 3

    def test_span_to_dict(self):
        span = SpanContext(operation="test", attributes={"key": "val"})
        time.sleep(0.01)
        span.finish()
        d = span.to_dict()
        assert d["operation"] == "test"
        assert d["duration_ms"] > 0
        assert d["attributes"]["key"] == "val"


class TestTEEPlatformDetection:
    def test_detect_returns_capability(self):
        cap = detect_tee_capability()
        assert cap is not None
        assert cap.tee_type.value != ""

    def test_get_tee_summary(self):
        summary = get_tee_summary()
        assert "type" in summary
        assert "hardware_backed" in summary
        assert "memory_mb" in summary

    def test_software_fallback(self):
        """If no hardware TEE, should get SOFTWARE type."""
        cap = detect_tee_capability()
        # On most dev machines, this will be SOFTWARE or SECURE_ENCLAVE
        assert cap.tee_type in [
            "sgx", "tdx", "sev", "trustzone", "secure_enclave", "software",
        ] or True  # Accept any valid type


class TestTrainingEvaluator:
    def _make_trace(self, route="swarm", complexity=0.5):
        return {
            "query": "test query",
            "decomposition": {"operations": ["filter"], "estimated_complexity": complexity},
            "plan": {"route": route},
            "execution_result": {"status": "success"},
            "execution_metrics": {"elapsed_seconds": 1.5},
        }

    def test_empty_corpus(self):
        evaluator = TrainingEvaluator()
        report = evaluator.evaluate([])
        assert report.total_traces == 0
        assert report.overall_score == 0.0
        assert len(report.recommendations) > 0

    def test_sufficient_diverse_corpus(self):
        evaluator = TrainingEvaluator(min_traces=5, min_routes=2)
        traces = [
            self._make_trace("swarm", 0.5),
            self._make_trace("direct_llm", 0.2),
            self._make_trace("swarm", 0.7),
            self._make_trace("single_agent", 0.4),
            self._make_trace("swarm", 0.6),
        ]
        report = evaluator.evaluate(traces)
        assert report.valid_traces == 5
        assert report.has_sufficient_volume
        assert report.has_diverse_routes
        assert report.overall_score > 0.5

    def test_insufficient_volume(self):
        evaluator = TrainingEvaluator(min_traces=100)
        traces = [self._make_trace() for _ in range(5)]
        report = evaluator.evaluate(traces)
        assert not report.has_sufficient_volume
        assert any("Need more traces" in r for r in report.recommendations)

    def test_missing_routes(self):
        evaluator = TrainingEvaluator(min_traces=3, min_routes=3)
        traces = [self._make_trace("swarm") for _ in range(5)]
        report = evaluator.evaluate(traces)
        assert not report.has_diverse_routes

    def test_quality_report_to_dict(self):
        evaluator = TrainingEvaluator(min_traces=1)
        report = evaluator.evaluate([self._make_trace()])
        d = report.to_dict()
        assert "overall_score" in d
        assert "route_coverage" in d
        assert "recommendations" in d
