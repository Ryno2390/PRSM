"""Tests for the data processing agent."""

import pytest
import json
import csv
import io

from prsm.compute.agents.data_processor import DataProcessor
from prsm.compute.agents.instruction_set import (
    AgentOp, AgentInstruction, InstructionManifest,
)


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def sample_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["state", "vehicle_type", "year", "count"])
    writer.writeheader()
    writer.writerows([
        {"state": "NC", "vehicle_type": "EV", "year": "2025", "count": "15000"},
        {"state": "NC", "vehicle_type": "ICE", "year": "2025", "count": "85000"},
        {"state": "NC", "vehicle_type": "EV", "year": "2026", "count": "22000"},
        {"state": "NC", "vehicle_type": "ICE", "year": "2026", "count": "78000"},
        {"state": "CA", "vehicle_type": "EV", "year": "2025", "count": "45000"},
        {"state": "CA", "vehicle_type": "EV", "year": "2026", "count": "58000"},
    ])
    return output.getvalue().encode()


@pytest.fixture
def sample_json():
    data = [
        {"state": "NC", "vehicle_type": "EV", "year": 2025, "count": 15000},
        {"state": "NC", "vehicle_type": "ICE", "year": 2025, "count": 85000},
        {"state": "NC", "vehicle_type": "EV", "year": 2026, "count": 22000},
    ]
    return json.dumps(data).encode()


class TestDataParsing:
    def test_parse_csv(self, processor, sample_csv):
        records = processor._parse_data(sample_csv)
        assert len(records) == 6
        assert records[0]["state"] == "NC"

    def test_parse_json(self, processor, sample_json):
        records = processor._parse_data(sample_json)
        assert len(records) == 3

    def test_parse_jsonl(self, processor):
        data = b'{"a": 1}\n{"a": 2}\n{"a": 3}'
        records = processor._parse_data(data)
        assert len(records) == 3

    def test_parse_empty(self, processor):
        assert processor._parse_data(b"") == []


class TestFilterOperation:
    def test_filter_by_state(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="NC vehicles",
            instructions=[AgentInstruction(op=AgentOp.FILTER, field="state", value="NC")],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["status"] == "success"
        assert result["count"] == 4  # 4 NC records

    def test_filter_by_type(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="EV only",
            instructions=[AgentInstruction(op=AgentOp.FILTER, field="vehicle_type", value="EV")],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["count"] == 4  # 4 EV records


class TestAggregationOperations:
    def test_count(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="total count",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["records"][0]["count"] == 6

    def test_sum(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="total vehicles",
            instructions=[AgentInstruction(op=AgentOp.SUM, field="count")],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["records"][0]["sum"] > 0

    def test_average(self, processor, sample_json):
        manifest = InstructionManifest(
            query="avg count",
            instructions=[AgentInstruction(op=AgentOp.AVERAGE, field="count")],
        )
        result = processor.execute(manifest, sample_json)
        assert result["records"][0]["average"] > 0

    def test_group_by(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="by state",
            instructions=[AgentInstruction(op=AgentOp.GROUP_BY, field="state")],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["count"] == 2  # CA and NC groups


class TestPipeline:
    def test_filter_then_count(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="NC EV count",
            instructions=[
                AgentInstruction(op=AgentOp.FILTER, field="state", value="NC"),
                AgentInstruction(op=AgentOp.FILTER, field="vehicle_type", value="EV"),
                AgentInstruction(op=AgentOp.COUNT),
            ],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["status"] == "success"
        assert result["records"][0]["count"] == 2  # NC + EV = 2 records

    def test_filter_group_count(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="EV by year",
            instructions=[
                AgentInstruction(op=AgentOp.FILTER, field="vehicle_type", value="EV"),
                AgentInstruction(op=AgentOp.GROUP_BY, field="year"),
            ],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["count"] == 2  # 2025 and 2026 groups

    def test_sort_and_limit(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="top 3",
            instructions=[
                AgentInstruction(op=AgentOp.SORT, field="count", params={"ascending": False}),
                AgentInstruction(op=AgentOp.LIMIT, value=3),
            ],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["count"] == 3

    def test_time_series(self, processor, sample_csv):
        manifest = InstructionManifest(
            query="trends by year",
            instructions=[
                AgentInstruction(op=AgentOp.FILTER, field="vehicle_type", value="EV"),
                AgentInstruction(op=AgentOp.TIME_SERIES, field="year", params={"metric_field": "count"}),
            ],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["count"] == 2  # 2025 and 2026

    def test_the_nada_scenario(self, processor, sample_csv):
        """The full NADA scenario from the development notes."""
        manifest = InstructionManifest(
            query="EV adoption trends in NC 2025 vs 2026",
            instructions=[
                AgentInstruction(op=AgentOp.FILTER, field="state", value="NC"),
                AgentInstruction(op=AgentOp.FILTER, field="vehicle_type", value="EV"),
                AgentInstruction(op=AgentOp.GROUP_BY, field="year"),
            ],
        )
        result = processor.execute(manifest, sample_csv)
        assert result["status"] == "success"
        # Should show 2 groups: 2025 and 2026
        assert result["count"] == 2
        years = {r["group"] for r in result["records"]}
        assert "2025" in years
        assert "2026" in years


class TestDeploymentConfig:
    def test_systemd_service_file_exists(self):
        from pathlib import Path
        assert Path("deploy/production/prsm-node.service").exists()

    def test_env_template_exists(self):
        from pathlib import Path
        assert Path("deploy/production/prsm.env.template").exists()

    def test_deployment_guide_exists(self):
        from pathlib import Path
        assert Path("deploy/production/DEPLOYMENT_GUIDE.md").exists()

    def test_bootstrap_env_override(self):
        import os
        from prsm.node.config import NodeConfig
        # Verify env var is read
        os.environ["PRSM_BOOTSTRAP_NODES"] = "ws://custom:9001,ws://custom2:9002"
        config = NodeConfig()
        # post_init should have picked up the env var
        assert config.bootstrap_nodes == ["ws://custom:9001", "ws://custom2:9002"]
        os.environ.pop("PRSM_BOOTSTRAP_NODES", None)
