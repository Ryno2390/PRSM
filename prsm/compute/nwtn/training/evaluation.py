"""
Training Evaluation
===================

Dataset quality scoring and fine-tune readiness assessment
for the NWTN training pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DatasetQualityReport:
    """Quality assessment of a training corpus."""
    total_traces: int = 0
    valid_traces: int = 0
    route_coverage: Dict[str, int] = None  # route -> count
    avg_complexity: float = 0.0
    complexity_distribution: Dict[str, int] = None  # range -> count
    has_diverse_routes: bool = False
    has_sufficient_volume: bool = False
    has_quality_metrics: bool = False
    overall_score: float = 0.0  # 0-1
    recommendations: List[str] = None

    def __post_init__(self):
        if self.route_coverage is None:
            self.route_coverage = {}
        if self.complexity_distribution is None:
            self.complexity_distribution = {}
        if self.recommendations is None:
            self.recommendations = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "valid_traces": self.valid_traces,
            "route_coverage": self.route_coverage,
            "avg_complexity": round(self.avg_complexity, 3),
            "has_diverse_routes": self.has_diverse_routes,
            "has_sufficient_volume": self.has_sufficient_volume,
            "overall_score": round(self.overall_score, 3),
            "recommendations": self.recommendations,
        }


class TrainingEvaluator:
    """Evaluates training corpus quality and fine-tune readiness."""

    def __init__(self, min_traces: int = 100, min_routes: int = 2):
        self.min_traces = min_traces
        self.min_routes = min_routes

    def evaluate(self, traces: List[Dict[str, Any]]) -> DatasetQualityReport:
        """Assess training corpus quality."""
        report = DatasetQualityReport(total_traces=len(traces))

        if not traces:
            report.recommendations.append("Corpus is empty — run queries to collect traces")
            return report

        # Count valid traces
        required = {"query", "decomposition", "plan", "execution_result"}
        valid = [t for t in traces if required.issubset(set(t.keys()))]
        report.valid_traces = len(valid)

        # Route coverage
        routes = {}
        complexities = []
        for t in valid:
            route = t.get("plan", {}).get("route", "unknown")
            routes[route] = routes.get(route, 0) + 1
            comp = t.get("decomposition", {}).get("estimated_complexity", 0)
            complexities.append(comp)

        report.route_coverage = routes
        report.avg_complexity = sum(complexities) / len(complexities) if complexities else 0

        # Complexity distribution
        dist = {"low (0-0.3)": 0, "medium (0.3-0.7)": 0, "high (0.7-1.0)": 0}
        for c in complexities:
            if c < 0.3:
                dist["low (0-0.3)"] += 1
            elif c < 0.7:
                dist["medium (0.3-0.7)"] += 1
            else:
                dist["high (0.7-1.0)"] += 1
        report.complexity_distribution = dist

        # Quality checks
        report.has_diverse_routes = len(routes) >= self.min_routes
        report.has_sufficient_volume = len(valid) >= self.min_traces
        report.has_quality_metrics = all(
            t.get("execution_metrics", {}).get("elapsed_seconds", 0) > 0
            for t in valid[:10]  # Check first 10
        ) if valid else False

        # Overall score (0-1)
        score = 0.0
        if report.has_sufficient_volume:
            score += 0.4
        elif report.valid_traces > 0:
            score += 0.4 * (report.valid_traces / self.min_traces)

        if report.has_diverse_routes:
            score += 0.3
        elif len(routes) > 0:
            score += 0.3 * (len(routes) / self.min_routes)

        if report.has_quality_metrics:
            score += 0.2

        # Bonus for balanced complexity
        if dist.get("medium (0.3-0.7)", 0) > len(valid) * 0.3:
            score += 0.1

        report.overall_score = min(score, 1.0)

        # Recommendations
        if not report.has_sufficient_volume:
            report.recommendations.append(
                f"Need more traces: {report.valid_traces}/{self.min_traces} minimum"
            )
        if not report.has_diverse_routes:
            missing = {"direct_llm", "single_agent", "swarm"} - set(routes.keys())
            if missing:
                report.recommendations.append(
                    f"Missing route types: {', '.join(missing)}"
                )
        if dist.get("high (0.7-1.0)", 0) == 0:
            report.recommendations.append(
                "No high-complexity queries — add complex analysis tasks"
            )

        return report
