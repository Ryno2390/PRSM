"""
PRSM Automated Distillation System
Democratizes creation of specialized distilled models for the PRSM ecosystem

This module provides comprehensive automated distillation capabilities:
- Teacher model analysis and knowledge extraction
- Optimal student architecture generation
- Multi-stage automated training pipelines
- Quality assurance and safety validation
- FTNS economic integration
- Marketplace deployment automation

The system enables any PRSM user to create high-quality, specialized
models without deep ML expertise, accelerating the growth of the
decentralized agent network.
"""

from .orchestrator import DistillationOrchestrator
from .models import (
    DistillationRequest, DistillationJob, DistillationStatus,
    TeacherAnalysis, StudentArchitecture, TrainingConfig,
    QualityMetrics, SafetyAssessment
)
from .knowledge_extractor import KnowledgeExtractor
from .architecture_generator import ArchitectureGenerator
from .training_pipeline import TrainingPipeline
from .evaluator import ModelEvaluator
from .safety_validator import SafetyValidator

__all__ = [
    "DistillationOrchestrator",
    "DistillationRequest",
    "DistillationJob", 
    "DistillationStatus",
    "TeacherAnalysis",
    "StudentArchitecture",
    "TrainingConfig",
    "QualityMetrics",
    "SafetyAssessment",
    "KnowledgeExtractor",
    "ArchitectureGenerator", 
    "TrainingPipeline",
    "ModelEvaluator",
    "SafetyValidator"
]