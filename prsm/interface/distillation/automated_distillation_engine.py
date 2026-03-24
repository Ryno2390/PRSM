"""
Re-export shim: prsm.interface.distillation.automated_distillation_engine
delegates to the canonical implementation in prsm.compute.distillation.
"""

from prsm.compute.distillation.automated_distillation_engine import (  # noqa: F401
    DistillationStrategy,
    ModelType,
    DistillationPhase,
    TeacherModel,
    StudentModel,
    DistillationDataset,
    DistillationJob,
    DistillationResult,
    AutomatedDistillationEngine,
    get_distillation_engine,
)
