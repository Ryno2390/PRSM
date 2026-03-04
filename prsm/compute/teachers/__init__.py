# PRSM Teachers Module
from .teacher_model import (
    DistilledTeacher,
    TeachingStrategy,
    StudentCapabilities,
    TeachingOutcome,
    TrainableTeacher,
    create_trainable_teacher,
    create_production_teacher,
    create_teacher_with_specialization
)
from .rlvr_engine import RLVREngine, VerifiableReward, TeacherWeights, RewardCalculation
from .curriculum import CurriculumGenerator, LearningGap, DifficultyProgression, CurriculumMetrics

# Training infrastructure (Phase 1.3)
from .training_config import (
    TrainingConfig,
    TrainingStrategy,
    Hyperparameters,
    EarlyStoppingConfig,
    CheckpointConfig,
    LoggingConfig,
    DataConfig,
    TrainingExample,
    TrainingResult,
    EvaluationResult,
    OptimizerType,
    LearningRateSchedule,
    CheckpointStrategy
)
from .trainer import (
    TeacherTrainer,
    TrainingDataset,
    DataLoaderManager,
    ModelWrapper,
    CheckpointManager,
    MetricsTracker,
    EarlyStopping,
    create_teacher_trainer
)

__all__ = [
    # Teacher model
    "DistilledTeacher",
    "TeachingStrategy",
    "StudentCapabilities",
    "TeachingOutcome",
    "TrainableTeacher",
    "create_trainable_teacher",
    "create_production_teacher",
    "create_teacher_with_specialization",
    # RLVR
    "RLVREngine",
    "VerifiableReward",
    "TeacherWeights",
    "RewardCalculation",
    # Curriculum
    "CurriculumGenerator",
    "LearningGap",
    "DifficultyProgression",
    "CurriculumMetrics",
    # Training infrastructure
    "TrainingConfig",
    "TrainingStrategy",
    "Hyperparameters",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DataConfig",
    "TrainingExample",
    "TrainingResult",
    "EvaluationResult",
    "OptimizerType",
    "LearningRateSchedule",
    "CheckpointStrategy",
    # Trainer
    "TeacherTrainer",
    "TrainingDataset",
    "DataLoaderManager",
    "ModelWrapper",
    "CheckpointManager",
    "MetricsTracker",
    "EarlyStopping",
    "create_teacher_trainer"
]