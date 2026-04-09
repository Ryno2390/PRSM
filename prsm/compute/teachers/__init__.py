# PRSM Teachers Module
# teachers/ is entirely legacy (scheduled for deletion in PR 3).
# real_teacher_implementation imports agents/executors/model_executor (deleted in PR 1).
# All imports are wrapped in try/except to avoid breaking the package.

try:
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
except (ImportError, ModuleNotFoundError):
    DistilledTeacher = None  # type: ignore[assignment,misc]
    TeachingStrategy = None  # type: ignore[assignment,misc]
    StudentCapabilities = None  # type: ignore[assignment,misc]
    TeachingOutcome = None  # type: ignore[assignment,misc]
    TrainableTeacher = None  # type: ignore[assignment,misc]
    create_trainable_teacher = None  # type: ignore[assignment]
    create_production_teacher = None  # type: ignore[assignment]
    create_teacher_with_specialization = None  # type: ignore[assignment]

try:
    from .rlvr_engine import RLVREngine, VerifiableReward, TeacherWeights, RewardCalculation
except (ImportError, ModuleNotFoundError):
    RLVREngine = None  # type: ignore[assignment,misc]
    VerifiableReward = None  # type: ignore[assignment,misc]
    TeacherWeights = None  # type: ignore[assignment,misc]
    RewardCalculation = None  # type: ignore[assignment,misc]

try:
    from .curriculum import CurriculumGenerator, LearningGap, DifficultyProgression, CurriculumMetrics
except (ImportError, ModuleNotFoundError):
    CurriculumGenerator = None  # type: ignore[assignment,misc]
    LearningGap = None  # type: ignore[assignment,misc]
    DifficultyProgression = None  # type: ignore[assignment,misc]
    CurriculumMetrics = None  # type: ignore[assignment,misc]

# Training infrastructure (Phase 1.3)
try:
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
except (ImportError, ModuleNotFoundError):
    TrainingConfig = None  # type: ignore[assignment,misc]
    TrainingStrategy = None  # type: ignore[assignment,misc]
    Hyperparameters = None  # type: ignore[assignment,misc]
    EarlyStoppingConfig = None  # type: ignore[assignment,misc]
    CheckpointConfig = None  # type: ignore[assignment,misc]
    LoggingConfig = None  # type: ignore[assignment,misc]
    DataConfig = None  # type: ignore[assignment,misc]
    TrainingExample = None  # type: ignore[assignment,misc]
    TrainingResult = None  # type: ignore[assignment,misc]
    EvaluationResult = None  # type: ignore[assignment,misc]
    OptimizerType = None  # type: ignore[assignment,misc]
    LearningRateSchedule = None  # type: ignore[assignment,misc]
    CheckpointStrategy = None  # type: ignore[assignment,misc]

try:
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
except (ImportError, ModuleNotFoundError):
    TeacherTrainer = None  # type: ignore[assignment,misc]
    TrainingDataset = None  # type: ignore[assignment,misc]
    DataLoaderManager = None  # type: ignore[assignment,misc]
    ModelWrapper = None  # type: ignore[assignment,misc]
    CheckpointManager = None  # type: ignore[assignment,misc]
    MetricsTracker = None  # type: ignore[assignment,misc]
    EarlyStopping = None  # type: ignore[assignment,misc]
    create_teacher_trainer = None  # type: ignore[assignment]

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
