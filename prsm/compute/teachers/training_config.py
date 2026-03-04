"""
Training Configuration for Teacher Models

Defines configuration dataclasses for training teacher models in the NWTN pipeline.
Supports various training strategies, hyperparameters, and checkpoint management.

🎯 PURPOSE:
- Centralized configuration for all training parameters
- Support for different training strategies (fine-tuning, distillation, RLVR)
- Checkpoint and logging configuration
- Early stopping and learning rate scheduling
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class TrainingStrategy(Enum):
    """Training strategy types for teacher models"""
    FINE_TUNING = "fine_tuning"
    DISTILLATION = "distillation"
    RLVR = "rlvr"  # Reinforcement Learning from Verifiable Rewards
    SUPERVISED = "supervised"
    FEW_SHOT = "few_shot"


class OptimizerType(Enum):
    """Supported optimizer types"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"


class LearningRateSchedule(Enum):
    """Learning rate schedule types"""
    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"


class CheckpointStrategy(Enum):
    """Checkpoint saving strategies"""
    BEST_ONLY = "best_only"  # Only save best model
    ALL = "all"  # Save all checkpoints
    LAST_N = "last_n"  # Keep last N checkpoints
    INTERVAL = "interval"  # Save at regular intervals


@dataclass
class Hyperparameters:
    """
    Training hyperparameters.
    
    Contains all hyperparameters needed for model training including
    learning rate, batch size, regularization, and optimization settings.
    """
    learning_rate: float = 1e-5
    batch_size: int = 8
    epochs: int = 3
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    
    # Optimizer settings
    optimizer: OptimizerType = OptimizerType.ADAMW
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Learning rate schedule
    lr_schedule: LearningRateSchedule = LearningRateSchedule.LINEAR_WARMUP
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 1000
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "optimizer": self.optimizer.value,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule.value,
            "lr_decay_rate": self.lr_decay_rate,
            "lr_decay_steps": self.lr_decay_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_amp": self.use_amp,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hyperparameters":
        """Create from dictionary"""
        return cls(
            learning_rate=data.get("learning_rate", 1e-5),
            batch_size=data.get("batch_size", 8),
            epochs=data.get("epochs", 3),
            max_grad_norm=data.get("max_grad_norm", 1.0),
            weight_decay=data.get("weight_decay", 0.01),
            warmup_steps=data.get("warmup_steps", 100),
            warmup_ratio=data.get("warmup_ratio", 0.1),
            optimizer=OptimizerType(data.get("optimizer", "adamw")),
            beta1=data.get("beta1", 0.9),
            beta2=data.get("beta2", 0.999),
            epsilon=data.get("epsilon", 1e-8),
            lr_schedule=LearningRateSchedule(data.get("lr_schedule", "linear_warmup")),
            lr_decay_rate=data.get("lr_decay_rate", 0.1),
            lr_decay_steps=data.get("lr_decay_steps", 1000),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
            use_amp=data.get("use_amp", True),
            dropout_rate=data.get("dropout_rate", 0.1),
            attention_dropout=data.get("attention_dropout", 0.1),
            hidden_dropout=data.get("hidden_dropout", 0.1)
        )


@dataclass
class EarlyStoppingConfig:
    """
    Early stopping configuration.
    
    Defines when to stop training early based on validation metrics.
    """
    enabled: bool = True
    patience: int = 3  # Number of epochs to wait before stopping
    min_delta: float = 0.001  # Minimum improvement to qualify as an improvement
    monitor_metric: str = "validation_loss"  # Metric to monitor
    mode: str = "min"  # "min" for loss, "max" for accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor_metric": self.monitor_metric,
            "mode": self.mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EarlyStoppingConfig":
        """Create from dictionary"""
        return cls(
            enabled=data.get("enabled", True),
            patience=data.get("patience", 3),
            min_delta=data.get("min_delta", 0.001),
            monitor_metric=data.get("monitor_metric", "validation_loss"),
            mode=data.get("mode", "min")
        )


@dataclass
class CheckpointConfig:
    """
    Checkpoint management configuration.
    
    Defines how and when to save model checkpoints during training.
    """
    checkpoint_dir: str = "models/nwtn_optimized/"
    checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.BEST_ONLY
    save_interval: int = 1  # Save every N epochs (for INTERVAL strategy)
    max_checkpoints: int = 5  # Maximum checkpoints to keep (for LAST_N strategy)
    save_optimizer_state: bool = True
    save_training_state: bool = True
    
    # Checkpoint naming
    checkpoint_prefix: str = "checkpoint"
    include_timestamp: bool = True
    include_epoch: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_strategy": self.checkpoint_strategy.value,
            "save_interval": self.save_interval,
            "max_checkpoints": self.max_checkpoints,
            "save_optimizer_state": self.save_optimizer_state,
            "save_training_state": self.save_training_state,
            "checkpoint_prefix": self.checkpoint_prefix,
            "include_timestamp": self.include_timestamp,
            "include_epoch": self.include_epoch
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointConfig":
        """Create from dictionary"""
        return cls(
            checkpoint_dir=data.get("checkpoint_dir", "models/nwtn_optimized/"),
            checkpoint_strategy=CheckpointStrategy(data.get("checkpoint_strategy", "best_only")),
            save_interval=data.get("save_interval", 1),
            max_checkpoints=data.get("max_checkpoints", 5),
            save_optimizer_state=data.get("save_optimizer_state", True),
            save_training_state=data.get("save_training_state", True),
            checkpoint_prefix=data.get("checkpoint_prefix", "checkpoint"),
            include_timestamp=data.get("include_timestamp", True),
            include_epoch=data.get("include_epoch", True)
        )
    
    def get_checkpoint_path(self, epoch: int, model_name: str = "teacher") -> Path:
        """Generate checkpoint file path"""
        parts = [self.checkpoint_prefix, model_name]
        
        if self.include_epoch:
            parts.append(f"epoch{epoch}")
        
        if self.include_timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        filename = "_".join(parts) + ".pt"
        return Path(self.checkpoint_dir) / filename


@dataclass
class LoggingConfig:
    """
    Training logging configuration.
    
    Defines how training progress is logged and tracked.
    """
    log_dir: str = "logs/training/"
    log_level: str = "INFO"
    log_interval: int = 10  # Log every N steps
    log_to_console: bool = True
    log_to_file: bool = True
    log_to_tensorboard: bool = False
    log_to_wandb: bool = False
    
    # Metrics to track
    track_memory: bool = True
    track_gpu_utilization: bool = True
    track_gradients: bool = False
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "log_dir": self.log_dir,
            "log_level": self.log_level,
            "log_interval": self.log_interval,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
            "log_to_tensorboard": self.log_to_tensorboard,
            "log_to_wandb": self.log_to_wandb,
            "track_memory": self.track_memory,
            "track_gpu_utilization": self.track_gpu_utilization,
            "track_gradients": self.track_gradients,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """Create from dictionary"""
        return cls(
            log_dir=data.get("log_dir", "logs/training/"),
            log_level=data.get("log_level", "INFO"),
            log_interval=data.get("log_interval", 10),
            log_to_console=data.get("log_to_console", True),
            log_to_file=data.get("log_to_file", True),
            log_to_tensorboard=data.get("log_to_tensorboard", False),
            log_to_wandb=data.get("log_to_wandb", False),
            track_memory=data.get("track_memory", True),
            track_gpu_utilization=data.get("track_gpu_utilization", True),
            track_gradients=data.get("track_gradients", False),
            experiment_name=data.get("experiment_name"),
            run_name=data.get("run_name")
        )


@dataclass
class DataConfig:
    """
    Training data configuration.
    
    Defines how training data is loaded, processed, and split.
    """
    data_dir: str = "data/nwtn_training/"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Data processing
    max_sequence_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"  # "max_length", "longest", "do_not_pad"
    
    # Data augmentation
    augmentation_enabled: bool = False
    augmentation_ratio: float = 0.1  # Ratio of augmented samples
    
    # Data splitting (if single file)
    validation_split: float = 0.1
    test_split: float = 0.1
    shuffle_data: bool = True
    seed: int = 42
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data_dir": self.data_dir,
            "train_file": self.train_file,
            "validation_file": self.validation_file,
            "test_file": self.test_file,
            "max_sequence_length": self.max_sequence_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "augmentation_enabled": self.augmentation_enabled,
            "augmentation_ratio": self.augmentation_ratio,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "shuffle_data": self.shuffle_data,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary"""
        return cls(
            data_dir=data.get("data_dir", "data/nwtn_training/"),
            train_file=data.get("train_file"),
            validation_file=data.get("validation_file"),
            test_file=data.get("test_file"),
            max_sequence_length=data.get("max_sequence_length", 2048),
            truncation=data.get("truncation", True),
            padding=data.get("padding", "max_length"),
            augmentation_enabled=data.get("augmentation_enabled", False),
            augmentation_ratio=data.get("augmentation_ratio", 0.1),
            validation_split=data.get("validation_split", 0.1),
            test_split=data.get("test_split", 0.1),
            shuffle_data=data.get("shuffle_data", True),
            seed=data.get("seed", 42),
            num_workers=data.get("num_workers", 4),
            pin_memory=data.get("pin_memory", True),
            prefetch_factor=data.get("prefetch_factor", 2)
        )


@dataclass
class TrainingConfig:
    """
    Complete training configuration for teacher models.
    
    This is the main configuration class that combines all aspects of training:
    - Hyperparameters
    - Early stopping
    - Checkpoint management
    - Logging
    - Data configuration
    
    Usage:
        config = TrainingConfig(
            model_name="llama3.1_teacher",
            hyperparameters=Hyperparameters(learning_rate=2e-5, epochs=5),
            strategy=TrainingStrategy.DISTILLATION
        )
        
        # Or load from file
        config = TrainingConfig.from_json("config.json")
    """
    # Model identification
    model_name: str = "teacher_model"
    model_type: str = "base"  # Base model type (e.g., "llama", "mistral", "command-r")
    strategy: TrainingStrategy = TrainingStrategy.SUPERVISED
    
    # Sub-configurations
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Backend configuration
    backend_type: str = "mock"  # "mock", "openai", "anthropic", "local"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    
    # Training control
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    deterministic: bool = False
    
    # Evaluation
    evaluation_strategy: str = "epoch"  # "epoch", "steps", "no"
    evaluation_steps: int = 500  # If evaluation_strategy is "steps"
    
    # Model-specific settings
    teacher_model_id: Optional[str] = None  # For distillation
    student_model_id: Optional[str] = None  # For distillation
    temperature: float = 1.0  # For distillation
    alpha: float = 0.5  # Distillation loss weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "strategy": self.strategy.value,
            "hyperparameters": self.hyperparameters.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "logging": self.logging.to_dict(),
            "data": self.data.to_dict(),
            "backend_type": self.backend_type,
            "device": self.device,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "evaluation_strategy": self.evaluation_strategy,
            "evaluation_steps": self.evaluation_steps,
            "teacher_model_id": self.teacher_model_id,
            "student_model_id": self.student_model_id,
            "temperature": self.temperature,
            "alpha": self.alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary"""
        return cls(
            model_name=data.get("model_name", "teacher_model"),
            model_type=data.get("model_type", "base"),
            strategy=TrainingStrategy(data.get("strategy", "supervised")),
            hyperparameters=Hyperparameters.from_dict(data.get("hyperparameters", {})),
            early_stopping=EarlyStoppingConfig.from_dict(data.get("early_stopping", {})),
            checkpoint=CheckpointConfig.from_dict(data.get("checkpoint", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            backend_type=data.get("backend_type", "mock"),
            device=data.get("device", "auto"),
            resume_from_checkpoint=data.get("resume_from_checkpoint"),
            seed=data.get("seed", 42),
            deterministic=data.get("deterministic", False),
            evaluation_strategy=data.get("evaluation_strategy", "epoch"),
            evaluation_steps=data.get("evaluation_steps", 500),
            teacher_model_id=data.get("teacher_model_id"),
            student_model_id=data.get("student_model_id"),
            temperature=data.get("temperature", 1.0),
            alpha=data.get("alpha", 0.5)
        )
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Validate hyperparameters
        if self.hyperparameters.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        if self.hyperparameters.learning_rate > 1:
            issues.append("Learning rate > 1 is unusually high")
        if self.hyperparameters.batch_size <= 0:
            issues.append("Batch size must be positive")
        if self.hyperparameters.epochs <= 0:
            issues.append("Epochs must be positive")
        
        # Validate early stopping
        if self.early_stopping.patience <= 0:
            issues.append("Early stopping patience must be positive")
        
        # Validate checkpoint config
        if self.checkpoint.max_checkpoints <= 0:
            issues.append("Max checkpoints must be positive")
        
        # Validate data config
        if self.data.validation_split < 0 or self.data.validation_split > 1:
            issues.append("Validation split must be between 0 and 1")
        if self.data.test_split < 0 or self.data.test_split > 1:
            issues.append("Test split must be between 0 and 1")
        if self.data.validation_split + self.data.test_split >= 1:
            issues.append("Validation + test split must be less than 1")
        
        # Validate distillation settings
        if self.strategy == TrainingStrategy.DISTILLATION:
            if not self.teacher_model_id:
                issues.append("Teacher model ID required for distillation")
            if not self.student_model_id:
                issues.append("Student model ID required for distillation")
            if self.alpha < 0 or self.alpha > 1:
                issues.append("Distillation alpha must be between 0 and 1")
        
        return issues


@dataclass
class TrainingExample:
    """
    Single training example for teacher model training.
    
    Represents one training sample with input, target, and metadata.
    """
    example_id: str
    input_text: str
    target_text: str
    domain: str = "general"
    difficulty: float = 0.5
    reasoning_type: str = "deductive"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "example_id": self.example_id,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "reasoning_type": self.reasoning_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """Create from dictionary"""
        return cls(
            example_id=data["example_id"],
            input_text=data["input_text"],
            target_text=data["target_text"],
            domain=data.get("domain", "general"),
            difficulty=data.get("difficulty", 0.5),
            reasoning_type=data.get("reasoning_type", "deductive"),
            metadata=data.get("metadata", {})
        )


@dataclass
class TrainingResult:
    """
    Result of a training run.
    
    Contains metrics, checkpoints, and training history.
    """
    success: bool
    model_name: str
    total_epochs: int
    total_steps: int
    training_time_seconds: float
    
    # Final metrics
    final_train_loss: float
    final_validation_loss: Optional[float] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Best metrics
    best_validation_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    
    # Checkpoint info
    final_checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    
    # Training history
    epoch_history: List[Dict[str, float]] = field(default_factory=list)
    step_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "model_name": self.model_name,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "training_time_seconds": self.training_time_seconds,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "final_metrics": self.final_metrics,
            "best_validation_loss": self.best_validation_loss,
            "best_epoch": self.best_epoch,
            "final_checkpoint_path": self.final_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "epoch_history": self.epoch_history,
            "step_history": self.step_history,
            "error_message": self.error_message
        }


@dataclass
class EvaluationResult:
    """
    Result of model evaluation.
    
    Contains metrics computed on evaluation dataset.
    """
    model_name: str
    dataset_name: str
    total_samples: int
    
    # Core metrics
    accuracy: float
    perplexity: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Domain-specific metrics
    domain_scores: Dict[str, float] = field(default_factory=dict)
    
    # Reasoning metrics
    reasoning_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Additional metrics
    average_latency_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # Per-sample results (optional)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "domain_scores": self.domain_scores,
            "reasoning_accuracy": self.reasoning_accuracy,
            "average_latency_ms": self.average_latency_ms,
            "total_time_seconds": self.total_time_seconds,
            "sample_results": self.sample_results
        }