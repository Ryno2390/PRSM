"""
Tests for Teacher Model Training Infrastructure (Phase 1.3)

Tests the training infrastructure including:
- TrainingConfig configuration
- TeacherTrainer training loop
- Evaluation metrics
- Checkpoint management
- TrainableTeacher integration
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import training infrastructure
from prsm.compute.teachers.training_config import (
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
from prsm.compute.teachers.trainer import (
    TeacherTrainer,
    TrainingDataset,
    DataLoaderManager,
    ModelWrapper,
    CheckpointManager,
    MetricsTracker,
    EarlyStopping,
    create_teacher_trainer
)
from prsm.compute.teachers.teacher_model import (
    TrainableTeacher,
    create_trainable_teacher
)
from prsm.core.models import TeacherModel, ModelType


# === Fixtures ===

@pytest.fixture
def sample_training_examples():
    """Create sample training examples for testing"""
    examples = []
    for i in range(20):
        example = TrainingExample(
            example_id=f"example_{i}",
            input_text=f"Input text for example {i}",
            target_text=f"Target text for example {i}",
            domain="science" if i % 2 == 0 else "mathematics",
            difficulty=0.3 + (i % 5) * 0.15,
            reasoning_type="deductive" if i % 3 == 0 else "inductive"
        )
        examples.append(example)
    return examples


@pytest.fixture
def basic_training_config():
    """Create a basic training configuration for testing"""
    return TrainingConfig(
        model_name="test_teacher",
        model_type="test",
        strategy=TrainingStrategy.SUPERVISED,
        hyperparameters=Hyperparameters(
            learning_rate=1e-5,
            batch_size=4,
            epochs=2,
            max_grad_norm=1.0,
            warmup_steps=10
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=2,
            min_delta=0.001
        ),
        checkpoint=CheckpointConfig(
            checkpoint_dir=tempfile.mkdtemp(),
            checkpoint_strategy=CheckpointStrategy.BEST_ONLY
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            log_interval=1
        ),
        data=DataConfig(
            data_dir="data/nwtn_training/",
            validation_split=0.2,
            test_split=0.1
        ),
        backend_type="mock"
    )


@pytest.fixture
def teacher_model():
    """Create a sample teacher model for testing"""
    return TeacherModel(
        teacher_id=uuid4(),
        name="Test Teacher",
        specialization="science",
        model_type=ModelType.TEACHER,
        performance_score=0.85,
        curriculum_ids=[],
        learning_sessions=[]
    )


# === TrainingConfig Tests ===

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        assert config.model_name == "teacher_model"
        assert config.strategy == TrainingStrategy.SUPERVISED
        assert config.hyperparameters.learning_rate == 1e-5
        assert config.hyperparameters.batch_size == 8
        assert config.hyperparameters.epochs == 3
        assert config.backend_type == "mock"
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            model_name="custom_teacher",
            strategy=TrainingStrategy.DISTILLATION,
            hyperparameters=Hyperparameters(
                learning_rate=2e-5,
                batch_size=16,
                epochs=5
            )
        )
        
        assert config.model_name == "custom_teacher"
        assert config.strategy == TrainingStrategy.DISTILLATION
        assert config.hyperparameters.learning_rate == 2e-5
        assert config.hyperparameters.batch_size == 16
        assert config.hyperparameters.epochs == 5
    
    def test_config_serialization(self):
        """Test configuration serialization to dict"""
        config = TrainingConfig(model_name="test")
        config_dict = config.to_dict()
        
        assert "model_name" in config_dict
        assert "hyperparameters" in config_dict
        assert "early_stopping" in config_dict
        assert "checkpoint" in config_dict
        
        # Test round-trip
        restored_config = TrainingConfig.from_dict(config_dict)
        assert restored_config.model_name == config.model_name
    
    def test_config_json_serialization(self, tmp_path):
        """Test configuration serialization to JSON file"""
        config = TrainingConfig(model_name="json_test")
        json_path = tmp_path / "config.json"
        
        config.to_json(str(json_path))
        assert json_path.exists()
        
        restored_config = TrainingConfig.from_json(str(json_path))
        assert restored_config.model_name == "json_test"
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = TrainingConfig()
        issues = config.validate()
        assert len(issues) == 0
        
        # Invalid learning rate
        config.hyperparameters.learning_rate = -0.1
        issues = config.validate()
        assert any("learning rate" in issue.lower() for issue in issues)
        
        # Invalid batch size
        config.hyperparameters.learning_rate = 1e-5
        config.hyperparameters.batch_size = 0
        issues = config.validate()
        assert any("batch size" in issue.lower() for issue in issues)
    
    def test_distillation_config_validation(self):
        """Test validation for distillation strategy"""
        config = TrainingConfig(
            strategy=TrainingStrategy.DISTILLATION,
            teacher_model_id=None,  # Missing teacher
            student_model_id=None   # Missing student
        )
        
        issues = config.validate()
        assert any("teacher model" in issue.lower() for issue in issues)
        assert any("student model" in issue.lower() for issue in issues)


class TestHyperparameters:
    """Tests for Hyperparameters dataclass"""
    
    def test_default_hyperparameters(self):
        """Test default hyperparameter values"""
        hp = Hyperparameters()
        
        assert hp.learning_rate == 1e-5
        assert hp.batch_size == 8
        assert hp.epochs == 3
        assert hp.optimizer == OptimizerType.ADAMW
        assert hp.lr_schedule == LearningRateSchedule.LINEAR_WARMUP
    
    def test_custom_hyperparameters(self):
        """Test custom hyperparameter values"""
        hp = Hyperparameters(
            learning_rate=5e-5,
            batch_size=32,
            epochs=10,
            optimizer=OptimizerType.ADAM,
            lr_schedule=LearningRateSchedule.COSINE
        )
        
        assert hp.learning_rate == 5e-5
        assert hp.batch_size == 32
        assert hp.epochs == 10
        assert hp.optimizer == OptimizerType.ADAM
        assert hp.lr_schedule == LearningRateSchedule.COSINE


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass"""
    
    def test_checkpoint_path_generation(self):
        """Test checkpoint path generation"""
        config = CheckpointConfig(
            checkpoint_dir="models/test",
            checkpoint_prefix="model",
            include_timestamp=True,
            include_epoch=True
        )
        
        path = config.get_checkpoint_path(epoch=5, model_name="teacher")
        
        assert "models/test" in str(path)
        assert "model" in str(path)
        assert "teacher" in str(path)
        assert "epoch5" in str(path)
    
    def test_checkpoint_path_without_timestamp(self):
        """Test checkpoint path without timestamp"""
        config = CheckpointConfig(
            checkpoint_dir="models/test",
            include_timestamp=False,
            include_epoch=True
        )
        
        path = config.get_checkpoint_path(epoch=3)
        
        assert "epoch3" in str(path)


# === TrainingExample Tests ===

class TestTrainingExample:
    """Tests for TrainingExample dataclass"""
    
    def test_example_creation(self):
        """Test creating a training example"""
        example = TrainingExample(
            example_id="test_1",
            input_text="What is 2+2?",
            target_text="2+2 equals 4",
            domain="mathematics",
            difficulty=0.5
        )
        
        assert example.example_id == "test_1"
        assert example.input_text == "What is 2+2?"
        assert example.target_text == "2+2 equals 4"
        assert example.domain == "mathematics"
        assert example.difficulty == 0.5
    
    def test_example_serialization(self):
        """Test example serialization"""
        example = TrainingExample(
            example_id="test_2",
            input_text="Test input",
            target_text="Test output",
            metadata={"key": "value"}
        )
        
        example_dict = example.to_dict()
        restored = TrainingExample.from_dict(example_dict)
        
        assert restored.example_id == example.example_id
        assert restored.input_text == example.input_text
        assert restored.metadata == example.metadata


# === DataLoaderManager Tests ===

class TestDataLoaderManager:
    """Tests for DataLoaderManager"""
    
    @pytest.mark.asyncio
    async def test_load_synthetic_data(self, basic_training_config):
        """Test loading synthetic data when no files exist"""
        manager = DataLoaderManager(basic_training_config)
        
        train_data, val_data, test_data = await manager.load_training_data()
        
        # Should generate synthetic examples
        assert len(train_data) > 0
        assert isinstance(train_data[0], TrainingExample)
    
    @pytest.mark.asyncio
    async def test_data_splitting(self, basic_training_config, sample_training_examples):
        """Test data splitting into train/val/test"""
        manager = DataLoaderManager(basic_training_config)
        
        # Split data
        train_ratio = 1.0 - basic_training_config.data.validation_split - basic_training_config.data.test_split
        train_data, remaining = manager._split_data(sample_training_examples, train_ratio)
        
        assert len(train_data) < len(sample_training_examples)
        assert len(train_data) + len(remaining) == len(sample_training_examples)
    
    def test_create_dataloader(self, basic_training_config, sample_training_examples):
        """Test creating a data loader"""
        manager = DataLoaderManager(basic_training_config)
        
        dataloader = manager.create_dataloader(
            sample_training_examples,
            batch_size=4,
            shuffle=False
        )
        
        # Should return either a DataLoader or list of batches
        assert dataloader is not None


# === ModelWrapper Tests ===

class TestModelWrapper:
    """Tests for ModelWrapper"""
    
    def test_model_wrapper_creation(self, basic_training_config):
        """Test creating a model wrapper"""
        wrapper = ModelWrapper(config=basic_training_config)
        
        assert wrapper.config == basic_training_config
        assert wrapper.model is None
    
    def test_parameter_management(self, basic_training_config):
        """Test parameter get/set"""
        wrapper = ModelWrapper(config=basic_training_config)
        
        # Set parameters
        params = {"weight": [[0.1, 0.2], [0.3, 0.4]], "bias": [0.0, 0.0]}
        wrapper.set_parameters(params)
        
        # Get parameters
        retrieved = wrapper.get_parameters()
        assert retrieved == params
    
    def test_gradient_operations(self, basic_training_config):
        """Test gradient operations"""
        wrapper = ModelWrapper(config=basic_training_config)
        
        # Zero gradients
        wrapper.zero_gradients()
        assert wrapper.get_gradients() == {}
        
        # Set gradients
        wrapper._gradients = {"weight": [[0.01, 0.02], [0.03, 0.04]]}
        assert len(wrapper.get_gradients()) == 1


# === CheckpointManager Tests ===

class TestCheckpointManager:
    """Tests for CheckpointManager"""
    
    def test_checkpoint_manager_creation(self):
        """Test creating checkpoint manager"""
        config = CheckpointConfig(checkpoint_dir=tempfile.mkdtemp())
        manager = CheckpointManager(config)
        
        assert manager.config == config
        assert manager.checkpoint_dir.exists()
    
    def test_save_checkpoint(self, basic_training_config):
        """Test saving a checkpoint"""
        manager = CheckpointManager(basic_training_config.checkpoint)
        model = ModelWrapper(config=basic_training_config)
        model.set_parameters({"weight": [[0.1]], "bias": [0.0]})
        
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=None,
            epoch=1,
            step=100,
            metrics={"loss": 0.5}
        )
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, basic_training_config):
        """Test loading a checkpoint"""
        manager = CheckpointManager(basic_training_config.checkpoint)
        model = ModelWrapper(config=basic_training_config)
        model.set_parameters({"weight": [[0.1]], "bias": [0.0]})
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=None,
            epoch=1,
            step=100,
            metrics={"loss": 0.5}
        )
        
        # Load into new model
        new_model = ModelWrapper(config=basic_training_config)
        checkpoint_data = manager.load_checkpoint(checkpoint_path, new_model)
        
        assert checkpoint_data["epoch"] == 1
        assert checkpoint_data["step"] == 100
        assert new_model.get_parameters() == model.get_parameters()
    
    def test_checkpoint_retention_best_only(self, basic_training_config):
        """Test checkpoint retention with BEST_ONLY strategy"""
        basic_training_config.checkpoint.checkpoint_strategy = CheckpointStrategy.BEST_ONLY
        manager = CheckpointManager(basic_training_config.checkpoint)
        model = ModelWrapper(config=basic_training_config)
        
        # Save multiple checkpoints
        for i in range(3):
            manager.save_checkpoint(
                model=model,
                optimizer=None,
                epoch=i,
                step=i * 100,
                metrics={"loss": 0.5 - i * 0.1},
                is_best=(i == 2)  # Last one is best
            )
        
        # Should only keep best checkpoint
        assert len(manager.list_checkpoints()) == 1


# === MetricsTracker Tests ===

class TestMetricsTracker:
    """Tests for MetricsTracker"""
    
    def test_step_metrics(self):
        """Test adding step metrics"""
        tracker = MetricsTracker()
        
        tracker.add_step_metric("loss", 0.5, step=1)
        tracker.add_step_metric("accuracy", 0.8, step=1)
        
        assert len(tracker.step_history) == 1
        assert tracker.step_history[0]["metrics"]["loss"] == 0.5
    
    def test_epoch_metrics(self):
        """Test epoch metric aggregation"""
        tracker = MetricsTracker()
        
        tracker.add_epoch_metric("loss", 0.5, epoch=1)
        tracker.add_epoch_metric("loss", 0.4, epoch=1)
        tracker.add_epoch_metric("loss", 0.3, epoch=1)
        
        metrics = tracker.finalize_epoch(epoch=1)
        
        assert metrics["loss"] == pytest.approx(0.4)  # Average
        assert metrics["epoch"] == 1
    
    def test_best_metric(self):
        """Test finding best metric"""
        tracker = MetricsTracker()
        
        tracker.epoch_history = [
            {"loss": 0.5, "epoch": 0},
            {"loss": 0.3, "epoch": 1},
            {"loss": 0.4, "epoch": 2}
        ]
        
        best_loss, best_epoch = tracker.get_best_metric("loss", mode="min")
        assert best_loss == 0.3
        assert best_epoch == 1
    
    def test_perplexity_calculation(self):
        """Test perplexity calculation"""
        tracker = MetricsTracker()
        
        perplexity = tracker.compute_perplexity(loss=2.0)
        expected = 7.389  # e^2
        
        # Allow for floating point differences
        assert abs(perplexity - expected) < 0.01
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        tracker = MetricsTracker()
        
        predictions = [1, 0, 1, 1, 0]
        targets = [1, 0, 0, 1, 1]
        
        accuracy = tracker.compute_accuracy(predictions, targets)
        assert accuracy == pytest.approx(0.6)  # 3/5 correct
    
    def test_f1_calculation(self):
        """Test F1 score calculation"""
        tracker = MetricsTracker()
        
        predictions = [1, 0, 1, 1, 0]
        targets = [1, 0, 0, 1, 1]
        
        precision, recall, f1 = tracker.compute_f1(predictions, targets)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


# === EarlyStopping Tests ===

class TestEarlyStopping:
    """Tests for EarlyStopping"""
    
    def test_no_improvement(self):
        """Test early stopping with no improvement"""
        config = EarlyStoppingConfig(patience=2, min_delta=0.01)
        stopping = EarlyStopping(config)
        
        # First metric
        should_stop = stopping.should_stop(0.5, 0)
        assert not should_stop
        
        # No improvement
        should_stop = stopping.should_stop(0.5, 1)
        assert not should_stop
        
        # Still no improvement
        should_stop = stopping.should_stop(0.5, 2)
        assert should_stop  # Patience exceeded
    
    def test_with_improvement(self):
        """Test early stopping with improvement"""
        config = EarlyStoppingConfig(patience=2, min_delta=0.01)
        stopping = EarlyStopping(config)
        
        # Improving metrics
        should_stop = stopping.should_stop(0.5, 0)
        assert not should_stop
        
        should_stop = stopping.should_stop(0.4, 1)
        assert not should_stop
        
        should_stop = stopping.should_stop(0.3, 2)
        assert not should_stop  # Still improving
    
    def test_disabled(self):
        """Test early stopping when disabled"""
        config = EarlyStoppingConfig(enabled=False)
        stopping = EarlyStopping(config)
        
        # Should never stop when disabled
        for i in range(10):
            should_stop = stopping.should_stop(0.5, i)
            assert not should_stop
    
    def test_reset(self):
        """Test resetting early stopping"""
        config = EarlyStoppingConfig(patience=2)
        stopping = EarlyStopping(config)
        
        stopping.should_stop(0.5, 0)
        stopping.should_stop(0.5, 1)
        stopping.should_stop(0.5, 2)
        
        stopping.reset()
        
        assert stopping.best_value is None
        assert stopping.patience_counter == 0


# === TeacherTrainer Tests ===

class TestTeacherTrainer:
    """Tests for TeacherTrainer"""
    
    @pytest.mark.asyncio
    async def test_trainer_creation(self, basic_training_config):
        """Test creating a teacher trainer"""
        trainer = await create_teacher_trainer(basic_training_config)
        
        assert trainer.config == basic_training_config
        assert trainer.data_loader_manager is not None
        assert trainer.checkpoint_manager is not None
        assert trainer.metrics_tracker is not None
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, basic_training_config):
        """Test model initialization"""
        trainer = TeacherTrainer(basic_training_config)
        await trainer._initialize_model()
        
        assert trainer.model is not None
        assert trainer.model.get_parameters() is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Allow more time for training
    async def test_training_loop(self, basic_training_config, sample_training_examples):
        """Test running a training loop"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        trainer = TeacherTrainer(basic_training_config)
        
        result = await trainer.train(training_data=minimal_examples)
        
        assert result.success
        assert result.total_epochs > 0
        assert result.training_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_evaluation(self, basic_training_config, sample_training_examples):
        """Test model evaluation"""
        trainer = TeacherTrainer(basic_training_config)
        await trainer._initialize_model()
        
        result = await trainer.evaluate(
            model=trainer.model,
            eval_data=sample_training_examples[:5]
        )
        
        assert result.model_name == basic_training_config.model_name
        assert result.total_samples == 5
        assert 0 <= result.accuracy <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Allow more time for training
    async def test_checkpoint_saving(self, basic_training_config, sample_training_examples):
        """Test checkpoint saving during training"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        trainer = TeacherTrainer(basic_training_config)
        
        result = await trainer.train(training_data=minimal_examples)
        
        assert result.success
        assert result.final_checkpoint_path is not None
        assert Path(result.final_checkpoint_path).exists()


# === TrainableTeacher Tests ===

class TestTrainableTeacher:
    """Tests for TrainableTeacher"""
    
    @pytest.mark.asyncio
    async def test_trainable_teacher_creation(self, teacher_model, basic_training_config):
        """Test creating a trainable teacher"""
        teacher = await create_trainable_teacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        assert teacher.teacher_model == teacher_model
        assert teacher.training_config == basic_training_config
        assert not teacher.is_trained
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_trainable_teacher_training(self, teacher_model, basic_training_config, sample_training_examples):
        """Test training a trainable teacher"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        teacher = TrainableTeacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        result = await teacher.train(training_data=minimal_examples)
        
        assert result.success
        assert teacher.is_trained
        assert teacher.training_result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_trainable_teacher_evaluation(self, teacher_model, basic_training_config, sample_training_examples):
        """Test evaluating a trainable teacher"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        teacher = TrainableTeacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        # Train first
        await teacher.train(training_data=minimal_examples)
        
        # Evaluate
        eval_result = await teacher.evaluate(eval_data=minimal_examples[:3])
        
        assert eval_result is not None
        assert eval_result.total_samples == 3
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_trainable_teacher_checkpoint(self, teacher_model, basic_training_config, sample_training_examples):
        """Test checkpoint save/load for trainable teacher"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        teacher = TrainableTeacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        # Train
        await teacher.train(training_data=minimal_examples)
        
        # Save checkpoint
        checkpoint_path = teacher.save_checkpoint()
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        
        # Create new teacher and load checkpoint
        new_teacher = TrainableTeacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        loaded = new_teacher.load_checkpoint(checkpoint_path)
        assert loaded
        assert new_teacher.is_trained
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_trainable_teacher_curriculum(self, teacher_model, basic_training_config, sample_training_examples):
        """Test curriculum generation with trained model"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        teacher = TrainableTeacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        # Train
        await teacher.train(training_data=minimal_examples)
        
        # Generate curriculum
        curriculum = await teacher.generate_curriculum_with_training(
            student_model="student_1",
            domain="science"
        )
        
        assert curriculum is not None
        assert curriculum.domain == "science"


# === Integration Tests ===

class TestTrainingIntegration:
    """Integration tests for training infrastructure"""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(90)
    async def test_full_training_pipeline(self, teacher_model, basic_training_config, sample_training_examples):
        """Test complete training pipeline"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        # Create trainable teacher
        teacher = await create_trainable_teacher(
            teacher_model=teacher_model,
            training_config=basic_training_config
        )
        
        # Initialize trainer
        await teacher.initialize_trainer()
        assert teacher.trainer is not None
        
        # Train
        train_result = await teacher.train(training_data=minimal_examples)
        assert train_result.success
        
        # Evaluate
        eval_result = await teacher.evaluate(eval_data=minimal_examples[:3])
        assert eval_result.accuracy >= 0
        
        # Save checkpoint
        checkpoint_path = teacher.save_checkpoint()
        assert checkpoint_path is not None
        
        # Generate curriculum
        curriculum = await teacher.generate_curriculum_with_training(
            student_model="test_student",
            domain="science"
        )
        assert curriculum is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_training_with_validation(self, basic_training_config, sample_training_examples):
        """Test training with validation data"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:8]
        basic_training_config.hyperparameters.epochs = 1
        basic_training_config.hyperparameters.batch_size = 2
        
        # Split data
        split_idx = int(len(minimal_examples) * 0.75)
        train_data = minimal_examples[:split_idx]
        val_data = minimal_examples[split_idx:]
        
        trainer = TeacherTrainer(basic_training_config)
        result = await trainer.train(
            training_data=train_data,
            validation_data=val_data
        )
        
        assert result.success
        assert result.final_validation_loss is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_early_stopping_during_training(self, basic_training_config, sample_training_examples):
        """Test early stopping during training"""
        # Use minimal examples for faster testing
        minimal_examples = sample_training_examples[:5]
        basic_training_config.hyperparameters.epochs = 3
        basic_training_config.hyperparameters.batch_size = 2
        
        # Configure early stopping with low patience
        basic_training_config.early_stopping.patience = 1
        
        trainer = TeacherTrainer(basic_training_config)
        result = await trainer.train(training_data=minimal_examples)
        
        # Should stop early or complete successfully
        assert result.success


# === Configuration Tests ===

class TestTrainingStrategies:
    """Tests for different training strategies"""
    
    def test_supervised_strategy(self):
        """Test supervised training strategy"""
        config = TrainingConfig(strategy=TrainingStrategy.SUPERVISED)
        assert config.strategy == TrainingStrategy.SUPERVISED
    
    def test_distillation_strategy(self):
        """Test distillation training strategy"""
        config = TrainingConfig(
            strategy=TrainingStrategy.DISTILLATION,
            teacher_model_id="teacher_v1",
            student_model_id="student_v1"
        )
        assert config.strategy == TrainingStrategy.DISTILLATION
        assert config.teacher_model_id == "teacher_v1"
    
    def test_rlvr_strategy(self):
        """Test RLVR training strategy"""
        config = TrainingConfig(strategy=TrainingStrategy.RLVR)
        assert config.strategy == TrainingStrategy.RLVR


class TestOptimizerTypes:
    """Tests for optimizer types"""
    
    def test_adam_optimizer(self):
        """Test Adam optimizer configuration"""
        hp = Hyperparameters(optimizer=OptimizerType.ADAM)
        assert hp.optimizer == OptimizerType.ADAM
    
    def test_adamw_optimizer(self):
        """Test AdamW optimizer configuration"""
        hp = Hyperparameters(optimizer=OptimizerType.ADAMW)
        assert hp.optimizer == OptimizerType.ADAMW
    
    def test_sgd_optimizer(self):
        """Test SGD optimizer configuration"""
        hp = Hyperparameters(optimizer=OptimizerType.SGD)
        assert hp.optimizer == OptimizerType.SGD


class TestLearningRateSchedules:
    """Tests for learning rate schedules"""
    
    def test_linear_warmup_schedule(self):
        """Test linear warmup schedule"""
        hp = Hyperparameters(lr_schedule=LearningRateSchedule.LINEAR_WARMUP)
        assert hp.lr_schedule == LearningRateSchedule.LINEAR_WARMUP
    
    def test_cosine_schedule(self):
        """Test cosine schedule"""
        hp = Hyperparameters(lr_schedule=LearningRateSchedule.COSINE)
        assert hp.lr_schedule == LearningRateSchedule.COSINE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])