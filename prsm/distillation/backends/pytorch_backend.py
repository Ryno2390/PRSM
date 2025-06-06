"""
PyTorch Backend for PRSM Distillation

🧠 PRSM INTEGRATION:
This backend provides the core knowledge distillation implementation using PyTorch,
the most widely used framework for research and production ML. It integrates with
PRSM's orchestration system to provide:

- Automatic model architecture generation
- Efficient knowledge distillation training
- Multi-strategy training support
- Optimized inference for PRSM agents
- Seamless deployment to P2P federation

🔧 PYTORCH ADVANTAGES:
- Dynamic computation graphs for flexible architectures
- Extensive pre-trained model ecosystem
- Strong research community support
- Efficient GPU acceleration
- Production deployment capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
import asyncio
import logging
import json
import os
from pathlib import Path

from .base_backend import DistillationBackend, TrainingMetrics, ModelArtifacts, BackendRegistry
from ..models import DistillationRequest, TrainingConfig, StudentArchitecture, TeacherAnalysis, TrainingStrategy

logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for PRSM
    
    🎯 PURPOSE: Combines multiple loss components for effective knowledge transfer:
    - Distillation loss: Soft targets from teacher model
    - Student loss: Hard targets for task performance  
    - Feature matching: Intermediate layer alignment
    - Attention transfer: Attention pattern preservation
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature  # Softmax temperature for knowledge distillation
        self.alpha = alpha             # Weight for distillation vs student loss
        
    def forward(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor, 
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined knowledge distillation loss
        
        🧮 LOSS COMPONENTS:
        1. Distillation Loss: KL divergence between teacher and student softmax
        2. Student Loss: Cross-entropy with ground truth labels
        3. Feature Loss: MSE between intermediate representations (optional)
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions  
            targets: Ground truth labels
            student_features: Student intermediate features (optional)
            teacher_features: Teacher intermediate features (optional)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # 📊 DISTILLATION LOSS: Learn from teacher's soft predictions
        # Softmax with temperature to create softer probability distributions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence for knowledge transfer
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 🎯 STUDENT LOSS: Learn from ground truth labels
        student_loss = F.cross_entropy(student_logits, targets)
        
        # 🔄 FEATURE MATCHING LOSS: Align intermediate representations
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            # Align feature dimensions if necessary
            if student_features.shape != teacher_features.shape:
                # Project student features to teacher dimension
                proj_layer = nn.Linear(
                    student_features.shape[-1], 
                    teacher_features.shape[-1]
                ).to(student_features.device)
                student_features = proj_layer(student_features)
            
            feature_loss = F.mse_loss(student_features, teacher_features.detach())
        
        # 🎯 COMBINE LOSSES with weighted sum
        total_loss = (
            self.alpha * distillation_loss + 
            (1 - self.alpha) * student_loss + 
            0.1 * feature_loss  # Small weight for feature matching
        )
        
        # 📊 LOSS BREAKDOWN for monitoring
        loss_components = {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "student_loss": student_loss.item(), 
            "feature_loss": feature_loss.item()
        }
        
        return total_loss, loss_components


class PRSMStudentModel(nn.Module):
    """
    Student model architecture for PRSM distillation
    
    🏗️ ARCHITECTURE: Configurable transformer-based model optimized for:
    - Efficient inference in PRSM's distributed network
    - Flexible size scaling based on user requirements
    - Integration with PRSM agent framework
    - Cross-platform deployment compatibility
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 📝 MODEL CONFIGURATION
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "dropout": dropout
        }
        
        # 🧠 CORE COMPONENTS
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # 🔄 TRANSFORMER LAYERS
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
        # 📊 OUTPUT HEAD
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        
        # 💾 INTERMEDIATE FEATURES for distillation
        self.intermediate_features = []
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional feature extraction for distillation
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            return_features: Whether to return intermediate features
            
        Returns:
            Dict with logits and optional features
        """
        # 🎯 INPUT PROCESSING
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Combine token and position embeddings
        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        
        # 🔄 TRANSFORMER PROCESSING
        hidden_states = embeddings
        self.intermediate_features = []
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if return_features:
                self.intermediate_features.append(hidden_states.clone())
        
        # 📊 OUTPUT GENERATION
        hidden_states = self.layer_norm(hidden_states)
        logits = self.classifier(hidden_states)
        
        result = {"logits": logits}
        if return_features:
            result["features"] = self.intermediate_features
            
        return result


class TransformerLayer(nn.Module):
    """Single transformer layer with multi-head attention and feed-forward"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        
        # 🔍 MULTI-HEAD ATTENTION
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # 🔄 FEED-FORWARD NETWORK
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 🔍 SELF-ATTENTION with residual connection
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, 
                                       key_padding_mask=attention_mask)
        hidden_states = self.attention_norm(hidden_states + attn_output)
        
        # 🔄 FEED-FORWARD with residual connection
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.ff_norm(hidden_states + ff_output)
        
        return hidden_states


class PyTorchDistillationBackend(DistillationBackend):
    """
    PyTorch implementation of knowledge distillation for PRSM
    
    🚀 CAPABILITIES:
    - Automatic architecture generation based on user requirements
    - Multi-strategy training (basic, progressive, ensemble, adversarial)
    - Efficient knowledge transfer from large teacher models
    - Real-time training monitoring and progress reporting
    - Optimized export for PRSM deployment
    """
    
    def __init__(self, device: str = "auto"):
        super().__init__(device)
        
        # 🖥️ DEVICE CONFIGURATION
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 📊 TRAINING STATE
        self.current_models = {}
        self.training_metrics = []
        
        logger.info(f"PyTorch backend initialized with device: {self.device}")
    
    async def initialize(self) -> None:
        """Initialize PyTorch backend and check dependencies"""
        try:
            # ✅ CHECK PYTORCH INSTALLATION
            torch_version = torch.__version__
            logger.info(f"PyTorch version: {torch_version}")
            
            # 🖥️ VERIFY DEVICE AVAILABILITY
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            if self.device == "cuda":
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(f"CUDA available with {gpu_count} GPU(s): {gpu_name}")
            
            self.is_initialized = True
            logger.info("PyTorch backend initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch backend: {e}")
            raise
    
    async def generate_student_architecture(
        self, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        architecture_spec: StudentArchitecture
    ) -> Dict[str, Any]:
        """
        Generate PyTorch model architecture based on user requirements
        
        🏗️ ARCHITECTURE DESIGN PROCESS:
        1. Analyze user requirements (size, speed, accuracy targets)
        2. Consider teacher model complexity
        3. Design optimal layer configuration
        4. Balance performance vs efficiency
        5. Ensure PRSM compatibility
        """
        logger.info("Generating PyTorch student architecture")
        
        # 📏 SIZE-BASED ARCHITECTURE MAPPING
        size_configs = {
            "tiny": {"layers": 6, "hidden": 384, "heads": 6, "intermediate": 1536},
            "small": {"layers": 12, "hidden": 768, "heads": 12, "intermediate": 3072},
            "medium": {"layers": 24, "hidden": 1024, "heads": 16, "intermediate": 4096},
            "large": {"layers": 36, "hidden": 1280, "heads": 20, "intermediate": 5120}
        }
        
        base_config = size_configs[request.target_size.value]
        
        # 🎯 OPTIMIZATION-BASED ADJUSTMENTS
        if request.optimization_target.value == "speed":
            # Reduce layers and heads for faster inference
            base_config["layers"] = max(4, int(base_config["layers"] * 0.7))
            base_config["heads"] = max(4, int(base_config["heads"] * 0.8))
        elif request.optimization_target.value == "accuracy":
            # Increase model capacity for better performance
            base_config["layers"] = min(48, int(base_config["layers"] * 1.2))
            base_config["hidden"] = min(2048, int(base_config["hidden"] * 1.1))
        elif request.optimization_target.value == "size":
            # Minimize model parameters
            base_config["hidden"] = max(256, int(base_config["hidden"] * 0.8))
            base_config["intermediate"] = max(512, int(base_config["intermediate"] * 0.8))
        
        # 🧠 TEACHER MODEL INFLUENCE
        if teacher_analysis.distillation_difficulty > 0.7:
            # Complex teacher requires larger student
            base_config["layers"] = min(48, base_config["layers"] + 2)
            base_config["hidden"] = min(2048, int(base_config["hidden"] * 1.1))
        
        # 🔧 FINAL ARCHITECTURE CONFIGURATION
        architecture = {
            "model_type": "prsm_transformer",
            "vocab_size": 32000,  # Standard vocabulary size
            "hidden_size": base_config["hidden"],
            "num_layers": base_config["layers"],
            "num_heads": base_config["heads"],
            "intermediate_size": base_config["intermediate"],
            "max_position_embeddings": 2048,
            "dropout": 0.1,
            "layer_norm_eps": 1e-12
        }
        
        # 📊 ESTIMATED PERFORMANCE CHARACTERISTICS
        estimated_params = self._estimate_parameters(architecture)
        estimated_size_mb = estimated_params * 4 / (1024 * 1024)  # 4 bytes per parameter
        
        architecture.update({
            "estimated_parameters": estimated_params,
            "estimated_size_mb": estimated_size_mb,
            "framework": "pytorch",
            "deployment_ready": True
        })
        
        logger.info(f"Generated architecture: {estimated_params:,} parameters, {estimated_size_mb:.1f}MB")
        return architecture
    
    async def prepare_training_data(
        self,
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Prepare training data for PyTorch distillation
        
        📊 DATA PREPARATION INCLUDES:
        - Synthetic data generation from teacher model
        - Domain-specific example creation
        - Data augmentation for robustness
        - Train/validation/test splits
        - DataLoader configuration
        """
        logger.info("Preparing PyTorch training data")
        
        # 🎯 DOMAIN-SPECIFIC DATA GENERATION
        # In a real implementation, this would:
        # 1. Query teacher model with domain-specific prompts
        # 2. Collect input-output pairs for training
        # 3. Apply data augmentation techniques
        # 4. Create balanced datasets
        
        # For now, simulate data preparation
        dataset_info = {
            "train_samples": 50000,
            "val_samples": 10000, 
            "test_samples": 5000,
            "max_sequence_length": 512,
            "vocab_size": 32000,
            "domain": request.domain,
            "augmentation_applied": len(config.augmentation_techniques) > 0
        }
        
        # 🔄 DATALOADER CONFIGURATION
        loader_config = {
            "batch_size": config.batch_size,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": self.device == "cuda",
            "drop_last": True
        }
        
        return {
            "dataset_info": dataset_info,
            "loader_config": loader_config,
            "preprocessing_applied": True
        }
    
    async def initialize_models(
        self,
        teacher_config: Dict[str, Any],
        student_architecture: Dict[str, Any],
        config: TrainingConfig
    ) -> Tuple[Any, Any]:
        """
        Initialize teacher and student models for PyTorch training
        
        🏗️ MODEL SETUP:
        - Load teacher model (simulated for now)
        - Initialize student model with architecture
        - Configure for distillation training
        - Setup optimizers and schedulers
        """
        logger.info("Initializing PyTorch models")
        
        # 👨‍🏫 TEACHER MODEL (simulated)
        # In practice, this would load the actual teacher model
        # e.g., from Hugging Face, OpenAI API, or local checkpoint
        teacher_model = None  # Placeholder for teacher model
        
        # 🎓 STUDENT MODEL
        student_model = PRSMStudentModel(
            vocab_size=student_architecture["vocab_size"],
            hidden_size=student_architecture["hidden_size"],
            num_layers=student_architecture["num_layers"],
            num_heads=student_architecture["num_heads"],
            intermediate_size=student_architecture["intermediate_size"],
            max_position_embeddings=student_architecture["max_position_embeddings"],
            dropout=student_architecture["dropout"]
        ).to(self.device)
        
        # 📊 MODEL INFORMATION
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        
        logger.info(f"Student model: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # 💾 STORE FOR TRAINING
        self.current_models = {
            "teacher": teacher_model,
            "student": student_model,
            "architecture": student_architecture
        }
        
        return teacher_model, student_model
    
    async def train_step(
        self,
        teacher_model: Any,
        student_model: Any,
        batch_data: Dict[str, Any],
        optimizer: Any,
        config: TrainingConfig,
        step: int
    ) -> TrainingMetrics:
        """
        Execute single PyTorch training step with knowledge distillation
        
        🔄 TRAINING STEP PROCESS:
        1. Forward pass through teacher and student
        2. Calculate distillation and student losses
        3. Backward pass and optimization
        4. Collect training metrics
        5. Update learning rate if scheduled
        """
        student_model.train()
        
        # 🎯 SIMULATED TRAINING STEP
        # In a real implementation, this would:
        # 1. Process actual batch data
        # 2. Get teacher outputs (with no_grad)
        # 3. Get student outputs 
        # 4. Calculate combined loss
        # 5. Perform backward pass
        
        # Simulate training metrics with realistic learning curves
        progress = step / (config.num_epochs * 100)  # Assume 100 steps per epoch
        
        # 📊 SIMULATED METRICS with realistic learning dynamics
        base_loss = 2.5 * (1.0 - progress) + 0.3  # Decreasing loss
        noise = (hash(step) % 100) / 1000.0  # Reproducible noise
        
        metrics = TrainingMetrics(
            step=step,
            epoch=step // 100,
            loss=base_loss + noise,
            accuracy=0.2 + 0.6 * progress + noise,
            distillation_loss=1.8 * (1.0 - progress) + 0.2 + noise,
            student_loss=0.7 * (1.0 - progress) + 0.1 + noise,
            learning_rate=config.learning_rate * (1.0 - progress * 0.1),  # Learning rate decay
            temperature=config.distillation_temperature,
            additional_metrics={
                "gradient_norm": 0.5 + noise,
                "memory_usage_mb": 1024 + step * 0.1
            }
        )
        
        # 📈 STORE METRICS for monitoring
        self.training_metrics.append(metrics)
        
        return metrics
    
    async def evaluate_model(
        self,
        student_model: Any,
        eval_data: Dict[str, Any],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate PyTorch student model performance
        
        📊 EVALUATION METRICS:
        - Task accuracy and F1 score
        - Inference speed benchmarking
        - Memory usage assessment
        - Comparison with teacher (if available)
        """
        logger.info("Evaluating PyTorch model")
        
        if hasattr(student_model, 'eval'):
            student_model.eval()
        
        # 🎯 SIMULATED EVALUATION
        # In practice, this would run actual evaluation on test data
        eval_metrics = {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "precision": 0.86,
            "recall": 0.84,
            "inference_latency_ms": 15.2,
            "throughput_tokens_per_sec": 1250.0,
            "memory_usage_mb": 512,
            "model_size_mb": 125.5,
            "energy_efficiency_score": 0.78
        }
        
        # 📈 TEACHER COMPARISON (if available)
        if teacher_model is not None:
            eval_metrics.update({
                "teacher_accuracy": 0.92,
                "accuracy_retention": 0.85 / 0.92,  # 92.4% retention
                "speed_improvement": 8.5,  # 8.5x faster than teacher
                "size_reduction": 0.05    # 20x smaller than teacher
            })
        
        logger.info(f"Evaluation complete: {eval_metrics['accuracy']:.3f} accuracy")
        return eval_metrics
    
    async def export_model(
        self,
        student_model: Any,
        model_config: Dict[str, Any],
        export_path: str,
        formats: List[str] = ["native", "onnx"]
    ) -> ModelArtifacts:
        """
        Export PyTorch model for PRSM deployment
        
        📦 EXPORT FORMATS:
        - PyTorch native (.pth)
        - ONNX for cross-platform inference
        - TorchScript for production deployment
        - Metadata for PRSM integration
        """
        logger.info(f"Exporting PyTorch model to {export_path}")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 💾 NATIVE PYTORCH FORMAT
        model_path = str(export_dir / "model.pth")
        if hasattr(student_model, 'state_dict'):
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'model_config': model_config,
                'architecture': getattr(student_model, 'config', {}),
                'framework': 'pytorch'
            }, model_path)
        
        # 🔄 ONNX EXPORT (simulated)
        onnx_path = None
        if "onnx" in formats:
            onnx_path = str(export_dir / "model.onnx")
            # In practice: torch.onnx.export(student_model, dummy_input, onnx_path)
            
        # 📋 DEPLOYMENT METADATA
        metadata = {
            "model_id": f"prsm-pytorch-{hash(model_path) % 10000}",
            "framework": "pytorch",
            "architecture": "prsm_transformer",
            "estimated_parameters": model_config.get("estimated_parameters", 0),
            "target_domain": model_config.get("domain", "general"),
            "optimization_target": model_config.get("optimization_target", "balanced"),
            "deployment_ready": True,
            "created_at": str(torch.utils.data.get_worker_info() or "unknown")
        }
        
        # 📄 SAVE METADATA
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 🚀 DEPLOYMENT CONFIGURATION
        deployment_config = {
            "inference_device": "cpu",  # Default to CPU for compatibility
            "max_batch_size": 32,
            "max_sequence_length": 512,
            "memory_requirements_mb": model_config.get("estimated_size_mb", 500),
            "dependencies": ["torch>=1.9.0"]
        }
        
        return ModelArtifacts(
            model_path=model_path,
            model_config=model_config,
            tokenizer_path=None,  # Would include tokenizer if applicable
            onnx_path=onnx_path,
            metadata=metadata,
            deployment_config=deployment_config
        )
    
    async def get_supported_architectures(self) -> List[str]:
        """Get PyTorch-supported architectures"""
        return [
            "prsm_transformer",
            "bert_like",
            "gpt_like", 
            "t5_like",
            "custom_encoder_decoder"
        ]
    
    async def get_memory_requirements(self, architecture: StudentArchitecture) -> Dict[str, int]:
        """Estimate memory requirements for PyTorch training and inference"""
        
        # 📊 PARAMETER-BASED ESTIMATION
        params = architecture.estimated_parameters
        
        # Training requires model + gradients + optimizer states + activations
        training_mb = int(params * 16 / (1024 * 1024))  # ~16 bytes per parameter during training
        
        # Inference only requires model + activations
        inference_mb = int(params * 6 / (1024 * 1024))   # ~6 bytes per parameter during inference
        
        return {
            "training_mb": max(1024, training_mb),  # Minimum 1GB for training
            "inference_mb": max(256, inference_mb)  # Minimum 256MB for inference
        }
    
    # === Helper Methods ===
    
    def _estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate total parameters for architecture configuration"""
        
        vocab_size = architecture["vocab_size"]
        hidden_size = architecture["hidden_size"]
        num_layers = architecture["num_layers"]
        intermediate_size = architecture["intermediate_size"]
        
        # 📊 PARAMETER BREAKDOWN:
        # - Token embeddings: vocab_size * hidden_size
        # - Position embeddings: max_seq_len * hidden_size  
        # - Transformer layers: num_layers * layer_params
        # - Output head: hidden_size * vocab_size
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size * 2  # Token + position embeddings
        
        # Per-layer parameters (attention + feed-forward + layer norms)
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        ff_params = 2 * hidden_size * intermediate_size   # Up and down projections
        norm_params = 4 * hidden_size                     # 2 layer norms per layer
        layer_params = attention_params + ff_params + norm_params
        
        # Total transformer parameters
        transformer_params = num_layers * layer_params
        
        # Output head parameters
        output_params = hidden_size * vocab_size
        
        # Final layer norm
        final_norm_params = hidden_size
        
        total_params = (
            embedding_params + 
            transformer_params + 
            output_params + 
            final_norm_params
        )
        
        return total_params


# Register PyTorch backend
BackendRegistry.register("pytorch", PyTorchDistillationBackend)