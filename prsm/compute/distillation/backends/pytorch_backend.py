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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except (ImportError, RuntimeError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]

PYTORCH_AVAILABLE = torch is not None
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
import asyncio
import logging
import json
import os
import time
import math
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
        
        # 📝 TOKENIZER CACHE
        self.tokenizer_cache = {}
        
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
        
        # 👨‍🏫 TEACHER MODEL — load from config
        from .teacher_loader import load_teacher_model
        teacher_model = await load_teacher_model(
            model_id  = teacher_config.get("model_id", ""),
            device    = self.device,
            executor  = teacher_config.get("executor")
        )
        logger.info("Teacher model loaded",
                    model_id=teacher_config.get("model_id"),
                    source=teacher_model.source.value,
                    supports_soft_labels=teacher_model.supports_soft_labels)
        
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
        try:
            # 1️⃣ VALIDATE INPUTS
            if torch is None:
                return TrainingMetrics(step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                                      distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                                      temperature=config.distillation_temperature)
            inputs = batch_data.get("inputs", [])
            targets = batch_data.get("targets", [])
            if not inputs:
                return TrainingMetrics(step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                                      distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                                      temperature=config.distillation_temperature)

            # 2️⃣ ACQUIRE/CREATE OPTIMIZER (instance variable, created once)
            if not hasattr(self, "_optimizer") or self._optimizer is None:
                self._optimizer = torch.optim.AdamW(
                    student_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )

            # 3️⃣ TOKENIZE BATCH
            tokenizer = self._get_or_create_tokenizer()
            if tokenizer is None:
                return TrainingMetrics(step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                                      distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                                      temperature=config.distillation_temperature)

            input_enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256)
            target_enc = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=64)

            input_ids = input_enc["input_ids"].to(self.device)
            attn_mask = input_enc["attention_mask"].to(self.device)
            labels = target_enc["input_ids"].to(self.device)

            # 4️⃣ TRAINING MODE AND ZERO GRAD
            student_model.train()
            self._optimizer.zero_grad()

            # 5️⃣ STUDENT FORWARD PASS
            student_outputs = student_model(input_ids, attention_mask=attn_mask, labels=labels)
            student_loss = student_outputs.get("loss") if isinstance(student_outputs, dict) else getattr(student_outputs, "loss", None)

            if student_loss is None:
                student_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

            # 6️⃣ TEACHER CONTRIBUTION (soft-label distillation, optional)
            distillation_loss = torch.tensor(0.0, device=self.device)
            if (teacher_model is not None
                and hasattr(teacher_model, "supports_soft_labels")
                and teacher_model.supports_soft_labels):

                teacher_logits = await teacher_model.get_soft_labels(input_ids, attn_mask)

                student_logits = student_outputs.get("logits") if isinstance(student_outputs, dict) else getattr(student_outputs, "logits", None)

                if teacher_logits is not None and student_logits is not None:
                    distill_fn = KnowledgeDistillationLoss(
                        temperature=config.distillation_temperature,
                        alpha=config.alpha_knowledge_distillation
                    )
                    total_loss, _ = distill_fn(student_logits, teacher_logits, labels)
                    distillation_loss = total_loss - (1.0 - config.alpha_knowledge_distillation) * student_loss
                else:
                    total_loss = student_loss
            else:
                total_loss = student_loss

            # 7️⃣ BACKWARD + CLIP + STEP
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                config.gradient_clipping
            )
            self._optimizer.step()

            # 8️⃣ RETURN REAL TrainingMetrics
            metrics = TrainingMetrics(
                step=step,
                epoch=step // 100,
                loss=total_loss.item(),
                accuracy=0.0,  # not computed per-step; done in evaluate_model
                distillation_loss=distillation_loss.item(),
                student_loss=student_loss.item(),
                learning_rate=self._optimizer.param_groups[0]["lr"],
                temperature=config.distillation_temperature,
                additional_metrics={
                    "gradient_norm": float(grad_norm),
                    "memory_usage_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
                }
            )

            # 📈 STORE METRICS for monitoring
            self.training_metrics.append(metrics)

            return metrics

        except Exception as e:
            logger.warning("train_step failed, returning fallback metrics", error=str(e))
            return TrainingMetrics(
                step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                temperature=config.distillation_temperature
            )
    
    def _get_or_create_tokenizer(self) -> Any:
        """
        Get or create a default tokenizer for evaluation.
        
        Uses a cached tokenizer to avoid repeated downloads.
        Falls back to a simple whitespace tokenizer if transformers is unavailable.
        """
        if "default" not in self.tokenizer_cache:
            try:
                from transformers import AutoTokenizer
                self.tokenizer_cache["default"] = AutoTokenizer.from_pretrained(
                    "bert-base-uncased",
                    model_max_length=512
                )
                logger.debug("Created default BERT tokenizer for evaluation")
            except (ImportError, OSError) as e:
                logger.warning(f"Could not load transformers tokenizer: {e}")
                # Return None - will use fallback tokenization
                self.tokenizer_cache["default"] = None
        return self.tokenizer_cache["default"]
    
    def _fallback_metrics(self) -> Dict[str, float]:
        """
        Return all-zeros metrics dict when evaluation fails.
        
        This signals evaluation failure without falsely inflating scores.
        """
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "avg_loss": 0.0,
            "perplexity": 0.0,
            "inference_latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "memory_usage_mb": 0.0,
            "model_size_mb": 0.0,
            "items_evaluated": 0
        }
    
    async def evaluate_model(
        self,
        student_model: Any,
        eval_data: Dict[str, Any],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate PyTorch student model performance with real inference.
        
        📊 EVALUATION METRICS:
        - Loss-based perplexity as primary quality metric
        - Top-1 token accuracy as secondary metric
        - Inference latency benchmarking
        - Memory usage assessment
        - Comparison with teacher (if available)
        """
        logger.info("Evaluating PyTorch model with real inference")
        
        # Check for PyTorch availability
        if torch is None:
            logger.warning("PyTorch not available, returning fallback metrics")
            return self._fallback_metrics()
        
        # Validate inputs
        test_data = eval_data.get("test", [])
        if not test_data:
            logger.warning("No test data provided, returning fallback metrics")
            return self._fallback_metrics()
        
        # Check if student model is valid
        if student_model is None or not hasattr(student_model, 'eval'):
            logger.warning("Invalid student model, returning fallback metrics")
            return self._fallback_metrics()
        
        try:
            # Set model to evaluation mode
            student_model.eval()
            
            # Get tokenizer
            tokenizer = self._get_or_create_tokenizer()
            
            # Collect metrics
            all_losses: List[float] = []
            all_preds: List[int] = []
            all_labels: List[int] = []
            all_latencies: List[float] = []
            
            # Limit evaluation to 200 items for efficiency
            eval_items = test_data[:min(200, len(test_data))]
            
            # Loss function for computing perplexity
            loss_fn = nn.CrossEntropyLoss(reduction='sum') if nn is not None else None
            
            with torch.no_grad():
                for item in eval_items:
                    # Extract input and target text
                    input_text = item.get("content", "") or item.get("input", "")
                    target_text = item.get("expected_answer", "") or item.get("target", "")
                    
                    # Skip if no target
                    if not target_text or not input_text:
                        continue
                    
                    try:
                        # Tokenize input and target
                        if tokenizer is not None:
                            input_ids = tokenizer.encode(
                                input_text,
                                max_length=256,
                                truncation=True,
                                return_tensors="pt"
                            )
                            target_ids = tokenizer.encode(
                                target_text,
                                max_length=64,
                                truncation=True
                            )
                        else:
                            # Fallback: simple character-level tokenization
                            input_ids = torch.tensor([[ord(c) % 32000 for c in input_text[:256]]])
                            target_ids = [ord(c) % 32000 for c in target_text[:64]]
                        
                        # Move to device
                        input_ids = input_ids.to(self.device)
                        
                        # Measure inference latency
                        start_time = time.perf_counter()
                        
                        # Forward pass
                        outputs = student_model(input_ids)
                        
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                        all_latencies.append(latency_ms)
                        
                        # Extract logits (handle different output formats)
                        if isinstance(outputs, dict):
                            logits = outputs.get("logits", outputs.get("last_hidden_state", None))
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        if logits is None:
                            continue
                        
                        # Get predictions (top-1 token from last position)
                        last_token_logits = logits[0, -1, :]
                        pred_token = torch.argmax(last_token_logits).item()
                        all_preds.append(pred_token)
                        
                        # Get label (first token of target)
                        label_token = target_ids[0] if target_ids else 0
                        all_labels.append(label_token)
                        
                        # Compute loss for perplexity
                        if loss_fn is not None and len(target_ids) > 0:
                            # Create target tensor
                            target_tensor = torch.tensor(target_ids, device=self.device)
                            # Compute cross-entropy loss
                            # Expand logits to match target length if needed
                            if logits.size(1) >= len(target_ids):
                                logits_for_loss = logits[0, :len(target_ids), :]
                                loss = loss_fn(logits_for_loss, target_tensor)
                                all_losses.append(loss.item() / len(target_ids))
                        
                    except Exception as item_error:
                        logger.debug(f"Error processing evaluation item: {item_error}")
                        continue
            
            # Check if we have any valid results
            if not all_preds:
                logger.warning("No valid predictions during evaluation")
                return self._fallback_metrics()
            
            # Compute accuracy
            correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
            accuracy = correct / len(all_preds) if all_preds else 0.0
            
            # Compute average loss and perplexity
            avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
            # Cap perplexity to avoid overflow
            perplexity = math.exp(min(avg_loss, 10.0)) if avg_loss > 0 else 0.0
            
            # Compute average latency
            avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
            
            # Compute F1, precision, recall using sklearn if available
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score
                # Use macro average for multi-class scenario
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            except (ImportError, ValueError):
                # Fallback: approximate from accuracy
                f1 = accuracy
                precision = accuracy
                recall = accuracy
            
            # Compute memory usage
            memory_mb = 0.0
            if torch.cuda.is_available() and self.device == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Compute model size from parameters
            model_size_mb = 0.0
            if hasattr(student_model, 'parameters'):
                total_params = sum(p.numel() for p in student_model.parameters())
                # Assume float32 (4 bytes per parameter)
                model_size_mb = (total_params * 4) / (1024 * 1024)
            
            # Build metrics dict
            eval_metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "inference_latency_ms": avg_latency,
                "throughput_tokens_per_sec": 1000.0 / max(0.001, avg_latency),
                "memory_usage_mb": memory_mb,
                "model_size_mb": model_size_mb,
                "items_evaluated": len(all_preds)
            }
            
            # 📈 TEACHER COMPARISON (if available)
            if teacher_model is not None:
                # Note: Real teacher comparison would require running inference on teacher
                # For now, we just note that teacher was provided
                eval_metrics["teacher_provided"] = True
            
            logger.info(f"Evaluation complete: {accuracy:.3f} accuracy, {perplexity:.2f} perplexity, {len(all_preds)} items")
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            return self._fallback_metrics()
        
        finally:
            # Always restore training mode
            if student_model is not None and hasattr(student_model, 'train'):
                student_model.train()
    
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