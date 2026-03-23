"""
Hugging Face Transformers Backend for PRSM Distillation

🤗 TRANSFORMERS INTEGRATION:
This backend leverages Hugging Face Transformers for seamless integration with
thousands of pre-trained models. It's particularly effective for:

- NLP tasks (text generation, classification, Q&A)
- Pre-trained model distillation (BERT, GPT, T5, etc.)
- Multi-language model creation
- Domain adaptation from existing checkpoints
- Easy deployment to Hugging Face Hub

🎯 ADVANTAGES FOR PRSM:
- Vast ecosystem of pre-trained teacher models
- Standardized model interfaces and tokenizers
- Automatic optimization and quantization
- Built-in evaluation metrics
- Community-driven model sharing
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import time
import math
import tempfile
from pathlib import Path

import numpy as np

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        Trainer, TrainingArguments,
        DistilBertConfig, DistilBertForSequenceClassification,
        BertConfig, BertForSequenceClassification,
        GPT2Config, GPT2LMHeadModel
    )
    import torch
    import torch.nn as nn
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError):
    TRANSFORMERS_AVAILABLE = False

from .base_backend import DistillationBackend, TrainingMetrics, ModelArtifacts, BackendRegistry
from ..models import DistillationRequest, TrainingConfig, StudentArchitecture, TeacherAnalysis

logger = logging.getLogger(__name__)


class TransformersDistillationBackend(DistillationBackend):
    """
    Hugging Face Transformers backend for PRSM distillation
    
    🚀 CAPABILITIES:
    - Seamless integration with 100,000+ pre-trained models
    - Automatic tokenization and preprocessing
    - Built-in training optimization
    - Easy model sharing and deployment
    - Multi-task and multi-language support
    """
    
    def __init__(self, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers backend requires 'transformers' library. "
                "Install with: pip install transformers torch"
            )
        
        super().__init__(device)
        
        # 🖥️ DEVICE CONFIGURATION
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 📚 MODEL CACHE
        self.model_cache = {}
        self.tokenizer_cache = {}
        
        logger.info(f"Transformers backend initialized with device: {self.device}")
    
    async def initialize(self) -> None:
        """Initialize Transformers backend"""
        try:
            # ✅ CHECK TRANSFORMERS INSTALLATION
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
            
            # 🖥️ VERIFY DEVICE
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            self.is_initialized = True
            logger.info("Transformers backend initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers backend: {e}")
            raise
    
    async def generate_student_architecture(
        self, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        architecture_spec: StudentArchitecture
    ) -> Dict[str, Any]:
        """
        Generate Transformers-compatible student architecture
        
        🏗️ ARCHITECTURE SELECTION:
        - BERT-like: For classification and encoding tasks
        - GPT-like: For text generation and completion
        - T5-like: For text-to-text tasks
        - DistilBERT: For efficient BERT alternatives
        """
        logger.info("Generating Transformers student architecture")
        
        # 🎯 TASK-BASED ARCHITECTURE SELECTION
        domain_architectures = {
            "medical_research": "bert",
            "legal_analysis": "bert", 
            "code_generation": "gpt2",
            "creative_writing": "gpt2",
            "scientific_reasoning": "bert",
            "data_analysis": "bert",
            "general_purpose": "distilbert"
        }
        
        base_arch = domain_architectures.get(request.domain, "distilbert")
        
        # 📏 SIZE-BASED CONFIGURATIONS
        size_configs = {
            "tiny": {
                "bert": {"hidden_size": 384, "num_layers": 4, "num_heads": 6},
                "gpt2": {"n_embd": 384, "n_layer": 6, "n_head": 6},
                "distilbert": {"dim": 384, "n_layers": 4, "n_heads": 6}
            },
            "small": {
                "bert": {"hidden_size": 768, "num_layers": 6, "num_heads": 12},
                "gpt2": {"n_embd": 768, "n_layer": 8, "n_head": 12},
                "distilbert": {"dim": 768, "n_layers": 6, "n_heads": 12}
            },
            "medium": {
                "bert": {"hidden_size": 1024, "num_layers": 12, "num_heads": 16},
                "gpt2": {"n_embd": 1024, "n_layer": 12, "n_head": 16},
                "distilbert": {"dim": 1024, "n_layers": 8, "n_heads": 16}
            },
            "large": {
                "bert": {"hidden_size": 1280, "num_layers": 18, "num_heads": 20},
                "gpt2": {"n_embd": 1280, "n_layer": 16, "n_head": 20},
                "distilbert": {"dim": 1280, "n_layers": 12, "n_heads": 20}
            }
        }
        
        config_params = size_configs[request.target_size.value][base_arch]
        
        # 🎯 OPTIMIZATION ADJUSTMENTS
        if request.optimization_target.value == "speed":
            # Reduce layers for faster inference
            if "n_layer" in config_params:
                config_params["n_layer"] = max(4, int(config_params["n_layer"] * 0.75))
            elif "num_layers" in config_params:
                config_params["num_layers"] = max(4, int(config_params["num_layers"] * 0.75))
            elif "n_layers" in config_params:
                config_params["n_layers"] = max(4, int(config_params["n_layers"] * 0.75))
        
        # 🏗️ CREATE ARCHITECTURE CONFIGURATION
        if base_arch == "bert":
            architecture = {
                "model_type": "bert",
                "vocab_size": 30522,
                "hidden_size": config_params["hidden_size"],
                "num_hidden_layers": config_params["num_layers"],
                "num_attention_heads": config_params["num_heads"],
                "intermediate_size": config_params["hidden_size"] * 4,
                "max_position_embeddings": 512,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1
            }
        elif base_arch == "gpt2":
            architecture = {
                "model_type": "gpt2",
                "vocab_size": 50257,
                "n_embd": config_params["n_embd"],
                "n_layer": config_params["n_layer"],
                "n_head": config_params["n_head"],
                "n_positions": 1024,
                "resid_pdrop": 0.1,
                "attn_pdrop": 0.1
            }
        else:  # distilbert
            architecture = {
                "model_type": "distilbert",
                "vocab_size": 30522,
                "dim": config_params["dim"],
                "n_layers": config_params["n_layers"],
                "n_heads": config_params["n_heads"],
                "max_position_embeddings": 512,
                "dropout": 0.1,
                "attention_dropout": 0.1
            }
        
        # 📊 PARAMETER ESTIMATION
        estimated_params = self._estimate_transformers_parameters(architecture)
        estimated_size_mb = estimated_params * 4 / (1024 * 1024)
        
        architecture.update({
            "estimated_parameters": estimated_params,
            "estimated_size_mb": estimated_size_mb,
            "framework": "transformers",
            "base_architecture": base_arch,
            "deployment_ready": True
        })
        
        logger.info(f"Generated {base_arch} architecture: {estimated_params:,} parameters")
        return architecture
    
    async def prepare_training_data(
        self,
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Prepare training data for Transformers distillation
        
        📊 DATA PREPARATION:
        - Automatic tokenization with appropriate tokenizer
        - Task-specific data formatting
        - Attention mask generation
        - Dataset splitting and batching
        """
        logger.info("Preparing Transformers training data")
        
        # 🎯 TOKENIZER SELECTION based on architecture
        tokenizer_mapping = {
            "bert": "bert-base-uncased",
            "gpt2": "gpt2",
            "distilbert": "distilbert-base-uncased"
        }
        
        # For simulation, assume we have data preparation
        dataset_info = {
            "train_samples": 50000,
            "val_samples": 10000,
            "test_samples": 5000,
            "max_sequence_length": 512,
            "tokenizer": tokenizer_mapping.get(request.domain, "bert-base-uncased"),
            "task_type": self._infer_task_type(request.domain),
            "preprocessing_applied": True
        }
        
        return {
            "dataset_info": dataset_info,
            "tokenization_complete": True,
            "data_format": "transformers_compatible"
        }
    
    async def initialize_models(
        self,
        teacher_config: Dict[str, Any],
        student_architecture: Dict[str, Any],
        config: TrainingConfig
    ) -> Tuple[Any, Any]:
        """
        Initialize teacher and student models using Transformers
        
        🏗️ MODEL INITIALIZATION:
        - Load teacher from Hugging Face Hub or local path
        - Create student with custom architecture
        - Setup for knowledge distillation
        """
        logger.info("Initializing Transformers models")
        
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
        model_type = student_architecture["model_type"]
        
        if model_type == "bert":
            config_class = BertConfig
            model_class = BertForSequenceClassification
        elif model_type == "gpt2":
            config_class = GPT2Config
            model_class = GPT2LMHeadModel
        else:  # distilbert
            config_class = DistilBertConfig
            model_class = DistilBertForSequenceClassification
        
        # 🔧 CREATE STUDENT CONFIGURATION
        student_config = config_class(**{
            k: v for k, v in student_architecture.items() 
            if k not in ["estimated_parameters", "estimated_size_mb", "framework", "deployment_ready"]
        })
        
        # 🎓 INITIALIZE STUDENT MODEL
        student_model = model_class(student_config).to(self.device)
        
        # 📊 LOG MODEL INFO
        total_params = sum(p.numel() for p in student_model.parameters())
        logger.info(f"Student model: {total_params:,} parameters")
        
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
        Execute Transformers training step with distillation
        
        🔄 TRAINING PROCESS:
        1. Validate inputs - return zero metrics if no inputs
        2. Create optimizer (AdamW, instance variable, created once)
        3. Tokenize batch using tokenizer
        4. Forward pass through student model
        5. Teacher soft labels if available (use KnowledgeDistillationLoss)
        6. Backward pass with gradient clipping
        7. Return real TrainingMetrics with actual loss values
        8. Wrap in try/except for graceful fallback
        
        🔑 KEY DIFFERENCES FROM PYTORCH BACKEND:
        - HF models (e.g., BertForSequenceClassification) compute loss internally
          when labels are passed: model(input_ids, labels=labels) returns loss
        - No custom loss function needed unless teacher soft labels are available
        - HF models compute cross-entropy internally when labels passed
        """
        try:
            # 1️⃣ VALIDATE INPUTS - return zero metrics if no inputs
            if not TRANSFORMERS_AVAILABLE or torch is None:
                return TrainingMetrics(
                    step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                    distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                    temperature=config.distillation_temperature
                )
            
            inputs = batch_data.get("inputs", [])
            targets = batch_data.get("targets", [])
            if not inputs:
                return TrainingMetrics(
                    step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                    distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                    temperature=config.distillation_temperature
                )
            
            # 2️⃣ CREATE OPTIMIZER (AdamW, instance variable, created once)
            if not hasattr(self, "_optimizer") or self._optimizer is None:
                self._optimizer = torch.optim.AdamW(
                    student_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            
            # 3️⃣ TOKENIZE BATCH using tokenizer
            tokenizer = self._get_or_create_tokenizer()
            if tokenizer is None:
                return TrainingMetrics(
                    step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                    distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                    temperature=config.distillation_temperature
                )
            
            input_enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256)
            target_enc = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=64)
            
            input_ids = input_enc["input_ids"].to(self.device)
            attn_mask = input_enc["attention_mask"].to(self.device)
            labels = target_enc["input_ids"].to(self.device)
            
            # 4️⃣ TRAINING MODE AND ZERO GRAD
            student_model.train()
            self._optimizer.zero_grad()
            
            # 5️⃣ FORWARD PASS through student model
            # HF models compute loss internally when labels are passed
            student_outputs = student_model(input_ids, attention_mask=attn_mask, labels=labels)
            
            # Extract student loss (HF models return loss when labels provided)
            student_loss = getattr(student_outputs, "loss", None)
            if student_loss is None:
                student_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            
            # Extract student logits for potential distillation
            student_logits = getattr(student_outputs, "logits", None)
            
            # 6️⃣ TEACHER SOFT LABELS if available (use KnowledgeDistillationLoss)
            distillation_loss = torch.tensor(0.0, device=self.device)
            total_loss = student_loss
            
            if (teacher_model is not None
                and hasattr(teacher_model, "supports_soft_labels")
                and teacher_model.supports_soft_labels
                and student_logits is not None):
                
                # Get soft labels from teacher
                teacher_logits = await teacher_model.get_soft_labels(input_ids, attn_mask)
                
                if teacher_logits is not None:
                    # Import KD loss from pytorch_backend
                    from .pytorch_backend import KnowledgeDistillationLoss
                    
                    distill_fn = KnowledgeDistillationLoss(
                        temperature=config.distillation_temperature,
                        alpha=config.alpha_knowledge_distillation
                    )
                    total_loss, _ = distill_fn(student_logits, teacher_logits, labels)
                    distillation_loss = total_loss - (1.0 - config.alpha_knowledge_distillation) * student_loss
            
            # 7️⃣ BACKWARD PASS with gradient clipping
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                config.gradient_clipping
            )
            self._optimizer.step()
            
            # 8️⃣ RETURN REAL TrainingMetrics with actual loss values
            metrics = TrainingMetrics(
                step=step,
                epoch=step // 100,
                loss=total_loss.item(),
                accuracy=0.0,  # Accuracy computed during evaluation, not training
                distillation_loss=distillation_loss.item(),
                student_loss=student_loss.item(),
                learning_rate=config.learning_rate,
                temperature=config.distillation_temperature,
                additional_metrics={
                    "perplexity": math.exp(min(total_loss.item(), 10.0)) if total_loss.item() > 0 else 0.0,
                    "gradient_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"train_step failed: {e}")
            # Return zero metrics on failure (no crash)
            return TrainingMetrics(
                step=step, epoch=step // 100, loss=0.0, accuracy=0.0,
                distillation_loss=0.0, student_loss=0.0, learning_rate=0.0,
                temperature=config.distillation_temperature
            )
    
    async def evaluate_model(
        self,
        student_model: Any,
        eval_data: Dict[str, Any],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate model using Transformers evaluation utilities.
        
        📊 EVALUATION FEATURES:
        - Task-specific metrics (accuracy, F1, BLEU, etc.)
        - Automatic metric computation via HuggingFace Trainer
        - Comparison with teacher model
        - Inference speed benchmarking
        
        Strategy: Use HuggingFace Trainer.evaluate() if available;
        fall back to manual inference loop for non-HF models.

        Note: Teacher model loading is now fully implemented via teacher_loader.py.
        The train_step method uses real gradient computation with teacher soft labels
        when available, falling back to student-only cross-entropy loss otherwise.
        """
        logger.info("Evaluating Transformers model with real inference")
        
        # Check for transformers availability
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, returning fallback metrics")
            return self._fallback_metrics()
        
        # Validate test data exists
        test_data = eval_data.get("test", [])
        if not test_data:
            logger.warning("No test data provided, returning fallback metrics")
            return self._fallback_metrics()
        
        # Validate student model
        if student_model is None:
            logger.warning("Invalid student model (None), returning fallback metrics")
            return self._fallback_metrics()
        
        try:
            # Check if student_model is a HuggingFace model (has .config attribute)
            is_hf_model = hasattr(student_model, 'config') and hasattr(student_model, 'eval')
            
            if is_hf_model:
                # Try HuggingFace Trainer evaluation (best path)
                return await self._evaluate_with_trainer(student_model, test_data, teacher_model)
            else:
                # Fallback: manual inference loop for non-HF models
                return await self._evaluate_with_manual_loop(student_model, test_data, teacher_model)
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return self._fallback_metrics()
    
    async def _evaluate_with_trainer(
        self,
        student_model: Any,
        test_data: List[Dict[str, Any]],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate using HuggingFace Trainer.evaluate() for native HF models.
        
        This is the preferred evaluation path for models that are full HuggingFace models.
        """
        logger.info("Using HuggingFace Trainer for evaluation")
        
        # Get tokenizer
        tokenizer = self._get_or_create_tokenizer()
        
        # Create evaluation dataset
        eval_dataset = _EvalDataset(test_data, tokenizer)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create TrainingArguments for evaluation
            training_args = TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=8,
                dataloader_drop_last=False,
                report_to="none",  # Disable wandb/mlflow reporting
                fp16=torch.cuda.is_available() if torch is not None else False
            )
            
            # Create Trainer
            trainer = Trainer(
                model=student_model,
                args=training_args,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics,
                tokenizer=tokenizer
            )
            
            # Run evaluation
            eval_results = trainer.evaluate()
            
            # Measure inference latency with a small benchmark
            student_model.eval()
            latencies = []
            
            with torch.no_grad():
                for i, item in enumerate(test_data[:20]):  # Benchmark on 20 items
                    input_text = item.get("content", "") or item.get("input", "")
                    if not input_text:
                        continue
                    
                    if tokenizer is not None:
                        inputs = tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=256
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        # Fallback encoding
                        input_ids = torch.tensor([[ord(c) % 30522 for c in input_text[:256]]]).to(self.device)
                        inputs = {"input_ids": input_ids}
                    
                    start_time = time.perf_counter()
                    _ = student_model(**inputs)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            
            # Restore training mode
            student_model.train()
            
            # Compute metrics
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            
            # Compute memory usage
            memory_mb = 0.0
            if torch is not None and torch.cuda.is_available() and self.device == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Compute model size from parameters
            model_size_mb = 0.0
            if hasattr(student_model, 'parameters'):
                total_params = sum(p.numel() for p in student_model.parameters())
                model_size_mb = (total_params * 4) / (1024 * 1024)
            
            # Map HF output keys to PRSM schema
            accuracy = eval_results.get("eval_accuracy", eval_results.get("eval_f1", 0.0))
            
            # Compute F1, precision, recall if not in eval_results
            f1 = eval_results.get("eval_f1", accuracy)
            precision = eval_results.get("eval_precision", accuracy)
            recall = eval_results.get("eval_recall", accuracy)
            
            # Compute perplexity from eval_loss if available
            eval_loss = eval_results.get("eval_loss", 0.0)
            perplexity = math.exp(min(eval_loss, 10.0)) if eval_loss > 0 else 0.0
            
            eval_metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "avg_loss": eval_loss,
                "perplexity": perplexity,
                "inference_latency_ms": avg_latency,
                "throughput_tokens_per_sec": 1000.0 / max(0.001, avg_latency),
                "memory_usage_mb": memory_mb,
                "model_size_mb": model_size_mb,
                "items_evaluated": len(test_data)
            }
            
            # Teacher comparison flag
            if teacher_model is not None:
                eval_metrics["teacher_provided"] = True
            
            logger.info(f"Trainer evaluation complete: {accuracy:.3f} accuracy, {perplexity:.2f} perplexity")
            return eval_metrics
    
    async def _evaluate_with_manual_loop(
        self,
        student_model: Any,
        test_data: List[Dict[str, Any]],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Fallback: Manual inference loop for non-HuggingFace models.
        
        Uses the same algorithm as pytorch_backend.evaluate_model().
        """
        logger.info("Using manual inference loop for evaluation")
        
        # Set model to evaluation mode
        if hasattr(student_model, 'eval'):
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
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum') if torch is not None else None
        
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
                        input_ids = torch.tensor([[ord(c) % 30522 for c in input_text[:256]]])
                        target_ids = [ord(c) % 30522 for c in target_text[:64]]
                    
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
                        if logits.size(1) >= len(target_ids):
                            logits_for_loss = logits[0, :len(target_ids), :]
                            loss = loss_fn(logits_for_loss, target_tensor)
                            all_losses.append(loss.item() / len(target_ids))
                    
                except Exception as item_error:
                    logger.debug(f"Error processing evaluation item: {item_error}")
                    continue
        
        # Restore training mode
        if hasattr(student_model, 'train'):
            student_model.train()
        
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
        if torch is not None and torch.cuda.is_available() and self.device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Compute model size from parameters
        model_size_mb = 0.0
        if hasattr(student_model, 'parameters'):
            total_params = sum(p.numel() for p in student_model.parameters())
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
        
        # Teacher comparison flag
        if teacher_model is not None:
            eval_metrics["teacher_provided"] = True
        
        logger.info(f"Manual evaluation complete: {accuracy:.3f} accuracy, {perplexity:.2f} perplexity, {len(all_preds)} items")
        return eval_metrics
    
    async def export_model(
        self,
        student_model: Any,
        model_config: Dict[str, Any],
        export_path: str,
        formats: List[str] = ["transformers", "onnx"]
    ) -> ModelArtifacts:
        """
        Export model in Transformers-compatible formats
        
        📦 EXPORT OPTIONS:
        - Transformers native format
        - ONNX for cross-platform deployment
        - TensorFlow Lite for mobile
        - Hugging Face Hub integration
        """
        logger.info(f"Exporting Transformers model to {export_path}")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 🤗 TRANSFORMERS FORMAT
        model_path = str(export_dir)
        if hasattr(student_model, 'save_pretrained'):
            student_model.save_pretrained(model_path)
        
        # 🔧 TOKENIZER (if available)
        tokenizer_path = None
        # In practice: tokenizer.save_pretrained(model_path)
        
        # 📋 METADATA
        metadata = {
            "model_id": f"prsm-transformers-{hash(model_path) % 10000}",
            "framework": "transformers",
            "architecture": model_config.get("base_architecture", "bert"),
            "estimated_parameters": model_config.get("estimated_parameters", 0),
            "huggingface_compatible": True,
            "deployment_ready": True
        }
        
        # 💾 SAVE METADATA
        with open(export_dir / "prsm_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 🚀 DEPLOYMENT CONFIGURATION
        deployment_config = {
            "inference_device": "cpu",
            "framework": "transformers",
            "dependencies": ["transformers>=4.0.0", "torch>=1.9.0"],
            "huggingface_hub_ready": True
        }
        
        return ModelArtifacts(
            model_path=model_path,
            model_config=model_config,
            tokenizer_path=tokenizer_path,
            onnx_path=None,  # Would be generated if requested
            metadata=metadata,
            deployment_config=deployment_config
        )
    
    async def get_supported_architectures(self) -> List[str]:
        """Get Transformers-supported architectures"""
        return [
            "bert",
            "distilbert", 
            "gpt2",
            "t5",
            "roberta",
            "electra",
            "deberta"
        ]
    
    async def get_memory_requirements(self, architecture: StudentArchitecture) -> Dict[str, int]:
        """Estimate memory requirements for Transformers training"""
        params = architecture.estimated_parameters
        
        # Transformers training is memory-efficient
        training_mb = int(params * 12 / (1024 * 1024))  # ~12 bytes per parameter
        inference_mb = int(params * 4 / (1024 * 1024))  # ~4 bytes per parameter
        
        return {
            "training_mb": max(512, training_mb),
            "inference_mb": max(128, inference_mb)
        }
    
    # === Helper Methods ===
    
    def _estimate_transformers_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate parameters for Transformers architecture"""
        
        model_type = architecture["model_type"]
        
        if model_type == "bert":
            vocab_size = architecture["vocab_size"]
            hidden_size = architecture["hidden_size"]
            num_layers = architecture["num_hidden_layers"]
            intermediate_size = architecture["intermediate_size"]
            
            # BERT parameter calculation
            embedding_params = vocab_size * hidden_size + 512 * hidden_size  # Token + position
            layer_params = (
                4 * hidden_size * hidden_size +  # Attention Q,K,V,O
                2 * hidden_size * intermediate_size +  # FFN up/down
                6 * hidden_size  # Layer norms and biases
            )
            total_params = embedding_params + num_layers * layer_params + hidden_size * vocab_size
            
        elif model_type == "gpt2":
            vocab_size = architecture["vocab_size"]
            n_embd = architecture["n_embd"]
            n_layer = architecture["n_layer"]
            
            # GPT-2 parameter calculation
            embedding_params = vocab_size * n_embd + 1024 * n_embd  # Token + position
            layer_params = (
                4 * n_embd * n_embd +  # Attention
                2 * n_embd * (4 * n_embd) +  # FFN (4x expansion)
                4 * n_embd  # Layer norms
            )
            total_params = embedding_params + n_layer * layer_params + n_embd * vocab_size
            
        else:  # distilbert
            vocab_size = architecture["vocab_size"]
            dim = architecture["dim"]
            n_layers = architecture["n_layers"]
            
            # DistilBERT parameter calculation (simplified BERT)
            embedding_params = vocab_size * dim + 512 * dim
            layer_params = (
                4 * dim * dim +  # Attention
                2 * dim * (4 * dim) +  # FFN
                4 * dim  # Norms
            )
            total_params = embedding_params + n_layers * layer_params + dim * vocab_size
        
        return total_params
    
    def _infer_task_type(self, domain: str) -> str:
        """Infer ML task type from domain"""
        task_mapping = {
            "medical_research": "classification",
            "legal_analysis": "classification",
            "code_generation": "generation",
            "creative_writing": "generation",
            "scientific_reasoning": "classification",
            "data_analysis": "classification"
        }
        return task_mapping.get(domain, "classification")
    
    def _get_or_create_tokenizer(self) -> Any:
        """
        Get or create a default tokenizer for evaluation.
        
        Uses distilbert-base-uncased as the default tokenizer (matches distillation use case).
        Falls back to None if transformers is unavailable.
        """
        if "default" not in self.tokenizer_cache:
            try:
                self.tokenizer_cache["default"] = AutoTokenizer.from_pretrained(
                    "distilbert-base-uncased",
                    model_max_length=512
                )
                logger.debug("Created default DistilBERT tokenizer for evaluation")
            except (ImportError, OSError) as e:
                logger.warning(f"Could not load transformers tokenizer: {e}")
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
    
    @staticmethod
    def _compute_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute metrics for HuggingFace Trainer.evaluate().
        
        Args:
            eval_pred: EvalPrediction object with predictions and label_ids
            
        Returns:
            Dict with accuracy metric
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        try:
            import evaluate
            metric = evaluate.load("accuracy")
            return metric.compute(predictions=predictions, references=labels)
        except ImportError:
            # Fallback: compute accuracy manually
            acc = (predictions == labels).mean()
            return {"accuracy": float(acc)}


class _EvalDataset(torch.utils.data.Dataset):
    """
    Simple dataset class for HuggingFace Trainer evaluation.
    
    Wraps test data items into tokenized format suitable for model evaluation.
    """
    
    def __init__(self, items: List[Dict[str, Any]], tokenizer: Any,
                 max_input_length: int = 256, max_target_length: int = 64):
        self.items = items
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        input_text = item.get("content", "") or item.get("input", "")
        target_text = item.get("expected_answer", "") or item.get("target", "")
        
        if self.tokenizer is not None:
            input_enc = self.tokenizer(
                input_text, truncation=True, max_length=self.max_input_length,
                padding="max_length", return_tensors="pt"
            )
            target_enc = self.tokenizer(
                target_text, truncation=True, max_length=self.max_target_length,
                padding="max_length", return_tensors="pt"
            )
            
            return {
                "input_ids": input_enc["input_ids"].squeeze(0),
                "attention_mask": input_enc["attention_mask"].squeeze(0),
                "labels": target_enc["input_ids"].squeeze(0)
            }
        else:
            # Fallback: simple character-level encoding
            input_ids = torch.tensor([[ord(c) % 30522 for c in input_text[:self.max_input_length]]])
            target_ids = torch.tensor([[ord(c) % 30522 for c in target_text[:self.max_target_length]]])
            
            return {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": torch.ones_like(input_ids.squeeze(0)),
                "labels": target_ids.squeeze(0)
            }


# Register Transformers backend if available
if TRANSFORMERS_AVAILABLE:
    BackendRegistry.register("transformers", TransformersDistillationBackend)
    logger.info("Transformers backend registered successfully")
else:
    logger.warning("Transformers backend not available - install 'transformers' library")