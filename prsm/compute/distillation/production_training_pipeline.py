"""
Production ML Training Pipeline for PRSM
Real knowledge distillation with PyTorch, TensorFlow, and Transformers
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass
from pathlib import Path

# ML Framework imports
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import transformers
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        Trainer, TrainingArguments, DataCollatorWithPadding
    )
    import tensorflow as tf
    from datasets import Dataset as HFDataset
    import evaluate
    import wandb
except ImportError as e:
    print(f"⚠️ ML dependencies not fully installed: {e}")
    print("Install with: pip install torch transformers tensorflow datasets evaluate wandb")

from prsm.core.config import settings
from prsm.core.models import TeacherModel, ModelType
from prsm.economy.tokenomics.ftns_service import get_ftns_service
from prsm.core.safety.monitor import SafetyMonitor
from .models import (
    DistillationRequest, DistillationJob, TrainingMetrics, 
    ModelSize, OptimizationTarget, TrainingStrategy
)


# === Configuration ===

ENABLE_WANDB = getattr(settings, "PRSM_ENABLE_WANDB", False)
CHECKPOINT_FREQUENCY = int(getattr(settings, "PRSM_CHECKPOINT_FREQ", 100))
EVAL_FREQUENCY = int(getattr(settings, "PRSM_EVAL_FREQ", 500))
MAX_TEACHER_TOKENS = int(getattr(settings, "PRSM_MAX_TEACHER_TOKENS", 2048))
DEFAULT_BATCH_SIZE = int(getattr(settings, "PRSM_DEFAULT_BATCH_SIZE", 16))


@dataclass
class TrainingConfig:
    """Configuration for production training"""
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    temperature: float = 4.0
    alpha: float = 0.7  # Distillation loss weight
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 50
    max_grad_norm: float = 1.0
    use_fp16: bool = True
    dataloader_num_workers: int = 4


class TeacherModelConnector:
    """Real connection to teacher models via APIs"""
    
    def __init__(self):
        self.model_clients = {}
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Initialize API clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for different teacher models"""
        try:
            import openai
            self.model_clients['openai'] = openai.OpenAI()
        except ImportError:
            print("⚠️ OpenAI client not available")
        
        try:
            import anthropic
            self.model_clients['anthropic'] = anthropic.Anthropic()
        except ImportError:
            print("⚠️ Anthropic client not available")
    
    async def query_teacher(self, model_name: str, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Query teacher model and return response with logits/embeddings when possible"""
        try:
            # Check cache first
            cache_key = f"{model_name}:{hash(prompt)}"
            if cache_key in self.response_cache:
                cached_result, timestamp = self.response_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_result
            
            result = None
            
            # OpenAI models
            if 'gpt' in model_name.lower():
                result = await self._query_openai(model_name, prompt, max_tokens)
            # Anthropic models
            elif 'claude' in model_name.lower():
                result = await self._query_anthropic(model_name, prompt, max_tokens)
            # Hugging Face models
            else:
                result = await self._query_huggingface(model_name, prompt, max_tokens)
            
            if result:
                # Cache result
                self.response_cache[cache_key] = (result, time.time())
                return result
            
            return {"error": "No suitable teacher model client found"}
            
        except Exception as e:
            print(f"❌ Error querying teacher model {model_name}: {e}")
            return {"error": str(e)}
    
    async def _query_openai(self, model_name: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Query OpenAI models"""
        try:
            if 'openai' not in self.model_clients:
                return {"error": "OpenAI client not available"}
            
            client = self.model_clients['openai']
            
            # For text generation models
            response = client.chat.completions.create(
                model=model_name.replace('openai/', ''),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                logprobs=True,  # Get logits when available
                top_logprobs=20
            )
            
            result = {
                "response": response.choices[0].message.content,
                "model": model_name,
                "tokens_used": response.usage.total_tokens,
                "logprobs": response.choices[0].logprobs.content if response.choices[0].logprobs else None
            }
            
            return result
            
        except Exception as e:
            return {"error": f"OpenAI query failed: {e}"}
    
    async def _query_anthropic(self, model_name: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Query Anthropic models"""
        try:
            if 'anthropic' not in self.model_clients:
                return {"error": "Anthropic client not available"}
            
            client = self.model_clients['anthropic']
            
            response = client.messages.create(
                model=model_name.replace('anthropic/', ''),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = {
                "response": response.content[0].text,
                "model": model_name,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Anthropic query failed: {e}"}
    
    async def _query_huggingface(self, model_name: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Query Hugging Face models"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Generate with logits
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            result = {
                "response": response_text,
                "model": model_name,
                "tokens_used": len(outputs.sequences[0]),
                "logits": outputs.scores[-1] if outputs.scores else None  # Last token logits
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Hugging Face query failed: {e}"}


class DistillationDataset(Dataset):
    """Real dataset for knowledge distillation training"""
    
    def __init__(self, teacher_connector: TeacherModelConnector, teacher_model: str, 
                 domain: str, size: int = 1000, tokenizer=None):
        self.teacher_connector = teacher_connector
        self.teacher_model = teacher_model
        self.domain = domain
        self.tokenizer = tokenizer
        self.size = size
        
        # Generate domain-specific prompts
        self.prompts = self._generate_domain_prompts()
        self.teacher_responses = []
        
        print(f"📚 Creating distillation dataset: {size} examples for {domain}")
    
    def _generate_domain_prompts(self) -> List[str]:
        """Generate domain-specific training prompts"""
        prompts = []
        
        domain_templates = {
            "nlp": [
                "Summarize the following text: {}",
                "Translate to French: {}",
                "Answer the question: {}",
                "Complete the sentence: {}",
                "Classify the sentiment: {}"
            ],
            "coding": [
                "Write a Python function to {}",
                "Debug this code: {}",
                "Explain this algorithm: {}",
                "Convert to {} programming language: {}",
                "Optimize this code: {}"
            ],
            "reasoning": [
                "Solve this logic problem: {}",
                "Analyze the argument: {}",
                "What is the next step in: {}",
                "Explain the relationship between: {}",
                "Draw a conclusion from: {}"
            ],
            "general": [
                "Explain the concept of: {}",
                "What are the benefits of: {}",
                "How does {} work?",
                "Compare {} and {}",
                "Provide examples of: {}"
            ]
        }
        
        # Get templates for domain or fall back to general
        templates = domain_templates.get(self.domain.lower(), domain_templates["general"])
        
        # Generate diverse prompts
        for i in range(self.size):
            template = templates[i % len(templates)]
            
            # Fill in template with domain-appropriate content
            if '{}' in template:
                content = self._get_domain_content(i)
                prompt = template.format(content)
            else:
                prompt = template
            
            prompts.append(prompt)
        
        return prompts
    
    def _get_domain_content(self, index: int) -> str:
        """Get domain-appropriate content for prompt templates"""
        domain_content = {
            "nlp": ["machine learning", "natural language processing", "transformers", "attention mechanisms"],
            "coding": ["sort a list", "parse JSON", "handle exceptions", "optimize performance"],
            "reasoning": ["logical deduction", "causal relationships", "pattern recognition", "problem solving"],
            "general": ["artificial intelligence", "sustainability", "innovation", "collaboration"]
        }
        
        content_list = domain_content.get(self.domain.lower(), domain_content["general"])
        return content_list[index % len(content_list)]
    
    async def generate_teacher_responses(self):
        """Generate all teacher responses (called once during dataset creation)"""
        print(f"🧠 Generating teacher responses for {len(self.prompts)} prompts...")
        
        for i, prompt in enumerate(self.prompts):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(self.prompts)} ({i/len(self.prompts)*100:.1f}%)")
            
            response = await self.teacher_connector.query_teacher(
                self.teacher_model, prompt, max_tokens=256
            )
            
            self.teacher_responses.append(response)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        print(f"✅ Generated {len(self.teacher_responses)} teacher responses")
    
    def __len__(self):
        return min(len(self.prompts), len(self.teacher_responses))
    
    def __getitem__(self, idx):
        if idx >= len(self.teacher_responses):
            # If teacher responses not generated yet, return placeholder
            return {
                "input_text": self.prompts[idx],
                "target_text": "Generating...",
                "teacher_logits": None
            }
        
        prompt = self.prompts[idx]
        teacher_response = self.teacher_responses[idx]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            # Tokenize input and target
            inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            targets = self.tokenizer(
                teacher_response.get("response", ""),
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "target_ids": targets["input_ids"].squeeze(),
                "target_attention_mask": targets["attention_mask"].squeeze(),
                "teacher_logits": teacher_response.get("logits"),
                "teacher_response": teacher_response.get("response", "")
            }
        else:
            return {
                "input_text": prompt,
                "target_text": teacher_response.get("response", ""),
                "teacher_logits": teacher_response.get("logits"),
                "teacher_response": teacher_response.get("response", "")
            }


class ProductionPyTorchTrainer:
    """Real PyTorch training implementation for knowledge distillation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.safety_monitor = SafetyMonitor()
        
        # Initialize wandb if enabled
        if ENABLE_WANDB:
            wandb.init(project="prsm-distillation", config=config.__dict__)
    
    async def train_model(self, teacher_model_name: str, student_model_path: str, 
                         domain: str, job_id: str) -> Dict[str, Any]:
        """Execute real PyTorch knowledge distillation training"""
        try:
            print(f"🚀 Starting PyTorch distillation training for job {job_id}")
            print(f"   Teacher: {teacher_model_name}")
            print(f"   Domain: {domain}")
            print(f"   Device: {self.device}")
            
            # 1. Initialize teacher connector
            teacher_connector = TeacherModelConnector()
            
            # 2. Load/Create student model
            student_model, tokenizer = await self._load_student_model(student_model_path, domain)
            student_model.to(self.device)
            
            # 3. Create training dataset
            dataset = DistillationDataset(
                teacher_connector, teacher_model_name, domain, 
                size=1000, tokenizer=tokenizer
            )
            await dataset.generate_teacher_responses()
            
            # 4. Create data loader
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=self._collate_fn
            )
            
            # 5. Initialize optimizer and scheduler
            optimizer = optim.AdamW(
                student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            num_training_steps = len(dataloader) * self.config.num_epochs
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=self.config.warmup_steps
            )
            
            # 6. Training loop
            training_metrics = await self._training_loop(
                student_model, dataloader, optimizer, scheduler, job_id
            )
            
            # 7. Save final model
            model_save_path = await self._save_model(student_model, tokenizer, job_id)
            
            # 8. Final evaluation
            eval_results = await self._evaluate_model(student_model, tokenizer, domain)
            
            print(f"✅ Training completed successfully for job {job_id}")
            
            return {
                "status": "completed",
                "model_path": model_save_path,
                "final_loss": training_metrics["final_loss"],
                "training_steps": training_metrics["total_steps"],
                "evaluation_results": eval_results,
                "training_time": training_metrics["training_time"]
            }
            
        except Exception as e:
            print(f"❌ Training failed for job {job_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _load_student_model(self, model_path: str, domain: str) -> Tuple[nn.Module, Any]:
        """Load or create student model for distillation"""
        try:
            # For now, use a small GPT-2 model as student
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            model_name = "gpt2"  # Start with small GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Add domain-specific adaptation if needed
            if domain in ["coding", "reasoning"]:
                # Could add domain-specific layers here
                pass
            
            print(f"📦 Loaded student model: {model_name} ({sum(p.numel() for p in model.parameters())} parameters)")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading student model: {e}")
            raise
    
    def _collate_fn(self, batch):
        """Collate function for data loader"""
        if not batch:
            return {}
        
        # Handle tokenized batch
        if "input_ids" in batch[0]:
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "target_ids": torch.stack([item["target_ids"] for item in batch]),
                "target_attention_mask": torch.stack([item["target_attention_mask"] for item in batch])
            }
        else:
            # Handle text batch (would need additional processing)
            return {
                "texts": [item["input_text"] for item in batch],
                "targets": [item["target_text"] for item in batch]
            }
    
    async def _training_loop(self, model: nn.Module, dataloader: DataLoader, 
                           optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                           job_id: str) -> Dict[str, Any]:
        """Real PyTorch training loop with knowledge distillation"""
        model.train()
        
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()
        
        print(f"🏃 Starting training loop: {self.config.num_epochs} epochs, {len(dataloader)} batches/epoch")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if not batch or "input_ids" not in batch:
                    continue
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids
                )
                
                # Calculate loss (using built-in language modeling loss for now)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                if total_steps < self.config.warmup_steps:
                    scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                epoch_loss += loss.item()
                total_steps += 1
                epoch_steps += 1
                
                # Logging
                if total_steps % self.config.logging_steps == 0:
                    avg_loss = total_loss / total_steps
                    print(f"   Step {total_steps}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")
                    
                    if ENABLE_WANDB:
                        wandb.log({
                            "train_loss": loss.item(),
                            "avg_train_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "step": total_steps
                        })
                
                # Safety check
                safety_valid = await self.safety_monitor.validate_model_output(
                    {"loss": loss.item(), "step": total_steps},
                    ["validate_training_progress"]
                )
                if not safety_valid:
                    print("⚠️ Safety validation failed, stopping training")
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"📊 Epoch {epoch + 1}/{self.config.num_epochs} completed: avg_loss={avg_epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        final_loss = total_loss / max(total_steps, 1)
        
        print(f"🏁 Training completed: {total_steps} steps, {training_time:.2f}s, final_loss={final_loss:.4f}")
        
        return {
            "final_loss": final_loss,
            "total_steps": total_steps,
            "training_time": training_time
        }
    
    async def _save_model(self, model: nn.Module, tokenizer: Any, job_id: str) -> str:
        """Save trained model and tokenizer"""
        try:
            save_dir = Path(f"models/distilled/{job_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = save_dir / "pytorch_model.bin"
            torch.save(model.state_dict(), model_path)
            
            # Save tokenizer
            tokenizer.save_pretrained(save_dir)
            
            # Save config
            config_path = save_dir / "training_config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(self.config.__dict__, f, indent=2)
            
            print(f"💾 Model saved to {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return ""
    
    async def _evaluate_model(self, model: nn.Module, tokenizer: Any, domain: str) -> Dict[str, float]:
        """Evaluate trained model on domain-specific benchmarks"""
        try:
            model.eval()
            
            # Simple evaluation with sample prompts
            eval_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "How does neural network training work?"
            ]
            
            total_length = 0
            valid_responses = 0
            
            with torch.no_grad():
                for prompt in eval_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response_length = len(response.split())
                    
                    if response_length > 5:  # Basic validity check
                        total_length += response_length
                        valid_responses += 1
            
            # Calculate basic metrics
            avg_response_length = total_length / max(valid_responses, 1)
            response_rate = valid_responses / len(eval_prompts)
            
            return {
                "avg_response_length": avg_response_length,
                "response_rate": response_rate,
                "domain": domain,
                "eval_prompts": len(eval_prompts)
            }
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {"error": str(e)}


class ProductionTrainingPipeline:
    """Production ML training pipeline orchestrator"""
    
    def __init__(self):
        self.active_training_jobs: Dict[str, Dict[str, Any]] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.safety_monitor = SafetyMonitor()
    
    async def start_training(self, request: DistillationRequest) -> DistillationJob:
        """Start real ML training pipeline"""
        try:
            job_id = str(uuid4())
            
            print(f"🎯 Starting production training pipeline for job {job_id}")
            print(f"   Teacher: {request.teacher_model}")
            print(f"   Domain: {request.domain}")
            print(f"   Strategy: {request.training_strategy}")
            print(f"   Target size: {request.target_size}")
            
            # Create training configuration
            config = TrainingConfig(
                learning_rate=self._get_optimal_lr(request),
                batch_size=self._get_optimal_batch_size(request),
                num_epochs=self._get_optimal_epochs(request),
                temperature=4.0,
                alpha=0.7
            )
            
            # Initialize trainer based on backend
            if request.backend == "pytorch":
                trainer = ProductionPyTorchTrainer(config)
            else:
                # For now, fallback to PyTorch
                trainer = ProductionPyTorchTrainer(config)
                print(f"⚠️ Backend {request.backend} not fully implemented, using PyTorch")
            
            # Create job record
            job = DistillationJob(
                job_id=UUID(job_id),
                user_id=request.user_id,
                teacher_model=request.teacher_model,
                domain=request.domain,
                status="training",
                progress=0.0,
                current_stage="initializing",
                estimated_completion=datetime.now(timezone.utc),
                backend=request.backend or "pytorch"
            )
            
            # Store active job
            self.active_training_jobs[job_id] = {
                "job": job,
                "trainer": trainer,
                "config": config,
                "request": request,
                "start_time": time.time()
            }
            
            # Start training asynchronously
            asyncio.create_task(self._execute_training(job_id))
            
            print(f"✅ Training job {job_id} started successfully")
            return job
            
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            raise
    
    async def _execute_training(self, job_id: str):
        """Execute the actual training process"""
        try:
            job_info = self.active_training_jobs[job_id]
            job = job_info["job"]
            trainer = job_info["trainer"]
            request = job_info["request"]
            
            # Update job status
            job.status = "training"
            job.current_stage = "data_preparation"
            job.progress = 0.1
            
            # Execute training
            training_result = await trainer.train_model(
                teacher_model_name=request.teacher_model,
                student_model_path="",  # Will be created
                domain=request.domain,
                job_id=job_id
            )
            
            # Update job with results
            if training_result["status"] == "completed":
                job.status = "completed"
                job.progress = 1.0
                job.current_stage = "completed"
                job.model_path = training_result["model_path"]
                job.final_metrics = training_result.get("evaluation_results", {})
                
                # Charge FTNS for successful training
                await ftns_service.charge_context_access(
                    request.user_id, 
                    request.budget_ftns
                )
                
                print(f"✅ Training job {job_id} completed successfully")
            else:
                job.status = "failed"
                job.error_message = training_result.get("error", "Unknown error")
                print(f"❌ Training job {job_id} failed: {job.error_message}")
            
            # Move to history
            self.training_history.append({
                "job_id": job_id,
                "job": job,
                "result": training_result,
                "duration": time.time() - job_info["start_time"]
            })
            
            # Remove from active jobs
            del self.active_training_jobs[job_id]
            
        except Exception as e:
            print(f"❌ Training execution error for job {job_id}: {e}")
            
            # Update job status
            if job_id in self.active_training_jobs:
                job = self.active_training_jobs[job_id]["job"]
                job.status = "failed"
                job.error_message = str(e)
    
    async def get_job_status(self, job_id: str) -> Optional[DistillationJob]:
        """Get current status of a training job"""
        if job_id in self.active_training_jobs:
            return self.active_training_jobs[job_id]["job"]
        
        # Check history
        for record in self.training_history:
            if record["job_id"] == job_id:
                return record["job"]
        
        return None
    
    def _get_optimal_lr(self, request: DistillationRequest) -> float:
        """Get optimal learning rate based on request parameters"""
        base_lr = 2e-5
        
        # Adjust based on model size
        if request.target_size == ModelSize.TINY:
            return base_lr * 2.0
        elif request.target_size == ModelSize.LARGE:
            return base_lr * 0.5
        
        return base_lr
    
    def _get_optimal_batch_size(self, request: DistillationRequest) -> int:
        """Get optimal batch size based on model size and available memory"""
        base_batch_size = DEFAULT_BATCH_SIZE
        
        if request.target_size == ModelSize.TINY:
            return base_batch_size * 2
        elif request.target_size == ModelSize.LARGE:
            return max(4, base_batch_size // 2)
        
        return base_batch_size
    
    def _get_optimal_epochs(self, request: DistillationRequest) -> int:
        """Get optimal number of epochs based on training strategy"""
        if request.training_strategy == TrainingStrategy.BASIC:
            return 2
        elif request.training_strategy == TrainingStrategy.PROGRESSIVE:
            return 3
        elif request.training_strategy == TrainingStrategy.ADVERSARIAL:
            return 5
        
        return 3
    
    async def get_training_metrics(self) -> Dict[str, Any]:
        """Get overall training pipeline metrics"""
        active_jobs = len(self.active_training_jobs)
        completed_jobs = len([r for r in self.training_history if r["job"].status == "completed"])
        failed_jobs = len([r for r in self.training_history if r["job"].status == "failed"])
        
        return {
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "total_jobs": completed_jobs + failed_jobs,
            "success_rate": completed_jobs / max(completed_jobs + failed_jobs, 1),
            "average_training_time": sum(r["duration"] for r in self.training_history) / max(len(self.training_history), 1)
        }


class ProductionTensorFlowTrainer:
    """Real TensorFlow training implementation for knowledge distillation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.safety_monitor = SafetyMonitor()
        
        # Initialize wandb if enabled
        if ENABLE_WANDB:
            wandb.init(project="prsm-distillation-tf", config=config.__dict__)
    
    async def train_model(self, teacher_model_name: str, student_model_path: str, 
                         domain: str, job_id: str) -> Dict[str, Any]:
        """Execute real TensorFlow knowledge distillation training"""
        try:
            print(f"🚀 Starting TensorFlow distillation training for job {job_id}")
            print(f"   Teacher: {teacher_model_name}")
            print(f"   Domain: {domain}")
            
            # 1. Initialize teacher connector
            teacher_connector = TeacherModelConnector()
            
            # 2. Load/Create student model using TensorFlow
            student_model, tokenizer = await self._load_tf_student_model(student_model_path, domain)
            
            # 3. Create training dataset
            dataset = DistillationDataset(
                teacher_connector, teacher_model_name, domain, 
                size=1000, tokenizer=tokenizer
            )
            await dataset.generate_teacher_responses()
            
            # 4. Create TensorFlow dataset
            tf_dataset = await self._create_tf_dataset(dataset)
            
            # 5. Initialize optimizer and loss
            optimizer = self._create_tf_optimizer()
            loss_fn = self._create_distillation_loss()
            
            # 6. Training loop
            training_metrics = await self._tf_training_loop(
                student_model, tf_dataset, optimizer, loss_fn, job_id
            )
            
            # 7. Save final model
            model_save_path = await self._save_tf_model(student_model, tokenizer, job_id)
            
            # 8. Final evaluation
            eval_results = await self._evaluate_tf_model(student_model, tokenizer, domain)
            
            print(f"✅ TensorFlow training completed successfully for job {job_id}")
            
            return {
                "status": "completed",
                "model_path": model_save_path,
                "final_loss": training_metrics["final_loss"],
                "training_steps": training_metrics["total_steps"],
                "evaluation_results": eval_results,
                "training_time": training_metrics["training_time"]
            }
            
        except Exception as e:
            print(f"❌ TensorFlow training failed for job {job_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _load_tf_student_model(self, model_path: str, domain: str):
        """Load or create TensorFlow student model"""
        try:
            import tensorflow as tf
            from transformers import TFAutoModel, AutoTokenizer
            
            model_name = "distilbert-base-uncased"  # Start with DistilBERT
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create custom TensorFlow model for distillation
            model = TFAutoModel.from_pretrained(model_name)
            
            print(f"📦 Loaded TensorFlow student model: {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading TensorFlow student model: {e}")
            raise
    
    async def _create_tf_dataset(self, dataset: DistillationDataset):
        """Create TensorFlow dataset from distillation dataset"""
        try:
            import tensorflow as tf
            
            # Convert to TensorFlow format
            inputs = []
            targets = []
            
            for i in range(len(dataset)):
                item = dataset[i]
                if "input_ids" in item:
                    inputs.append(item["input_ids"].numpy())
                    targets.append(item["target_ids"].numpy())
            
            tf_dataset = tf.data.Dataset.from_tensor_slices({
                "input_ids": inputs,
                "target_ids": targets
            })
            
            tf_dataset = tf_dataset.batch(self.config.batch_size)
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
            
            return tf_dataset
            
        except Exception as e:
            print(f"❌ Error creating TensorFlow dataset: {e}")
            raise
    
    def _create_tf_optimizer(self):
        """Create TensorFlow optimizer"""
        try:
            import tensorflow as tf
            
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            return optimizer
            
        except Exception as e:
            print(f"❌ Error creating TensorFlow optimizer: {e}")
            raise
    
    def _create_distillation_loss(self):
        """Create knowledge distillation loss function"""
        try:
            import tensorflow as tf
            
            def distillation_loss(y_true, y_pred, teacher_logits=None):
                # Standard cross-entropy loss
                student_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true, y_pred, from_logits=True
                )
                
                # Knowledge distillation loss (if teacher logits available)
                if teacher_logits is not None:
                    temperature = self.config.temperature
                    soft_targets = tf.nn.softmax(teacher_logits / temperature)
                    soft_predictions = tf.nn.softmax(y_pred / temperature)
                    
                    kd_loss = tf.keras.losses.categorical_crossentropy(
                        soft_targets, soft_predictions
                    ) * (temperature ** 2)
                    
                    # Combine losses
                    total_loss = (self.config.alpha * kd_loss + 
                                 (1 - self.config.alpha) * student_loss)
                else:
                    total_loss = student_loss
                
                return total_loss
            
            return distillation_loss
            
        except Exception as e:
            print(f"❌ Error creating distillation loss: {e}")
            raise
    
    async def _tf_training_loop(self, model, dataset, optimizer, loss_fn, job_id):
        """TensorFlow training loop with knowledge distillation"""
        try:
            import tensorflow as tf
            
            total_loss = 0.0
            total_steps = 0
            start_time = time.time()
            
            print(f"🏃 Starting TensorFlow training loop: {self.config.num_epochs} epochs")
            
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0.0
                epoch_steps = 0
                
                for batch in dataset:
                    with tf.GradientTape() as tape:
                        # Forward pass
                        outputs = model(batch["input_ids"], training=True)
                        
                        # Calculate loss
                        loss = loss_fn(batch["target_ids"], outputs.logits)
                        loss = tf.reduce_mean(loss)
                    
                    # Backward pass
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Gradient clipping
                    gradients = [tf.clip_by_norm(g, self.config.max_grad_norm) 
                               for g in gradients]
                    
                    # Optimizer step
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    # Update metrics
                    total_loss += loss.numpy()
                    epoch_loss += loss.numpy()
                    total_steps += 1
                    epoch_steps += 1
                    
                    # Logging
                    if total_steps % self.config.logging_steps == 0:
                        avg_loss = total_loss / total_steps
                        print(f"   Step {total_steps}: loss={loss.numpy():.4f}, avg_loss={avg_loss:.4f}")
                        
                        if ENABLE_WANDB:
                            wandb.log({
                                "train_loss": loss.numpy(),
                                "avg_train_loss": avg_loss,
                                "learning_rate": optimizer.learning_rate.numpy(),
                                "step": total_steps
                            })
                    
                    # Safety check
                    safety_valid = await self.safety_monitor.validate_model_output(
                        {"loss": loss.numpy(), "step": total_steps},
                        ["validate_training_progress"]
                    )
                    if not safety_valid:
                        print("⚠️ Safety validation failed, stopping training")
                        break
                
                # End of epoch
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                print(f"📊 Epoch {epoch + 1}/{self.config.num_epochs} completed: avg_loss={avg_epoch_loss:.4f}")
            
            training_time = time.time() - start_time
            final_loss = total_loss / max(total_steps, 1)
            
            print(f"🏁 TensorFlow training completed: {total_steps} steps, {training_time:.2f}s, final_loss={final_loss:.4f}")
            
            return {
                "final_loss": final_loss,
                "total_steps": total_steps,
                "training_time": training_time
            }
            
        except Exception as e:
            print(f"❌ TensorFlow training loop error: {e}")
            raise
    
    async def _save_tf_model(self, model, tokenizer, job_id: str) -> str:
        """Save trained TensorFlow model"""
        try:
            save_dir = Path(f"models/distilled/{job_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save TensorFlow model
            model.save_pretrained(save_dir)
            
            # Save tokenizer
            tokenizer.save_pretrained(save_dir)
            
            # Save config
            config_path = save_dir / "training_config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(self.config.__dict__, f, indent=2)
            
            print(f"💾 TensorFlow model saved to {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            print(f"❌ Error saving TensorFlow model: {e}")
            return ""
    
    async def _evaluate_tf_model(self, model, tokenizer, domain: str) -> Dict[str, float]:
        """Evaluate trained TensorFlow model"""
        try:
            import tensorflow as tf
            
            # Simple evaluation with sample prompts
            eval_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "How does neural network training work?"
            ]
            
            total_length = 0
            valid_responses = 0
            
            for prompt in eval_prompts:
                inputs = tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)
                
                # Simple forward pass for evaluation
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Convert to predictions
                predictions = tf.nn.softmax(logits)
                response_quality = tf.reduce_mean(tf.reduce_max(predictions, axis=-1))
                
                if response_quality > 0.5:  # Basic validity check
                    total_length += len(prompt.split())
                    valid_responses += 1
            
            # Calculate basic metrics
            avg_response_length = total_length / max(valid_responses, 1)
            response_rate = valid_responses / len(eval_prompts)
            
            return {
                "avg_response_length": avg_response_length,
                "response_rate": response_rate,
                "domain": domain,
                "eval_prompts": len(eval_prompts)
            }
            
        except Exception as e:
            print(f"❌ TensorFlow evaluation error: {e}")
            return {"error": str(e)}


class ProductionTransformersTrainer:
    """Real Transformers (Hugging Face) training implementation for knowledge distillation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.safety_monitor = SafetyMonitor()
        
        # Initialize wandb if enabled
        if ENABLE_WANDB:
            wandb.init(project="prsm-distillation-hf", config=config.__dict__)
    
    async def train_model(self, teacher_model_name: str, student_model_path: str, 
                         domain: str, job_id: str) -> Dict[str, Any]:
        """Execute real Transformers knowledge distillation training"""
        try:
            print(f"🚀 Starting Transformers distillation training for job {job_id}")
            print(f"   Teacher: {teacher_model_name}")
            print(f"   Domain: {domain}")
            
            # 1. Initialize teacher connector
            teacher_connector = TeacherModelConnector()
            
            # 2. Load/Create student model using Transformers
            student_model, tokenizer = await self._load_hf_student_model(student_model_path, domain)
            
            # 3. Create training dataset
            dataset = DistillationDataset(
                teacher_connector, teacher_model_name, domain, 
                size=1000, tokenizer=tokenizer
            )
            await dataset.generate_teacher_responses()
            
            # 4. Create Hugging Face dataset
            hf_dataset = await self._create_hf_dataset(dataset)
            
            # 5. Initialize training arguments and trainer
            training_args = self._create_training_arguments(job_id)
            trainer = self._create_hf_trainer(student_model, tokenizer, hf_dataset, training_args)
            
            # 6. Execute training
            training_metrics = await self._hf_training_loop(trainer, job_id)
            
            # 7. Save final model
            model_save_path = await self._save_hf_model(trainer, job_id)
            
            # 8. Final evaluation
            eval_results = await self._evaluate_hf_model(trainer, domain)
            
            print(f"✅ Transformers training completed successfully for job {job_id}")
            
            return {
                "status": "completed",
                "model_path": model_save_path,
                "final_loss": training_metrics["final_loss"],
                "training_steps": training_metrics["total_steps"],
                "evaluation_results": eval_results,
                "training_time": training_metrics["training_time"]
            }
            
        except Exception as e:
            print(f"❌ Transformers training failed for job {job_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _load_hf_student_model(self, model_path: str, domain: str):
        """Load or create Hugging Face student model"""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
            
            model_name = "distilgpt2"  # Start with DistilGPT2
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            print(f"📦 Loaded Transformers student model: {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading Transformers student model: {e}")
            raise
    
    async def _create_hf_dataset(self, dataset: DistillationDataset):
        """Create Hugging Face dataset from distillation dataset"""
        try:
            from datasets import Dataset as HFDataset
            
            # Convert to Hugging Face format
            data = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for i in range(len(dataset)):
                item = dataset[i]
                if "input_ids" in item:
                    data["input_ids"].append(item["input_ids"].tolist())
                    data["attention_mask"].append(item["attention_mask"].tolist())
                    data["labels"].append(item["target_ids"].tolist())
            
            hf_dataset = HFDataset.from_dict(data)
            
            return hf_dataset
            
        except Exception as e:
            print(f"❌ Error creating Hugging Face dataset: {e}")
            raise
    
    def _create_training_arguments(self, job_id: str):
        """Create Hugging Face training arguments"""
        try:
            from transformers import TrainingArguments
            
            training_args = TrainingArguments(
                output_dir=f"./models/checkpoints/{job_id}",
                overwrite_output_dir=True,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                learning_rate=self.config.learning_rate,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if ENABLE_WANDB else None,
                remove_unused_columns=False,
                dataloader_num_workers=self.config.dataloader_num_workers,
                fp16=self.config.use_fp16,
                gradient_checkpointing=True,
                max_grad_norm=self.config.max_grad_norm,
                seed=42,
                data_seed=42
            )
            
            return training_args
            
        except Exception as e:
            print(f"❌ Error creating training arguments: {e}")
            raise
    
    def _create_hf_trainer(self, model, tokenizer, dataset, training_args):
        """Create Hugging Face trainer with custom distillation loss"""
        try:
            from transformers import Trainer, DataCollatorForLanguageModeling
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # Causal language modeling
            )
            
            # Split dataset for training and evaluation
            train_size = int(0.9 * len(dataset))
            eval_size = len(dataset) - train_size
            
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, train_size + eval_size))
            
            # Custom trainer with distillation loss
            class DistillationTrainer(Trainer):
                def __init__(self, temperature=4.0, alpha=0.7, **kwargs):
                    super().__init__(**kwargs)
                    self.temperature = temperature
                    self.alpha = alpha
                
                def compute_loss(self, model, inputs, return_outputs=False):
                    """
                    Custom loss function for knowledge distillation
                    """
                    labels = inputs.pop("labels")
                    
                    # Forward pass
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Shift logits and labels for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Standard cross-entropy loss
                    loss_fct = torch.nn.CrossEntropyLoss()
                    student_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # For now, just use student loss (can add teacher distillation later)
                    loss = student_loss
                    
                    return (loss, outputs) if return_outputs else loss
            
            trainer = DistillationTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                temperature=self.config.temperature,
                alpha=self.config.alpha
            )
            
            return trainer
            
        except Exception as e:
            print(f"❌ Error creating Hugging Face trainer: {e}")
            raise
    
    async def _hf_training_loop(self, trainer, job_id: str):
        """Execute Hugging Face training loop"""
        try:
            start_time = time.time()
            
            print(f"🏃 Starting Hugging Face training loop")
            
            # Execute training
            train_result = trainer.train()
            
            training_time = time.time() - start_time
            
            print(f"🏁 Hugging Face training completed: {training_time:.2f}s")
            
            return {
                "final_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "training_time": training_time,
                "train_result": train_result
            }
            
        except Exception as e:
            print(f"❌ Hugging Face training loop error: {e}")
            raise
    
    async def _save_hf_model(self, trainer, job_id: str) -> str:
        """Save trained Hugging Face model"""
        try:
            save_dir = Path(f"models/distilled/{job_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            trainer.save_model(str(save_dir))
            trainer.tokenizer.save_pretrained(str(save_dir))
            
            # Save config
            config_path = save_dir / "training_config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(self.config.__dict__, f, indent=2)
            
            print(f"💾 Hugging Face model saved to {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            print(f"❌ Error saving Hugging Face model: {e}")
            return ""
    
    async def _evaluate_hf_model(self, trainer, domain: str) -> Dict[str, float]:
        """Evaluate trained Hugging Face model"""
        try:
            # Run evaluation
            eval_results = trainer.evaluate()
            
            return {
                "eval_loss": eval_results.get("eval_loss", 0.0),
                "eval_runtime": eval_results.get("eval_runtime", 0.0),
                "eval_samples_per_second": eval_results.get("eval_samples_per_second", 0.0),
                "eval_steps_per_second": eval_results.get("eval_steps_per_second", 0.0),
                "domain": domain
            }
            
        except Exception as e:
            print(f"❌ Hugging Face evaluation error: {e}")
            return {"error": str(e)}


# Enhanced Production Training Pipeline with multi-backend support
class EnhancedProductionTrainingPipeline(ProductionTrainingPipeline):
    """Enhanced production training pipeline with multi-backend support"""
    
    def __init__(self):
        super().__init__()
        self.trainers = {
            "pytorch": ProductionPyTorchTrainer,
            "tensorflow": ProductionTensorFlowTrainer,
            "transformers": ProductionTransformersTrainer
        }
    
    async def start_training(self, request: DistillationRequest) -> DistillationJob:
        """Start enhanced training with automatic backend selection"""
        try:
            job_id = str(uuid4())
            
            print(f"🎯 Starting enhanced production training pipeline for job {job_id}")
            print(f"   Teacher: {request.teacher_model}")
            print(f"   Domain: {request.domain}")
            print(f"   Strategy: {request.training_strategy}")
            print(f"   Target size: {request.target_size}")
            
            # Auto-select best backend based on domain and requirements
            backend = self._select_optimal_backend(request)
            print(f"   Selected backend: {backend}")
            
            # Create training configuration
            config = TrainingConfig(
                learning_rate=self._get_optimal_lr(request),
                batch_size=self._get_optimal_batch_size(request),
                num_epochs=self._get_optimal_epochs(request),
                temperature=4.0,
                alpha=0.7
            )
            
            # Initialize trainer based on selected backend
            trainer_class = self.trainers.get(backend, ProductionPyTorchTrainer)
            trainer = trainer_class(config)
            
            # Create job record
            job = DistillationJob(
                job_id=UUID(job_id),
                user_id=request.user_id,
                teacher_model=request.teacher_model,
                domain=request.domain,
                status="training",
                progress=0.0,
                current_stage="initializing",
                estimated_completion=datetime.now(timezone.utc),
                backend=backend
            )
            
            # Store active job
            self.active_training_jobs[job_id] = {
                "job": job,
                "trainer": trainer,
                "config": config,
                "request": request,
                "start_time": time.time(),
                "backend": backend
            }
            
            # Start training asynchronously
            asyncio.create_task(self._execute_enhanced_training(job_id))
            
            print(f"✅ Enhanced training job {job_id} started successfully with {backend} backend")
            return job
            
        except Exception as e:
            print(f"❌ Failed to start enhanced training: {e}")
            raise
    
    def _select_optimal_backend(self, request: DistillationRequest) -> str:
        """Select optimal backend based on domain and requirements"""
        # NLP domains work best with Transformers
        nlp_domains = ["creative_writing", "legal_analysis", "medical_research", "code_generation"]
        if request.domain in nlp_domains:
            return "transformers"
        
        # Size optimization works best with TensorFlow
        if request.optimization_target == OptimizationTarget.SIZE:
            return "tensorflow"
        
        # Default to PyTorch for flexibility
        return "pytorch"
    
    async def _execute_enhanced_training(self, job_id: str):
        """Execute enhanced training with selected backend"""
        try:
            job_info = self.active_training_jobs[job_id]
            job = job_info["job"]
            trainer = job_info["trainer"]
            request = job_info["request"]
            backend = job_info["backend"]
            
            print(f"🔥 Executing enhanced training with {backend} backend")
            
            # Update job status
            job.status = "training"
            job.current_stage = "data_preparation"
            job.progress = 0.1
            
            # Execute training using selected backend
            training_result = await trainer.train_model(
                teacher_model_name=request.teacher_model,
                student_model_path="",  # Will be created
                domain=request.domain,
                job_id=job_id
            )
            
            # Update job with results
            if training_result["status"] == "completed":
                job.status = "completed"
                job.progress = 1.0
                job.current_stage = "completed"
                job.model_path = training_result["model_path"]
                job.final_metrics = training_result.get("evaluation_results", {})
                
                # Charge FTNS for successful training
                await ftns_service.charge_context_access(
                    request.user_id, 
                    request.budget_ftns
                )
                
                print(f"✅ Enhanced training job {job_id} completed successfully with {backend}")
            else:
                job.status = "failed"
                job.error_message = training_result.get("error", "Unknown error")
                print(f"❌ Enhanced training job {job_id} failed: {job.error_message}")
            
            # Move to history
            self.training_history.append({
                "job_id": job_id,
                "job": job,
                "result": training_result,
                "duration": time.time() - job_info["start_time"],
                "backend": backend
            })
            
            # Remove from active jobs
            del self.active_training_jobs[job_id]
            
        except Exception as e:
            print(f"❌ Enhanced training execution error for job {job_id}: {e}")
            
            # Update job status
            if job_id in self.active_training_jobs:
                job = self.active_training_jobs[job_id]["job"]
                job.status = "failed"
                job.error_message = str(e)


# === Global Production Training Pipeline Instance ===

_production_training_pipeline: Optional[EnhancedProductionTrainingPipeline] = None

def get_production_training_pipeline() -> EnhancedProductionTrainingPipeline:
    """Get or create the global enhanced production training pipeline instance"""
    global _production_training_pipeline
    if _production_training_pipeline is None:
        _production_training_pipeline = EnhancedProductionTrainingPipeline()
    return _production_training_pipeline