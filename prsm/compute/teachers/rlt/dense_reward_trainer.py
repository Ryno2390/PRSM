"""
RLT Dense Reward Trainer Implementation

Implements Sakana AI's dense reward methodology for training teacher models
focused on effective student distillation rather than traditional problem-solving.

Key Innovation: Teachers receive both question AND solution, then generate
instructive explanations that maximize student comprehension.

Reward System:
- r_SS: Student Solution understanding (log probabilities on solution tokens)
- r_KL: Logical continuity (KL divergence between teacher/student distributions)
"""

import asyncio
import math
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncGenerator
from uuid import UUID, uuid4
import structlog

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ...core.models import PRSMBaseModel

logger = structlog.get_logger()


@dataclass
class RLTTrainingConfig:
    """Configuration for RLT dense reward training"""
    # Reward system parameters
    alpha: float = 0.1  # Weight for r_KL (logical continuity)
    beta: float = 0.01  # KL regularization term
    gamma: float = 0.99  # Discount factor for long sequences
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 16
    max_explanation_length: int = 2048
    num_training_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Evaluation parameters
    student_evaluation_batch_size: int = 8
    min_comprehension_threshold: float = 0.6
    quality_assessment_frequency: int = 100  # steps
    
    # Model parameters
    teacher_model_name: str = "microsoft/DialoGPT-medium"
    student_model_name: str = "microsoft/DialoGPT-small"
    max_sequence_length: int = 1024


class RLTTrainingDataset(Dataset):
    """Dataset for RLT training with question+solution pairs"""
    
    def __init__(
        self,
        questions: List[str],
        solutions: List[str],
        tokenizer,
        max_length: int = 1024
    ):
        self.questions = questions
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        question = self.questions[idx]
        solution = self.solutions[idx]
        
        # Format input as: Question: {question}\nSolution: {solution}\nExplain:
        input_text = f"Question: {question}\nSolution: {solution}\nExplain:"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "question": question,
            "solution": solution
        }


class RLTDenseRewardTrainer:
    """
    Core RLT training pipeline implementing Sakana's dense reward methodology.
    
    Trains teacher models to generate effective explanations given question+solution pairs,
    using student comprehension as the primary reward signal.
    """
    
    def __init__(
        self,
        config: Optional[RLTTrainingConfig] = None,
        teacher_model = None,
        student_model = None,
        tokenizer = None
    ):
        self.config = config or RLTTrainingConfig()
        self.logger = logger.bind(component="RLTDenseRewardTrainer")
        
        # Initialize models and tokenizer
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        
        # Training state
        self.training_step = 0
        self.total_rewards = []
        self.quality_scores = []
        
        # Performance tracking
        self.metrics = {
            "avg_r_ss": 0.0,
            "avg_r_kl": 0.0,
            "avg_total_reward": 0.0,
            "explanation_quality": 0.0,
            "student_comprehension": 0.0
        }
        
    async def initialize_models(self):
        """Initialize teacher and student models with proper configuration"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and Transformers required for RLT training")
            
        try:
            # Initialize teacher model (the one being trained)
            if self.teacher_model is None:
                self.logger.info("Initializing teacher model", model=self.config.teacher_model_name)
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    self.config.teacher_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            # Initialize student model (for evaluation)
            if self.student_model is None:
                self.logger.info("Initializing student model", model=self.config.student_model_name)
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    self.config.student_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            # Initialize tokenizer
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.teacher_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.teacher_model = self.teacher_model.to(device)
            self.student_model = self.student_model.to(device)
            
            self.logger.info("Models initialized successfully", device=str(device))
            
        except Exception as e:
            self.logger.error("Failed to initialize models", error=str(e))
            raise
    
    def compute_rss_reward(
        self,
        explanation: str,
        student_response: str,
        solution: str,
        question: str
    ) -> float:
        """
        Compute r_SS reward: Student Solution understanding
        
        Measures how well the student understands the solution given the explanation.
        Uses log probabilities over solution tokens with average and minimum operations.
        """
        try:
            if not PYTORCH_AVAILABLE:
                return 0.0
                
            # Format student input: Question + Explanation
            student_input = f"Question: {question}\nExplanation: {explanation}\nSolution:"
            
            # Tokenize input and solution
            input_encoding = self.tokenizer(
                student_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            solution_encoding = self.tokenizer(
                solution,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            # Get student model predictions
            self.student_model.eval()
            with torch.no_grad():
                outputs = self.student_model(**input_encoding)
                logits = outputs.logits
                
                # Get log probabilities for solution tokens
                log_probs = F.log_softmax(logits, dim=-1)
                solution_token_ids = solution_encoding["input_ids"][0]
                
                # Extract log probabilities for each solution token
                solution_log_probs = []
                for i, token_id in enumerate(solution_token_ids):
                    if i < log_probs.size(1):
                        token_log_prob = log_probs[0, i, token_id].item()
                        solution_log_probs.append(token_log_prob)
                
                if not solution_log_probs:
                    return 0.0
                    
                # Compute r_SS as average + alpha * minimum (as per Sakana paper)
                avg_log_prob = np.mean(solution_log_probs)
                min_log_prob = np.min(solution_log_probs)
                
                r_ss = avg_log_prob + self.config.alpha * min_log_prob
                
                return float(r_ss)
                
        except Exception as e:
            self.logger.warning("Error computing r_SS reward", error=str(e))
            return 0.0
    
    def compute_rkl_reward(
        self,
        explanation: str,
        question: str,
        solution: str
    ) -> float:
        """
        Compute r_KL reward: Logical continuity assessment
        
        Measures whether the explanation tokens are interpretable logical continuations
        from the student's perspective compared to the teacher's distribution.
        """
        try:
            if not PYTORCH_AVAILABLE:
                return 0.0
                
            # Format inputs for teacher and student
            teacher_input = f"Question: {question}\nSolution: {solution}\nExplain:"
            student_input = f"Question: {question}\nExplain:"
            
            # Tokenize explanation
            explanation_encoding = self.tokenizer(
                explanation,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Get teacher distribution
            teacher_encoding = self.tokenizer(
                teacher_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            # Get student distribution
            student_encoding = self.tokenizer(
                student_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            self.teacher_model.eval()
            self.student_model.eval()
            
            with torch.no_grad():
                # Get teacher probabilities
                teacher_outputs = self.teacher_model(**teacher_encoding)
                teacher_logits = teacher_outputs.logits
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                # Get student probabilities
                student_outputs = self.student_model(**student_encoding)
                student_logits = student_outputs.logits
                student_probs = F.softmax(student_logits, dim=-1)
                
                # Compute KL divergence between distributions
                # KL(teacher || student) for explanation tokens
                explanation_tokens = explanation_encoding["input_ids"][0]
                
                kl_divergences = []
                min_length = min(teacher_probs.size(1), student_probs.size(1))
                
                for i in range(min(len(explanation_tokens), min_length)):
                    teacher_dist = teacher_probs[0, i, :]
                    student_dist = student_probs[0, i, :]
                    
                    # Add small epsilon for numerical stability
                    eps = 1e-8
                    teacher_dist = teacher_dist + eps
                    student_dist = student_dist + eps
                    
                    # Compute KL divergence
                    kl_div = F.kl_div(
                        student_dist.log(),
                        teacher_dist,
                        reduction='sum'
                    ).item()
                    
                    kl_divergences.append(kl_div)
                
                if not kl_divergences:
                    return 0.0
                    
                # r_KL is negative KL divergence (lower divergence = higher reward)
                avg_kl = np.mean(kl_divergences)
                r_kl = -avg_kl  # Negative because we want to minimize KL divergence
                
                return float(r_kl)
                
        except Exception as e:
            self.logger.warning("Error computing r_KL reward", error=str(e))
            return 0.0
    
    def compute_total_reward(
        self,
        explanation: str,
        student_response: str,
        solution: str,
        question: str
    ) -> Dict[str, float]:
        """
        Compute total RLT reward combining r_SS and r_KL components.
        
        Returns:
            Dictionary containing individual rewards and total reward
        """
        r_ss = self.compute_rss_reward(explanation, student_response, solution, question)
        r_kl = self.compute_rkl_reward(explanation, question, solution)
        
        # Total reward as per Sakana methodology
        total_reward = r_ss + self.config.alpha * r_kl
        
        return {
            "r_ss": r_ss,
            "r_kl": r_kl,
            "total_reward": total_reward,
            "explanation_length": len(explanation.split()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def generate_explanation(
        self,
        question: str,
        solution: str,
        max_length: int = 512
    ) -> str:
        """
        Generate explanation using teacher model given question and solution.
        """
        try:
            # Format input with question and solution
            input_text = f"Question: {question}\nSolution: {solution}\nExplain:"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            # Move to device
            device = next(self.teacher_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate explanation
            self.teacher_model.eval()
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated tokens
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].size(1):],
                    skip_special_tokens=True
                )
                
                return generated_text.strip()
                
        except Exception as e:
            self.logger.error("Error generating explanation", error=str(e))
            return ""
    
    async def train_with_dense_rewards(
        self,
        questions: List[str],
        solutions: List[str],
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop implementing RLT dense reward methodology.
        
        Args:
            questions: List of question strings
            solutions: List of corresponding solution strings
            num_epochs: Number of training epochs (defaults to config)
            
        Returns:
            Training metrics and results
        """
        if not questions or not solutions or len(questions) != len(solutions):
            raise ValueError("Questions and solutions must be non-empty and same length")
            
        await self.initialize_models()
        
        num_epochs = num_epochs or self.config.num_training_epochs
        
        self.logger.info(
            "Starting RLT dense reward training",
            num_samples=len(questions),
            num_epochs=num_epochs,
            batch_size=self.config.batch_size
        )
        
        # Create dataset and dataloader
        dataset = RLTTrainingDataset(
            questions, solutions, self.tokenizer, self.config.max_sequence_length
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.teacher_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training metrics
        epoch_metrics = []
        total_steps = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_rewards = []
            epoch_losses = []
            
            self.teacher_model.train()
            
            for batch_idx, batch in enumerate(dataloader):
                step_start = time.time()
                
                # Generate explanations for batch
                batch_rewards = []
                batch_explanations = []
                
                for i in range(len(batch["question"])):
                    question = batch["question"][i]
                    solution = batch["solution"][i]
                    
                    # Generate explanation
                    explanation = await self.generate_explanation(question, solution)
                    batch_explanations.append(explanation)
                    
                    # Generate student response (simplified - could be more sophisticated)
                    student_response = f"Based on the explanation: {explanation[:100]}..."
                    
                    # Compute rewards
                    rewards = self.compute_total_reward(
                        explanation, student_response, solution, question
                    )
                    batch_rewards.append(rewards)
                    
                # Compute loss based on rewards (REINFORCE-style)
                total_loss = 0.0
                for i, rewards in enumerate(batch_rewards):
                    # Use negative reward as loss (we want to maximize reward)
                    loss = -rewards["total_reward"]
                    total_loss += loss
                    
                avg_loss = total_loss / len(batch_rewards)
                epoch_losses.append(avg_loss)
                
                # Backward pass
                optimizer.zero_grad()
                # Note: In a full implementation, you'd compute gradients through the generation process
                # This is a simplified version for demonstration
                
                # Track metrics
                avg_rewards = {
                    "r_ss": np.mean([r["r_ss"] for r in batch_rewards]),
                    "r_kl": np.mean([r["r_kl"] for r in batch_rewards]),
                    "total_reward": np.mean([r["total_reward"] for r in batch_rewards])
                }
                epoch_rewards.append(avg_rewards)
                
                total_steps += 1
                step_time = time.time() - step_start
                
                if batch_idx % 10 == 0:
                    self.logger.info(
                        "Training step",
                        epoch=epoch,
                        batch=batch_idx,
                        avg_reward=avg_rewards["total_reward"],
                        step_time=step_time
                    )
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            epoch_avg_reward = np.mean([r["total_reward"] for r in epoch_rewards])
            epoch_avg_loss = np.mean(epoch_losses)
            
            epoch_metric = {
                "epoch": epoch,
                "avg_reward": epoch_avg_reward,
                "avg_loss": epoch_avg_loss,
                "epoch_time": epoch_time,
                "total_steps": total_steps
            }
            epoch_metrics.append(epoch_metric)
            
            self.logger.info(
                "Epoch completed",
                **epoch_metric
            )
        
        # Final training results
        training_results = {
            "total_epochs": num_epochs,
            "total_steps": total_steps,
            "final_avg_reward": epoch_metrics[-1]["avg_reward"] if epoch_metrics else 0.0,
            "epoch_metrics": epoch_metrics,
            "config": self.config.__dict__
        }
        
        self.logger.info("RLT training completed", **training_results)
        return training_results
    
    async def evaluate_explanation_quality(
        self,
        question: str,
        solution: str,
        explanation: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a generated explanation using multiple metrics.
        """
        # Generate student response for evaluation
        student_response = f"Based on the explanation: {explanation}"
        
        # Compute rewards
        rewards = self.compute_total_reward(explanation, student_response, solution, question)
        
        # Additional quality metrics
        quality_metrics = {
            **rewards,
            "explanation_coherence": self._assess_coherence(explanation),
            "explanation_completeness": self._assess_completeness(explanation, solution),
            "student_comprehension_score": rewards["r_ss"]  # Use r_SS as comprehension proxy
        }
        
        return quality_metrics
    
    def _assess_coherence(self, explanation: str) -> float:
        """Simple coherence assessment based on text properties"""
        if not explanation:
            return 0.0
            
        # Simple heuristics for coherence
        sentences = explanation.split('.')
        if len(sentences) < 2:
            return 0.3
            
        # Check for logical connectors
        connectors = ['because', 'therefore', 'since', 'thus', 'hence', 'so']
        connector_count = sum(1 for conn in connectors if conn in explanation.lower())
        
        coherence_score = min(1.0, 0.5 + (connector_count * 0.1))
        return coherence_score
    
    def _assess_completeness(self, explanation: str, solution: str) -> float:
        """Simple completeness assessment"""
        if not explanation or not solution:
            return 0.0
            
        # Check if explanation mentions key elements from solution
        solution_words = set(solution.lower().split())
        explanation_words = set(explanation.lower().split())
        
        overlap = len(solution_words.intersection(explanation_words))
        completeness = min(1.0, overlap / max(len(solution_words), 1))
        
        return completeness


# Example usage and testing functionality
async def test_rlt_trainer():
    """Test function for RLT trainer with sample data"""
    # Sample training data
    questions = [
        "What is the derivative of x^2?",
        "How do you solve a quadratic equation?",
        "What is the Pythagorean theorem?"
    ]
    
    solutions = [
        "The derivative of x^2 is 2x",
        "Use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
        "In a right triangle, a² + b² = c² where c is the hypotenuse"
    ]
    
    # Initialize trainer
    config = RLTTrainingConfig(
        batch_size=2,
        num_training_epochs=1,
        learning_rate=1e-4
    )
    
    trainer = RLTDenseRewardTrainer(config=config)
    
    # Run training
    results = await trainer.train_with_dense_rewards(questions, solutions)
    
    print("Training Results:")
    print(f"Final Average Reward: {results['final_avg_reward']:.4f}")
    print(f"Total Steps: {results['total_steps']}")
    
    return results


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_rlt_trainer())