"""
Student Comprehension Evaluator for RLT Framework

Implements sophisticated student comprehension assessment using multiple evaluation
strategies to measure how well students understand explanations and solutions.

Key Features:
- Log probability analysis for solution understanding
- Attention pattern analysis for explanation focus
- Multi-metric comprehension scoring
- Real-time feedback for teacher improvement
"""

import asyncio
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
import structlog

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM,
        AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from prsm.core.models import PRSMBaseModel

logger = structlog.get_logger()


@dataclass
class ComprehensionMetrics:
    """Comprehensive metrics for student comprehension assessment"""
    solution_understanding: float  # How well student understands the solution
    explanation_coherence: float   # How coherent the explanation appears to student
    logical_flow: float           # Whether explanation follows logical progression
    key_concept_grasp: float      # Understanding of key concepts in explanation
    overall_comprehension: float   # Weighted overall comprehension score
    confidence_score: float       # Confidence in the assessment
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "solution_understanding": self.solution_understanding,
            "explanation_coherence": self.explanation_coherence,
            "logical_flow": self.logical_flow,
            "key_concept_grasp": self.key_concept_grasp,
            "overall_comprehension": self.overall_comprehension,
            "confidence_score": self.confidence_score
        }


@dataclass
class EvaluationConfig:
    """Configuration for student comprehension evaluation"""
    # Model parameters
    student_model_name: str = "microsoft/DialoGPT-small"
    max_sequence_length: int = 1024
    
    # Evaluation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    num_evaluation_samples: int = 5
    
    # Comprehension thresholds
    min_comprehension_threshold: float = 0.6
    high_comprehension_threshold: float = 0.8
    
    # Metric weights
    solution_weight: float = 0.4
    coherence_weight: float = 0.2
    logical_flow_weight: float = 0.2
    concept_grasp_weight: float = 0.2


class StudentCompressionEvaluator:
    """
    Advanced student comprehension evaluator implementing multiple assessment strategies.
    
    Uses language model probabilities, attention patterns, and semantic analysis
    to evaluate how well students understand teacher explanations.
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        student_model = None,
        tokenizer = None
    ):
        self.config = config or EvaluationConfig()
        self.logger = logger.bind(component="StudentCompressionEvaluator")
        
        # Models and tokenizer
        self.student_model = student_model
        self.tokenizer = tokenizer
        
        # Evaluation cache
        self.evaluation_cache = {}
        self.comprehension_history = []
        
        # Performance tracking
        self.evaluation_count = 0
        self.avg_comprehension_score = 0.0
        
    async def initialize_models(self):
        """Initialize student model and tokenizer for evaluation"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and Transformers required for comprehension evaluation")
            
        try:
            # Initialize student model
            if self.student_model is None:
                self.logger.info("Initializing student model", model=self.config.student_model_name)
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    self.config.student_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    output_attentions=True,  # Enable attention output for analysis
                )
                
            # Initialize tokenizer
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.student_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            # Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.student_model = self.student_model.to(device)
            
            self.logger.info("Student model initialized successfully", device=str(device))
            
        except Exception as e:
            self.logger.error("Failed to initialize student model", error=str(e))
            raise
    
    async def evaluate_solution_understanding(
        self,
        question: str,
        explanation: str,
        solution: str
    ) -> float:
        """
        Evaluate how well the student understands the solution given the explanation.
        
        Uses log probabilities over solution tokens to measure understanding.
        """
        try:
            if not PYTORCH_AVAILABLE:
                return 0.0
                
            # Format student input: Question + Explanation -> Solution
            student_input = f"Question: {question}\nExplanation: {explanation}\nBased on this explanation, the solution is:"
            
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
            
            # Move to device
            device = next(self.student_model.parameters()).device
            input_encoding = {k: v.to(device) for k, v in input_encoding.items()}
            solution_encoding = {k: v.to(device) for k, v in solution_encoding.items()}
            
            # Get student model predictions
            self.student_model.eval()
            with torch.no_grad():
                outputs = self.student_model(**input_encoding)
                logits = outputs.logits
                
                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Extract log probabilities for solution tokens
                solution_token_ids = solution_encoding["input_ids"][0]
                solution_log_probs = []
                
                # Get probabilities for each solution token
                for i, token_id in enumerate(solution_token_ids):
                    if i < log_probs.size(1) - 1:  # Ensure we don't go out of bounds
                        token_log_prob = log_probs[0, i, token_id].item()
                        solution_log_probs.append(token_log_prob)
                
                if not solution_log_probs:
                    return 0.0
                
                # Compute understanding score
                # Use both average and minimum as per Sakana methodology
                avg_log_prob = np.mean(solution_log_probs)
                min_log_prob = np.min(solution_log_probs)
                
                # Normalize to 0-1 range (log probs are negative)
                understanding_score = math.exp(avg_log_prob + 0.1 * min_log_prob)
                understanding_score = min(1.0, max(0.0, understanding_score))
                
                return understanding_score
                
        except Exception as e:
            self.logger.warning("Error evaluating solution understanding", error=str(e))
            return 0.0
    
    async def evaluate_explanation_coherence(
        self,
        question: str,
        explanation: str
    ) -> float:
        """
        Evaluate the coherence of explanation from student's perspective.
        
        Measures how well the explanation flows logically and makes sense.
        """
        try:
            # Format coherence evaluation prompt
            coherence_input = f"Question: {question}\nExplanation: {explanation}\nIs this explanation coherent and easy to follow? (Yes/No):"
            
            # Tokenize
            inputs = self.tokenizer(
                coherence_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            # Move to device
            device = next(self.student_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            self.student_model.eval()
            with torch.no_grad():
                outputs = self.student_model(**inputs)
                logits = outputs.logits
                
                # Get probabilities for "Yes" and "No" tokens
                yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                
                # Get last token probabilities
                last_token_logits = logits[0, -1, :]
                last_token_probs = F.softmax(last_token_logits, dim=-1)
                
                yes_prob = last_token_probs[yes_token_id].item()
                no_prob = last_token_probs[no_token_id].item()
                
                # Normalize to get coherence score
                total_prob = yes_prob + no_prob
                if total_prob > 0:
                    coherence_score = yes_prob / total_prob
                else:
                    coherence_score = 0.5  # Default neutral score
                
                return coherence_score
                
        except Exception as e:
            self.logger.warning("Error evaluating explanation coherence", error=str(e))
            return 0.5  # Default neutral score
    
    async def evaluate_logical_flow(
        self,
        explanation: str
    ) -> float:
        """
        Evaluate the logical flow of the explanation.
        
        Analyzes step-by-step progression and logical connectors.
        """
        try:
            if not explanation:
                return 0.0
            
            # Analyze logical structure
            sentences = explanation.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 0.3  # Too short for good logical flow
            
            # Check for logical connectors
            logical_connectors = [
                'first', 'second', 'third', 'then', 'next', 'after', 'before',
                'because', 'since', 'therefore', 'thus', 'hence', 'so',
                'however', 'although', 'while', 'whereas', 'on the other hand'
            ]
            
            connector_count = 0
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for connector in logical_connectors:
                    if connector in sentence_lower:
                        connector_count += 1
                        break
            
            # Score based on connector usage and sentence structure
            connector_ratio = connector_count / len(sentences)
            
            # Check for step-wise progression
            step_indicators = ['step', 'stage', 'phase', '1.', '2.', '3.', 'a)', 'b)', 'c)']
            step_count = sum(1 for sentence in sentences 
                           for indicator in step_indicators 
                           if indicator in sentence.lower())
            
            step_score = min(1.0, step_count / max(len(sentences) * 0.3, 1))
            
            # Combine scores
            logical_flow_score = (connector_ratio * 0.6 + step_score * 0.4)
            logical_flow_score = min(1.0, logical_flow_score)
            
            return logical_flow_score
            
        except Exception as e:
            self.logger.warning("Error evaluating logical flow", error=str(e))
            return 0.5
    
    async def evaluate_key_concept_grasp(
        self,
        question: str,
        explanation: str,
        solution: str
    ) -> float:
        """
        Evaluate how well the explanation helps grasp key concepts.
        
        Measures concept coverage and clarity of key terms.
        """
        try:
            # Extract key terms from question and solution
            question_words = set(question.lower().split())
            solution_words = set(solution.lower().split())
            explanation_words = set(explanation.lower().split())
            
            # Identify potentially important terms (longer words, mathematical terms)
            important_words = set()
            
            for word in question_words.union(solution_words):
                if len(word) > 4 or any(char.isdigit() for char in word):
                    important_words.add(word)
            
            # Check concept coverage in explanation
            covered_concepts = important_words.intersection(explanation_words)
            
            if not important_words:
                return 1.0  # No specific concepts to cover
            
            concept_coverage = len(covered_concepts) / len(important_words)
            
            # Check for concept elaboration (explanation should be longer for complex concepts)
            elaboration_score = min(1.0, len(explanation.split()) / 50)
            
            # Combine scores
            concept_grasp_score = (concept_coverage * 0.7 + elaboration_score * 0.3)
            concept_grasp_score = min(1.0, concept_grasp_score)
            
            return concept_grasp_score
            
        except Exception as e:
            self.logger.warning("Error evaluating concept grasp", error=str(e))
            return 0.5
    
    async def extract_log_probabilities(
        self,
        input_text: str,
        target_tokens: List[str]
    ) -> List[float]:
        """
        Extract log probabilities for specific target tokens.
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            # Move to device
            device = next(self.student_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            self.student_model.eval()
            with torch.no_grad():
                outputs = self.student_model(**inputs)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Extract probabilities for target tokens
                target_log_probs = []
                for token in target_tokens:
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                    if token_ids:
                        token_id = token_ids[0]
                        # Get probability from last position
                        token_log_prob = log_probs[0, -1, token_id].item()
                        target_log_probs.append(token_log_prob)
                
                return target_log_probs
                
        except Exception as e:
            self.logger.warning("Error extracting log probabilities", error=str(e))
            return []
    
    async def compute_comprehension_score(
        self,
        question: str,
        explanation: str,
        solution: str
    ) -> ComprehensionMetrics:
        """
        Compute comprehensive student comprehension score using all evaluation methods.
        
        Returns:
            ComprehensionMetrics object with detailed assessment
        """
        await self.initialize_models()
        
        try:
            # Evaluate individual components
            solution_understanding = await self.evaluate_solution_understanding(
                question, explanation, solution
            )
            
            explanation_coherence = await self.evaluate_explanation_coherence(
                question, explanation
            )
            
            logical_flow = await self.evaluate_logical_flow(explanation)
            
            key_concept_grasp = await self.evaluate_key_concept_grasp(
                question, explanation, solution
            )
            
            # Compute weighted overall score
            overall_comprehension = (
                solution_understanding * self.config.solution_weight +
                explanation_coherence * self.config.coherence_weight +
                logical_flow * self.config.logical_flow_weight +
                key_concept_grasp * self.config.concept_grasp_weight
            )
            
            # Compute confidence based on consistency of individual scores
            scores = [solution_understanding, explanation_coherence, logical_flow, key_concept_grasp]
            score_variance = np.var(scores)
            confidence_score = max(0.0, 1.0 - score_variance)
            
            # Create metrics object
            metrics = ComprehensionMetrics(
                solution_understanding=solution_understanding,
                explanation_coherence=explanation_coherence,
                logical_flow=logical_flow,
                key_concept_grasp=key_concept_grasp,
                overall_comprehension=overall_comprehension,
                confidence_score=confidence_score
            )
            
            # Update tracking
            self.evaluation_count += 1
            self.comprehension_history.append(metrics)
            
            # Update running average
            self.avg_comprehension_score = (
                (self.avg_comprehension_score * (self.evaluation_count - 1) + overall_comprehension) 
                / self.evaluation_count
            )
            
            self.logger.info(
                "Comprehension evaluation completed",
                overall_score=overall_comprehension,
                confidence=confidence_score,
                evaluation_count=self.evaluation_count
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Error computing comprehension score", error=str(e))
            # Return default metrics on error
            return ComprehensionMetrics(
                solution_understanding=0.0,
                explanation_coherence=0.0,
                logical_flow=0.0,
                key_concept_grasp=0.0,
                overall_comprehension=0.0,
                confidence_score=0.0
            )
    
    async def batch_evaluate_comprehension(
        self,
        questions: List[str],
        explanations: List[str],
        solutions: List[str]
    ) -> List[ComprehensionMetrics]:
        """
        Evaluate comprehension for multiple question-explanation-solution triplets.
        """
        if len(questions) != len(explanations) or len(explanations) != len(solutions):
            raise ValueError("Questions, explanations, and solutions must have same length")
        
        results = []
        for i, (question, explanation, solution) in enumerate(zip(questions, explanations, solutions)):
            self.logger.info(f"Evaluating comprehension {i+1}/{len(questions)}")
            metrics = await self.compute_comprehension_score(question, explanation, solution)
            results.append(metrics)
        
        return results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed.
        """
        if not self.comprehension_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate aggregate statistics
        all_overall_scores = [m.overall_comprehension for m in self.comprehension_history]
        all_solution_scores = [m.solution_understanding for m in self.comprehension_history]
        all_coherence_scores = [m.explanation_coherence for m in self.comprehension_history]
        all_logical_scores = [m.logical_flow for m in self.comprehension_history]
        all_concept_scores = [m.key_concept_grasp for m in self.comprehension_history]
        all_confidence_scores = [m.confidence_score for m in self.comprehension_history]
        
        summary = {
            "total_evaluations": self.evaluation_count,
            "average_comprehension": self.avg_comprehension_score,
            "comprehension_statistics": {
                "mean": np.mean(all_overall_scores),
                "std": np.std(all_overall_scores),
                "min": np.min(all_overall_scores),
                "max": np.max(all_overall_scores),
                "median": np.median(all_overall_scores)
            },
            "component_averages": {
                "solution_understanding": np.mean(all_solution_scores),
                "explanation_coherence": np.mean(all_coherence_scores),
                "logical_flow": np.mean(all_logical_scores),
                "key_concept_grasp": np.mean(all_concept_scores),
                "confidence": np.mean(all_confidence_scores)
            },
            "evaluation_config": self.config.__dict__
        }
        
        return summary


# Example usage and testing
async def test_comprehension_evaluator():
    """Test function for comprehension evaluator"""
    # Sample data
    question = "What is the derivative of x^2?"
    explanation = "To find the derivative of x^2, we use the power rule. First, we identify that x^2 is a power function with exponent 2. Then, we apply the power rule: bring down the exponent (2) and multiply by x raised to the power of (2-1). Therefore, the derivative is 2x."
    solution = "The derivative of x^2 is 2x"
    
    # Initialize evaluator
    config = EvaluationConfig()
    evaluator = StudentCompressionEvaluator(config=config)
    
    # Evaluate comprehension
    metrics = await evaluator.compute_comprehension_score(question, explanation, solution)
    
    print("Comprehension Evaluation Results:")
    print(f"Overall Comprehension: {metrics.overall_comprehension:.3f}")
    print(f"Solution Understanding: {metrics.solution_understanding:.3f}")
    print(f"Explanation Coherence: {metrics.explanation_coherence:.3f}")
    print(f"Logical Flow: {metrics.logical_flow:.3f}")
    print(f"Key Concept Grasp: {metrics.key_concept_grasp:.3f}")
    print(f"Confidence Score: {metrics.confidence_score:.3f}")
    
    return metrics


# Alias for backward compatibility (common typo fix)
StudentComprehensionEvaluator = StudentCompressionEvaluator


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_comprehension_evaluator())