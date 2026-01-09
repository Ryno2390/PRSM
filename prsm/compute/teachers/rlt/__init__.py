"""
RLT (Reinforcement Learning Teachers) Framework for PRSM

Based on Sakana AI's "Reinforcement Learning Teachers of Test Time Scaling" research.
Implements dense reward training for teacher models focused on effective student distillation.

Key Components:
- RLTDenseRewardTrainer: Core training pipeline with dense rewards
- StudentCompressionEvaluator: Student comprehension assessment
- RLTFormatter: Input/output formatting for question+solution methodology
- DomainTransfer: Zero-shot domain transfer capabilities
- QualityMonitor: Real-time explanation quality monitoring
"""

from .dense_reward_trainer import RLTDenseRewardTrainer
from .student_comprehension_evaluator import StudentCompressionEvaluator
from .explanation_formatter import RLTFormatter
from .quality_monitor import RLTQualityMonitor

__all__ = [
    "RLTDenseRewardTrainer",
    "StudentCompressionEvaluator", 
    "RLTFormatter",
    "RLTQualityMonitor"
]