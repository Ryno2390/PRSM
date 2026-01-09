"""
PRSM Evolution System

Darwin Gödel Machine (DGM) enhanced evolution capabilities for self-improving AI infrastructure.
Enables recursive capability growth through archive-based exploration and empirical validation.
"""

from .archive import EvolutionArchive, SolutionNode, ArchiveStats, GenealogyTree
from .models import (
    EvaluationResult, ModificationProposal, SafetyValidationResult, 
    ModificationResult, SelectionStrategy, PerformanceStats
)
from .self_modification import SelfModifyingComponent, ModificationValidator
from .exploration import OpenEndedExplorationEngine, SteppingStoneAnalyzer

__all__ = [
    'EvolutionArchive',
    'SolutionNode', 
    'ArchiveStats',
    'GenealogyTree',
    'EvaluationResult',
    'ModificationProposal',
    'SafetyValidationResult',
    'ModificationResult',
    'SelectionStrategy',
    'PerformanceStats',
    'SelfModifyingComponent',
    'ModificationValidator',
    'OpenEndedExplorationEngine',
    'SteppingStoneAnalyzer'
]