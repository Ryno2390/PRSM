"""
Evolution Archive System

Core archive infrastructure for storing and managing solution evolution.
Implements DGM-style archive with genealogy tracking and stepping-stone discovery.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid
import hashlib
from math import exp
import random

from .models import (
    ComponentType, SafetyStatus, RiskLevel, SelectionStrategy, 
    EvaluationResult, PerformanceStats, GenealogyTree, GenealogyNode, 
    ArchiveStats, SynchronizationResult
)
from prsm.core.ipfs_client import IPFSClient
from prsm.core.database_service import DatabaseService


logger = logging.getLogger(__name__)


class SolutionNode(BaseModel):
    """
    Individual solution node in the evolution archive.
    Represents a discovered configuration/solution with genealogy tracking.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Genealogy tracking
    parent_ids: List[str] = Field(default_factory=list)
    child_ids: List[str] = Field(default_factory=list)
    generation: int = 0
    
    # Solution content
    component_type: ComponentType
    configuration: Dict[str, Any] = Field(default_factory=dict)
    code_changes: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance tracking
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    evaluation_history: List[EvaluationResult] = Field(default_factory=list)
    best_performance: float = 0.0
    average_performance: float = 0.0
    
    # Evolution metadata
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_evaluation: Optional[datetime] = None
    modification_count: int = 0
    
    # Safety and validation
    safety_status: SafetyStatus = SafetyStatus.PENDING
    safety_violations: List[str] = Field(default_factory=list)
    validation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Archive management
    is_active: bool = True
    is_stepping_stone: bool = False
    stepping_stone_score: float = 0.0
    novelty_score: float = 0.0
    
    # Resource tracking
    compute_cost: float = 0.0
    storage_size_mb: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def performance(self) -> float:
        """Get the latest performance score."""
        if self.evaluation_history:
            return self.evaluation_history[-1].performance_score
        return 0.0
    
    @property
    def child_count(self) -> int:
        """Get number of children (for novelty calculations)."""
        return len(self.child_ids)
    
    def add_evaluation(self, evaluation: EvaluationResult):
        """Add new evaluation result."""
        self.evaluation_history.append(evaluation)
        self.last_evaluation = evaluation.timestamp
        
        # Update performance statistics
        performances = [e.performance_score for e in self.evaluation_history]
        self.best_performance = max(performances)
        self.average_performance = sum(performances) / len(performances)
        
        # Update performance metrics
        self.performance_metrics.update({
            'latest_score': evaluation.performance_score,
            'success_rate': evaluation.task_success_rate,
            'evaluation_count': len(self.evaluation_history)
        })
    
    def add_child(self, child_id: str):
        """Add child solution."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def calculate_selection_weight(self, strategy: SelectionStrategy) -> float:
        """Calculate selection weight based on strategy."""
        if strategy == SelectionStrategy.PERFORMANCE_WEIGHTED:
            # Sigmoid-scaled performance weight
            return 1 / (1 + exp(-10 * (self.performance - 0.5)))
        
        elif strategy == SelectionStrategy.NOVELTY_WEIGHTED:
            # Inverse of child count for novelty
            return 1 / (1 + self.child_count)
        
        elif strategy == SelectionStrategy.QUALITY_DIVERSITY:
            # Balance performance and novelty
            perf_weight = 1 / (1 + exp(-10 * (self.performance - 0.5)))
            novelty_weight = 1 / (1 + self.child_count)
            diversity_bonus = 1 + self.novelty_score
            return perf_weight * novelty_weight * diversity_bonus
        
        elif strategy == SelectionStrategy.STEPPING_STONE_FOCUSED:
            # Focus on solutions that could be stepping stones
            return self.stepping_stone_score * (1 + self.novelty_score)
        
        else:  # RANDOM
            return 1.0
    
    def get_content_hash(self) -> str:
        """Generate content hash for deduplication."""
        content = {
            'configuration': self.configuration,
            'code_changes': self.code_changes,
            'component_type': self.component_type.value
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class EvolutionArchive:
    """
    Evolution archive for storing and managing solution evolution.
    Implements DGM-style archive with genealogy tracking, selection strategies,
    and distributed synchronization capabilities.
    """
    
    def __init__(
        self, 
        archive_id: str, 
        component_type: ComponentType,
        storage_backend: Optional[Any] = None,
        ipfs_client: Optional[IPFSClient] = None,
        database_service: Optional[DatabaseService] = None
    ):
        self.archive_id = archive_id
        self.component_type = component_type
        self.storage_backend = storage_backend
        self.ipfs_client = ipfs_client
        self.database_service = database_service
        
        # In-memory archive
        self.solutions: Dict[str, SolutionNode] = {}
        self.genealogy_tree = GenealogyTree(root_solution_id="")
        
        # Archive metadata
        self.creation_timestamp = datetime.utcnow()
        self.last_sync_timestamp: Optional[datetime] = None
        self.total_evaluations = 0
        
        # Performance tracking
        self.performance_history: List[Tuple[datetime, float]] = []
        self.breakthrough_threshold = 0.1  # 10% improvement for breakthrough
        
        # Deduplication
        self.content_hashes: Dict[str, str] = {}  # hash -> solution_id
        
    async def add_solution(self, solution: SolutionNode) -> str:
        """Add solution to archive with deduplication."""
        
        # Check for duplicates
        content_hash = solution.get_content_hash()
        if content_hash in self.content_hashes:
            logger.info(f"Duplicate solution detected, skipping: {solution.id}")
            return self.content_hashes[content_hash]
        
        # Set initial root if this is the first solution
        if not self.solutions and not self.genealogy_tree.root_solution_id:
            self.genealogy_tree.root_solution_id = solution.id
        
        # Add to archive
        self.solutions[solution.id] = solution
        self.content_hashes[content_hash] = solution.id
        
        # Update genealogy tree
        genealogy_node = GenealogyNode(
            solution_id=solution.id,
            parent_ids=solution.parent_ids,
            child_ids=solution.child_ids,
            generation=solution.generation,
            creation_timestamp=solution.creation_timestamp,
            performance_score=solution.performance
        )
        self.genealogy_tree.add_node(genealogy_node)
        
        # Update parent-child relationships
        for parent_id in solution.parent_ids:
            if parent_id in self.solutions:
                self.solutions[parent_id].add_child(solution.id)
        
        # Persist to storage
        if self.storage_backend:
            await self._persist_solution(solution)
        
        logger.info(f"Added solution {solution.id} to archive {self.archive_id}")
        return solution.id
    
    async def get_solution(self, solution_id: str) -> Optional[SolutionNode]:
        """Get solution by ID."""
        if solution_id in self.solutions:
            return self.solutions[solution_id]
        
        # Try loading from storage
        if self.storage_backend:
            solution = await self._load_solution(solution_id)
            if solution:
                self.solutions[solution_id] = solution
                return solution
        
        return None
    
    async def select_parents(
        self, 
        k_parents: int, 
        strategy: SelectionStrategy = SelectionStrategy.QUALITY_DIVERSITY,
        exclude_perfect: bool = True
    ) -> List[SolutionNode]:
        """Select parent solutions for evolution using specified strategy."""
        
        # Get eligible solutions
        eligible_solutions = [
            solution for solution in self.solutions.values()
            if solution.is_active and 
            (not exclude_perfect or solution.performance < 0.99) and
            solution.safety_status in [SafetyStatus.SAFE, SafetyStatus.PENDING]
        ]
        
        if not eligible_solutions:
            logger.warning("No eligible solutions for parent selection")
            return []
        
        if len(eligible_solutions) <= k_parents:
            return eligible_solutions
        
        # Calculate selection weights
        weights = []
        for solution in eligible_solutions:
            weight = solution.calculate_selection_weight(strategy)
            weights.append(weight)
        
        # Weighted sampling without replacement
        selected_parents = []
        for _ in range(min(k_parents, len(eligible_solutions))):
            if not weights:
                break
                
            # Sample based on weights
            total_weight = sum(weights)
            if total_weight == 0:
                # Fallback to random selection
                idx = random.randint(0, len(eligible_solutions) - 1)
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0
                idx = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if cumulative >= r:
                        idx = i
                        break
            
            # Add selected parent and remove from consideration
            selected_parents.append(eligible_solutions[idx])
            eligible_solutions.pop(idx)
            weights.pop(idx)
        
        logger.info(f"Selected {len(selected_parents)} parents using {strategy}")
        return selected_parents
    
    async def get_genealogy(self, solution_id: str) -> GenealogyTree:
        """Get genealogy tree for a solution."""
        if solution_id not in self.solutions:
            raise ValueError(f"Solution {solution_id} not found in archive")
        
        # Return subtree rooted at the solution
        subtree = GenealogyTree(root_solution_id=solution_id)
        
        # Add all descendants
        descendants = self.genealogy_tree.get_descendants(solution_id, generations_forward=10)
        for node in descendants:
            subtree.add_node(node)
        
        return subtree
    
    async def archive_statistics(self) -> ArchiveStats:
        """Get comprehensive archive statistics."""
        if not self.solutions:
            return ArchiveStats(
                total_solutions=0,
                active_solutions=0,
                generations=0,
                average_performance=0.0,
                best_performance=0.0,
                performance_improvement_rate=0.0,
                diversity_score=0.0,
                stepping_stones_discovered=0,
                breakthrough_solutions=0,
                safety_violations=0,
                last_updated=datetime.utcnow()
            )
        
        active_solutions = [s for s in self.solutions.values() if s.is_active]
        performances = [s.performance for s in active_solutions]
        
        # Calculate performance metrics
        avg_performance = sum(performances) / len(performances) if performances else 0.0
        best_performance = max(performances) if performances else 0.0
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.performance_history) >= 2:
            recent_perf = sum(p[1] for p in self.performance_history[-10:]) / min(10, len(self.performance_history))
            early_perf = sum(p[1] for p in self.performance_history[:10]) / min(10, len(self.performance_history))
            if early_perf > 0:
                improvement_rate = (recent_perf - early_perf) / early_perf
        
        # Calculate diversity score (based on configuration diversity)
        diversity_score = await self._calculate_diversity_score()
        
        # Count special solutions
        stepping_stones = sum(1 for s in self.solutions.values() if s.is_stepping_stone)
        breakthroughs = await self._count_breakthrough_solutions()
        safety_violations = sum(len(s.safety_violations) for s in self.solutions.values())
        
        # Calculate generations
        generations = max((s.generation for s in self.solutions.values()), default=0)
        
        return ArchiveStats(
            total_solutions=len(self.solutions),
            active_solutions=len(active_solutions),
            generations=generations,
            average_performance=avg_performance,
            best_performance=best_performance,
            performance_improvement_rate=improvement_rate,
            diversity_score=diversity_score,
            stepping_stones_discovered=stepping_stones,
            breakthrough_solutions=breakthroughs,
            safety_violations=safety_violations,
            last_updated=datetime.utcnow()
        )
    
    async def identify_stepping_stones(self, lookback_generations: int = 5) -> List[SolutionNode]:
        """Identify solutions that could serve as stepping stones."""
        stepping_stones = []
        
        # Get breakthrough solutions (top 10% performers)
        all_solutions = list(self.solutions.values())
        all_solutions.sort(key=lambda x: x.performance, reverse=True)
        top_10_percent = int(len(all_solutions) * 0.1) or 1
        breakthrough_solutions = all_solutions[:top_10_percent]
        
        for breakthrough in breakthrough_solutions:
            # Analyze genealogy to find enabling ancestors
            ancestors = self.genealogy_tree.get_ancestors(breakthrough.id, lookback_generations)
            
            for ancestor_node in ancestors:
                ancestor = self.solutions.get(ancestor_node.solution_id)
                if not ancestor:
                    continue
                
                # Check if ancestor was significantly worse but enabled breakthrough
                performance_gap = breakthrough.performance - ancestor.performance
                if performance_gap > self.breakthrough_threshold:
                    # This ancestor might be a stepping stone
                    ancestor.is_stepping_stone = True
                    ancestor.stepping_stone_score = performance_gap * (1 + ancestor.novelty_score)
                    stepping_stones.append(ancestor)
        
        # Remove duplicates and sort by stepping stone score
        unique_stepping_stones = list({s.id: s for s in stepping_stones}.values())
        unique_stepping_stones.sort(key=lambda x: x.stepping_stone_score, reverse=True)
        
        logger.info(f"Identified {len(unique_stepping_stones)} stepping stones")
        return unique_stepping_stones
    
    async def get_breakthrough_solutions(self, threshold: float = 0.1) -> List[SolutionNode]:
        """Get solutions that represent significant breakthroughs."""
        if not self.solutions:
            return []
        
        breakthroughs = []
        sorted_solutions = sorted(self.solutions.values(), key=lambda x: x.creation_timestamp)
        
        current_best = 0.0
        for solution in sorted_solutions:
            if solution.performance > current_best + threshold:
                breakthroughs.append(solution)
                current_best = solution.performance
        
        return breakthroughs
    
    async def prune_archive(self, max_solutions: int = 10000) -> int:
        """Prune archive to maintain manageable size while preserving diversity."""
        if len(self.solutions) <= max_solutions:
            return 0
        
        # Calculate pruning priorities
        solutions_with_priority = []
        for solution in self.solutions.values():
            priority = self._calculate_pruning_priority(solution)
            solutions_with_priority.append((solution, priority))
        
        # Sort by priority (higher priority = keep)
        solutions_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top solutions
        solutions_to_keep = solutions_with_priority[:max_solutions]
        solutions_to_remove = solutions_with_priority[max_solutions:]
        
        # Remove low-priority solutions
        removed_count = 0
        for solution, _ in solutions_to_remove:
            if not solution.is_stepping_stone:  # Never remove stepping stones
                del self.solutions[solution.id]
                removed_count += 1
        
        logger.info(f"Pruned {removed_count} solutions from archive")
        return removed_count
    
    def _calculate_pruning_priority(self, solution: SolutionNode) -> float:
        """Calculate priority score for pruning decisions."""
        priority = 0.0
        
        # Performance contribution
        priority += solution.performance * 0.3
        
        # Recency bonus
        age_days = (datetime.utcnow() - solution.creation_timestamp).days
        recency_bonus = max(0, 1 - age_days / 365)  # Decay over a year
        priority += recency_bonus * 0.2
        
        # Stepping stone bonus
        if solution.is_stepping_stone:
            priority += solution.stepping_stone_score * 0.3
        
        # Diversity bonus
        priority += solution.novelty_score * 0.2
        
        return priority
    
    async def _calculate_diversity_score(self) -> float:
        """Calculate diversity score of archive solutions."""
        if len(self.solutions) < 2:
            return 0.0
        
        # Simplified diversity calculation based on configuration differences
        configurations = [s.configuration for s in self.solutions.values()]
        
        # Calculate pairwise differences
        total_differences = 0
        comparison_count = 0
        
        for i, config1 in enumerate(configurations):
            for config2 in configurations[i+1:]:
                differences = self._calculate_config_difference(config1, config2)
                total_differences += differences
                comparison_count += 1
        
        if comparison_count == 0:
            return 0.0
        
        return total_differences / comparison_count
    
    def _calculate_config_difference(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate difference between two configurations."""
        all_keys = set(config1.keys()) | set(config2.keys())
        if not all_keys:
            return 0.0
        
        different_keys = 0
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            if val1 != val2:
                different_keys += 1
        
        return different_keys / len(all_keys)
    
    async def _count_breakthrough_solutions(self) -> int:
        """Count solutions that represent breakthroughs."""
        breakthroughs = await self.get_breakthrough_solutions()
        return len(breakthroughs)
    
    async def _persist_solution(self, solution: SolutionNode):
        """Persist solution to storage backend."""
        if self.database_service:
            # Store in database
            await self.database_service.store_solution(self.archive_id, solution)
        
        if self.ipfs_client:
            # Store in IPFS for distributed access
            solution_data = solution.dict()
            ipfs_hash = await self.ipfs_client.add_json(solution_data)
            logger.debug(f"Solution {solution.id} stored in IPFS: {ipfs_hash}")
    
    async def _load_solution(self, solution_id: str) -> Optional[SolutionNode]:
        """Load solution from storage backend."""
        if self.database_service:
            solution_data = await self.database_service.get_solution(self.archive_id, solution_id)
            if solution_data:
                return SolutionNode(**solution_data)
        
        return None
    
    async def synchronize_with_peers(self, peer_archives: List[str]) -> SynchronizationResult:
        """Synchronize archive with peer nodes."""
        start_time = datetime.utcnow()
        solutions_shared = 0
        solutions_received = 0
        conflicts_resolved = 0
        errors = []
        
        try:
            # This would implement actual peer synchronization
            # For now, return a mock result
            logger.info(f"Synchronizing archive {self.archive_id} with {len(peer_archives)} peers")
            
            # Mock synchronization logic would go here
            await asyncio.sleep(0.1)  # Simulate network delay
            
            self.last_sync_timestamp = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Archive synchronization failed: {e}")
            errors.append(str(e))
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return SynchronizationResult(
            solutions_shared=solutions_shared,
            solutions_received=solutions_received,
            conflicts_resolved=conflicts_resolved,
            synchronization_time_seconds=duration,
            bandwidth_used_mb=0.0,  # Would be calculated in real implementation
            errors=errors,
            timestamp=datetime.utcnow()
        )