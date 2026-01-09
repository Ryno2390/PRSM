"""
Open-Ended Exploration Engine

DGM-style exploration algorithms for discovering breakthrough innovations
through stepping-stone preservation and quality-diversity optimization.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from math import exp, sqrt
import numpy as np
from collections import defaultdict

from .models import (
    ComponentType, SelectionStrategy, EvaluationResult, 
    NetworkEvolutionResult, SynchronizationResult
)
from .archive import EvolutionArchive, SolutionNode


logger = logging.getLogger(__name__)


@dataclass
class ExplorationMetrics:
    """Metrics for tracking exploration effectiveness."""
    exploration_diversity: float
    stepping_stones_found: int
    breakthrough_rate: float
    novelty_score: float
    quality_diversity_ratio: float
    stagnation_periods: int
    last_breakthrough: Optional[datetime] = None


class OpenEndedExplorationEngine:
    """
    Open-ended exploration engine implementing DGM-style discovery algorithms.
    Balances exploitation of high-performing solutions with exploration of novel territories.
    """
    
    def __init__(self, archive: EvolutionArchive):
        self.archive = archive
        self.exploration_history: List[ExplorationMetrics] = []
        
        # Selection strategy configuration
        self.selection_strategies = {
            SelectionStrategy.PERFORMANCE_WEIGHTED: self._performance_weighted_selection,
            SelectionStrategy.NOVELTY_WEIGHTED: self._novelty_weighted_selection,
            SelectionStrategy.QUALITY_DIVERSITY: self._quality_diversity_selection,
            SelectionStrategy.STEPPING_STONE_FOCUSED: self._stepping_stone_focused_selection,
            SelectionStrategy.RANDOM: self._random_selection
        }
        
        # Exploration parameters
        self.diversity_threshold = 0.1
        self.breakthrough_threshold = 0.15
        self.stagnation_threshold = 10  # generations without improvement
        self.exploration_temperature = 1.0  # Controls exploration vs exploitation
        
        # Adaptive strategy parameters
        self.strategy_performance: Dict[SelectionStrategy, float] = defaultdict(float)
        self.strategy_usage_count: Dict[SelectionStrategy, int] = defaultdict(int)
        
    async def select_parents_for_evolution(
        self, 
        k_parents: int, 
        strategy: Optional[SelectionStrategy] = None,
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """
        Select parent solutions for evolution using adaptive strategy selection.
        
        Args:
            k_parents: Number of parents to select
            strategy: Specific strategy to use (if None, auto-select)
            focus_areas: Areas to focus exploration on
            
        Returns:
            List of selected parent solutions
        """
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = await self._select_exploration_strategy()
        
        # Execute selection strategy
        selection_function = self.selection_strategies[strategy]
        parents = await selection_function(k_parents, focus_areas)
        
        # Update strategy performance tracking
        self.strategy_usage_count[strategy] += 1
        
        # Log selection details
        logger.info(
            f"Selected {len(parents)} parents using {strategy.value} strategy. "
            f"Focus areas: {focus_areas or 'None'}"
        )
        
        return parents
    
    async def _select_exploration_strategy(self) -> SelectionStrategy:
        """Adaptively select exploration strategy based on current archive state."""
        archive_stats = await self.archive.archive_statistics()
        
        # Check for stagnation
        if await self._detect_stagnation():
            logger.info("Stagnation detected, increasing exploration")
            return random.choice([
                SelectionStrategy.NOVELTY_WEIGHTED,
                SelectionStrategy.STEPPING_STONE_FOCUSED,
                SelectionStrategy.RANDOM
            ])
        
        # Check diversity level
        if archive_stats.diversity_score < self.diversity_threshold:
            logger.info("Low diversity detected, focusing on novelty")
            return SelectionStrategy.NOVELTY_WEIGHTED
        
        # Check for recent breakthroughs
        recent_breakthroughs = await self._count_recent_breakthroughs()
        if recent_breakthroughs == 0:
            logger.info("No recent breakthroughs, exploring stepping stones")
            return SelectionStrategy.STEPPING_STONE_FOCUSED
        
        # Default to quality-diversity balance
        return SelectionStrategy.QUALITY_DIVERSITY
    
    async def _performance_weighted_selection(
        self, 
        k_parents: int, 
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """Select parents weighted by performance (sigmoid-scaled)."""
        eligible_solutions = await self._get_eligible_solutions(focus_areas)
        
        if not eligible_solutions:
            return []
        
        weights = []
        for solution in eligible_solutions:
            # Sigmoid-scaled performance weight (DGM formula)
            weight = 1 / (1 + exp(-10 * (solution.performance - 0.5)))
            weights.append(weight)
        
        return self._weighted_sample(eligible_solutions, weights, k_parents)
    
    async def _novelty_weighted_selection(
        self, 
        k_parents: int, 
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """Select parents weighted by novelty (inverse of child count)."""
        eligible_solutions = await self._get_eligible_solutions(focus_areas)
        
        if not eligible_solutions:
            return []
        
        weights = []
        for solution in eligible_solutions:
            # Novelty weight (inverse of child count)
            weight = 1 / (1 + solution.child_count)
            weights.append(weight)
        
        return self._weighted_sample(eligible_solutions, weights, k_parents)
    
    async def _quality_diversity_selection(
        self, 
        k_parents: int, 
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """Select parents balancing quality and diversity (DGM approach)."""
        eligible_solutions = await self._get_eligible_solutions(focus_areas)
        
        if not eligible_solutions:
            return []
        
        weights = []
        for solution in eligible_solutions:
            # Performance component (sigmoid-scaled)
            perf_weight = 1 / (1 + exp(-10 * (solution.performance - 0.5)))
            
            # Novelty component (inverse of child count)
            novelty_weight = 1 / (1 + solution.child_count)
            
            # Diversity bonus based on configuration uniqueness
            diversity_bonus = 1 + solution.novelty_score
            
            # Combined weight
            weight = perf_weight * novelty_weight * diversity_bonus
            weights.append(weight)
        
        return self._weighted_sample(eligible_solutions, weights, k_parents)
    
    async def _stepping_stone_focused_selection(
        self, 
        k_parents: int, 
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """Select parents focusing on potential stepping stones."""
        eligible_solutions = await self._get_eligible_solutions(focus_areas)
        
        if not eligible_solutions:
            return []
        
        # Update stepping stone scores
        await self._update_stepping_stone_scores(eligible_solutions)
        
        weights = []
        for solution in eligible_solutions:
            # Focus on stepping stone potential
            weight = solution.stepping_stone_score * (1 + solution.novelty_score)
            
            # Add exploration bonus for less-explored areas
            if solution.child_count < 2:
                weight *= 2.0
            
            weights.append(weight)
        
        return self._weighted_sample(eligible_solutions, weights, k_parents)
    
    async def _random_selection(
        self, 
        k_parents: int, 
        focus_areas: Optional[List[str]] = None
    ) -> List[SolutionNode]:
        """Random selection for maximum exploration."""
        eligible_solutions = await self._get_eligible_solutions(focus_areas)
        
        if not eligible_solutions:
            return []
        
        return random.sample(eligible_solutions, min(k_parents, len(eligible_solutions)))
    
    async def _get_eligible_solutions(self, focus_areas: Optional[List[str]] = None) -> List[SolutionNode]:
        """Get solutions eligible for parent selection."""
        all_solutions = list(self.archive.solutions.values())
        
        # Filter active, safe solutions with performance < 1.0
        eligible = [
            solution for solution in all_solutions
            if (solution.is_active and 
                solution.performance < 0.99 and  # Avoid perfect solutions
                solution.safety_status.value in ['SAFE', 'PENDING'])
        ]
        
        # Apply focus area filtering if specified
        if focus_areas:
            filtered = []
            for solution in eligible:
                config_keys = set(solution.configuration.keys())
                if any(area in config_keys for area in focus_areas):
                    filtered.append(solution)
            eligible = filtered if filtered else eligible
        
        return eligible
    
    def _weighted_sample(
        self, 
        solutions: List[SolutionNode], 
        weights: List[float], 
        k: int
    ) -> List[SolutionNode]:
        """Sample solutions without replacement using weights."""
        if not solutions or not weights:
            return []
        
        k = min(k, len(solutions))
        selected = []
        remaining_solutions = solutions.copy()
        remaining_weights = weights.copy()
        
        for _ in range(k):
            if not remaining_weights or not remaining_solutions:
                break
            
            # Normalize weights
            total_weight = sum(remaining_weights)
            if total_weight == 0:
                # Fallback to random selection
                idx = random.randint(0, len(remaining_solutions) - 1)
            else:
                # Weighted sampling
                r = random.uniform(0, total_weight)
                cumulative = 0
                idx = 0
                for i, weight in enumerate(remaining_weights):
                    cumulative += weight
                    if cumulative >= r:
                        idx = i
                        break
            
            # Add selected solution and remove from consideration
            selected.append(remaining_solutions[idx])
            remaining_solutions.pop(idx)
            remaining_weights.pop(idx)
        
        return selected
    
    async def _update_stepping_stone_scores(self, solutions: List[SolutionNode]):
        """Update stepping stone scores for solutions."""
        breakthrough_solutions = await self.archive.get_breakthrough_solutions()
        
        for solution in solutions:
            score = 0.0
            
            # Check if this solution led to breakthroughs
            for breakthrough in breakthrough_solutions:
                genealogy = await self.archive.get_genealogy(breakthrough.id)
                ancestors = genealogy.get_ancestors(breakthrough.id, generations_back=5)
                
                if any(ancestor.solution_id == solution.id for ancestor in ancestors):
                    # This solution contributed to a breakthrough
                    performance_gap = breakthrough.performance - solution.performance
                    if performance_gap > self.breakthrough_threshold:
                        score += performance_gap
            
            # Update stepping stone score
            solution.stepping_stone_score = score
            if score > 0:
                solution.is_stepping_stone = True
    
    async def _detect_stagnation(self) -> bool:
        """Detect if exploration has stagnated."""
        if len(self.archive.performance_history) < self.stagnation_threshold:
            return False
        
        # Check if performance has plateaued
        recent_performances = [
            perf for _, perf in self.archive.performance_history[-self.stagnation_threshold:]
        ]
        
        if len(set(recent_performances)) <= 2:  # Very little variation
            return True
        
        # Check improvement rate
        old_avg = np.mean(recent_performances[:self.stagnation_threshold//2])
        new_avg = np.mean(recent_performances[self.stagnation_threshold//2:])
        
        improvement_rate = (new_avg - old_avg) / old_avg if old_avg > 0 else 0
        
        return improvement_rate < 0.01  # Less than 1% improvement
    
    async def _count_recent_breakthroughs(self, days_back: int = 7) -> int:
        """Count breakthroughs in recent time period."""
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        breakthrough_solutions = await self.archive.get_breakthrough_solutions()
        
        recent_breakthroughs = [
            solution for solution in breakthrough_solutions
            if solution.creation_timestamp > cutoff_time
        ]
        
        return len(recent_breakthroughs)
    
    async def calculate_exploration_metrics(self) -> ExplorationMetrics:
        """Calculate comprehensive exploration metrics."""
        archive_stats = await self.archive.archive_statistics()
        
        # Calculate breakthrough rate
        total_solutions = len(self.archive.solutions)
        breakthrough_solutions = await self.archive.get_breakthrough_solutions()
        breakthrough_rate = len(breakthrough_solutions) / total_solutions if total_solutions > 0 else 0
        
        # Calculate novelty score
        novelty_score = await self._calculate_novelty_score()
        
        # Detect stagnation periods
        stagnation_periods = await self._count_stagnation_periods()
        
        # Find last breakthrough
        last_breakthrough = None
        if breakthrough_solutions:
            last_breakthrough = max(
                solution.creation_timestamp for solution in breakthrough_solutions
            )
        
        metrics = ExplorationMetrics(
            exploration_diversity=archive_stats.diversity_score,
            stepping_stones_found=archive_stats.stepping_stones_discovered,
            breakthrough_rate=breakthrough_rate,
            novelty_score=novelty_score,
            quality_diversity_ratio=self._calculate_quality_diversity_ratio(),
            stagnation_periods=stagnation_periods,
            last_breakthrough=last_breakthrough
        )
        
        self.exploration_history.append(metrics)
        return metrics
    
    async def _calculate_novelty_score(self) -> float:
        """Calculate overall novelty score of the archive."""
        solutions = list(self.archive.solutions.values())
        if len(solutions) < 2:
            return 0.0
        
        total_novelty = 0.0
        comparison_count = 0
        
        # Calculate pairwise novelty
        for i, solution1 in enumerate(solutions):
            for solution2 in solutions[i+1:]:
                novelty = self._calculate_pairwise_novelty(solution1, solution2)
                total_novelty += novelty
                comparison_count += 1
        
        return total_novelty / comparison_count if comparison_count > 0 else 0.0
    
    def _calculate_pairwise_novelty(self, solution1: SolutionNode, solution2: SolutionNode) -> float:
        """Calculate novelty between two solutions."""
        # Compare configurations
        config1_keys = set(solution1.configuration.keys())
        config2_keys = set(solution2.configuration.keys())
        
        all_keys = config1_keys | config2_keys
        if not all_keys:
            return 0.0
        
        different_keys = 0
        for key in all_keys:
            val1 = solution1.configuration.get(key)
            val2 = solution2.configuration.get(key)
            if val1 != val2:
                different_keys += 1
        
        return different_keys / len(all_keys)
    
    def _calculate_quality_diversity_ratio(self) -> float:
        """Calculate ratio of quality to diversity in archive."""
        solutions = list(self.archive.solutions.values())
        if not solutions:
            return 0.0
        
        # Quality: average performance
        quality = sum(solution.performance for solution in solutions) / len(solutions)
        
        # Diversity: average novelty score
        diversity = sum(solution.novelty_score for solution in solutions) / len(solutions)
        
        if diversity == 0:
            return float('inf') if quality > 0 else 0.0
        
        return quality / diversity
    
    async def _count_stagnation_periods(self) -> int:
        """Count number of stagnation periods in exploration history."""
        if len(self.exploration_history) < 5:
            return 0
        
        stagnation_count = 0
        in_stagnation = False
        
        for i in range(1, len(self.exploration_history)):
            current_breakthrough_rate = self.exploration_history[i].breakthrough_rate
            previous_breakthrough_rate = self.exploration_history[i-1].breakthrough_rate
            
            improvement = current_breakthrough_rate - previous_breakthrough_rate
            
            if improvement < 0.01:  # Less than 1% improvement
                if not in_stagnation:
                    stagnation_count += 1
                    in_stagnation = True
            else:
                in_stagnation = False
        
        return stagnation_count


class SteppingStoneAnalyzer:
    """
    Analyzer for identifying and tracking stepping stone solutions.
    Specialized in detecting solutions that enable future breakthroughs.
    """
    
    def __init__(self, archive: EvolutionArchive):
        self.archive = archive
        self.stepping_stone_candidates: Set[str] = set()
        self.validated_stepping_stones: Set[str] = set()
        
    async def identify_potential_stepping_stones(self) -> List[SolutionNode]:
        """Identify solutions that could become stepping stones."""
        all_solutions = list(self.archive.solutions.values())
        candidates = []
        
        for solution in all_solutions:
            score = await self._calculate_stepping_stone_potential(solution)
            if score > 0.1:  # Threshold for stepping stone potential
                solution.stepping_stone_score = score
                candidates.append(solution)
                self.stepping_stone_candidates.add(solution.id)
        
        # Sort by potential score
        candidates.sort(key=lambda x: x.stepping_stone_score, reverse=True)
        
        logger.info(f"Identified {len(candidates)} potential stepping stones")
        return candidates
    
    async def _calculate_stepping_stone_potential(self, solution: SolutionNode) -> float:
        """Calculate potential of solution to become a stepping stone."""
        score = 0.0
        
        # Factor 1: Novelty (unexplored areas often lead to breakthroughs)
        novelty_score = 1 / (1 + solution.child_count)
        score += novelty_score * 0.3
        
        # Factor 2: Partial success (not perfect, but promising)
        if 0.3 < solution.performance < 0.8:
            score += 0.4
        
        # Factor 3: Recent creation (new ideas might not be fully explored)
        age_days = (datetime.utcnow() - solution.creation_timestamp).days
        if age_days < 30:  # Less than a month old
            score += 0.2
        
        # Factor 4: Configuration uniqueness
        uniqueness = await self._calculate_configuration_uniqueness(solution)
        score += uniqueness * 0.1
        
        return score
    
    async def _calculate_configuration_uniqueness(self, solution: SolutionNode) -> float:
        """Calculate how unique a solution's configuration is."""
        other_solutions = [
            s for s in self.archive.solutions.values() 
            if s.id != solution.id
        ]
        
        if not other_solutions:
            return 1.0
        
        max_similarity = 0.0
        for other in other_solutions:
            similarity = self._calculate_configuration_similarity(
                solution.configuration, 
                other.configuration
            )
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _calculate_configuration_similarity(
        self, 
        config1: Dict[str, any], 
        config2: Dict[str, any]
    ) -> float:
        """Calculate similarity between two configurations."""
        all_keys = set(config1.keys()) | set(config2.keys())
        if not all_keys:
            return 1.0
        
        matching_keys = 0
        for key in all_keys:
            if config1.get(key) == config2.get(key):
                matching_keys += 1
        
        return matching_keys / len(all_keys)
    
    async def validate_stepping_stone(self, solution_id: str) -> bool:
        """Validate if a solution actually served as a stepping stone."""
        if solution_id not in self.archive.solutions:
            return False
        
        solution = self.archive.solutions[solution_id]
        
        # Check if any descendants achieved significant improvement
        genealogy = await self.archive.get_genealogy(solution_id)
        descendants = genealogy.get_descendants(solution_id, generations_forward=3)
        
        for descendant_node in descendants:
            descendant = self.archive.solutions.get(descendant_node.solution_id)
            if descendant:
                improvement = descendant.performance - solution.performance
                if improvement > 0.15:  # 15% improvement threshold
                    self.validated_stepping_stones.add(solution_id)
                    logger.info(f"Validated stepping stone: {solution_id}")
                    return True
        
        return False
    
    async def analyze_stepping_stone_patterns(self) -> Dict[str, any]:
        """Analyze patterns in validated stepping stones."""
        validated_solutions = [
            self.archive.solutions[sid] for sid in self.validated_stepping_stones
            if sid in self.archive.solutions
        ]
        
        if not validated_solutions:
            return {"pattern_count": 0}
        
        # Analyze common characteristics
        performance_range = self._analyze_performance_patterns(validated_solutions)
        config_patterns = self._analyze_configuration_patterns(validated_solutions)
        temporal_patterns = self._analyze_temporal_patterns(validated_solutions)
        
        return {
            "pattern_count": len(validated_solutions),
            "performance_patterns": performance_range,
            "configuration_patterns": config_patterns,
            "temporal_patterns": temporal_patterns
        }
    
    def _analyze_performance_patterns(self, solutions: List[SolutionNode]) -> Dict[str, float]:
        """Analyze performance patterns in stepping stones."""
        performances = [s.performance for s in solutions]
        
        return {
            "mean_performance": np.mean(performances),
            "std_performance": np.std(performances),
            "min_performance": min(performances),
            "max_performance": max(performances)
        }
    
    def _analyze_configuration_patterns(self, solutions: List[SolutionNode]) -> Dict[str, any]:
        """Analyze configuration patterns in stepping stones."""
        # Find most common configuration keys
        key_counts = defaultdict(int)
        for solution in solutions:
            for key in solution.configuration.keys():
                key_counts[key] += 1
        
        common_keys = [
            key for key, count in key_counts.items() 
            if count >= len(solutions) * 0.5  # Present in at least 50% of solutions
        ]
        
        return {
            "common_configuration_keys": common_keys,
            "total_unique_keys": len(key_counts),
            "average_config_size": np.mean([len(s.configuration) for s in solutions])
        }
    
    def _analyze_temporal_patterns(self, solutions: List[SolutionNode]) -> Dict[str, any]:
        """Analyze temporal patterns in stepping stone discovery."""
        creation_times = [s.creation_timestamp for s in solutions]
        
        if len(creation_times) < 2:
            return {"discovery_rate": 0}
        
        # Calculate discovery rate
        time_span = (max(creation_times) - min(creation_times)).total_seconds() / 3600  # hours
        discovery_rate = len(solutions) / time_span if time_span > 0 else 0
        
        return {
            "discovery_rate_per_hour": discovery_rate,
            "first_discovery": min(creation_times),
            "latest_discovery": max(creation_times),
            "discovery_span_hours": time_span
        }