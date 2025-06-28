"""
Advanced Open-Ended Exploration Engine

Sophisticated exploration algorithms for breakthrough discovery through
stepping-stone analysis, multi-objective optimization, and adaptive strategies.

Implements Phase 3.1 of the DGM-Enhanced Evolution System roadmap.
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from math import exp, sqrt, log, isnan, isinf
import numpy as np
from collections import defaultdict, deque
from itertools import combinations
import statistics

from .models import (
    ComponentType, SelectionStrategy, EvaluationResult,
    PerformanceStats
)
from .archive import EvolutionArchive, SolutionNode
from .exploration import OpenEndedExplorationEngine, ExplorationMetrics

logger = logging.getLogger(__name__)


@dataclass
class SteppingStone:
    """Represents a solution that enables future breakthroughs."""
    
    solution_id: str
    discovery_timestamp: datetime
    
    # Stepping stone characteristics
    breakthrough_potential: float  # Estimated potential for enabling breakthroughs
    novelty_score: float          # How different from existing solutions
    exploration_value: float      # Value for future exploration
    
    # Historical validation
    breakthroughs_enabled: List[str] = field(default_factory=list)  # Solution IDs enabled
    generations_to_breakthrough: int = 0
    performance_gap_bridged: float = 0.0
    
    # Quality metrics
    reliability_score: float = 0.0
    versatility_score: float = 0.0
    efficiency_score: float = 0.0
    
    @property
    def validated_stepping_stone(self) -> bool:
        """Check if this has been validated as a stepping stone."""
        return len(self.breakthroughs_enabled) > 0
    
    @property
    def stepping_stone_score(self) -> float:
        """Calculate overall stepping stone score."""
        base_score = (
            self.breakthrough_potential * 0.4 +
            self.novelty_score * 0.3 +
            self.exploration_value * 0.3
        )
        
        # Bonus for validated stepping stones
        if self.validated_stepping_stone:
            validation_bonus = len(self.breakthroughs_enabled) * 0.1
            base_score += validation_bonus
        
        return min(1.0, base_score)


@dataclass
class ExplorationFrontier:
    """Represents the current exploration frontier."""
    
    frontier_solutions: List[str]  # Solution IDs on the frontier
    frontier_timestamp: datetime
    
    # Frontier characteristics
    diversity_score: float
    quality_range: Tuple[float, float]
    unexplored_directions: List[Dict[str, Any]]
    
    # Exploration progress
    exploration_rate: float
    stagnation_risk: float
    breakthrough_potential: float


@dataclass
class BreakthroughEvent:
    """Represents a detected breakthrough."""
    
    breakthrough_id: str
    solution_id: str
    detection_timestamp: datetime
    
    # Breakthrough characteristics
    performance_improvement: float
    significance_level: float
    enabling_stepping_stones: List[str]
    
    # Impact analysis
    novelty_introduced: float
    exploration_impact: float
    solution_quality: float
    
    @property
    def breakthrough_magnitude(self) -> str:
        """Classify breakthrough magnitude."""
        if self.performance_improvement > 0.3:
            return "revolutionary"
        elif self.performance_improvement > 0.15:
            return "major"
        elif self.performance_improvement > 0.05:
            return "minor"
        else:
            return "incremental"


class AdvancedExplorationEngine(OpenEndedExplorationEngine):
    """
    Advanced exploration engine with sophisticated stepping-stone discovery,
    multi-objective optimization, and adaptive exploration strategies.
    
    Extends the base exploration engine with:
    - Sophisticated stepping-stone identification and validation
    - Multi-objective exploration balancing quality, diversity, and novelty
    - Breakthrough detection and analysis
    - Adaptive exploration strategy selection
    - Exploration frontier management
    """
    
    def __init__(self, archive: EvolutionArchive):
        super().__init__(archive)
        
        # Advanced exploration components
        self.stepping_stones: Dict[str, SteppingStone] = {}
        self.breakthrough_events: List[BreakthroughEvent] = []
        self.exploration_frontiers: deque = deque(maxlen=100)  # Recent frontiers
        
        # Exploration strategy parameters
        self.stepping_stone_threshold = 0.6  # Minimum score for stepping stone candidacy
        self.breakthrough_threshold = 0.1   # Minimum improvement for breakthrough
        self.novelty_decay_rate = 0.1       # Rate at which novelty decays
        self.exploration_temperature = 1.0  # Controls exploration vs exploitation
        
        # Multi-objective weights (sum to 1.0)
        self.objective_weights = {
            "quality": 0.4,
            "diversity": 0.3,
            "novelty": 0.2,
            "exploration_value": 0.1
        }
        
        # Adaptive parameters
        self.exploration_history = deque(maxlen=50)
        self.strategy_performance = defaultdict(list)
        self.auto_adaptation_enabled = True
    
    async def advanced_parent_selection(
        self,
        k_parents: int,
        objectives: Optional[List[str]] = None,
        exploration_bias: float = 0.0
    ) -> List[SolutionNode]:
        """
        Advanced parent selection with multi-objective optimization.
        
        Args:
            k_parents: Number of parents to select
            objectives: Specific objectives to optimize (quality, diversity, novelty, exploration)
            exploration_bias: Bias towards exploration (0.0 = balanced, 1.0 = max exploration)
            
        Returns:
            List of selected parent solutions
        """
        
        # Update exploration temperature based on archive state
        await self._update_exploration_temperature()
        
        # Identify current exploration frontier
        frontier = await self._identify_exploration_frontier()
        self.exploration_frontiers.append(frontier)
        
        # Get eligible solutions
        eligible_solutions = await self._get_eligible_solutions_advanced()
        
        if not eligible_solutions:
            logger.warning("No eligible solutions for advanced parent selection")
            return []
        
        # Calculate multi-objective scores
        solution_scores = await self._calculate_multi_objective_scores(
            eligible_solutions, 
            objectives or ["quality", "diversity", "novelty"],
            exploration_bias
        )
        
        # Apply exploration strategy
        strategy = await self._select_adaptive_exploration_strategy()
        selected_parents = await self._execute_advanced_selection_strategy(
            eligible_solutions,
            solution_scores,
            k_parents,
            strategy
        )
        
        # Record selection for adaptive learning
        await self._record_selection_outcome(selected_parents, strategy)
        
        logger.info(f"Advanced selection: {len(selected_parents)} parents using {strategy}")
        return selected_parents
    
    async def detect_stepping_stones(self, lookback_generations: int = 10) -> List[SteppingStone]:
        """
        Detect and analyze stepping stones using sophisticated algorithms.
        
        Args:
            lookback_generations: How many generations to analyze
            
        Returns:
            List of identified stepping stones
        """
        
        logger.info("Detecting stepping stones with advanced analysis")
        
        # Get all solutions for analysis
        all_solutions = list(self.archive.solutions.values())
        
        if len(all_solutions) < 5:
            logger.info("Insufficient solutions for stepping stone analysis")
            return []
        
        # Identify breakthrough solutions
        breakthrough_solutions = await self._identify_breakthrough_solutions()
        
        stepping_stone_candidates = []
        
        for solution in all_solutions:
            if solution.id in [b.solution_id for b in breakthrough_solutions]:
                continue  # Skip breakthrough solutions themselves
            
            # Calculate stepping stone potential
            potential_score = await self._calculate_stepping_stone_potential(
                solution, breakthrough_solutions, lookback_generations
            )
            
            if potential_score > self.stepping_stone_threshold:
                stepping_stone = SteppingStone(
                    solution_id=solution.id,
                    discovery_timestamp=solution.creation_timestamp,
                    breakthrough_potential=potential_score,
                    novelty_score=await self._calculate_solution_novelty(solution),
                    exploration_value=await self._calculate_exploration_value(solution)
                )
                
                # Validate stepping stone through genealogy analysis
                await self._validate_stepping_stone(stepping_stone, breakthrough_solutions)
                
                stepping_stone_candidates.append(stepping_stone)
                self.stepping_stones[solution.id] = stepping_stone
        
        # Sort by stepping stone score
        stepping_stone_candidates.sort(key=lambda x: x.stepping_stone_score, reverse=True)
        
        logger.info(f"Detected {len(stepping_stone_candidates)} stepping stones")
        return stepping_stone_candidates
    
    async def detect_breakthroughs(self, window_size: int = 20) -> List[BreakthroughEvent]:
        """
        Detect breakthrough solutions using advanced statistical analysis.
        
        Args:
            window_size: Number of recent solutions to analyze
            
        Returns:
            List of detected breakthrough events
        """
        
        recent_solutions = list(self.archive.solutions.values())[-window_size:]
        
        if len(recent_solutions) < 10:
            return []
        
        breakthroughs = []
        
        # Sort by creation time
        recent_solutions.sort(key=lambda x: x.creation_timestamp)
        
        # Calculate moving baseline performance
        baseline_window = 10
        
        for i in range(baseline_window, len(recent_solutions)):
            current_solution = recent_solutions[i]
            baseline_solutions = recent_solutions[max(0, i-baseline_window):i]
            
            if not current_solution.evaluation_history or not baseline_solutions:
                continue
            
            current_performance = current_solution.evaluation_history[-1].performance_score
            baseline_performances = [
                s.evaluation_history[-1].performance_score 
                for s in baseline_solutions 
                if s.evaluation_history
            ]
            
            if not baseline_performances:
                continue
            
            baseline_mean = statistics.mean(baseline_performances)
            baseline_std = statistics.stdev(baseline_performances) if len(baseline_performances) > 1 else 0.1
            
            # Check for breakthrough
            performance_improvement = current_performance - baseline_mean
            
            if performance_improvement > self.breakthrough_threshold:
                # Calculate statistical significance
                z_score = performance_improvement / baseline_std if baseline_std > 0 else 0
                significance = min(0.99, max(0, 1 - exp(-abs(z_score))))
                
                # Only consider significant breakthroughs
                if significance > 0.8:
                    breakthrough = BreakthroughEvent(
                        breakthrough_id=f"breakthrough_{len(self.breakthrough_events)}",
                        solution_id=current_solution.id,
                        detection_timestamp=datetime.utcnow(),
                        performance_improvement=performance_improvement,
                        significance_level=significance,
                        enabling_stepping_stones=await self._find_enabling_stepping_stones(current_solution),
                        novelty_introduced=await self._calculate_solution_novelty(current_solution),
                        exploration_impact=await self._calculate_exploration_impact(current_solution),
                        solution_quality=current_performance
                    )
                    
                    breakthroughs.append(breakthrough)
                    self.breakthrough_events.append(breakthrough)
        
        logger.info(f"Detected {len(breakthroughs)} breakthrough events")
        return breakthroughs
    
    async def analyze_exploration_progress(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of exploration progress and effectiveness.
        
        Returns:
            Detailed exploration analysis
        """
        
        # Detect recent stepping stones and breakthroughs
        stepping_stones = await self.detect_stepping_stones()
        breakthroughs = await self.detect_breakthroughs()
        
        # Calculate exploration metrics
        archive_stats = await self.archive.archive_statistics()
        current_frontier = await self._identify_exploration_frontier()
        
        # Analyze breakthrough patterns
        breakthrough_analysis = await self._analyze_breakthrough_patterns(breakthroughs)
        
        # Assess exploration efficiency
        efficiency_metrics = await self._calculate_exploration_efficiency()
        
        # Predict future breakthrough potential
        breakthrough_prediction = await self._predict_breakthrough_potential()
        
        return {
            "archive_overview": {
                "total_solutions": archive_stats.total_solutions,
                "generations": archive_stats.generations,
                "diversity_score": archive_stats.diversity_score,
                "best_performance": archive_stats.best_performance
            },
            "stepping_stones": {
                "total_detected": len(stepping_stones),
                "validated_count": len([s for s in stepping_stones if s.validated_stepping_stone]),
                "average_score": statistics.mean([s.stepping_stone_score for s in stepping_stones]) if stepping_stones else 0,
                "top_stepping_stones": [
                    {
                        "solution_id": s.solution_id[:8],
                        "score": s.stepping_stone_score,
                        "breakthroughs_enabled": len(s.breakthroughs_enabled)
                    }
                    for s in stepping_stones[:5]
                ]
            },
            "breakthroughs": {
                "total_detected": len(breakthroughs),
                "recent_count": len([b for b in breakthroughs if 
                                   (datetime.utcnow() - b.detection_timestamp).days <= 7]),
                "breakthrough_analysis": breakthrough_analysis
            },
            "exploration_frontier": {
                "frontier_size": len(current_frontier.frontier_solutions),
                "diversity_score": current_frontier.diversity_score,
                "stagnation_risk": current_frontier.stagnation_risk,
                "breakthrough_potential": current_frontier.breakthrough_potential
            },
            "efficiency_metrics": efficiency_metrics,
            "breakthrough_prediction": breakthrough_prediction,
            "recommendations": await self._generate_exploration_recommendations(
                stepping_stones, breakthroughs, current_frontier
            )
        }
    
    async def _update_exploration_temperature(self):
        """Update exploration temperature based on recent progress."""
        
        if len(self.exploration_history) < 10:
            return
        
        # Calculate recent diversity and breakthrough rate
        recent_diversity = statistics.mean([h.exploration_diversity for h in self.exploration_history[-10:]])
        recent_breakthroughs = len([b for b in self.breakthrough_events if 
                                  (datetime.utcnow() - b.detection_timestamp).days <= 7])
        
        # Adjust temperature
        if recent_diversity < 0.2:  # Low diversity
            self.exploration_temperature = min(2.0, self.exploration_temperature * 1.1)
        elif recent_breakthroughs == 0:  # No recent breakthroughs
            self.exploration_temperature = min(2.0, self.exploration_temperature * 1.05)
        elif recent_breakthroughs > 2:  # Many breakthroughs
            self.exploration_temperature = max(0.5, self.exploration_temperature * 0.95)
        
        logger.debug(f"Updated exploration temperature: {self.exploration_temperature:.3f}")
    
    async def _identify_exploration_frontier(self) -> ExplorationFrontier:
        """Identify the current exploration frontier."""
        
        all_solutions = list(self.archive.solutions.values())
        
        if not all_solutions:
            return ExplorationFrontier(
                frontier_solutions=[],
                frontier_timestamp=datetime.utcnow(),
                diversity_score=0.0,
                quality_range=(0.0, 0.0),
                unexplored_directions=[],
                exploration_rate=0.0,
                stagnation_risk=0.0,
                breakthrough_potential=0.0
            )
        
        # Find Pareto frontier (solutions not dominated by others)
        frontier_solutions = []
        performances = []
        
        for solution in all_solutions:
            if solution.evaluation_history:
                performance = solution.evaluation_history[-1].performance_score
                novelty = await self._calculate_solution_novelty(solution)
                
                # Check if this solution is on the Pareto frontier
                is_frontier = True
                for other in all_solutions:
                    if other.id == solution.id or not other.evaluation_history:
                        continue
                    
                    other_performance = other.evaluation_history[-1].performance_score
                    other_novelty = await self._calculate_solution_novelty(other)
                    
                    # Check if other dominates this solution
                    if (other_performance >= performance and other_novelty >= novelty and
                        (other_performance > performance or other_novelty > novelty)):
                        is_frontier = False
                        break
                
                if is_frontier:
                    frontier_solutions.append(solution.id)
                    performances.append(performance)
        
        # Calculate frontier characteristics
        diversity_score = await self._calculate_frontier_diversity(frontier_solutions)
        quality_range = (min(performances), max(performances)) if performances else (0.0, 0.0)
        
        # Assess exploration state
        stagnation_risk = await self._assess_stagnation_risk()
        breakthrough_potential = await self._assess_breakthrough_potential()
        
        return ExplorationFrontier(
            frontier_solutions=frontier_solutions,
            frontier_timestamp=datetime.utcnow(),
            diversity_score=diversity_score,
            quality_range=quality_range,
            unexplored_directions=await self._identify_unexplored_directions(),
            exploration_rate=len(frontier_solutions) / len(all_solutions) if all_solutions else 0,
            stagnation_risk=stagnation_risk,
            breakthrough_potential=breakthrough_potential
        )
    
    async def _calculate_multi_objective_scores(
        self,
        solutions: List[SolutionNode],
        objectives: List[str],
        exploration_bias: float
    ) -> Dict[str, Dict[str, float]]:
        """Calculate multi-objective scores for solutions."""
        
        scores = {}
        
        for solution in solutions:
            solution_scores = {}
            
            # Quality objective
            if "quality" in objectives:
                quality_score = solution.performance if hasattr(solution, 'performance') else 0.0
                solution_scores["quality"] = quality_score
            
            # Diversity objective
            if "diversity" in objectives:
                diversity_score = await self._calculate_solution_diversity(solution, solutions)
                solution_scores["diversity"] = diversity_score
            
            # Novelty objective
            if "novelty" in objectives:
                novelty_score = await self._calculate_solution_novelty(solution)
                solution_scores["novelty"] = novelty_score
            
            # Exploration value objective
            if "exploration_value" in objectives:
                exploration_score = await self._calculate_exploration_value(solution)
                solution_scores["exploration_value"] = exploration_score
            
            # Calculate weighted combination
            weighted_score = 0.0
            for objective in objectives:
                weight = self.objective_weights.get(objective, 1.0 / len(objectives))
                weighted_score += solution_scores.get(objective, 0.0) * weight
            
            # Apply exploration bias
            if exploration_bias > 0:
                exploration_component = solution_scores.get("exploration_value", 0.0)
                weighted_score = (1 - exploration_bias) * weighted_score + exploration_bias * exploration_component
            
            solution_scores["weighted"] = weighted_score
            scores[solution.id] = solution_scores
        
        return scores
    
    async def _calculate_solution_novelty(self, solution: SolutionNode) -> float:
        """Calculate novelty score for a solution."""
        
        other_solutions = [s for s in self.archive.solutions.values() if s.id != solution.id]
        
        if not other_solutions:
            return 1.0
        
        # Calculate configuration similarity to other solutions
        max_similarity = 0.0
        
        for other in other_solutions:
            similarity = await self._calculate_configuration_similarity(
                solution.configuration, other.configuration
            )
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of maximum similarity
        novelty = 1.0 - max_similarity
        
        # Apply temporal decay (older solutions contribute less to novelty calculation)
        age_days = (datetime.utcnow() - solution.creation_timestamp).days
        age_factor = exp(-age_days * self.novelty_decay_rate)
        
        return novelty * age_factor
    
    async def _calculate_solution_diversity(self, solution: SolutionNode, solution_set: List[SolutionNode]) -> float:
        """Calculate diversity of solution relative to a set."""
        
        if len(solution_set) <= 1:
            return 1.0
        
        # Calculate average distance to other solutions
        distances = []
        
        for other in solution_set:
            if other.id != solution.id:
                distance = await self._calculate_solution_distance(solution, other)
                distances.append(distance)
        
        return statistics.mean(distances) if distances else 0.0
    
    async def _calculate_solution_distance(self, solution1: SolutionNode, solution2: SolutionNode) -> float:
        """Calculate distance between two solutions."""
        
        # Euclidean distance in configuration space
        config1 = solution1.configuration
        config2 = solution2.configuration
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        if not all_keys:
            return 0.0
        
        total_distance = 0.0
        
        for key in all_keys:
            val1 = config1.get(key, 0)
            val2 = config2.get(key, 0)
            
            # Normalize different value types
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                distance = abs(val1 - val2)
            elif isinstance(val1, str) and isinstance(val2, str):
                distance = 0.0 if val1 == val2 else 1.0
            elif isinstance(val1, bool) and isinstance(val2, bool):
                distance = 0.0 if val1 == val2 else 1.0
            else:
                distance = 0.0 if val1 == val2 else 1.0
            
            total_distance += distance ** 2
        
        return sqrt(total_distance / len(all_keys))
    
    async def _calculate_exploration_value(self, solution: SolutionNode) -> float:
        """Calculate exploration value of a solution."""
        
        exploration_value = 0.0
        
        # Value based on unexplored configuration space
        unexplored_bonus = 1.0 / (1.0 + solution.child_count)
        exploration_value += unexplored_bonus * 0.4
        
        # Value based on potential for further improvement
        if solution.evaluation_history:
            current_performance = solution.evaluation_history[-1].performance_score
            improvement_potential = 1.0 - current_performance  # Room for improvement
            exploration_value += improvement_potential * 0.3
        
        # Value based on configuration uniqueness
        novelty = await self._calculate_solution_novelty(solution)
        exploration_value += novelty * 0.3
        
        return min(1.0, exploration_value)
    
    async def _calculate_configuration_similarity(self, config1: Dict, config2: Dict) -> float:
        """Calculate similarity between two configurations."""
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        if not all_keys:
            return 1.0
        
        matching_keys = 0
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 == val2:
                matching_keys += 1
        
        return matching_keys / len(all_keys)
    
    async def _select_adaptive_exploration_strategy(self) -> str:
        """Select exploration strategy based on current archive state."""
        
        if not self.auto_adaptation_enabled:
            return "quality_diversity"
        
        # Analyze recent exploration performance
        recent_breakthroughs = len([b for b in self.breakthrough_events if 
                                  (datetime.utcnow() - b.detection_timestamp).days <= 14])
        
        archive_stats = await self.archive.archive_statistics()
        
        # Strategy selection logic
        if archive_stats.diversity_score < 0.2:
            return "novelty_focused"
        elif recent_breakthroughs == 0 and len(self.breakthrough_events) > 0:
            return "stepping_stone_focused"
        elif archive_stats.best_performance < 0.5:
            return "quality_focused"
        else:
            return "quality_diversity"
    
    async def _execute_advanced_selection_strategy(
        self,
        solutions: List[SolutionNode],
        scores: Dict[str, Dict[str, float]],
        k_parents: int,
        strategy: str
    ) -> List[SolutionNode]:
        """Execute advanced selection strategy."""
        
        if strategy == "novelty_focused":
            return await self._novelty_focused_selection(solutions, scores, k_parents)
        elif strategy == "stepping_stone_focused":
            return await self._stepping_stone_focused_selection(solutions, scores, k_parents)
        elif strategy == "quality_focused":
            return await self._quality_focused_selection(solutions, scores, k_parents)
        else:  # quality_diversity
            return await self._quality_diversity_selection(solutions, scores, k_parents)
    
    async def _novelty_focused_selection(
        self, 
        solutions: List[SolutionNode], 
        scores: Dict[str, Dict[str, float]], 
        k_parents: int
    ) -> List[SolutionNode]:
        """Selection focused on novelty and exploration."""
        
        # Sort by novelty score
        sorted_solutions = sorted(
            solutions,
            key=lambda s: scores[s.id].get("novelty", 0.0),
            reverse=True
        )
        
        return sorted_solutions[:k_parents]
    
    async def _stepping_stone_focused_selection(
        self,
        solutions: List[SolutionNode],
        scores: Dict[str, Dict[str, float]],
        k_parents: int
    ) -> List[SolutionNode]:
        """Selection focused on stepping stone potential."""
        
        # Prioritize known stepping stones and high exploration value
        solution_priorities = []
        
        for solution in solutions:
            priority = 0.0
            
            # Bonus for known stepping stones
            if solution.id in self.stepping_stones:
                stepping_stone = self.stepping_stones[solution.id]
                priority += stepping_stone.stepping_stone_score * 0.6
            
            # Bonus for exploration value
            priority += scores[solution.id].get("exploration_value", 0.0) * 0.4
            
            solution_priorities.append((solution, priority))
        
        # Sort by priority and select top k
        solution_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [solution for solution, _ in solution_priorities[:k_parents]]
    
    async def _quality_focused_selection(
        self,
        solutions: List[SolutionNode],
        scores: Dict[str, Dict[str, float]],
        k_parents: int
    ) -> List[SolutionNode]:
        """Selection focused on quality/performance."""
        
        # Sort by quality score
        sorted_solutions = sorted(
            solutions,
            key=lambda s: scores[s.id].get("quality", 0.0),
            reverse=True
        )
        
        return sorted_solutions[:k_parents]
    
    async def _quality_diversity_selection(
        self,
        solutions: List[SolutionNode],
        scores: Dict[str, Dict[str, float]],
        k_parents: int
    ) -> List[SolutionNode]:
        """Balanced quality-diversity selection."""
        
        # Use weighted score that balances quality and diversity
        sorted_solutions = sorted(
            solutions,
            key=lambda s: scores[s.id].get("weighted", 0.0),
            reverse=True
        )
        
        return sorted_solutions[:k_parents]
    
    # Additional helper methods for comprehensive analysis
    
    async def _calculate_frontier_diversity(self, frontier_solutions: List[str]) -> float:
        """Calculate diversity score for frontier solutions."""
        
        if len(frontier_solutions) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i, solution_id_1 in enumerate(frontier_solutions):
            solution_1 = self.archive.solutions.get(solution_id_1)
            if not solution_1:
                continue
                
            for solution_id_2 in frontier_solutions[i+1:]:
                solution_2 = self.archive.solutions.get(solution_id_2)
                if not solution_2:
                    continue
                
                distance = await self._calculate_solution_distance(solution_1, solution_2)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    async def _assess_stagnation_risk(self) -> float:
        """Assess risk of exploration stagnation."""
        
        # Simple stagnation risk based on recent diversity
        if len(self.exploration_history) < 5:
            return 0.0
        
        recent_diversity = [h.exploration_diversity for h in self.exploration_history[-5:]]
        diversity_trend = recent_diversity[-1] - recent_diversity[0]
        
        # High risk if diversity is declining
        if diversity_trend < -0.1:
            return 0.8
        elif diversity_trend < 0.0:
            return 0.4
        else:
            return 0.1
    
    async def _assess_breakthrough_potential(self) -> float:
        """Assess potential for breakthroughs."""
        
        # Based on stepping stones and recent performance improvements
        stepping_stone_count = len([s for s in self.stepping_stones.values() if s.stepping_stone_score > 0.7])
        recent_breakthroughs = len([b for b in self.breakthrough_events if 
                                  (datetime.utcnow() - b.detection_timestamp).days <= 14])
        
        # Normalize to 0-1 range
        potential = min(1.0, (stepping_stone_count * 0.2) + (recent_breakthroughs * 0.3) + 0.1)
        return potential
    
    async def _identify_unexplored_directions(self) -> List[Dict[str, Any]]:
        """Identify unexplored directions in the solution space."""
        
        # Simple heuristic based on configuration space analysis
        all_solutions = list(self.archive.solutions.values())
        
        if not all_solutions:
            return []
        
        # Analyze configuration space coverage
        config_analysis = {}
        for solution in all_solutions:
            for key, value in solution.configuration.items():
                if key not in config_analysis:
                    config_analysis[key] = set()
                config_analysis[key].add(str(value))
        
        # Identify potentially unexplored combinations
        unexplored = []
        for key, values in config_analysis.items():
            if len(values) < 3:  # Fewer than 3 different values explored
                unexplored.append({
                    "parameter": key,
                    "explored_values": len(values),
                    "exploration_opportunity": 1.0 - (len(values) / 5.0)  # Assume 5 is good coverage
                })
        
        return unexplored
    
    async def _identify_breakthrough_solutions(self) -> List[BreakthroughEvent]:
        """Identify solutions that represent breakthroughs."""
        
        # Return recent breakthrough events
        return [b for b in self.breakthrough_events if 
                (datetime.utcnow() - b.detection_timestamp).days <= 30]
    
    async def _calculate_stepping_stone_potential(
        self,
        solution: SolutionNode,
        breakthrough_solutions: List[BreakthroughEvent],
        lookback_generations: int
    ) -> float:
        """Calculate stepping stone potential for a solution."""
        
        potential_score = 0.0
        
        # Check if this solution is in the genealogy of breakthrough solutions
        for breakthrough in breakthrough_solutions:
            breakthrough_solution = self.archive.solutions.get(breakthrough.solution_id)
            if not breakthrough_solution:
                continue
            
            # Check genealogical relationship
            genealogy = await self.archive.get_genealogy(breakthrough.solution_id)
            ancestors = genealogy.get_ancestors(breakthrough.solution_id, lookback_generations)
            
            if any(ancestor.solution_id == solution.id for ancestor in ancestors):
                # This solution is an ancestor of a breakthrough
                performance_gap = breakthrough.performance_improvement
                potential_score += performance_gap * 0.5
        
        # Add base potential based on novelty and exploration value
        novelty = await self._calculate_solution_novelty(solution)
        exploration_value = await self._calculate_exploration_value(solution)
        
        potential_score += novelty * 0.3 + exploration_value * 0.2
        
        return min(1.0, potential_score)
    
    async def _validate_stepping_stone(
        self,
        stepping_stone: SteppingStone,
        breakthrough_solutions: List[BreakthroughEvent]
    ):
        """Validate stepping stone by checking its actual contribution to breakthroughs."""
        
        for breakthrough in breakthrough_solutions:
            breakthrough_solution = self.archive.solutions.get(breakthrough.solution_id)
            if not breakthrough_solution:
                continue
            
            # Check if stepping stone enabled this breakthrough
            genealogy = await self.archive.get_genealogy(breakthrough.solution_id)
            ancestors = genealogy.get_ancestors(breakthrough.solution_id, 10)
            
            if any(ancestor.solution_id == stepping_stone.solution_id for ancestor in ancestors):
                stepping_stone.breakthroughs_enabled.append(breakthrough.solution_id)
                stepping_stone.performance_gap_bridged += breakthrough.performance_improvement
                
                # Calculate generations to breakthrough
                for i, ancestor in enumerate(ancestors):
                    if ancestor.solution_id == stepping_stone.solution_id:
                        stepping_stone.generations_to_breakthrough = i + 1
                        break
    
    async def _find_enabling_stepping_stones(self, solution: SolutionNode) -> List[str]:
        """Find stepping stones that enabled this solution."""
        
        enabling_stones = []
        
        # Get genealogy
        genealogy = await self.archive.get_genealogy(solution.id)
        ancestors = genealogy.get_ancestors(solution.id, 10)
        
        for ancestor in ancestors:
            if ancestor.solution_id in self.stepping_stones:
                enabling_stones.append(ancestor.solution_id)
        
        return enabling_stones
    
    async def _calculate_exploration_impact(self, solution: SolutionNode) -> float:
        """Calculate the exploration impact of a solution."""
        
        # Impact based on descendants generated
        descendant_count = len(solution.child_ids)
        
        # Impact based on novelty preservation
        novelty = await self._calculate_solution_novelty(solution)
        
        # Combined impact score
        impact = min(1.0, (descendant_count / 10.0) * 0.6 + novelty * 0.4)
        
        return impact
    
    async def _get_eligible_solutions_advanced(self) -> List[SolutionNode]:
        """Get eligible solutions with advanced filtering."""
        
        all_solutions = list(self.archive.solutions.values())
        
        # Filter for active, safe solutions
        eligible = [
            solution for solution in all_solutions
            if (solution.is_active and 
                solution.safety_status.value in ['SAFE', 'PENDING'] and
                solution.performance < 0.99)  # Avoid perfect solutions
        ]
        
        # Additional filtering based on exploration state
        if len(eligible) > 50:  # If we have many solutions, be more selective
            # Prefer more recent solutions and stepping stones
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            eligible = [
                s for s in eligible 
                if (s.creation_timestamp > recent_cutoff or 
                    s.id in self.stepping_stones or
                    s.performance > 0.7)
            ]
        
        return eligible
    
    async def _record_selection_outcome(self, selected_parents: List[SolutionNode], strategy: str):
        """Record selection outcome for adaptive learning."""
        
        outcome = {
            "strategy": strategy,
            "parents_selected": len(selected_parents),
            "timestamp": datetime.utcnow(),
            "average_performance": statistics.mean([p.performance for p in selected_parents]) if selected_parents else 0
        }
        
        # Create proper exploration record
        exploration_record = type('ExplorationRecord', (), {
            'strategy': outcome['strategy'],
            'timestamp': outcome['timestamp'],
            'average_performance': outcome['average_performance'],
            'exploration_diversity': 0.5,  # Default diversity
            'breakthrough_rate': 0.1,      # Default rate
            'novelty_score': 0.3,          # Default novelty
            'quality_diversity_ratio': outcome['average_performance']
        })()
        self.exploration_history.append(exploration_record)
        self.strategy_performance[strategy].append(outcome["average_performance"])
    
    async def _analyze_breakthrough_patterns(self, breakthroughs: List[BreakthroughEvent]) -> Dict[str, Any]:
        """Analyze patterns in breakthrough events."""
        
        if not breakthroughs:
            return {"no_breakthroughs": True}
        
        # Analyze breakthrough magnitudes
        magnitudes = [b.breakthrough_magnitude for b in breakthroughs]
        magnitude_counts = {mag: magnitudes.count(mag) for mag in set(magnitudes)}
        
        # Analyze performance improvements
        improvements = [b.performance_improvement for b in breakthroughs]
        avg_improvement = statistics.mean(improvements)
        
        # Analyze novelty introduction
        novelties = [b.novelty_introduced for b in breakthroughs]
        avg_novelty = statistics.mean(novelties)
        
        return {
            "total_breakthroughs": len(breakthroughs),
            "magnitude_distribution": magnitude_counts,
            "average_improvement": avg_improvement,
            "average_novelty": avg_novelty,
            "most_impactful": max(breakthroughs, key=lambda b: b.performance_improvement).solution_id
        }
    
    async def _calculate_exploration_efficiency(self) -> Dict[str, float]:
        """Calculate exploration efficiency metrics."""
        
        archive_stats = await self.archive.archive_statistics()
        
        efficiency = {
            "solutions_per_generation": archive_stats.total_solutions / max(1, archive_stats.generations),
            "performance_per_solution": archive_stats.best_performance / max(1, archive_stats.total_solutions),
            "diversity_maintenance": archive_stats.diversity_score,
            "breakthrough_rate": len(self.breakthrough_events) / max(1, archive_stats.total_solutions)
        }
        
        return efficiency
    
    async def _predict_breakthrough_potential(self) -> Dict[str, Any]:
        """Predict future breakthrough potential."""
        
        # Simple heuristic based on current state
        stepping_stone_quality = statistics.mean([s.stepping_stone_score for s in self.stepping_stones.values()]) if self.stepping_stones else 0
        recent_diversity = self.exploration_history[-1].exploration_diversity if self.exploration_history else 0.5
        
        prediction = {
            "near_term_potential": min(1.0, stepping_stone_quality + recent_diversity * 0.5),
            "exploration_momentum": len(self.exploration_history) / 50.0,  # Normalize to expected history length
            "diversity_trend": "increasing" if len(self.exploration_history) >= 2 and 
                             self.exploration_history[-1].exploration_diversity > self.exploration_history[-2].exploration_diversity 
                             else "stable"
        }
        
        return prediction
    
    async def _generate_exploration_recommendations(
        self, 
        stepping_stones: List[SteppingStone], 
        breakthroughs: List[BreakthroughEvent], 
        current_frontier: ExplorationFrontier
    ) -> List[str]:
        """Generate recommendations for exploration strategy."""
        
        recommendations = []
        
        # Stepping stone recommendations
        if len(stepping_stones) < 3:
            recommendations.append("Increase exploration diversity to discover more stepping stones")
        
        # Breakthrough recommendations
        if not breakthroughs:
            recommendations.append("Focus on quality-diversity balance to enable breakthrough discovery")
        elif len(breakthroughs) > 5:
            recommendations.append("Consider exploitation phase to consolidate breakthrough gains")
        
        # Frontier recommendations
        if current_frontier.stagnation_risk > 0.7:
            recommendations.append("High stagnation risk - implement novelty-focused exploration")
        
        if current_frontier.diversity_score < 0.3:
            recommendations.append("Low frontier diversity - increase exploration temperature")
        
        if not recommendations:
            recommendations.append("Exploration progressing well - maintain current strategy")
        
        return recommendations