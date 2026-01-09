"""
Phase 3 Advanced DGM Evolution Demo (Optimized)

Fast demonstration of Phase 3 advanced evolution capabilities including:
- Stepping-stone discovery and validation
- Breakthrough detection with statistical significance
- Multi-objective parent selection
- Advanced performance evaluation framework
- Exploration frontier management

Optimized for quick execution while demonstrating core capabilities.
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, List
import statistics

# Set up minimal logging
logging.basicConfig(level=logging.WARNING)

# Core imports
from .archive import EvolutionArchive, SolutionNode
from .models import ComponentType, EvaluationResult
from .advanced_exploration import AdvancedExplorationEngine, SteppingStone, BreakthroughEvent
from .performance_framework import AdvancedPerformanceFramework, EvaluationTier


class FastMockSolutionGenerator:
    """Optimized mock solution generator."""
    
    def __init__(self):
        self.configs = [
            {"strategy": "adaptive", "threshold": 0.8, "factor": 0.7},
            {"strategy": "optimized", "threshold": 0.85, "factor": 0.8},
            {"strategy": "focused", "threshold": 0.9, "factor": 0.6},
            {"strategy": "balanced", "threshold": 0.75, "factor": 0.75}
        ]
    
    def generate_solution(self, generation: int = 0, parent_ids: List[str] = None) -> SolutionNode:
        """Generate a solution with realistic performance characteristics."""
        
        config = random.choice(self.configs).copy()
        config["generation"] = generation
        
        solution = SolutionNode(
            parent_ids=parent_ids or [],
            component_type=ComponentType.TASK_ORCHESTRATOR,
            configuration=config,
            generation=generation
        )
        
        # Calculate performance based on configuration and generation
        base_performance = 0.4 + config["threshold"] * 0.3 + config["factor"] * 0.2
        if generation > 0:
            base_performance += random.uniform(-0.05, 0.2)  # Evolution variance
        
        # Add initial evaluation
        evaluation = EvaluationResult(
            solution_id=solution.id,
            component_type=ComponentType.TASK_ORCHESTRATOR,
            performance_score=max(0.1, min(0.99, base_performance)),
            task_success_rate=random.uniform(0.75, 0.95),
            tasks_evaluated=random.randint(15, 40),
            tasks_successful=random.randint(12, 38),
            evaluation_duration_seconds=random.uniform(20, 80),
            evaluation_tier="comprehensive",
            evaluator_version="1.0",
            benchmark_suite="fast_demo"
        )
        solution.add_evaluation(evaluation)
        
        return solution


async def fast_phase3_demo():
    """Fast demonstration of Phase 3 capabilities."""
    
    print("🚀 Phase 3 Advanced DGM Evolution Demo (Optimized)")
    print("=" * 60)
    
    # Initialize systems
    print("\n📋 Initializing Systems...")
    archive = EvolutionArchive("phase3_demo", ComponentType.TASK_ORCHESTRATOR)
    exploration_engine = AdvancedExplorationEngine(archive)
    performance_framework = AdvancedPerformanceFramework(archive)
    generator = FastMockSolutionGenerator()
    
    print("✅ Systems initialized")
    
    # Phase 1: Build Archive History
    print("\n🧬 Phase 1: Building Evolution History")
    print("-" * 40)
    
    all_solutions = []
    
    # Generate 3 generations with 5-8 solutions each
    for gen in range(3):
        print(f"   Generation {gen}...")
        generation_solutions = []
        
        for i in range(random.randint(5, 8)):
            if gen == 0:
                solution = generator.generate_solution(generation=gen)
            else:
                # Select parents from previous generations
                possible_parents = [s for s in all_solutions if 
                                 archive.solutions[s].generation < gen]
                parents = random.sample(possible_parents, min(2, len(possible_parents)))
                solution = generator.generate_solution(generation=gen, parent_ids=parents)
            
            solution_id = await archive.add_solution(solution)
            generation_solutions.append(solution_id)
            all_solutions.append(solution_id)
            
            # Add breakthrough potential for later generations
            if gen >= 2 and random.random() < 0.3:
                breakthrough_eval = EvaluationResult(
                    solution_id=solution_id,
                    component_type=ComponentType.TASK_ORCHESTRATOR,
                    performance_score=min(0.98, solution.performance + random.uniform(0.15, 0.3)),
                    task_success_rate=random.uniform(0.85, 0.98),
                    tasks_evaluated=random.randint(25, 50),
                    tasks_successful=random.randint(22, 48),
                    evaluation_duration_seconds=random.uniform(30, 100),
                    evaluation_tier="comprehensive",
                    evaluator_version="1.0",
                    benchmark_suite="breakthrough_demo"
                )
                solution.add_evaluation(breakthrough_eval)
    
    archive_stats = await archive.archive_statistics()
    print(f"✅ Generated {archive_stats.total_solutions} solutions")
    print(f"   Best Performance: {archive_stats.best_performance:.3f}")
    print(f"   Diversity: {archive_stats.diversity_score:.3f}")
    
    # Phase 2: Stepping Stone Discovery
    print("\n🔍 Phase 2: Stepping Stone Discovery")
    print("-" * 40)
    
    stepping_stones = await exploration_engine.detect_stepping_stones(lookback_generations=3)
    
    print(f"🎯 Stepping Stone Analysis:")
    print(f"   Detected: {len(stepping_stones)}")
    
    if stepping_stones:
        top_stone = max(stepping_stones, key=lambda s: s.stepping_stone_score)
        print(f"   Top Stone: {top_stone.solution_id[:8]} (score: {top_stone.stepping_stone_score:.3f})")
        print(f"   Breakthrough Potential: {top_stone.breakthrough_potential:.3f}")
        print(f"   Novelty Score: {top_stone.novelty_score:.3f}")
    
    # Phase 3: Breakthrough Detection
    print("\n💥 Phase 3: Breakthrough Detection")
    print("-" * 40)
    
    breakthroughs = await exploration_engine.detect_breakthroughs(window_size=15)
    
    print(f"🚀 Breakthrough Analysis:")
    print(f"   Detected: {len(breakthroughs)}")
    
    if breakthroughs:
        for i, breakthrough in enumerate(breakthroughs):
            print(f"   {i+1}. {breakthrough.breakthrough_magnitude.title()}")
            print(f"      Solution: {breakthrough.solution_id[:8]}")
            print(f"      Improvement: {breakthrough.performance_improvement:.3f}")
            print(f"      Significance: {breakthrough.significance_level:.3f}")
    
    # Phase 4: Advanced Performance Evaluation
    print("\n📊 Phase 4: Advanced Performance Evaluation")
    print("-" * 40)
    
    # Evaluate top 3 solutions
    top_solutions = sorted(
        archive.solutions.values(),
        key=lambda s: s.performance,
        reverse=True
    )[:3]
    
    print("   Evaluating top solutions...")
    
    evaluation_results = {}
    for solution in top_solutions:
        evaluation = await performance_framework.evaluate_solution(
            solution,
            tier=EvaluationTier.QUICK,
            baseline_comparison=True
        )
        evaluation_results[solution.id] = evaluation
        print(f"     {solution.id[:8]}: {evaluation.performance_score:.3f}")
    
    # Phase 5: Multi-Objective Selection
    print("\n⚖️ Phase 5: Multi-Objective Parent Selection")
    print("-" * 40)
    
    print("   Testing selection strategies...")
    
    strategies = [
        ("Quality-Diversity", ["quality", "diversity"], 0.0),
        ("Novelty-Focused", ["novelty", "exploration_value"], 0.4),
        ("High Exploration", ["quality", "diversity", "novelty"], 0.8)
    ]
    
    for name, objectives, bias in strategies:
        parents = await exploration_engine.advanced_parent_selection(
            k_parents=3,
            objectives=objectives,
            exploration_bias=bias
        )
        
        if parents:
            avg_perf = statistics.mean(p.performance for p in parents)
            print(f"     {name}: {len(parents)} parents, avg performance: {avg_perf:.3f}")
        else:
            print(f"     {name}: No parents selected")
    
    # Phase 6: Comprehensive Analysis
    print("\n🔬 Phase 6: Exploration Progress Analysis")
    print("-" * 40)
    
    analysis = await exploration_engine.analyze_exploration_progress()
    
    print("📈 Progress Summary:")
    archive_overview = analysis["archive_overview"]
    print(f"   Solutions: {archive_overview['total_solutions']}")
    print(f"   Generations: {archive_overview['generations']}")
    print(f"   Best Performance: {archive_overview['best_performance']:.3f}")
    print(f"   Diversity: {archive_overview['diversity_score']:.3f}")
    
    stepping_analysis = analysis["stepping_stones"]
    print(f"\n🎯 Stepping Stones:")
    print(f"   Total: {stepping_analysis['total_detected']}")
    print(f"   Validated: {stepping_analysis['validated_count']}")
    print(f"   Average Score: {stepping_analysis['average_score']:.3f}")
    
    breakthrough_analysis = analysis["breakthroughs"]
    print(f"\n💥 Breakthroughs:")
    print(f"   Total: {breakthrough_analysis['total_detected']}")
    print(f"   Recent: {breakthrough_analysis['recent_count']}")
    
    frontier_analysis = analysis["exploration_frontier"]
    print(f"\n🌍 Exploration Frontier:")
    print(f"   Size: {frontier_analysis['frontier_size']}")
    print(f"   Diversity: {frontier_analysis['diversity_score']:.3f}")
    print(f"   Stagnation Risk: {frontier_analysis['stagnation_risk']:.3f}")
    print(f"   Breakthrough Potential: {frontier_analysis['breakthrough_potential']:.3f}")
    
    # Phase 7: System Insights
    print("\n💡 Phase 7: Performance Framework Insights")
    print("-" * 40)
    
    insights = await performance_framework.get_evaluation_insights()
    
    if "evaluation_summary" in insights:
        summary = insights["evaluation_summary"]
        print("📋 Evaluation Summary:")
        print(f"   Evaluations: {summary['total_evaluations']}")
        print(f"   Average Performance: {summary['average_performance']:.3f}")
        print(f"   Trend: {summary['performance_trend']}")
    
    if "recommendations" in insights:
        print(f"\n💡 Recommendations:")
        for rec in insights["recommendations"][:3]:  # Show top 3
            print(f"   • {rec}")
    
    # Final Summary
    print("\n🎉 Phase 3 Demo Complete!")
    print("=" * 60)
    
    final_stats = await archive.archive_statistics()
    performance_improvement = final_stats.best_performance - 0.5  # Assume 0.5 baseline
    
    print("✨ Advanced Capabilities Demonstrated:")
    print(f"   🔍 Stepping-stone discovery: {len(stepping_stones)} identified")
    print(f"   💥 Breakthrough detection: {len(breakthroughs)} found")
    print(f"   ⚖️ Multi-objective selection: 3 strategies tested")
    print(f"   📊 Advanced evaluation: Statistical analysis with confidence intervals")
    print(f"   🌍 Frontier management: Stagnation and breakthrough prediction")
    print(f"   🧬 Evolution progress: {performance_improvement:.3f} improvement achieved")
    
    print(f"\n🚀 Phase 3 Implementation Successfully Demonstrates:")
    print("   ✅ Sophisticated stepping-stone identification and validation")
    print("   ✅ Advanced breakthrough detection with statistical significance")
    print("   ✅ Multi-objective exploration with adaptive strategies")
    print("   ✅ Comprehensive performance evaluation framework")
    print("   ✅ Open-ended exploration with frontier management")
    print("   ✅ Quality-diversity optimization for sustainable evolution")
    
    return {
        "solutions_generated": final_stats.total_solutions,
        "stepping_stones_found": len(stepping_stones),
        "breakthroughs_detected": len(breakthroughs),
        "performance_improvement": performance_improvement,
        "final_diversity": final_stats.diversity_score,
        "demo_success": True
    }


if __name__ == "__main__":
    result = asyncio.run(fast_phase3_demo())
    print(f"\n📊 Demo Results: {result}")