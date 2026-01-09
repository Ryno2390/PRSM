#!/usr/bin/env python3
"""
Multi-Dimensional Breakthrough Ranking System
Demonstrates how sophisticated assessment enables better ranking than arbitrary thresholds
"""

from enhanced_breakthrough_assessment import *
from typing import List, Dict, Callable
import pandas as pd

class BreakthroughRanker:
    """Advanced ranking system using multi-dimensional profiles"""
    
    def __init__(self, organization_type: str = 'industry'):
        self.org_type = organization_type
        self.ranking_strategies = {
            'quality_weighted': self._quality_weighted_score,
            'risk_adjusted': self._risk_adjusted_score,
            'time_weighted': self._time_weighted_score,
            'portfolio_optimized': self._portfolio_optimized_score,
            'commercial_focused': self._commercial_focused_score,
            'innovation_focused': self._innovation_focused_score
        }
    
    def rank_breakthroughs(self, breakthroughs: List[BreakthroughProfile], 
                          strategy: str = 'quality_weighted',
                          filters: Dict = None) -> List[Tuple[BreakthroughProfile, float, str]]:
        """
        Rank breakthroughs using specified strategy and optional filters
        Returns: List of (breakthrough, score, reasoning) tuples
        """
        
        # Apply filters first
        filtered_breakthroughs = self._apply_filters(breakthroughs, filters or {})
        
        # Calculate scores using specified strategy
        ranking_func = self.ranking_strategies[strategy]
        scored_breakthroughs = []
        
        for breakthrough in filtered_breakthroughs:
            score, reasoning = ranking_func(breakthrough)
            scored_breakthroughs.append((breakthrough, score, reasoning))
        
        # Sort by score (descending)
        return sorted(scored_breakthroughs, key=lambda x: x[1], reverse=True)
    
    def _apply_filters(self, breakthroughs: List[BreakthroughProfile], filters: Dict) -> List[BreakthroughProfile]:
        """Apply filtering criteria"""
        filtered = breakthroughs
        
        # Time horizon filter
        if 'max_time_horizon' in filters:
            max_horizon = filters['max_time_horizon']
            horizon_order = [TimeHorizon.IMMEDIATE, TimeHorizon.NEAR_TERM, TimeHorizon.LONG_TERM, TimeHorizon.BLUE_SKY]
            max_index = horizon_order.index(max_horizon)
            filtered = [b for b in filtered if horizon_order.index(b.time_horizon) <= max_index]
        
        # Risk level filter
        if 'max_risk_level' in filters:
            max_risk = filters['max_risk_level']
            risk_order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.EXTREME]
            max_index = risk_order.index(max_risk)
            filtered = [b for b in filtered if risk_order.index(b.risk_level) <= max_index]
        
        # Category filter
        if 'categories' in filters:
            allowed_categories = filters['categories']
            filtered = [b for b in filtered if b.category in allowed_categories]
        
        # Minimum success probability
        if 'min_success_probability' in filters:
            min_prob = filters['min_success_probability']
            filtered = [b for b in filtered if b.success_probability >= min_prob]
        
        # Market size filter
        if 'min_market_size' in filters:
            # Simple keyword matching - could be more sophisticated
            size_keywords = filters['min_market_size']
            filtered = [b for b in filtered if any(keyword in b.market_size_estimate.lower() for keyword in size_keywords)]
        
        return filtered
    
    def _quality_weighted_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Balanced quality assessment across all dimensions"""
        
        # Core quality dimensions
        scientific_weight = 0.25
        commercial_weight = 0.25
        feasibility_weight = 0.25
        evidence_weight = 0.25
        
        base_score = (
            breakthrough.scientific_novelty * scientific_weight +
            breakthrough.commercial_potential * commercial_weight +
            breakthrough.technical_feasibility * feasibility_weight +
            breakthrough.evidence_strength * evidence_weight
        )
        
        # Adjust for success probability
        risk_adjusted = base_score * breakthrough.success_probability
        
        reasoning = f"Balanced score: Sci({breakthrough.scientific_novelty:.2f}) + Com({breakthrough.commercial_potential:.2f}) + Tech({breakthrough.technical_feasibility:.2f}) + Ev({breakthrough.evidence_strength:.2f}) × Prob({breakthrough.success_probability:.2f})"
        
        return risk_adjusted, reasoning
    
    def _risk_adjusted_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Prioritize low-risk, high-probability breakthroughs"""
        
        # Base quality
        quality = (breakthrough.scientific_novelty + breakthrough.commercial_potential + 
                  breakthrough.technical_feasibility + breakthrough.evidence_strength) / 4
        
        # Heavy penalty for high risk
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MODERATE: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.2
        }
        
        # Bonus for immediate timeline
        time_multipliers = {
            TimeHorizon.IMMEDIATE: 1.2,
            TimeHorizon.NEAR_TERM: 1.0,
            TimeHorizon.LONG_TERM: 0.7,
            TimeHorizon.BLUE_SKY: 0.4
        }
        
        risk_adjusted = quality * risk_multipliers[breakthrough.risk_level] * time_multipliers[breakthrough.time_horizon]
        
        reasoning = f"Risk-adjusted: Quality({quality:.2f}) × Risk({risk_multipliers[breakthrough.risk_level]}) × Time({time_multipliers[breakthrough.time_horizon]})"
        
        return risk_adjusted, reasoning
    
    def _commercial_focused_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Prioritize commercial viability and market opportunity"""
        
        # Heavy weight on commercial factors
        commercial_score = breakthrough.commercial_potential * 0.4
        competitive_score = breakthrough.competitive_advantage * 0.3
        feasibility_score = breakthrough.technical_feasibility * 0.2
        strategic_score = breakthrough.strategic_alignment * 0.1
        
        base_score = commercial_score + competitive_score + feasibility_score + strategic_score
        
        # Market size bonus
        market_multiplier = 1.0
        if 'billion' in breakthrough.market_size_estimate.lower():
            market_multiplier = 1.5
        elif 'million' in breakthrough.market_size_estimate.lower():
            market_multiplier = 1.2
        
        final_score = base_score * market_multiplier * breakthrough.success_probability
        
        reasoning = f"Commercial focus: Market({market_multiplier:.1f}x) × Success({breakthrough.success_probability:.2f}) × Commercial mix"
        
        return final_score, reasoning
    
    def _innovation_focused_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Prioritize scientific innovation and paradigm shifts"""
        
        # Heavy weight on innovation
        novelty_score = breakthrough.scientific_novelty * 0.5
        evidence_score = breakthrough.evidence_strength * 0.3
        feasibility_score = breakthrough.technical_feasibility * 0.2
        
        base_score = novelty_score + evidence_score + feasibility_score
        
        # Category bonus for revolutionary breakthroughs
        category_multipliers = {
            BreakthroughCategory.REVOLUTIONARY: 1.5,
            BreakthroughCategory.HIGH_IMPACT: 1.2,
            BreakthroughCategory.SPECULATIVE: 1.3,  # High novelty bonus
            BreakthroughCategory.INCREMENTAL: 0.8,
            BreakthroughCategory.NICHE: 0.9
        }
        
        final_score = base_score * category_multipliers[breakthrough.category]
        
        reasoning = f"Innovation focus: Novelty({breakthrough.scientific_novelty:.2f}) + Evidence({breakthrough.evidence_strength:.2f}) × Category({category_multipliers[breakthrough.category]:.1f}x)"
        
        return final_score, reasoning
    
    def _time_weighted_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Prioritize near-term implementation opportunities"""
        
        quality = (breakthrough.scientific_novelty + breakthrough.commercial_potential + 
                  breakthrough.technical_feasibility + breakthrough.evidence_strength) / 4
        
        # Strong preference for immediate and near-term
        time_weights = {
            TimeHorizon.IMMEDIATE: 1.5,
            TimeHorizon.NEAR_TERM: 1.2,
            TimeHorizon.LONG_TERM: 0.6,
            TimeHorizon.BLUE_SKY: 0.2
        }
        
        final_score = quality * time_weights[breakthrough.time_horizon] * breakthrough.success_probability
        
        reasoning = f"Time-weighted: Quality({quality:.2f}) × Timeline({time_weights[breakthrough.time_horizon]:.1f}x) × Success({breakthrough.success_probability:.2f})"
        
        return final_score, reasoning
    
    def _portfolio_optimized_score(self, breakthrough: BreakthroughProfile) -> Tuple[float, str]:
        """Balance portfolio across risk/reward spectrum"""
        
        base_quality = (breakthrough.scientific_novelty + breakthrough.commercial_potential + 
                       breakthrough.technical_feasibility + breakthrough.evidence_strength) / 4
        
        # Portfolio balance scoring - want mix of safe bets and moonshots
        risk_portfolio_weights = {
            RiskLevel.LOW: 1.0,      # Safe bets
            RiskLevel.MODERATE: 1.1,  # Sweet spot
            RiskLevel.HIGH: 0.8,     # Selective moonshots
            RiskLevel.EXTREME: 0.9   # Few revolutionary bets
        }
        
        time_portfolio_weights = {
            TimeHorizon.IMMEDIATE: 1.2,   # Quick wins
            TimeHorizon.NEAR_TERM: 1.3,   # Core pipeline
            TimeHorizon.LONG_TERM: 0.9,   # Future options
            TimeHorizon.BLUE_SKY: 0.7     # Research bets
        }
        
        portfolio_score = (base_quality * 
                          risk_portfolio_weights[breakthrough.risk_level] * 
                          time_portfolio_weights[breakthrough.time_horizon] *
                          breakthrough.success_probability)
        
        reasoning = f"Portfolio balanced: Quality × Risk({risk_portfolio_weights[breakthrough.risk_level]:.1f}) × Time({time_portfolio_weights[breakthrough.time_horizon]:.1f}) × Success"
        
        return portfolio_score, reasoning

def demonstrate_multi_dimensional_ranking():
    """Show how multi-dimensional ranking works in practice"""
    
    # Create sample breakthroughs with different profiles
    sample_breakthroughs = [
        BreakthroughProfile(
            discovery_id="gecko_adhesion",
            description="Gecko-inspired reversible adhesion",
            source_papers=["gecko_2024"],
            scientific_novelty=0.8,
            commercial_potential=0.7,
            technical_feasibility=0.6,
            evidence_strength=0.8,
            category=BreakthroughCategory.HIGH_IMPACT,
            time_horizon=TimeHorizon.NEAR_TERM,
            risk_level=RiskLevel.MODERATE,
            market_size_estimate="Hundreds of millions (specialized applications)",
            competitive_advantage=0.7,
            resource_requirements="Moderate investment",
            strategic_alignment=0.8,
            next_steps=["Prototype development"],
            success_probability=0.6,
            value_at_risk="Development costs",
            upside_potential="Major market opportunity",
            key_assumptions=["Technical feasibility"],
            failure_modes=["Manufacturing challenges"],
            sensitivity_factors=["Material costs"]
        ),
        BreakthroughProfile(
            discovery_id="smart_fastener",
            description="IoT-enabled smart fastening system",
            source_papers=["iot_2024", "smart_materials_2024"],
            scientific_novelty=0.5,
            commercial_potential=0.9,
            technical_feasibility=0.9,
            evidence_strength=0.7,
            category=BreakthroughCategory.INCREMENTAL,
            time_horizon=TimeHorizon.IMMEDIATE,
            risk_level=RiskLevel.LOW,
            market_size_estimate="Multi-billion (IoT market)",
            competitive_advantage=0.6,
            resource_requirements="Minimal investment",
            strategic_alignment=0.9,
            next_steps=["Market validation"],
            success_probability=0.8,
            value_at_risk="Low development risk",
            upside_potential="Massive market",
            key_assumptions=["Market acceptance"],
            failure_modes=["Competition"],
            sensitivity_factors=["Technology adoption"]
        ),
        BreakthroughProfile(
            discovery_id="quantum_fastening",
            description="Quantum-mechanical bonding system",
            source_papers=["quantum_2024"],
            scientific_novelty=0.95,
            commercial_potential=0.3,
            technical_feasibility=0.2,
            evidence_strength=0.4,
            category=BreakthroughCategory.REVOLUTIONARY,
            time_horizon=TimeHorizon.BLUE_SKY,
            risk_level=RiskLevel.EXTREME,
            market_size_estimate="Unknown potential",
            competitive_advantage=0.9,
            resource_requirements="Massive investment",
            strategic_alignment=0.3,
            next_steps=["Basic research"],
            success_probability=0.1,
            value_at_risk="High research costs",
            upside_potential="Paradigm shift",
            key_assumptions=["Fundamental physics"],
            failure_modes=["Technical impossibility"],
            sensitivity_factors=["Research breakthroughs"]
        )
    ]
    
    ranker = BreakthroughRanker()
    
    print("🎯 MULTI-DIMENSIONAL RANKING DEMONSTRATION")
    print("=" * 80)
    
    # Demonstrate different ranking strategies
    strategies = ['quality_weighted', 'risk_adjusted', 'commercial_focused', 'innovation_focused']
    
    for strategy in strategies:
        print(f"\n📊 RANKING STRATEGY: {strategy.upper().replace('_', ' ')}")
        print("-" * 60)
        
        ranked = ranker.rank_breakthroughs(sample_breakthroughs, strategy=strategy)
        
        for i, (breakthrough, score, reasoning) in enumerate(ranked, 1):
            print(f"{i}. {breakthrough.discovery_id} (Score: {score:.3f})")
            print(f"   {reasoning}")
            print(f"   Category: {breakthrough.category.value}, Risk: {breakthrough.risk_level.value}")
    
    # Demonstrate filtering
    print(f"\n🔍 FILTERING EXAMPLES")
    print("-" * 60)
    
    # Filter for low-risk, near-term opportunities
    filters = {
        'max_risk_level': RiskLevel.MODERATE,
        'max_time_horizon': TimeHorizon.NEAR_TERM,
        'min_success_probability': 0.5
    }
    
    filtered_ranked = ranker.rank_breakthroughs(
        sample_breakthroughs, 
        strategy='risk_adjusted',
        filters=filters
    )
    
    print("Low-risk, near-term opportunities (Success ≥50%):")
    for i, (breakthrough, score, reasoning) in enumerate(filtered_ranked, 1):
        print(f"{i}. {breakthrough.discovery_id} - {breakthrough.description}")
        print(f"   Success probability: {breakthrough.success_probability:.1%}")
    
    # Filter for high-innovation breakthroughs
    innovation_filters = {
        'categories': [BreakthroughCategory.REVOLUTIONARY, BreakthroughCategory.HIGH_IMPACT]
    }
    
    innovation_ranked = ranker.rank_breakthroughs(
        sample_breakthroughs,
        strategy='innovation_focused', 
        filters=innovation_filters
    )
    
    print(f"\nHigh-innovation breakthroughs:")
    for i, (breakthrough, score, reasoning) in enumerate(innovation_ranked, 1):
        print(f"{i}. {breakthrough.discovery_id} - Novelty: {breakthrough.scientific_novelty:.2f}")

def show_ranking_advantages():
    """Compare old threshold vs new multi-dimensional ranking"""
    
    print(f"\n🆚 OLD THRESHOLD vs NEW MULTI-DIMENSIONAL RANKING")
    print("=" * 80)
    
    print("❌ OLD WAY (Arbitrary Thresholds):")
    print("   • Score 0.749 → 'PROMISING' (3rd tier)")  
    print("   • Score 0.750 → 'HIGH_POTENTIAL' (2nd tier)")
    print("   • Same breakthrough, different label based on 0.001 difference!")
    print("   • No context about WHY it's high-potential")
    print("   • No guidance on WHEN to pursue it")
    print("   • No assessment of RISK or resource requirements")
    
    print(f"\n✅ NEW WAY (Multi-Dimensional):")
    print("   • Quality-weighted rank: #2 of 50 (top 4%)")
    print("   • Risk-adjusted rank: #8 of 50 (low-risk preference)")  
    print("   • Commercial rank: #1 of 50 (market-ready)")
    print("   • Innovation rank: #15 of 50 (incremental advancement)")
    print("   • CONTEXT: 'High commercial potential, moderate risk, 2-year timeline'")
    print("   • ACTIONABLE: 'Immediate prototyping recommended'")
    print("   • HONEST: '60% success probability, requires $2M investment'")

if __name__ == "__main__":
    demonstrate_multi_dimensional_ranking()
    show_ranking_advantages()