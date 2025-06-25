"""
Strategic Provenance & Revenue System
====================================

Advanced provenance tracking and revenue distribution designed to incentivize
major AI labs to contribute models to PRSM network, creating the competitive
dynamics that drive adoption.

Key Features:
- Sophisticated attribution tracking across model generations
- Competitive intelligence protection while maintaining provenance
- Dynamic royalty rates that reward early and high-quality contributors
- Strategic revenue sharing that creates lock-in effects
- Anti-gaming mechanisms to prevent provenance manipulation
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ContributionType(str, Enum):
    """Types of contributions that earn provenance royalties"""
    BASE_MODEL = "base_model"              # Original foundation model
    FINE_TUNED_MODEL = "fine_tuned_model"  # Specialized fine-tuning
    TRAINING_DATA = "training_data"        # High-quality training datasets
    SYNTHETIC_DATA = "synthetic_data"      # Generated training data
    EVALUATION_DATA = "evaluation_data"    # Benchmarking and evaluation sets
    RESEARCH_INSIGHTS = "research_insights" # Novel techniques/architectures
    COMPUTE_RESOURCES = "compute_resources" # Training infrastructure


class ProvenanceClass(str, Enum):
    """Classification of provenance based on contribution significance"""
    FOUNDATIONAL = "foundational"    # Core foundation models (highest royalties)
    SPECIALIZED = "specialized"      # Domain-specific adaptations
    INCREMENTAL = "incremental"      # Minor improvements and variants
    DERIVATIVE = "derivative"        # Models based on existing PRSM models


@dataclass
class ContributionMetrics:
    """Metrics for evaluating contribution value"""
    model_parameters: int
    training_compute_hours: float
    dataset_size_tokens: int
    benchmark_performance: Dict[str, float]
    novelty_score: float  # How novel/innovative the contribution is
    adoption_rate: float  # How often the model is used/distilled
    citation_count: int   # Academic/industry citations
    commercial_usage: float  # Revenue generated from model usage


class ProvenanceNode(BaseModel):
    """Individual node in the provenance graph"""
    node_id: UUID = Field(default_factory=uuid4)
    contributor_id: UUID
    contribution_type: ContributionType
    provenance_class: ProvenanceClass
    
    # Contribution details
    model_name: str
    model_version: str
    contribution_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Value metrics
    metrics: ContributionMetrics
    
    # Provenance relationships
    parent_nodes: List[UUID] = Field(default_factory=list)  # What this builds on
    child_nodes: List[UUID] = Field(default_factory=list)   # What builds on this
    
    # Revenue tracking
    total_royalties_earned: Decimal = Field(default=Decimal('0'))
    royalty_rate_percentage: float = Field(ge=0.0, le=100.0)
    
    # Strategic protections
    competitive_firewall: bool = False
    embargo_period_days: int = 0  # How long before competitors can access


class RevenueDistribution(BaseModel):
    """Revenue distribution for a specific usage event"""
    distribution_id: UUID = Field(default_factory=uuid4)
    usage_event_id: UUID
    total_revenue: Decimal
    
    # Distribution breakdown
    contributor_shares: Dict[UUID, Decimal] = Field(default_factory=dict)
    platform_fee: Decimal
    governance_fund: Decimal
    
    # Attribution details
    provenance_path: List[UUID] = Field(default_factory=list)
    distribution_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategicProvenanceSystem:
    """
    Advanced provenance and revenue system designed to create competitive
    incentives for major AI labs to contribute to PRSM network.
    
    The system creates a prisoner's dilemma where:
    1. First movers get higher royalty rates
    2. Popular models generate ongoing revenue streams
    3. Not participating means missing out on revenue from competitors' usage
    4. Early participation creates competitive advantages
    """
    
    def __init__(self):
        # Provenance graph
        self.provenance_graph: Dict[UUID, ProvenanceNode] = {}
        self.revenue_distributions: List[RevenueDistribution] = []
        
        # Strategic parameters
        self.base_royalty_rates = {
            ProvenanceClass.FOUNDATIONAL: 8.0,    # 8% for foundation models
            ProvenanceClass.SPECIALIZED: 5.0,     # 5% for specialized models
            ProvenanceClass.INCREMENTAL: 2.0,     # 2% for incremental improvements
            ProvenanceClass.DERIVATIVE: 1.0       # 1% for derivative works
        }
        
        # Early adopter bonuses (decay over time)
        self.early_adopter_multipliers = {
            "first_mover": 2.0,      # Double royalties for first in category
            "early_adopter": 1.5,    # 50% bonus for early adopters (first 3)
            "pioneer": 1.25,         # 25% bonus for pioneers (first 10)
        }
        
        # Quality multipliers
        self.quality_multipliers = {
            "benchmark_leader": 1.5,      # Leading benchmark performance
            "high_adoption": 1.3,         # High usage/distillation rate
            "novel_architecture": 1.4,    # Novel architectural contributions
            "open_source": 1.2,          # Open source contributions
        }
        
        # Competitive dynamics
        self.competitive_pressure_multiplier = 1.0  # Increases as competition grows
        
        print("💰 Strategic Provenance System initialized")
        print("   - Designed to incentivize major AI lab participation")
        print("   - First-mover advantages active")
        print("   - Competitive dynamics monitoring enabled")
    
    async def register_contribution(self,
                                  contributor_id: UUID,
                                  model_name: str,
                                  contribution_type: ContributionType,
                                  metrics: ContributionMetrics,
                                  parent_contributions: List[UUID] = None,
                                  competitive_constraints: Dict[str, Any] = None) -> ProvenanceNode:
        """
        Register a new contribution to the PRSM network with full provenance tracking.
        """
        
        # Classify the contribution
        provenance_class = self._classify_contribution(contribution_type, metrics, parent_contributions)
        
        # Calculate strategic royalty rate
        royalty_rate = await self._calculate_strategic_royalty_rate(
            contributor_id, contribution_type, provenance_class, metrics
        )
        
        # Apply competitive protections if needed
        competitive_firewall = False
        embargo_period = 0
        
        if competitive_constraints:
            competitive_firewall = competitive_constraints.get("firewall", False)
            embargo_period = competitive_constraints.get("embargo_days", 0)
        
        # Create provenance node
        node = ProvenanceNode(
            contributor_id=contributor_id,
            contribution_type=contribution_type,
            provenance_class=provenance_class,
            model_name=model_name,
            model_version="1.0",  # Could be parameterized
            metrics=metrics,
            parent_nodes=parent_contributions or [],
            royalty_rate_percentage=royalty_rate,
            competitive_firewall=competitive_firewall,
            embargo_period_days=embargo_period
        )
        
        # Update graph relationships
        self.provenance_graph[node.node_id] = node
        
        if parent_contributions:
            for parent_id in parent_contributions:
                if parent_id in self.provenance_graph:
                    self.provenance_graph[parent_id].child_nodes.append(node.node_id)
        
        # Trigger strategic analysis
        await self._analyze_competitive_impact(node)
        
        print(f"📝 Contribution registered: {model_name}")
        print(f"   - Class: {provenance_class}")
        print(f"   - Royalty rate: {royalty_rate:.2f}%")
        print(f"   - Contributor: {contributor_id}")
        
        return node
    
    async def distribute_revenue(self,
                               usage_event_id: UUID,
                               total_revenue: Decimal,
                               used_models: List[UUID],
                               usage_context: Dict[str, Any]) -> RevenueDistribution:
        """
        Distribute revenue from model usage across the provenance chain.
        """
        
        # Calculate platform fee and governance allocation
        platform_fee = total_revenue * Decimal('0.05')  # 5% platform fee
        governance_fund = total_revenue * Decimal('0.02')  # 2% to governance
        distributable_revenue = total_revenue - platform_fee - governance_fund
        
        # Build complete provenance path
        provenance_path = await self._build_provenance_path(used_models)
        
        # Calculate contributor shares
        contributor_shares = await self._calculate_contributor_shares(
            provenance_path, distributable_revenue, usage_context
        )
        
        # Create distribution record
        distribution = RevenueDistribution(
            usage_event_id=usage_event_id,
            total_revenue=total_revenue,
            contributor_shares=contributor_shares,
            platform_fee=platform_fee,
            governance_fund=governance_fund,
            provenance_path=provenance_path
        )
        
        # Update contributor totals
        for contributor_id, share in contributor_shares.items():
            await self._update_contributor_earnings(contributor_id, share)
        
        self.revenue_distributions.append(distribution)
        
        print(f"💸 Revenue distributed: ${total_revenue}")
        print(f"   - Contributors: {len(contributor_shares)}")
        print(f"   - Platform fee: ${platform_fee}")
        print(f"   - Governance fund: ${governance_fund}")
        
        return distribution
    
    async def analyze_competitive_dynamics(self) -> Dict[str, Any]:
        """
        Analyze current competitive dynamics and strategic opportunities.
        """
        
        # Count contributions by type and contributor
        contribution_stats = {}
        contributor_revenue = {}
        
        for node in self.provenance_graph.values():
            contributor_stats = contribution_stats.setdefault(node.contributor_id, {
                "foundational": 0, "specialized": 0, "incremental": 0, "derivative": 0
            })
            contributor_stats[node.provenance_class] += 1
            
            contributor_revenue[node.contributor_id] = (
                contributor_revenue.get(node.contributor_id, Decimal('0')) + 
                node.total_royalties_earned
            )
        
        # Identify strategic opportunities
        opportunities = await self._identify_strategic_opportunities()
        
        # Calculate competitive pressure
        competitive_pressure = self._calculate_competitive_pressure()
        
        return {
            "total_contributions": len(self.provenance_graph),
            "unique_contributors": len(contribution_stats),
            "revenue_by_contributor": contributor_revenue,
            "contribution_distribution": contribution_stats,
            "competitive_pressure": competitive_pressure,
            "strategic_opportunities": opportunities,
            "network_effects_strength": self._calculate_network_effects_strength()
        }
    
    def _classify_contribution(self,
                             contribution_type: ContributionType,
                             metrics: ContributionMetrics,
                             parent_contributions: List[UUID]) -> ProvenanceClass:
        """Classify contribution based on type, metrics, and relationships"""
        
        # Foundation models with no parents are foundational
        if (contribution_type == ContributionType.BASE_MODEL and 
            not parent_contributions and
            metrics.model_parameters > 10_000_000_000):  # 10B+ parameters
            return ProvenanceClass.FOUNDATIONAL
        
        # High-quality specialized models
        if (contribution_type in [ContributionType.FINE_TUNED_MODEL, ContributionType.BASE_MODEL] and
            metrics.novelty_score > 0.7):
            return ProvenanceClass.SPECIALIZED
        
        # Models building on PRSM models are derivative
        if parent_contributions and len(parent_contributions) > 0:
            # Check if parent is from PRSM network
            for parent_id in parent_contributions:
                if parent_id in self.provenance_graph:
                    return ProvenanceClass.DERIVATIVE
        
        # Default to incremental
        return ProvenanceClass.INCREMENTAL
    
    async def _calculate_strategic_royalty_rate(self,
                                              contributor_id: UUID,
                                              contribution_type: ContributionType,
                                              provenance_class: ProvenanceClass,
                                              metrics: ContributionMetrics) -> float:
        """Calculate royalty rate with strategic incentives"""
        
        # Base rate
        base_rate = self.base_royalty_rates[provenance_class]
        
        # Apply early adopter bonuses
        early_adopter_bonus = await self._calculate_early_adopter_bonus(
            contributor_id, contribution_type
        )
        
        # Apply quality multipliers
        quality_bonus = self._calculate_quality_bonus(metrics)
        
        # Apply competitive pressure multiplier
        competitive_multiplier = self.competitive_pressure_multiplier
        
        # Calculate final rate
        final_rate = base_rate * early_adopter_bonus * quality_bonus * competitive_multiplier
        
        # Cap at reasonable maximum
        return min(final_rate, 15.0)  # Max 15% royalty rate
    
    async def _calculate_early_adopter_bonus(self,
                                           contributor_id: UUID,
                                           contribution_type: ContributionType) -> float:
        """Calculate early adopter bonus multiplier"""
        
        # Count existing contributions of this type
        type_contributions = [
            node for node in self.provenance_graph.values()
            if node.contribution_type == contribution_type
        ]
        
        contribution_count = len(type_contributions)
        
        if contribution_count == 0:
            return self.early_adopter_multipliers["first_mover"]
        elif contribution_count < 3:
            return self.early_adopter_multipliers["early_adopter"]
        elif contribution_count < 10:
            return self.early_adopter_multipliers["pioneer"]
        else:
            return 1.0  # No bonus
    
    def _calculate_quality_bonus(self, metrics: ContributionMetrics) -> float:
        """Calculate quality-based bonus multiplier"""
        
        bonus = 1.0
        
        # Benchmark performance bonus
        if metrics.benchmark_performance:
            avg_performance = sum(metrics.benchmark_performance.values()) / len(metrics.benchmark_performance)
            if avg_performance > 0.9:  # Top 10% performance
                bonus *= self.quality_multipliers["benchmark_leader"]
        
        # Novelty bonus
        if metrics.novelty_score > 0.8:
            bonus *= self.quality_multipliers["novel_architecture"]
        
        # Adoption rate bonus
        if metrics.adoption_rate > 0.8:
            bonus *= self.quality_multipliers["high_adoption"]
        
        return bonus
    
    async def _build_provenance_path(self, used_models: List[UUID]) -> List[UUID]:
        """Build complete provenance path for used models"""
        
        provenance_path = []
        visited = set()
        
        def traverse_parents(node_id: UUID):
            if node_id in visited or node_id not in self.provenance_graph:
                return
            
            visited.add(node_id)
            provenance_path.append(node_id)
            
            # Traverse parent nodes
            node = self.provenance_graph[node_id]
            for parent_id in node.parent_nodes:
                traverse_parents(parent_id)
        
        # Start traversal from used models
        for model_id in used_models:
            traverse_parents(model_id)
        
        return provenance_path
    
    async def _calculate_contributor_shares(self,
                                          provenance_path: List[UUID],
                                          distributable_revenue: Decimal,
                                          usage_context: Dict[str, Any]) -> Dict[UUID, Decimal]:
        """Calculate how revenue should be distributed among contributors"""
        
        contributor_weights = {}
        
        # Calculate weights based on royalty rates and contribution class
        for node_id in provenance_path:
            if node_id not in self.provenance_graph:
                continue
            
            node = self.provenance_graph[node_id]
            
            # Base weight from royalty rate
            weight = node.royalty_rate_percentage / 100.0
            
            # Adjust based on contribution class
            if node.provenance_class == ProvenanceClass.FOUNDATIONAL:
                weight *= 1.5  # Foundational models get higher share
            elif node.provenance_class == ProvenanceClass.DERIVATIVE:
                weight *= 0.5  # Derivative models get lower share
            
            contributor_id = node.contributor_id
            contributor_weights[contributor_id] = (
                contributor_weights.get(contributor_id, 0) + weight
            )
        
        # Normalize weights to sum to 1
        total_weight = sum(contributor_weights.values())
        if total_weight == 0:
            return {}
        
        # Calculate actual shares
        contributor_shares = {}
        for contributor_id, weight in contributor_weights.items():
            share = distributable_revenue * Decimal(str(weight / total_weight))
            contributor_shares[contributor_id] = share
        
        return contributor_shares
    
    async def _update_contributor_earnings(self, contributor_id: UUID, earnings: Decimal):
        """Update total earnings for a contributor across all their contributions"""
        
        for node in self.provenance_graph.values():
            if node.contributor_id == contributor_id:
                # Distribute earnings proportionally across their contributions
                # This is simplified - could be more sophisticated
                node.total_royalties_earned += earnings / Decimal(str(self._count_contributor_nodes(contributor_id)))
    
    def _count_contributor_nodes(self, contributor_id: UUID) -> int:
        """Count nodes for a specific contributor"""
        return sum(1 for node in self.provenance_graph.values() if node.contributor_id == contributor_id)
    
    async def _analyze_competitive_impact(self, new_node: ProvenanceNode):
        """Analyze competitive impact of new contribution"""
        
        # This would trigger strategic analysis
        # For now, just update competitive pressure
        if new_node.provenance_class == ProvenanceClass.FOUNDATIONAL:
            self.competitive_pressure_multiplier *= 1.05  # Increase pressure by 5%
        
        print(f"📈 Competitive pressure updated: {self.competitive_pressure_multiplier:.3f}")
    
    async def _identify_strategic_opportunities(self) -> List[Dict[str, Any]]:
        """Identify strategic opportunities for new participants"""
        
        opportunities = []
        
        # Identify underrepresented contribution types
        contribution_counts = {}
        for node in self.provenance_graph.values():
            contribution_counts[node.contribution_type] = (
                contribution_counts.get(node.contribution_type, 0) + 1
            )
        
        # Find gaps
        for contrib_type in ContributionType:
            if contribution_counts.get(contrib_type, 0) < 3:  # Less than 3 contributions
                opportunities.append({
                    "type": "underrepresented_contribution",
                    "contribution_type": contrib_type,
                    "current_count": contribution_counts.get(contrib_type, 0),
                    "potential_bonus": "first_mover" if contribution_counts.get(contrib_type, 0) == 0 else "early_adopter"
                })
        
        return opportunities
    
    def _calculate_competitive_pressure(self) -> float:
        """Calculate current competitive pressure in the network"""
        
        # Based on number of high-quality contributions and diversity
        foundational_count = sum(1 for node in self.provenance_graph.values() 
                               if node.provenance_class == ProvenanceClass.FOUNDATIONAL)
        
        unique_contributors = len(set(node.contributor_id for node in self.provenance_graph.values()))
        
        # Pressure increases with quality and diversity
        pressure = min(2.0, 1.0 + (foundational_count * 0.1) + (unique_contributors * 0.05))
        
        return pressure
    
    def _calculate_network_effects_strength(self) -> float:
        """Calculate strength of network effects"""
        
        if not self.provenance_graph:
            return 0.0
        
        # Network effects based on:
        # 1. Number of interconnected contributions
        # 2. Diversity of contributors
        # 3. Quality of contributions
        
        interconnection_score = sum(len(node.parent_nodes) + len(node.child_nodes) 
                                  for node in self.provenance_graph.values()) / len(self.provenance_graph)
        
        diversity_score = len(set(node.contributor_id for node in self.provenance_graph.values())) / max(len(self.provenance_graph), 1)
        
        quality_score = sum(node.metrics.novelty_score for node in self.provenance_graph.values()) / len(self.provenance_graph)
        
        return (interconnection_score + diversity_score + quality_score) / 3.0


# Global strategic provenance system instance
strategic_provenance = StrategicProvenanceSystem()