"""
Micro-Node Empowerment Layer
============================

Prevents the Bitcoin mining pattern where hobbyists get squeezed out by 
industrial operations. Creates sustainable incentives for small participants
while institutions scale up.

Key Features:
- Logarithmic rewards curve that reduces emissions as node scale increases
- Edge node incentives for running small but reliable model shards
- Node credit system based on diversity, uptime, and contribution usefulness
- Protection mechanisms against compute oligopolies
- Sustainable economics for individual researchers and small teams
"""

import asyncio
import math
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field


class NodeScale(str, Enum):
    """Node scale categories"""
    MICRO = "micro"          # Individual researchers, laptops/small GPUs
    SMALL = "small"          # Small teams, workstation-class hardware
    MEDIUM = "medium"        # Research groups, small server clusters
    LARGE = "large"          # Enterprises, data center operations
    MASSIVE = "massive"      # Hyperscale operations


class ContributionType(str, Enum):
    """Types of node contributions beyond raw compute"""
    COMPUTE_POWER = "compute_power"        # Raw computational resources
    DATA_QUALITY = "data_quality"          # High-quality, verified data
    MODEL_DIVERSITY = "model_diversity"    # Unique model architectures
    UPTIME_RELIABILITY = "uptime_reliability"  # Consistent availability
    GEOGRAPHIC_DIVERSITY = "geographic_diversity"  # Geographic distribution
    DOMAIN_EXPERTISE = "domain_expertise"  # Specialized knowledge domains
    INNOVATION = "innovation"              # Novel techniques and research
    COMMUNITY_SUPPORT = "community_support"  # Documentation, tutorials, help


@dataclass
class NodeCapabilities:
    """Technical capabilities of a node"""
    compute_power_tflops: float
    storage_capacity_gb: float
    bandwidth_mbps: float
    uptime_percentage: float
    geographic_location: str
    specialized_domains: List[str]
    supported_model_types: List[str]


class NodeCreditScore(BaseModel):
    """Multi-dimensional credit score for node contribution value"""
    node_id: UUID
    
    # Core metrics (0.0 to 1.0)
    compute_score: float = Field(ge=0.0, le=1.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    diversity_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    innovation_score: float = Field(ge=0.0, le=1.0)
    
    # Composite score
    overall_score: float = Field(ge=0.0, le=1.0)
    
    # Temporal tracking
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score_history: List[Tuple[datetime, float]] = Field(default_factory=list)


class RewardMultiplier(BaseModel):
    """Reward multipliers for different contribution types"""
    base_multiplier: float = 1.0
    scale_adjustment: float = 1.0      # Logarithmic scale penalty/bonus
    diversity_bonus: float = 1.0       # Geographic/domain diversity
    reliability_bonus: float = 1.0     # Uptime and consistency
    innovation_bonus: float = 1.0      # Novel contributions
    
    # Final multiplier (product of all factors)
    final_multiplier: float = Field(default=1.0)


class MicroNodeEmpowerment:
    """
    Empowerment layer that prevents large institutional players from 
    squeezing out smaller participants through pure scale advantages.
    
    Uses logarithmic reward curves and multi-dimensional value assessment
    to ensure sustainable participation across all scales.
    """
    
    def __init__(self):
        # Node registry
        self.registered_nodes: Dict[UUID, Dict[str, Any]] = {}
        self.node_credit_scores: Dict[UUID, NodeCreditScore] = {}
        
        # Reward curve parameters
        self.logarithmic_curve_parameters = {
            "base_reward": 100.0,           # Base FTNS reward per period
            "scale_logarithm_base": 10.0,   # Log base for scale adjustment
            "max_scale_penalty": 0.1,       # Minimum multiplier for massive nodes
            "micro_node_bonus": 2.0,        # Extra bonus for micro nodes
        }
        
        # Diversity incentives
        self.diversity_bonuses = {
            "geographic_unique": 1.5,       # Unique geographic regions
            "domain_expertise": 1.3,        # Specialized domain knowledge
            "model_architecture": 1.4,      # Novel model architectures
            "underrepresented_region": 2.0, # Developing regions
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "minimum_uptime": 0.95,         # 95% uptime required
            "data_quality_score": 0.8,      # 80% data quality minimum
            "community_contribution": 0.1,  # Some community engagement required
        }
        
        print("🌱 Micro-Node Empowerment Layer initialized")
        print("   - Logarithmic reward curves active")
        print("   - Multi-dimensional value assessment enabled")
        print("   - Anti-centralization protection deployed")
    
    async def register_node(self,
                           node_id: UUID,
                           operator_info: Dict[str, Any],
                           capabilities: NodeCapabilities,
                           contribution_types: List[ContributionType]) -> Dict[str, Any]:
        """
        Register a new node with capability assessment and scale classification.
        """
        
        # Classify node scale
        node_scale = self._classify_node_scale(capabilities)
        
        # Calculate initial credit score
        credit_score = await self._calculate_node_credit_score(
            node_id, capabilities, contribution_types, []  # No history yet
        )
        
        # Calculate reward multiplier
        reward_multiplier = await self._calculate_reward_multiplier(
            node_scale, capabilities, credit_score
        )
        
        # Register node
        node_registration = {
            "node_id": node_id,
            "operator_info": operator_info,
            "capabilities": capabilities,
            "contribution_types": contribution_types,
            "node_scale": node_scale,
            "registration_time": datetime.now(timezone.utc),
            "status": "active"
        }
        
        self.registered_nodes[node_id] = node_registration
        self.node_credit_scores[node_id] = credit_score
        
        print(f"🌐 Node registered: {node_id}")
        print(f"   - Scale: {node_scale}")
        print(f"   - Credit score: {credit_score.overall_score:.3f}")
        print(f"   - Reward multiplier: {reward_multiplier.final_multiplier:.3f}")
        
        return {
            "registration_successful": True,
            "node_scale": node_scale,
            "credit_score": credit_score,
            "reward_multiplier": reward_multiplier,
            "expected_daily_rewards": self._estimate_daily_rewards(reward_multiplier),
            "empowerment_benefits": self._get_empowerment_benefits(node_scale)
        }
    
    async def calculate_periodic_rewards(self,
                                       assessment_period: timedelta = timedelta(days=1)) -> Dict[UUID, Decimal]:
        """
        Calculate periodic rewards for all registered nodes using the 
        logarithmic curve and multi-dimensional assessment.
        """
        
        rewards = {}
        total_rewards_pool = Decimal('10000')  # Example daily pool
        
        # Calculate relative weights for all nodes
        node_weights = {}
        total_weight = 0.0
        
        for node_id, node_info in self.registered_nodes.items():
            if node_info["status"] != "active":
                continue
            
            # Get current credit score
            credit_score = self.node_credit_scores.get(node_id)
            if not credit_score:
                continue
            
            # Calculate reward multiplier
            reward_multiplier = await self._calculate_reward_multiplier(
                node_info["node_scale"],
                node_info["capabilities"],
                credit_score
            )
            
            # Weight combines credit score and multiplier
            weight = credit_score.overall_score * reward_multiplier.final_multiplier
            node_weights[node_id] = weight
            total_weight += weight
        
        # Distribute rewards proportionally
        for node_id, weight in node_weights.items():
            if total_weight > 0:
                proportion = weight / total_weight
                reward = total_rewards_pool * Decimal(str(proportion))
                rewards[node_id] = reward
        
        print(f"💰 Periodic rewards calculated for {len(rewards)} nodes")
        print(f"   - Total pool: {total_rewards_pool} FTNS")
        print(f"   - Average reward: {total_rewards_pool / len(rewards) if rewards else 0:.2f} FTNS")
        
        return rewards
    
    async def update_node_performance(self,
                                    node_id: UUID,
                                    performance_metrics: Dict[str, Any],
                                    contribution_history: List[Dict[str, Any]]) -> NodeCreditScore:
        """
        Update node credit score based on recent performance and contributions.
        """
        
        if node_id not in self.registered_nodes:
            raise ValueError(f"Node {node_id} not registered")
        
        node_info = self.registered_nodes[node_id]
        capabilities = node_info["capabilities"]
        contribution_types = node_info["contribution_types"]
        
        # Calculate updated credit score
        new_credit_score = await self._calculate_node_credit_score(
            node_id, capabilities, contribution_types, contribution_history
        )
        
        # Update score history
        old_score = self.node_credit_scores.get(node_id)
        if old_score:
            new_credit_score.score_history = old_score.score_history.copy()
        
        new_credit_score.score_history.append((
            datetime.now(timezone.utc),
            new_credit_score.overall_score
        ))
        
        # Keep only last 30 days of history
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        new_credit_score.score_history = [
            (date, score) for date, score in new_credit_score.score_history
            if date >= cutoff_date
        ]
        
        self.node_credit_scores[node_id] = new_credit_score
        
        return new_credit_score
    
    async def analyze_network_health(self) -> Dict[str, Any]:
        """
        Analyze the health of the micro-node network and institutional balance.
        """
        
        # Scale distribution
        scale_distribution = {}
        total_nodes = len(self.registered_nodes)
        
        for node_info in self.registered_nodes.values():
            scale = node_info["node_scale"]
            scale_distribution[scale] = scale_distribution.get(scale, 0) + 1
        
        # Calculate percentages
        scale_percentages = {
            scale: (count / total_nodes) * 100 if total_nodes > 0 else 0
            for scale, count in scale_distribution.items()
        }
        
        # Geographic distribution
        geographic_distribution = {}
        for node_info in self.registered_nodes.values():
            location = node_info["capabilities"].geographic_location
            geographic_distribution[location] = geographic_distribution.get(location, 0) + 1
        
        # Credit score distribution
        credit_scores = [score.overall_score for score in self.node_credit_scores.values()]
        avg_credit_score = sum(credit_scores) / len(credit_scores) if credit_scores else 0
        
        # Network health assessment
        health_score = self._calculate_network_health_score(
            scale_percentages, geographic_distribution, credit_scores
        )
        
        return {
            "total_nodes": total_nodes,
            "scale_distribution": scale_distribution,
            "scale_percentages": scale_percentages,
            "geographic_distribution": geographic_distribution,
            "average_credit_score": avg_credit_score,
            "network_health_score": health_score,
            "decentralization_index": self._calculate_decentralization_index(scale_percentages),
            "recommendations": self._generate_network_recommendations(scale_percentages, health_score)
        }
    
    def _classify_node_scale(self, capabilities: NodeCapabilities) -> NodeScale:
        """Classify node based on computational capabilities"""
        
        compute_power = capabilities.compute_power_tflops
        
        if compute_power < 1.0:  # Less than 1 TFLOP (consumer hardware)
            return NodeScale.MICRO
        elif compute_power < 10.0:  # 1-10 TFLOPS (workstation class)
            return NodeScale.SMALL
        elif compute_power < 100.0:  # 10-100 TFLOPS (small clusters)
            return NodeScale.MEDIUM
        elif compute_power < 1000.0:  # 100-1000 TFLOPS (enterprise)
            return NodeScale.LARGE
        else:  # 1+ PFLOPS (hyperscale)
            return NodeScale.MASSIVE
    
    async def _calculate_node_credit_score(self,
                                         node_id: UUID,
                                         capabilities: NodeCapabilities,
                                         contribution_types: List[ContributionType],
                                         contribution_history: List[Dict[str, Any]]) -> NodeCreditScore:
        """Calculate multi-dimensional credit score for a node"""
        
        # Compute score (normalized and capped to prevent dominance)
        compute_score = min(1.0, math.log10(capabilities.compute_power_tflops + 1) / 3.0)
        
        # Reliability score based on uptime
        reliability_score = min(1.0, capabilities.uptime_percentage)
        
        # Diversity score based on unique contributions
        diversity_factors = []
        
        # Geographic diversity
        if capabilities.geographic_location not in ["us-east", "us-west", "eu-west"]:  # Common regions
            diversity_factors.append(0.3)
        
        # Domain expertise diversity
        if len(capabilities.specialized_domains) > 0:
            diversity_factors.append(0.3)
        
        # Model type diversity
        if len(capabilities.supported_model_types) > 2:
            diversity_factors.append(0.2)
        
        diversity_score = min(1.0, sum(diversity_factors))
        
        # Quality score based on contribution history
        quality_score = 0.5  # Base score
        if contribution_history:
            # Analyze recent contributions for quality indicators
            recent_quality_metrics = [
                contrib.get("quality_rating", 0.5) 
                for contrib in contribution_history[-10:]  # Last 10 contributions
            ]
            if recent_quality_metrics:
                quality_score = sum(recent_quality_metrics) / len(recent_quality_metrics)
        
        # Innovation score based on novel contributions
        innovation_score = 0.3  # Base score
        for contrib_type in contribution_types:
            if contrib_type in [ContributionType.INNOVATION, ContributionType.MODEL_DIVERSITY]:
                innovation_score += 0.2
        innovation_score = min(1.0, innovation_score)
        
        # Calculate overall score (weighted average)
        weights = {
            "compute": 0.2,      # Lower weight to prevent dominance
            "reliability": 0.25,
            "diversity": 0.25,
            "quality": 0.2,
            "innovation": 0.1
        }
        
        overall_score = (
            compute_score * weights["compute"] +
            reliability_score * weights["reliability"] +
            diversity_score * weights["diversity"] +
            quality_score * weights["quality"] +
            innovation_score * weights["innovation"]
        )
        
        return NodeCreditScore(
            node_id=node_id,
            compute_score=compute_score,
            reliability_score=reliability_score,
            diversity_score=diversity_score,
            quality_score=quality_score,
            innovation_score=innovation_score,
            overall_score=overall_score
        )
    
    async def _calculate_reward_multiplier(self,
                                         node_scale: NodeScale,
                                         capabilities: NodeCapabilities,
                                         credit_score: NodeCreditScore) -> RewardMultiplier:
        """Calculate reward multiplier using logarithmic curve and bonuses"""
        
        # Base multiplier
        base_multiplier = 1.0
        
        # Logarithmic scale adjustment (reduces rewards for larger nodes)
        if node_scale == NodeScale.MICRO:
            scale_adjustment = self.logarithmic_curve_parameters["micro_node_bonus"]
        elif node_scale == NodeScale.SMALL:
            scale_adjustment = 1.5
        elif node_scale == NodeScale.MEDIUM:
            scale_adjustment = 1.0
        elif node_scale == NodeScale.LARGE:
            scale_adjustment = 0.5
        else:  # MASSIVE
            scale_adjustment = self.logarithmic_curve_parameters["max_scale_penalty"]
        
        # Diversity bonus
        diversity_bonus = 1.0
        
        # Geographic diversity
        if capabilities.geographic_location not in ["us-east", "us-west", "eu-west"]:
            diversity_bonus *= self.diversity_bonuses["geographic_unique"]
        
        # Domain expertise
        if len(capabilities.specialized_domains) > 0:
            diversity_bonus *= self.diversity_bonuses["domain_expertise"]
        
        # Reliability bonus
        reliability_bonus = 1.0
        if capabilities.uptime_percentage > 0.99:  # 99%+ uptime
            reliability_bonus = 1.3
        elif capabilities.uptime_percentage > 0.95:  # 95%+ uptime
            reliability_bonus = 1.1
        
        # Innovation bonus based on credit score
        innovation_bonus = 1.0 + (credit_score.innovation_score * 0.5)
        
        # Calculate final multiplier
        final_multiplier = (
            base_multiplier * 
            scale_adjustment * 
            diversity_bonus * 
            reliability_bonus * 
            innovation_bonus
        )
        
        return RewardMultiplier(
            base_multiplier=base_multiplier,
            scale_adjustment=scale_adjustment,
            diversity_bonus=diversity_bonus,
            reliability_bonus=reliability_bonus,
            innovation_bonus=innovation_bonus,
            final_multiplier=final_multiplier
        )
    
    def _estimate_daily_rewards(self, reward_multiplier: RewardMultiplier) -> float:
        """Estimate daily FTNS rewards for a node"""
        base_daily_reward = self.logarithmic_curve_parameters["base_reward"]
        return base_daily_reward * reward_multiplier.final_multiplier
    
    def _get_empowerment_benefits(self, node_scale: NodeScale) -> List[str]:
        """Get empowerment benefits for different node scales"""
        
        benefits = {
            NodeScale.MICRO: [
                "2x reward multiplier for micro nodes",
                "Priority support for individual researchers",
                "Educational resources and tutorials",
                "Community mentorship programs"
            ],
            NodeScale.SMALL: [
                "1.5x reward multiplier for small teams",
                "Collaboration matching with larger nodes",
                "Access to shared computational resources",
                "Grant application assistance"
            ],
            NodeScale.MEDIUM: [
                "Standard reward rates",
                "Institutional partnership opportunities",
                "Advanced API access",
                "Research collaboration networks"
            ],
            NodeScale.LARGE: [
                "Enterprise support and SLAs",
                "Custom integration assistance",
                "Governance participation rights",
                "Strategic partnership opportunities"
            ],
            NodeScale.MASSIVE: [
                "Hyperscale infrastructure support",
                "Custom protocol development",
                "Global network coordination",
                "Policy influence and advisory roles"
            ]
        }
        
        return benefits.get(node_scale, [])
    
    def _calculate_network_health_score(self,
                                      scale_percentages: Dict[str, float],
                                      geographic_distribution: Dict[str, int],
                                      credit_scores: List[float]) -> float:
        """Calculate overall network health score"""
        
        # Scale diversity (want good distribution across scales)
        micro_small_percentage = scale_percentages.get("micro", 0) + scale_percentages.get("small", 0)
        scale_health = min(1.0, micro_small_percentage / 40.0)  # Target 40% micro+small
        
        # Geographic diversity
        geo_diversity = min(1.0, len(geographic_distribution) / 20.0)  # Target 20 regions
        
        # Credit score distribution
        avg_credit_score = sum(credit_scores) / len(credit_scores) if credit_scores else 0
        score_health = avg_credit_score
        
        # Combined health score
        return (scale_health + geo_diversity + score_health) / 3.0
    
    def _calculate_decentralization_index(self, scale_percentages: Dict[str, float]) -> float:
        """Calculate decentralization index (higher = more decentralized)"""
        
        # Perfect decentralization would be even distribution
        # But we want to favor smaller nodes, so weight accordingly
        target_distribution = {
            "micro": 30.0,
            "small": 25.0,
            "medium": 20.0,
            "large": 15.0,
            "massive": 10.0
        }
        
        # Calculate deviation from target
        total_deviation = 0.0
        for scale, target_pct in target_distribution.items():
            actual_pct = scale_percentages.get(scale, 0)
            deviation = abs(actual_pct - target_pct)
            total_deviation += deviation
        
        # Convert to index (0-1, higher is better)
        max_possible_deviation = 200.0  # Maximum theoretical deviation
        decentralization_index = 1.0 - (total_deviation / max_possible_deviation)
        
        return max(0.0, decentralization_index)
    
    def _generate_network_recommendations(self,
                                        scale_percentages: Dict[str, float],
                                        health_score: float) -> List[str]:
        """Generate recommendations for improving network health"""
        
        recommendations = []
        
        # Check micro/small node representation
        micro_small_pct = scale_percentages.get("micro", 0) + scale_percentages.get("small", 0)
        if micro_small_pct < 30:
            recommendations.append("Increase incentives for micro and small nodes")
        
        # Check for excessive concentration
        massive_pct = scale_percentages.get("massive", 0)
        if massive_pct > 25:
            recommendations.append("Implement stronger scale penalties for massive nodes")
        
        # Overall health
        if health_score < 0.7:
            recommendations.append("Focus on network diversity and geographic distribution")
        
        return recommendations


# Global micro-node empowerment instance
micro_node_empowerment = MicroNodeEmpowerment()