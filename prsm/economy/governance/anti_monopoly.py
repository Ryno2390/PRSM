"""
Anti-Monopoly Governance Framework
==================================

Prevents centralization and ensures fair competition as PRSM scales to
institutional adoption. Implements constitutional safeguards against
the Bitcoin mining centralization pattern.

Key Mechanisms:
- Progressive voting weights with centralization penalties
- Mandatory diversity requirements for critical decisions
- Competitive firewall enforcement
- Sunset clauses for large participant privileges
- Automatic rebalancing when concentration thresholds are exceeded
"""

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ConcentrationAlert(str, Enum):
    """Centralization warning levels"""
    GREEN = "green"        # Healthy distribution
    YELLOW = "yellow"      # Moderate concentration
    ORANGE = "orange"      # High concentration
    RED = "red"           # Dangerous concentration
    CRITICAL = "critical"  # Emergency intervention required


class InterventionType(str, Enum):
    """Types of anti-monopoly interventions"""
    WEIGHT_REDUCTION = "weight_reduction"       # Reduce governance weight
    VOTING_SUSPENSION = "voting_suspension"     # Temporary voting suspension
    FORCED_DIVESTITURE = "forced_divestiture"   # Break up large participants
    COMPETITIVE_FIREWALL = "competitive_firewall"  # Isolate competitors
    DIVERSITY_REQUIREMENT = "diversity_requirement"  # Mandate diverse coalitions
    SUNSET_ENFORCEMENT = "sunset_enforcement"   # Enforce privilege expiration


@dataclass
class ConcentrationMetrics:
    """Metrics for measuring network concentration"""
    gini_coefficient: float  # 0 = perfect equality, 1 = total inequality
    herfindahl_index: float  # Sum of squared market shares
    top_1_share: float      # Largest participant's share
    top_3_share: float      # Top 3 participants' combined share
    top_5_share: float      # Top 5 participants' combined share
    effective_participants: int  # Number of meaningful participants


class AntiMonopolyIntervention(BaseModel):
    """Record of anti-monopoly intervention"""
    intervention_id: UUID = Field(default_factory=uuid4)
    intervention_type: InterventionType
    target_participants: List[UUID]
    trigger_metrics: ConcentrationMetrics
    justification: str
    implemented_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None


class AntiMonopolyGovernance:
    """
    Constitutional framework preventing centralization and monopolization
    as PRSM scales to institutional adoption.
    
    Designed to prevent the Bitcoin mining centralization pattern where
    hobbyists are pushed out by industrial-scale operations.
    """
    
    def __init__(self):
        # Concentration thresholds
        self.concentration_thresholds = {
            ConcentrationAlert.YELLOW: {
                "gini_coefficient": 0.6,
                "top_1_share": 0.15,
                "top_3_share": 0.35,
                "herfindahl_index": 0.15
            },
            ConcentrationAlert.ORANGE: {
                "gini_coefficient": 0.7,
                "top_1_share": 0.25,
                "top_3_share": 0.50,
                "herfindahl_index": 0.25
            },
            ConcentrationAlert.RED: {
                "gini_coefficient": 0.8,
                "top_1_share": 0.35,
                "top_3_share": 0.65,
                "herfindahl_index": 0.35
            },
            ConcentrationAlert.CRITICAL: {
                "gini_coefficient": 0.9,
                "top_1_share": 0.50,
                "top_3_share": 0.80,
                "herfindahl_index": 0.50
            }
        }
        
        # Constitutional constraints
        self.constitutional_limits = {
            "max_single_participant": 0.15,      # No single entity > 15%
            "max_three_participants": 0.35,      # Top 3 < 35% combined
            "min_effective_participants": 10,    # At least 10 meaningful participants
            "max_gini_coefficient": 0.75,        # Maximum inequality allowed
            "competitive_separation_required": True,
            "diversity_coalition_threshold": 0.67,  # Supermajority decisions need diverse coalitions
        }
        
        # Intervention history
        self.interventions: List[AntiMonopolyIntervention] = []
        self.active_interventions: Dict[UUID, AntiMonopolyIntervention] = {}
        
        # Monitoring
        self.last_assessment: Optional[datetime] = None
        self.current_alert_level = ConcentrationAlert.GREEN
        
        print("⚖️ Anti-Monopoly Governance Framework initialized")
        print("   - Constitutional limits active")
        print("   - Concentration monitoring enabled")
        print("   - Competitive firewall enforcement ready")
    
    async def assess_concentration_risk(self, 
                                      participants: Dict[UUID, Any],
                                      weights: Dict[UUID, float]) -> Tuple[ConcentrationAlert, ConcentrationMetrics]:
        """
        Assess current concentration risk across multiple dimensions.
        """
        
        if not participants:
            return ConcentrationAlert.GREEN, ConcentrationMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate concentration metrics
        weight_values = list(weights.values())
        weight_values.sort(reverse=True)
        
        metrics = ConcentrationMetrics(
            gini_coefficient=self._calculate_gini_coefficient(weight_values),
            herfindahl_index=sum(w**2 for w in weight_values),
            top_1_share=weight_values[0] if weight_values else 0,
            top_3_share=sum(weight_values[:3]),
            top_5_share=sum(weight_values[:5]),
            effective_participants=len([w for w in weight_values if w >= 0.01])  # > 1% weight
        )
        
        # Determine alert level
        alert_level = self._determine_alert_level(metrics)
        
        # Update monitoring state
        self.last_assessment = datetime.now(timezone.utc)
        self.current_alert_level = alert_level
        
        print(f"📊 Concentration assessment: {alert_level}")
        print(f"   - Gini coefficient: {metrics.gini_coefficient:.3f}")
        print(f"   - Top participant share: {metrics.top_1_share:.3f}")
        print(f"   - Effective participants: {metrics.effective_participants}")
        
        return alert_level, metrics
    
    async def enforce_constitutional_limits(self,
                                          participants: Dict[UUID, Any],
                                          weights: Dict[UUID, float],
                                          proposed_changes: Dict[UUID, float] = None) -> Dict[str, Any]:
        """
        Enforce constitutional limits and trigger interventions if necessary.
        """
        
        # Assess current state
        alert_level, metrics = await self.assess_concentration_risk(participants, weights)
        
        # Check if intervention is needed
        interventions_needed = []
        
        if alert_level in [ConcentrationAlert.ORANGE, ConcentrationAlert.RED, ConcentrationAlert.CRITICAL]:
            interventions_needed = await self._determine_required_interventions(
                alert_level, metrics, participants, weights
            )
        
        # Apply interventions
        enforcement_results = {
            "alert_level": alert_level,
            "metrics": metrics,
            "interventions_applied": [],
            "constitutional_violations": [],
            "adjusted_weights": weights.copy()
        }
        
        for intervention_type in interventions_needed:
            result = await self._apply_intervention(
                intervention_type, participants, weights, metrics
            )
            enforcement_results["interventions_applied"].append(result)
            
            # Update weights if intervention modified them
            if "adjusted_weights" in result:
                enforcement_results["adjusted_weights"] = result["adjusted_weights"]
        
        return enforcement_results
    
    async def validate_supermajority_diversity(self,
                                             voting_coalition: List[UUID],
                                             participants: Dict[UUID, Any]) -> bool:
        """
        Validate that supermajority decisions include sufficient diversity
        to prevent coordinated monopolization.
        """
        
        if len(voting_coalition) < 3:
            return False  # Need at least 3 participants for diversity
        
        # Check tier diversity
        tiers = set()
        competitive_groups = set()
        
        for participant_id in voting_coalition:
            participant = participants[participant_id]
            tiers.add(participant.tier)
            
            # Group competitive participants
            name_lower = participant.institution_name.lower()
            if any(competitor in name_lower for competitor in ["openai", "anthropic", "deepmind"]):
                competitive_groups.add("frontier_ai")
            elif any(tech in name_lower for tech in ["google", "microsoft", "meta", "amazon"]):
                competitive_groups.add("big_tech")
            else:
                competitive_groups.add("other")
        
        # Require diversity across tiers and competitive groups
        tier_diversity = len(tiers) >= 2
        competitive_diversity = len(competitive_groups) >= 2
        
        return tier_diversity and competitive_diversity
    
    async def monitor_competitive_coordination(self,
                                             participants: Dict[UUID, Any],
                                             recent_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor for suspicious coordination between competitors that could
        indicate collusion or anti-competitive behavior.
        """
        
        coordination_signals = {
            "suspicious_voting_patterns": [],
            "potential_collusion": [],
            "competitive_firewall_violations": [],
            "market_manipulation_indicators": []
        }
        
        # Analyze voting patterns for coordination
        if len(recent_votes) >= 10:  # Need sufficient data
            voting_correlations = self._analyze_voting_correlations(recent_votes, participants)
            
            # Flag unusually high correlations between competitors
            for (participant_a, participant_b), correlation in voting_correlations.items():
                if correlation > 0.8 and self._are_competitors(participant_a, participant_b, participants):
                    coordination_signals["suspicious_voting_patterns"].append({
                        "participants": [participant_a, participant_b],
                        "correlation": correlation,
                        "concern_level": "high" if correlation > 0.9 else "medium"
                    })
        
        return coordination_signals
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) == 1:
            return 0.0
        
        values_sorted = sorted(values)
        n = len(values)
        cumulative_sum = sum((i + 1) * value for i, value in enumerate(values_sorted))
        
        if sum(values) == 0:
            return 0.0
        
        return (2 * cumulative_sum) / (n * sum(values)) - (n + 1) / n
    
    def _determine_alert_level(self, metrics: ConcentrationMetrics) -> ConcentrationAlert:
        """Determine alert level based on concentration metrics"""
        
        # Check critical thresholds first
        critical = self.concentration_thresholds[ConcentrationAlert.CRITICAL]
        if (metrics.gini_coefficient >= critical["gini_coefficient"] or
            metrics.top_1_share >= critical["top_1_share"] or
            metrics.herfindahl_index >= critical["herfindahl_index"]):
            return ConcentrationAlert.CRITICAL
        
        # Check red thresholds
        red = self.concentration_thresholds[ConcentrationAlert.RED]
        if (metrics.gini_coefficient >= red["gini_coefficient"] or
            metrics.top_1_share >= red["top_1_share"] or
            metrics.herfindahl_index >= red["herfindahl_index"]):
            return ConcentrationAlert.RED
        
        # Check orange thresholds
        orange = self.concentration_thresholds[ConcentrationAlert.ORANGE]
        if (metrics.gini_coefficient >= orange["gini_coefficient"] or
            metrics.top_1_share >= orange["top_1_share"] or
            metrics.herfindahl_index >= orange["herfindahl_index"]):
            return ConcentrationAlert.ORANGE
        
        # Check yellow thresholds
        yellow = self.concentration_thresholds[ConcentrationAlert.YELLOW]
        if (metrics.gini_coefficient >= yellow["gini_coefficient"] or
            metrics.top_1_share >= yellow["top_1_share"] or
            metrics.herfindahl_index >= yellow["herfindahl_index"]):
            return ConcentrationAlert.YELLOW
        
        return ConcentrationAlert.GREEN
    
    async def _determine_required_interventions(self,
                                              alert_level: ConcentrationAlert,
                                              metrics: ConcentrationMetrics,
                                              participants: Dict[UUID, Any],
                                              weights: Dict[UUID, float]) -> List[InterventionType]:
        """Determine what interventions are needed based on concentration level"""
        
        interventions = []
        
        if alert_level == ConcentrationAlert.CRITICAL:
            # Emergency interventions
            if metrics.top_1_share > self.constitutional_limits["max_single_participant"]:
                interventions.append(InterventionType.WEIGHT_REDUCTION)
            
            interventions.append(InterventionType.FORCED_DIVESTITURE)
            interventions.append(InterventionType.DIVERSITY_REQUIREMENT)
            
        elif alert_level == ConcentrationAlert.RED:
            # Strong interventions
            if metrics.top_1_share > self.constitutional_limits["max_single_participant"]:
                interventions.append(InterventionType.WEIGHT_REDUCTION)
            
            interventions.append(InterventionType.COMPETITIVE_FIREWALL)
            interventions.append(InterventionType.DIVERSITY_REQUIREMENT)
            
        elif alert_level == ConcentrationAlert.ORANGE:
            # Moderate interventions
            interventions.append(InterventionType.COMPETITIVE_FIREWALL)
            
            if metrics.top_3_share > self.constitutional_limits["max_three_participants"]:
                interventions.append(InterventionType.WEIGHT_REDUCTION)
        
        return interventions
    
    async def _apply_intervention(self,
                                intervention_type: InterventionType,
                                participants: Dict[UUID, Any],
                                weights: Dict[UUID, float],
                                metrics: ConcentrationMetrics) -> Dict[str, Any]:
        """Apply specific anti-monopoly intervention"""
        
        if intervention_type == InterventionType.WEIGHT_REDUCTION:
            return await self._apply_weight_reduction(participants, weights, metrics)
        
        elif intervention_type == InterventionType.COMPETITIVE_FIREWALL:
            return await self._enforce_competitive_firewall(participants)
        
        elif intervention_type == InterventionType.DIVERSITY_REQUIREMENT:
            return await self._enforce_diversity_requirements(participants)
        
        elif intervention_type == InterventionType.FORCED_DIVESTITURE:
            return await self._initiate_forced_divestiture(participants, weights, metrics)
        
        else:
            return {"intervention_type": intervention_type, "status": "not_implemented"}
    
    async def _apply_weight_reduction(self,
                                    participants: Dict[UUID, Any],
                                    weights: Dict[UUID, float],
                                    metrics: ConcentrationMetrics) -> Dict[str, Any]:
        """Reduce governance weights of overly concentrated participants"""
        
        adjusted_weights = weights.copy()
        reductions = {}
        
        # Sort participants by weight
        sorted_participants = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Reduce weights of top participants if they exceed limits
        for participant_id, weight in sorted_participants[:3]:  # Top 3
            if weight > self.constitutional_limits["max_single_participant"]:
                new_weight = self.constitutional_limits["max_single_participant"] * 0.9  # 10% buffer
                reduction = weight - new_weight
                adjusted_weights[participant_id] = new_weight
                reductions[participant_id] = reduction
                
                print(f"🔻 Weight reduction applied: {participants[participant_id].institution_name}")
                print(f"   - Old weight: {weight:.3f}")
                print(f"   - New weight: {new_weight:.3f}")
        
        return {
            "intervention_type": InterventionType.WEIGHT_REDUCTION,
            "status": "applied",
            "adjusted_weights": adjusted_weights,
            "reductions": reductions,
            "affected_participants": list(reductions.keys())
        }
    
    async def _enforce_competitive_firewall(self, participants: Dict[UUID, Any]) -> Dict[str, Any]:
        """Enforce competitive firewalls between direct competitors"""
        
        firewall_enforcements = []
        
        # Identify competitive clusters
        competitive_clusters = self._identify_competitive_clusters(participants)
        
        for cluster_name, cluster_participants in competitive_clusters.items():
            if len(cluster_participants) > 1:
                # Enforce firewall between cluster members
                for participant_id in cluster_participants:
                    firewall_enforcements.append({
                        "participant_id": participant_id,
                        "cluster": cluster_name,
                        "firewall_level": "strict"
                    })
        
        return {
            "intervention_type": InterventionType.COMPETITIVE_FIREWALL,
            "status": "applied",
            "firewall_enforcements": firewall_enforcements,
            "clusters_affected": len(competitive_clusters)
        }
    
    async def _enforce_diversity_requirements(self, participants: Dict[UUID, Any]) -> Dict[str, Any]:
        """Enforce diversity requirements for future governance decisions"""
        
        diversity_requirements = {
            "minimum_tiers_required": 2,
            "minimum_competitive_groups_required": 2,
            "maximum_single_tier_percentage": 0.6,
            "supermajority_diversity_validation": True
        }
        
        return {
            "intervention_type": InterventionType.DIVERSITY_REQUIREMENT,
            "status": "applied",
            "requirements": diversity_requirements,
            "effective_immediately": True
        }
    
    async def _initiate_forced_divestiture(self,
                                         participants: Dict[UUID, Any],
                                         weights: Dict[UUID, float],
                                         metrics: ConcentrationMetrics) -> Dict[str, Any]:
        """Initiate forced divestiture for critically concentrated participants"""
        
        # This is the most extreme intervention - only for critical situations
        divestiture_targets = []
        
        # Target participants with > 40% individual weight
        for participant_id, weight in weights.items():
            if weight > 0.4:
                divestiture_targets.append({
                    "participant_id": participant_id,
                    "current_weight": weight,
                    "required_divestiture": weight - 0.15,  # Bring down to 15% max
                    "timeline_days": 90
                })
        
        return {
            "intervention_type": InterventionType.FORCED_DIVESTITURE,
            "status": "initiated",
            "targets": divestiture_targets,
            "timeline": "90 days",
            "appeal_process_available": True
        }
    
    def _identify_competitive_clusters(self, participants: Dict[UUID, Any]) -> Dict[str, List[UUID]]:
        """Identify competitive clusters among participants"""
        
        clusters = {
            "frontier_ai_labs": [],
            "big_tech": [],
            "cloud_providers": [],
            "hardware_vendors": []
        }
        
        for participant_id, participant in participants.items():
            name_lower = participant.institution_name.lower()
            
            # Frontier AI labs
            if any(lab in name_lower for lab in ["openai", "anthropic", "deepmind", "cohere"]):
                clusters["frontier_ai_labs"].append(participant_id)
            
            # Big tech companies
            elif any(tech in name_lower for tech in ["google", "microsoft", "meta", "amazon", "apple"]):
                clusters["big_tech"].append(participant_id)
            
            # Cloud providers
            elif any(cloud in name_lower for cloud in ["aws", "azure", "gcp", "oracle", "alibaba"]):
                clusters["cloud_providers"].append(participant_id)
            
            # Hardware vendors
            elif any(hw in name_lower for hw in ["nvidia", "amd", "intel", "qualcomm"]):
                clusters["hardware_vendors"].append(participant_id)
        
        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}
    
    def _analyze_voting_correlations(self,
                                   recent_votes: List[Dict[str, Any]],
                                   participants: Dict[UUID, Any]) -> Dict[Tuple[UUID, UUID], float]:
        """Analyze voting correlations between participants"""
        
        # This would implement sophisticated correlation analysis
        # For now, return empty dict
        return {}
    
    def _are_competitors(self,
                        participant_a: UUID,
                        participant_b: UUID,
                        participants: Dict[UUID, Any]) -> bool:
        """Check if two participants are direct competitors"""
        
        name_a = participants[participant_a].institution_name.lower()
        name_b = participants[participant_b].institution_name.lower()
        
        # Check if they're in the same competitive cluster
        competitive_groups = [
            ["openai", "anthropic", "deepmind", "cohere"],
            ["google", "microsoft", "meta", "amazon", "apple"],
            ["aws", "azure", "gcp", "oracle"],
            ["nvidia", "amd", "intel", "qualcomm"]
        ]
        
        for group in competitive_groups:
            in_group_a = any(comp in name_a for comp in group)
            in_group_b = any(comp in name_b for comp in group)
            
            if in_group_a and in_group_b:
                return True
        
        return False


# Global anti-monopoly governance instance
anti_monopoly_governance = AntiMonopolyGovernance()