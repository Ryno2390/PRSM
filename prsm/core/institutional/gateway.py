"""
PRSM Institutional Gateway
=========================

Enterprise-grade gateway for major AI labs and institutional participants.
Designed to handle the inevitable network effects where large players dominate
compute/storage while maintaining decentralized governance and fair competition.

Key Features:
- High-bandwidth model integration for frontier AI labs
- Enterprise SLA guarantees and priority access tiers
- Advanced provenance tracking with competitive intelligence protection
- Anti-monopoly safeguards and governance participation requirements
- Graduated onboarding for different institutional scales
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from dataclasses import dataclass

from pydantic import BaseModel, Field


class InstitutionTier(str, Enum):
    """Institutional participation tiers"""
    HOBBYIST = "hobbyist"          # Individual researchers, small teams
    STARTUP = "startup"            # AI startups, research groups  
    ENTERPRISE = "enterprise"      # Large tech companies
    FRONTIER_LAB = "frontier_lab"  # OpenAI, Anthropic, DeepMind scale
    CONSORTIUM = "consortium"      # Multi-institutional collaborations


class ParticipationMode(str, Enum):
    """How institutions can participate"""
    COMPUTE_PROVIDER = "compute_provider"     # Provide computational resources
    MODEL_CONTRIBUTOR = "model_contributor"   # Contribute model weights/training
    DATA_PROVIDER = "data_provider"          # Provide training data
    INFRASTRUCTURE = "infrastructure"         # Network infrastructure, storage
    GOVERNANCE_ONLY = "governance_only"      # Governance participation without resources


@dataclass
class InstitutionalCapacity:
    """Institutional computational and resource capacity"""
    compute_tflops: float
    storage_petabytes: float
    bandwidth_gbps: float
    model_parameters: int
    research_personnel: int
    annual_ai_budget_usd: float


class InstitutionalParticipant(BaseModel):
    """Major institutional participant in PRSM network"""
    participant_id: UUID = Field(default_factory=uuid4)
    institution_name: str
    tier: InstitutionTier
    participation_modes: List[ParticipationMode]
    capacity: InstitutionalCapacity
    governance_weight: float = Field(ge=0.0, le=1.0)
    anti_monopoly_score: float = Field(ge=0.0, le=1.0)  # Prevents excessive centralization
    join_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Competitive protections
    competitive_firewall: bool = True  # Isolate from direct competitors
    data_sovereignty: Dict[str, Any] = Field(default_factory=dict)
    
    # Economic terms
    royalty_share_percentage: float = Field(ge=0.0, le=100.0)
    priority_access_level: int = Field(ge=1, le=5)
    sla_guarantees: Dict[str, Any] = Field(default_factory=dict)


class InstitutionalGateway:
    """
    Enterprise gateway managing institutional participation in PRSM.
    
    Handles the transition from hobbyist to institutional dominance while
    maintaining decentralized governance and preventing monopolization.
    """
    
    def __init__(self):
        self.participants: Dict[UUID, InstitutionalParticipant] = {}
        self.tier_quotas = {
            InstitutionTier.FRONTIER_LAB: 0.25,    # Max 25% network control
            InstitutionTier.ENTERPRISE: 0.45,      # Max 45% combined enterprise control
            InstitutionTier.STARTUP: 0.20,         # Reserve 20% for startups
            InstitutionTier.HOBBYIST: 0.10         # Protect 10% for individuals
        }
        
        # Anti-centralization mechanisms
        self.max_single_participant_weight = 0.15  # No single entity > 15%
        self.competitive_separation_required = True
        
        print("🏛️ Institutional Gateway initialized")
        print("   - Designed for enterprise-scale AI lab integration")
        print("   - Anti-monopoly protections active")
        print("   - Competitive firewalls enabled")
    
    async def onboard_institution(self, 
                                institution_name: str,
                                capacity: InstitutionalCapacity,
                                requested_modes: List[ParticipationMode],
                                competitive_constraints: Dict[str, Any] = None) -> InstitutionalParticipant:
        """
        Onboard a new institutional participant with appropriate tier assignment
        and anti-monopoly protections.
        """
        
        # Determine tier based on capacity
        tier = self._assess_institutional_tier(capacity)
        
        # Calculate governance weight with anti-centralization limits
        governance_weight = self._calculate_governance_weight(tier, capacity)
        
        # Apply competitive protections
        competitive_firewall = self._requires_competitive_firewall(institution_name, tier)
        
        # Create participant
        participant = InstitutionalParticipant(
            institution_name=institution_name,
            tier=tier,
            participation_modes=requested_modes,
            capacity=capacity,
            governance_weight=governance_weight,
            anti_monopoly_score=self._calculate_anti_monopoly_score(capacity, tier),
            competitive_firewall=competitive_firewall,
            royalty_share_percentage=self._calculate_royalty_share(tier, capacity),
            priority_access_level=self._assign_priority_level(tier),
            sla_guarantees=self._generate_sla_guarantees(tier, capacity)
        )
        
        # Validate against centralization limits
        if await self._validate_decentralization_constraints(participant):
            self.participants[participant.participant_id] = participant
            
            print(f"🎯 Institutional participant onboarded: {institution_name}")
            print(f"   - Tier: {tier}")
            print(f"   - Governance weight: {governance_weight:.3f}")
            print(f"   - Priority level: {participant.priority_access_level}")
            
            return participant
        else:
            raise ValueError("Onboarding would violate decentralization constraints")
    
    async def coordinate_competitive_dynamics(self) -> Dict[str, Any]:
        """
        Coordinate competitive dynamics to encourage major AI lab participation
        while preventing monopolization.
        """
        
        # Identify competitive clusters
        competitive_clusters = self._identify_competitive_clusters()
        
        # Calculate network effects and incentives
        network_effects = {
            "total_compute_tflops": sum(p.capacity.compute_tflops for p in self.participants.values()),
            "total_storage_pb": sum(p.capacity.storage_petabytes for p in self.participants.values()),
            "frontier_lab_count": len([p for p in self.participants.values() if p.tier == InstitutionTier.FRONTIER_LAB]),
            "competitive_separation_index": self._calculate_competitive_separation(),
        }
        
        # Generate participation incentives for non-participants
        participation_incentives = self._generate_participation_incentives()
        
        return {
            "network_effects": network_effects,
            "competitive_clusters": competitive_clusters,
            "participation_incentives": participation_incentives,
            "centralization_risk": self._assess_centralization_risk(),
            "next_target_institutions": self._identify_target_institutions()
        }
    
    async def integrate_with_cdn_infrastructure(self, participant_id: UUID) -> Dict[str, Any]:
        """
        Integrate institutional participant with PRSM CDN infrastructure,
        enabling enterprise nodes to participate in decentralized content delivery.
        """
        
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        participant = self.participants[participant_id]
        
        # Import CDN components
        from ..infrastructure.cdn_layer import prsm_cdn, NodeType, BandwidthMetrics, GeographicLocation
        from ..infrastructure.sybil_resistance import sybil_resistance
        
        # Determine CDN node type based on institutional tier
        cdn_node_type_mapping = {
            InstitutionTier.FRONTIER_LAB: NodeType.CORE_PRSM,
            InstitutionTier.ENTERPRISE: NodeType.ENTERPRISE_GATEWAY,
            InstitutionTier.STARTUP: NodeType.EDGE_NODE,
            InstitutionTier.HOBBYIST: NodeType.MICRO_CACHE
        }
        
        node_type = cdn_node_type_mapping[participant.tier]
        
        # Estimate geographic location (would be provided during registration in production)
        location = GeographicLocation(
            continent="North America",
            country="United States", 
            region="West Coast",
            latitude=37.7749,
            longitude=-122.4194
        )
        
        # Create bandwidth metrics based on institutional capacity
        bandwidth_metrics = BandwidthMetrics(
            download_mbps=min(participant.capacity.bandwidth_gbps * 1000, 100000),  # Cap at 100 Gbps
            upload_mbps=min(participant.capacity.bandwidth_gbps * 1000 * 0.8, 80000),  # 80% of download
            latency_ms=max(1.0, 50.0 / (participant.capacity.bandwidth_gbps + 1)),  # Better capacity = lower latency
            packet_loss_rate=max(0.001, 0.01 / (participant.capacity.compute_tflops / 1000 + 1)),  # Better compute = better infrastructure
            uptime_percentage=min(0.9999, 0.95 + (participant.tier.value == "frontier_lab") * 0.05),  # Tier-based uptime
            geographic_location=location
        )
        
        # Register CDN node
        cdn_node = await prsm_cdn.register_cdn_node(
            node_type=node_type,
            operator_id=participant_id,
            storage_capacity_gb=participant.capacity.storage_petabytes * 1000 * 1000,  # Convert PB to GB
            bandwidth_metrics=bandwidth_metrics
        )
        
        # Fast-track validation for institutional participants
        institutional_voucher = f"{participant.institution_name.lower().replace(' ', '_')}@institutional.prsm.ai"
        
        validation_status = await sybil_resistance.validate_new_node(
            node_id=cdn_node.node_id,
            claimed_capabilities={
                "bandwidth_mbps": bandwidth_metrics.download_mbps,
                "storage_gb": cdn_node.storage_capacity_gb,
                "institutional_tier": participant.tier.value,
                "geographic_location": f"{location.continent},{location.country},{location.region}"
            },
            institutional_voucher=institutional_voucher
        )
        
        # Create enterprise-specific CDN configuration
        enterprise_config = {
            "dedicated_capacity_reserved": participant.capacity.storage_petabytes * 0.1,  # Reserve 10% for PRSM
            "priority_bandwidth_allocation": participant.priority_access_level,
            "competitive_firewall_active": participant.competitive_firewall,
            "sla_guarantees": participant.sla_guarantees,
            "revenue_sharing_model": {
                "base_ftns_per_gb": 0.01 * participant.priority_access_level,  # Higher tier = better rates
                "institutional_bonus_multiplier": {
                    InstitutionTier.FRONTIER_LAB: 2.5,
                    InstitutionTier.ENTERPRISE: 2.0,
                    InstitutionTier.STARTUP: 1.5,
                    InstitutionTier.HOBBYIST: 1.0
                }[participant.tier],
                "early_adopter_bonus": 1.5 if len(self.participants) <= 5 else 1.0  # First 5 get bonus
            }
        }
        
        print(f"🌐 CDN integration completed for {participant.institution_name}")
        print(f"   - CDN Node ID: {cdn_node.node_id}")
        print(f"   - Node Type: {node_type}")
        print(f"   - Storage Capacity: {cdn_node.storage_capacity_gb:,.0f} GB")
        print(f"   - Bandwidth: {bandwidth_metrics.download_mbps:,.0f} Mbps")
        print(f"   - Validation Status: {validation_status}")
        
        return {
            "cdn_node_id": cdn_node.node_id,
            "node_type": node_type,
            "validation_status": validation_status,
            "enterprise_config": enterprise_config,
            "estimated_monthly_ftns_earnings": self._estimate_cdn_earnings(participant, enterprise_config),
            "integration_timestamp": datetime.now(timezone.utc)
        }
    
    def _estimate_cdn_earnings(self, participant: InstitutionalParticipant, enterprise_config: Dict[str, Any]) -> float:
        """Estimate monthly FTNS earnings from CDN participation"""
        
        # Base calculation: storage * usage_rate * ftns_per_gb * days_per_month
        base_storage_gb = participant.capacity.storage_petabytes * 1000 * 1000 * 0.1  # 10% utilization
        usage_factor = {
            InstitutionTier.FRONTIER_LAB: 0.8,  # High usage due to popularity
            InstitutionTier.ENTERPRISE: 0.6,
            InstitutionTier.STARTUP: 0.4,
            InstitutionTier.HOBBYIST: 0.2
        }[participant.tier]
        
        daily_gb_served = base_storage_gb * usage_factor
        ftns_per_gb = enterprise_config["revenue_sharing_model"]["base_ftns_per_gb"]
        institutional_multiplier = enterprise_config["revenue_sharing_model"]["institutional_bonus_multiplier"]
        early_adopter_bonus = enterprise_config["revenue_sharing_model"]["early_adopter_bonus"]
        
        monthly_earnings = (daily_gb_served * ftns_per_gb * institutional_multiplier * 
                          early_adopter_bonus * 30)  # 30 days
        
        return float(monthly_earnings)
    
    def _assess_institutional_tier(self, capacity: InstitutionalCapacity) -> InstitutionTier:
        """Assess appropriate tier based on institutional capacity"""
        
        # Frontier lab thresholds (OpenAI/Anthropic/DeepMind scale)
        if (capacity.compute_tflops > 100000 and  # 100+ PetaFLOPS
            capacity.model_parameters > 1000000000000 and  # 1T+ parameters
            capacity.annual_ai_budget_usd > 1000000000):  # $1B+ budget
            return InstitutionTier.FRONTIER_LAB
        
        # Enterprise thresholds (Google, Microsoft, Meta scale)
        elif (capacity.compute_tflops > 10000 and
              capacity.model_parameters > 100000000000 and  # 100B+ parameters
              capacity.annual_ai_budget_usd > 100000000):  # $100M+ budget
            return InstitutionTier.ENTERPRISE
        
        # Startup thresholds
        elif (capacity.compute_tflops > 100 and
              capacity.annual_ai_budget_usd > 1000000):  # $1M+ budget
            return InstitutionTier.STARTUP
        
        else:
            return InstitutionTier.HOBBYIST
    
    def _calculate_governance_weight(self, tier: InstitutionTier, capacity: InstitutionalCapacity) -> float:
        """Calculate governance weight with anti-centralization limits"""
        
        # Base weights by tier
        base_weights = {
            InstitutionTier.FRONTIER_LAB: 0.12,   # Substantial but not dominant
            InstitutionTier.ENTERPRISE: 0.08,
            InstitutionTier.STARTUP: 0.04,
            InstitutionTier.HOBBYIST: 0.01
        }
        
        base_weight = base_weights[tier]
        
        # Capacity multiplier (but capped to prevent dominance)
        capacity_factor = min(2.0, (capacity.compute_tflops / 10000) ** 0.5)
        
        # Apply centralization limits
        proposed_weight = base_weight * capacity_factor
        return min(proposed_weight, self.max_single_participant_weight)
    
    def _requires_competitive_firewall(self, institution_name: str, tier: InstitutionTier) -> bool:
        """Determine if competitive firewall is required"""
        
        # Major competitors that need separation
        major_competitors = {
            "openai", "anthropic", "deepmind", "google", "microsoft", "meta", 
            "amazon", "apple", "nvidia", "tesla", "xai"
        }
        
        institution_lower = institution_name.lower()
        
        return (tier in [InstitutionTier.FRONTIER_LAB, InstitutionTier.ENTERPRISE] and
                any(competitor in institution_lower for competitor in major_competitors))
    
    def _calculate_anti_monopoly_score(self, capacity: InstitutionalCapacity, tier: InstitutionTier) -> float:
        """Calculate anti-monopoly score (higher = less centralization risk)"""
        
        # Base score by tier (frontier labs have lower scores due to centralization risk)
        tier_scores = {
            InstitutionTier.HOBBYIST: 1.0,
            InstitutionTier.STARTUP: 0.9,
            InstitutionTier.ENTERPRISE: 0.7,
            InstitutionTier.FRONTIER_LAB: 0.5
        }
        
        base_score = tier_scores[tier]
        
        # Penalize excessive capacity concentration
        capacity_penalty = min(0.3, capacity.compute_tflops / 1000000)  # Penalty for > 1 ExaFLOP
        
        return max(0.1, base_score - capacity_penalty)
    
    def _calculate_royalty_share(self, tier: InstitutionTier, capacity: InstitutionalCapacity) -> float:
        """Calculate royalty share percentage for contributed models"""
        
        # Base royalty rates
        base_rates = {
            InstitutionTier.FRONTIER_LAB: 15.0,   # Higher rates to incentivize participation
            InstitutionTier.ENTERPRISE: 12.0,
            InstitutionTier.STARTUP: 10.0,
            InstitutionTier.HOBBYIST: 8.0
        }
        
        return base_rates[tier]
    
    def _assign_priority_level(self, tier: InstitutionTier) -> int:
        """Assign priority access level"""
        
        priority_levels = {
            InstitutionTier.FRONTIER_LAB: 5,  # Highest priority
            InstitutionTier.ENTERPRISE: 4,
            InstitutionTier.STARTUP: 3,
            InstitutionTier.HOBBYIST: 2
        }
        
        return priority_levels[tier]
    
    def _generate_sla_guarantees(self, tier: InstitutionTier, capacity: InstitutionalCapacity) -> Dict[str, Any]:
        """Generate SLA guarantees based on tier and capacity"""
        
        sla_templates = {
            InstitutionTier.FRONTIER_LAB: {
                "uptime_percentage": 99.99,
                "max_latency_ms": 50,
                "priority_queue_position": 1,
                "dedicated_support": True,
                "custom_integration": True
            },
            InstitutionTier.ENTERPRISE: {
                "uptime_percentage": 99.95,
                "max_latency_ms": 100,
                "priority_queue_position": 2,
                "dedicated_support": True,
                "custom_integration": False
            },
            InstitutionTier.STARTUP: {
                "uptime_percentage": 99.9,
                "max_latency_ms": 200,
                "priority_queue_position": 3,
                "dedicated_support": False,
                "custom_integration": False
            },
            InstitutionTier.HOBBYIST: {
                "uptime_percentage": 99.5,
                "max_latency_ms": 500,
                "priority_queue_position": 4,
                "dedicated_support": False,
                "custom_integration": False
            }
        }
        
        return sla_templates[tier]
    
    async def _validate_decentralization_constraints(self, new_participant: InstitutionalParticipant) -> bool:
        """Validate that adding this participant doesn't violate decentralization constraints"""
        
        # Check single participant weight limit
        if new_participant.governance_weight > self.max_single_participant_weight:
            return False
        
        # Check tier quotas
        tier_total = sum(p.governance_weight for p in self.participants.values() 
                        if p.tier == new_participant.tier)
        tier_total += new_participant.governance_weight
        
        if tier_total > self.tier_quotas[new_participant.tier]:
            return False
        
        return True
    
    def _identify_competitive_clusters(self) -> Dict[str, List[str]]:
        """Identify competitive clusters among participants"""
        
        # This would use more sophisticated analysis in production
        clusters = {
            "frontier_ai_labs": [],
            "big_tech": [],
            "ai_startups": [],
            "research_institutions": []
        }
        
        for participant in self.participants.values():
            name_lower = participant.institution_name.lower()
            
            if participant.tier == InstitutionTier.FRONTIER_LAB:
                if any(lab in name_lower for lab in ["openai", "anthropic", "deepmind"]):
                    clusters["frontier_ai_labs"].append(participant.institution_name)
            elif participant.tier == InstitutionTier.ENTERPRISE:
                if any(tech in name_lower for tech in ["google", "microsoft", "meta", "amazon"]):
                    clusters["big_tech"].append(participant.institution_name)
        
        return clusters
    
    def _calculate_competitive_separation(self) -> float:
        """Calculate how well competitive separation is maintained"""
        
        # This would implement sophisticated competitive analysis
        # For now, return a reasonable baseline
        return 0.85  # 85% separation maintained
    
    def _generate_participation_incentives(self) -> Dict[str, Any]:
        """Generate incentives for non-participating institutions"""
        
        return {
            "first_mover_advantage": "Early participants get higher royalty rates",
            "network_effects_multiplier": "Larger network = more valuable models",
            "competitive_pressure": "Competitors joining creates urgency",
            "revenue_opportunity": "Proven model marketplace with growing demand",
            "defensive_necessity": "Join or risk being disrupted by participants"
        }
    
    def _assess_centralization_risk(self) -> Dict[str, float]:
        """Assess current centralization risk across dimensions"""
        
        total_compute = sum(p.capacity.compute_tflops for p in self.participants.values())
        total_governance = sum(p.governance_weight for p in self.participants.values())
        
        # Calculate Gini coefficient for compute distribution
        compute_distribution = [p.capacity.compute_tflops for p in self.participants.values()]
        compute_gini = self._calculate_gini_coefficient(compute_distribution)
        
        # Calculate governance concentration
        governance_distribution = [p.governance_weight for p in self.participants.values()]
        governance_gini = self._calculate_gini_coefficient(governance_distribution)
        
        return {
            "compute_centralization": compute_gini,
            "governance_centralization": governance_gini,
            "single_participant_risk": max(p.governance_weight for p in self.participants.values()) if self.participants else 0.0,
            "tier_balance_score": self._calculate_tier_balance()
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for distribution inequality"""
        if not values:
            return 0.0
        
        values_sorted = sorted(values)
        n = len(values)
        cumulative_sum = sum((i + 1) * value for i, value in enumerate(values_sorted))
        
        return (2 * cumulative_sum) / (n * sum(values)) - (n + 1) / n
    
    def _calculate_tier_balance(self) -> float:
        """Calculate how balanced participation is across tiers"""
        
        tier_counts = {}
        for participant in self.participants.values():
            tier_counts[participant.tier] = tier_counts.get(participant.tier, 0) + 1
        
        # Perfect balance would be equal representation
        if not tier_counts:
            return 1.0
        
        max_count = max(tier_counts.values())
        min_count = min(tier_counts.values())
        
        return 1.0 - (max_count - min_count) / max_count
    
    def _identify_target_institutions(self) -> List[Dict[str, Any]]:
        """Identify key institutions to target for participation"""
        
        # This would use market intelligence and competitive analysis
        targets = [
            {"name": "OpenAI", "priority": "critical", "rationale": "Frontier model leader"},
            {"name": "Anthropic", "priority": "critical", "rationale": "Constitutional AI pioneer"},
            {"name": "Google DeepMind", "priority": "critical", "rationale": "Research powerhouse"},
            {"name": "Microsoft", "priority": "high", "rationale": "Cloud infrastructure leader"},
            {"name": "Meta", "priority": "high", "rationale": "Open source AI advocate"},
            {"name": "Nvidia", "priority": "medium", "rationale": "Hardware infrastructure"},
        ]
        
        # Filter out institutions already participating
        participating_names = {p.institution_name.lower() for p in self.participants.values()}
        
        return [target for target in targets 
                if not any(target["name"].lower() in name for name in participating_names)]


# Global institutional gateway instance
institutional_gateway = InstitutionalGateway()