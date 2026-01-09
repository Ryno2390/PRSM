#!/usr/bin/env python3
"""
Enhanced Breakthrough Assessment System
More nuanced evaluation beyond arbitrary thresholds
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json

class BreakthroughCategory(Enum):
    REVOLUTIONARY = "revolutionary"  # Paradigm-shifting discoveries
    HIGH_IMPACT = "high_impact"      # Significant advancement in field
    INCREMENTAL = "incremental"      # Meaningful improvement
    SPECULATIVE = "speculative"      # High potential, high uncertainty
    NICHE = "niche"                  # Valuable for specific applications

class TimeHorizon(Enum):
    IMMEDIATE = "0-2_years"          # Ready for implementation
    NEAR_TERM = "2-5_years"          # Needs development
    LONG_TERM = "5-10_years"         # Research required
    BLUE_SKY = "10+_years"           # Fundamental breakthroughs

class RiskLevel(Enum):
    LOW = "low"                      # Proven concepts, clear path
    MODERATE = "moderate"            # Some technical challenges
    HIGH = "high"                    # Major hurdles to overcome
    EXTREME = "extreme"              # Fundamental unknowns

@dataclass
class BreakthroughProfile:
    """Multi-dimensional breakthrough assessment"""
    
    # Core Assessment
    discovery_id: str
    description: str
    source_papers: List[str]
    
    # Multi-Dimensional Scores (0-1)
    scientific_novelty: float
    commercial_potential: float
    technical_feasibility: float
    evidence_strength: float
    
    # Contextual Assessment
    category: BreakthroughCategory
    time_horizon: TimeHorizon
    risk_level: RiskLevel
    
    # Strategic Factors
    market_size_estimate: str           # "Small niche" to "Multi-billion"
    competitive_advantage: float        # How defensible (0-1)
    resource_requirements: str          # "Minimal" to "Massive investment"
    strategic_alignment: float          # Fit with org goals (0-1)
    
    # Implementation Path
    next_steps: List[str]
    success_probability: float          # Overall success likelihood
    value_at_risk: str                 # Potential downside
    upside_potential: str              # Potential value creation
    
    # Uncertainty Factors
    key_assumptions: List[str]
    failure_modes: List[str]
    sensitivity_factors: List[str]

class EnhancedBreakthroughAssessor:
    """More sophisticated breakthrough evaluation"""
    
    def __init__(self):
        # Instead of fixed thresholds, use assessment frameworks
        self.assessment_frameworks = {
            'academic': {
                'weights': {'scientific_novelty': 0.4, 'evidence_strength': 0.3, 
                           'technical_feasibility': 0.2, 'commercial_potential': 0.1}
            },
            'industry': {
                'weights': {'commercial_potential': 0.4, 'technical_feasibility': 0.3,
                           'competitive_advantage': 0.2, 'strategic_alignment': 0.1}
            },
            'startup': {
                'weights': {'commercial_potential': 0.3, 'time_horizon': 0.25,
                           'resource_requirements': 0.25, 'risk_level': 0.2}
            },
            'vc_fund': {
                'weights': {'market_size_estimate': 0.3, 'competitive_advantage': 0.25,
                           'success_probability': 0.25, 'time_horizon': 0.2}
            }
        }
    
    def assess_breakthrough(self, mapping_data: Dict, context: Dict = None) -> BreakthroughProfile:
        """Comprehensive breakthrough assessment"""
        
        # Extract basic info
        discovery_id = mapping_data.get('discovery_id', 'unknown')
        description = mapping_data.get('description', 'No description')
        source_papers = mapping_data.get('source_papers', [])
        
        # Calculate dimensional scores
        scientific_novelty = self._assess_scientific_novelty(mapping_data)
        commercial_potential = self._assess_commercial_potential(mapping_data)
        technical_feasibility = self._assess_technical_feasibility(mapping_data)
        evidence_strength = self._assess_evidence_strength(mapping_data)
        
        # Determine categorical assessments
        category = self._categorize_breakthrough(mapping_data, scientific_novelty, commercial_potential)
        time_horizon = self._estimate_time_horizon(technical_feasibility, evidence_strength)
        risk_level = self._assess_risk_level(mapping_data, technical_feasibility)
        
        # Strategic analysis
        market_size = self._estimate_market_size(mapping_data)
        competitive_advantage = self._assess_competitive_advantage(mapping_data)
        resource_requirements = self._estimate_resource_requirements(mapping_data)
        strategic_alignment = self._assess_strategic_alignment(mapping_data, context)
        
        # Implementation analysis
        next_steps = self._define_next_steps(category, time_horizon, risk_level)
        success_probability = self._calculate_success_probability(
            technical_feasibility, evidence_strength, risk_level
        )
        value_at_risk = self._assess_downside_risk(mapping_data, risk_level)
        upside_potential = self._assess_upside_potential(mapping_data, market_size)
        
        # Uncertainty analysis
        key_assumptions = self._identify_key_assumptions(mapping_data)
        failure_modes = self._identify_failure_modes(mapping_data, technical_feasibility)
        sensitivity_factors = self._identify_sensitivity_factors(mapping_data)
        
        return BreakthroughProfile(
            discovery_id=discovery_id,
            description=description,
            source_papers=source_papers,
            scientific_novelty=scientific_novelty,
            commercial_potential=commercial_potential,
            technical_feasibility=technical_feasibility,
            evidence_strength=evidence_strength,
            category=category,
            time_horizon=time_horizon,
            risk_level=risk_level,
            market_size_estimate=market_size,
            competitive_advantage=competitive_advantage,
            resource_requirements=resource_requirements,
            strategic_alignment=strategic_alignment,
            next_steps=next_steps,
            success_probability=success_probability,
            value_at_risk=value_at_risk,
            upside_potential=upside_potential,
            key_assumptions=key_assumptions,
            failure_modes=failure_modes,
            sensitivity_factors=sensitivity_factors
        )
    
    def _categorize_breakthrough(self, mapping_data: Dict, novelty: float, commercial: float) -> BreakthroughCategory:
        """Determine breakthrough category based on multiple factors"""
        
        if novelty > 0.8 and commercial > 0.7:
            return BreakthroughCategory.REVOLUTIONARY
        elif novelty > 0.6 or commercial > 0.7:
            return BreakthroughCategory.HIGH_IMPACT
        elif novelty > 0.4 and commercial > 0.5:
            return BreakthroughCategory.INCREMENTAL
        elif novelty > 0.7 and commercial < 0.4:
            return BreakthroughCategory.SPECULATIVE
        else:
            return BreakthroughCategory.NICHE
    
    def _estimate_time_horizon(self, feasibility: float, evidence: float) -> TimeHorizon:
        """Estimate realistic time to implementation"""
        
        readiness_score = (feasibility + evidence) / 2
        
        if readiness_score > 0.8:
            return TimeHorizon.IMMEDIATE
        elif readiness_score > 0.6:
            return TimeHorizon.NEAR_TERM
        elif readiness_score > 0.4:
            return TimeHorizon.LONG_TERM
        else:
            return TimeHorizon.BLUE_SKY
    
    def _assess_risk_level(self, mapping_data: Dict, feasibility: float) -> RiskLevel:
        """Assess implementation risk level"""
        
        # Check for risk indicators
        constraints = mapping_data.get('constraints', [])
        uncertainty_factors = len(constraints)
        
        risk_score = (1 - feasibility) + (uncertainty_factors * 0.1)
        
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MODERATE
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _estimate_market_size(self, mapping_data: Dict) -> str:
        """Estimate potential market size"""
        
        target_domain = mapping_data.get('target_domain', '')
        commercial_score = self._assess_commercial_potential(mapping_data)
        
        if 'fastening' in target_domain.lower():
            if commercial_score > 0.7:
                return "Multi-billion (global fastening market)"
            elif commercial_score > 0.5:
                return "Hundreds of millions (specialized applications)"
            else:
                return "Tens of millions (niche markets)"
        
        return "Market size requires analysis"
    
    def _calculate_success_probability(self, feasibility: float, evidence: float, risk: RiskLevel) -> float:
        """Calculate overall probability of successful implementation"""
        
        base_probability = (feasibility + evidence) / 2
        
        risk_multipliers = {
            RiskLevel.LOW: 0.9,
            RiskLevel.MODERATE: 0.7,
            RiskLevel.HIGH: 0.4,
            RiskLevel.EXTREME: 0.2
        }
        
        return base_probability * risk_multipliers[risk]
    
    # Additional helper methods would continue with similar logic...
    def _assess_scientific_novelty(self, mapping_data: Dict) -> float:
        # Simplified implementation
        return 0.6
    
    def _assess_commercial_potential(self, mapping_data: Dict) -> float:
        # Simplified implementation  
        return 0.5
    
    def _assess_technical_feasibility(self, mapping_data: Dict) -> float:
        # Simplified implementation
        return 0.7
    
    def _assess_evidence_strength(self, mapping_data: Dict) -> float:
        # Simplified implementation
        return 0.6
    
    def _assess_competitive_advantage(self, mapping_data: Dict) -> float:
        return 0.6
    
    def _estimate_resource_requirements(self, mapping_data: Dict) -> str:
        return "Moderate investment required"
    
    def _assess_strategic_alignment(self, mapping_data: Dict, context: Dict) -> float:
        return 0.7
    
    def _define_next_steps(self, category: BreakthroughCategory, time_horizon: TimeHorizon, risk: RiskLevel) -> List[str]:
        return ["Proof of concept development", "Market validation", "Technical feasibility study"]
    
    def _assess_downside_risk(self, mapping_data: Dict, risk: RiskLevel) -> str:
        return "Development cost and opportunity cost"
    
    def _assess_upside_potential(self, mapping_data: Dict, market_size: str) -> str:
        return "Significant market opportunity if successful"
    
    def _identify_key_assumptions(self, mapping_data: Dict) -> List[str]:
        return ["Technical feasibility assumptions", "Market acceptance assumptions"]
    
    def _identify_failure_modes(self, mapping_data: Dict, feasibility: float) -> List[str]:
        return ["Technical implementation challenges", "Market adoption barriers"]
    
    def _identify_sensitivity_factors(self, mapping_data: Dict) -> List[str]:
        return ["Material costs", "Manufacturing scalability"]

def generate_portfolio_recommendations(breakthroughs: List[BreakthroughProfile], 
                                     organization_type: str = 'industry') -> Dict:
    """Generate portfolio-level recommendations"""
    
    recommendations = {
        'immediate_action': [],
        'research_pipeline': [],
        'monitoring_list': [],
        'strategic_bets': [],
        'resource_allocation': {},
        'risk_assessment': {}
    }
    
    for breakthrough in breakthroughs:
        if breakthrough.time_horizon == TimeHorizon.IMMEDIATE and breakthrough.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]:
            recommendations['immediate_action'].append(breakthrough)
        elif breakthrough.category in [BreakthroughCategory.HIGH_IMPACT, BreakthroughCategory.REVOLUTIONARY]:
            recommendations['strategic_bets'].append(breakthrough)
        elif breakthrough.success_probability > 0.6:
            recommendations['research_pipeline'].append(breakthrough)
        else:
            recommendations['monitoring_list'].append(breakthrough)
    
    return recommendations

# Example of more nuanced assessment
def demonstrate_enhanced_assessment():
    """Show how enhanced assessment works"""
    
    sample_mapping = {
        'discovery_id': 'bio_adhesion_001',
        'description': 'Gecko-inspired reversible adhesion system',
        'source_papers': ['gecko_study_2024', 'biomimetic_adhesion_2023'],
        'target_domain': 'fastening_technology',
        'confidence': 0.75,
        'innovation_potential': 0.85,
        'constraints': ['manufacturing_complexity', 'material_costs']
    }
    
    assessor = EnhancedBreakthroughAssessor()
    profile = assessor.assess_breakthrough(sample_mapping)
    
    print("🔍 ENHANCED BREAKTHROUGH ASSESSMENT:")
    print(f"Category: {profile.category.value}")
    print(f"Time Horizon: {profile.time_horizon.value}")
    print(f"Risk Level: {profile.risk_level.value}")
    print(f"Success Probability: {profile.success_probability:.1%}")
    print(f"Market Size: {profile.market_size_estimate}")
    print(f"Next Steps: {', '.join(profile.next_steps[:2])}")

if __name__ == "__main__":
    demonstrate_enhanced_assessment()