#!/usr/bin/env python3
"""
Calibrated Breakthrough Ranking System
Realistic quality scoring calibrated for genuine breakthrough discovery assessment
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

from breakthrough_ranker import BreakthroughScore, RankedBreakthrough, BreakthroughRanker

class CalibratedBreakthroughRanker(BreakthroughRanker):
    """
    Calibrated ranker with realistic quality scoring
    
    Key calibration changes:
    1. Much higher thresholds for quality tiers
    2. Stricter semantic matching requirements
    3. More conservative confidence assessment
    4. Realistic commercial potential evaluation
    """
    
    def __init__(self):
        super().__init__()
        
        # CALIBRATED: Much stricter quality thresholds
        self.calibrated_thresholds = {
            'breakthrough': 0.85,      # Was 0.8, now much higher
            'high_potential': 0.75,    # Was 0.7, now higher  
            'promising': 0.65,         # Was 0.6, now higher
            'moderate': 0.55,          # Was 0.5, now higher
            'low_priority': 0.45,      # Was 0.4, now higher
            'minimal': 0.0             # Below 0.45
        }
        
        # CALIBRATED: Stricter commercial relevance keywords
        self.high_value_keywords = [
            'adhesion', 'fastening', 'attachment', 'bonding', 'grip', 'stick',
            'manufacturing', 'scalable', 'cost_effective', 'commercial'
        ]
        
        # CALIBRATED: Require stronger novelty evidence
        self.strong_novelty_keywords = [
            'breakthrough', 'novel', 'first', 'unprecedented', 'revolutionary',
            'innovative', 'disruptive', 'unique'
        ]
        
        # CALIBRATED: Require proven feasibility evidence
        self.proven_feasibility_keywords = [
            'demonstrated', 'proven', 'validated', 'tested', 'fabricated',
            'manufactured', 'implemented', 'successful'
        ]
    
    def _calculate_novelty_score(self, text: str, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative novelty scoring"""
        
        score = 0.0
        
        # Require STRONG novelty evidence (not just any biomimetic content)
        strong_novelty_matches = sum(1 for keyword in self.strong_novelty_keywords if keyword in text.lower())
        if strong_novelty_matches > 0:
            score += min(0.4, strong_novelty_matches * 0.2)  # Much higher bar
        
        # Cross-domain distance bonus only for truly distant mappings
        bio_indicators = ['bio', 'biological', 'natural', 'organism']
        tech_indicators = ['engineering', 'synthetic', 'artificial', 'manufactured']
        
        bio_count = sum(1 for indicator in bio_indicators if indicator in text.lower())
        tech_count = sum(1 for indicator in tech_indicators if indicator in text.lower())
        
        # Require BOTH bio AND tech indicators for cross-domain bonus
        if bio_count >= 2 and tech_count >= 2:
            score += 0.3
        elif bio_count >= 1 and tech_count >= 1:
            score += 0.1
        
        # High confidence + high innovation must be EXCEPTIONAL
        confidence = mapping_data.get('confidence', 0.0)
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        
        # Much stricter requirements for genuine novelty
        if confidence > 0.8 and innovation_potential > 0.9:
            score += 0.3
        elif confidence > 0.7 and innovation_potential > 0.8:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_commercial_score(self, text: str, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative commercial scoring"""
        
        score = 0.0
        
        # Require DIRECT commercial relevance (not just generic terms)
        direct_commercial_matches = sum(1 for keyword in self.high_value_keywords if keyword in text.lower())
        
        # Much higher bar for commercial relevance
        if direct_commercial_matches >= 3:
            score += 0.4
        elif direct_commercial_matches >= 2:
            score += 0.2
        elif direct_commercial_matches >= 1:
            score += 0.1
        
        # Target domain must be genuinely relevant
        target_domain = mapping_data.get('target_domain', '')
        
        # Only give market bonus for truly relevant mappings
        if target_domain == 'fastening_technology':
            # Check if the source is genuinely related to fastening
            fastening_relevance = sum(1 for term in ['adhesion', 'attachment', 'fastening', 'grip', 'stick'] 
                                    if term in text.lower())
            if fastening_relevance >= 2:
                score += 0.3
            elif fastening_relevance >= 1:
                score += 0.15
        
        # Innovation potential bonus only for high commercial relevance
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        if direct_commercial_matches >= 2:
            score += innovation_potential * 0.1  # Much smaller bonus
        
        return min(1.0, score)
    
    def _calculate_feasibility_score(self, text: str, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative feasibility scoring"""
        
        score = 0.0
        
        # Require PROVEN feasibility evidence
        proven_matches = sum(1 for keyword in self.proven_feasibility_keywords if keyword in text.lower())
        
        if proven_matches >= 3:
            score += 0.4
        elif proven_matches >= 2:
            score += 0.25
        elif proven_matches >= 1:
            score += 0.1
        
        # Mapping confidence must be VERY high for feasibility bonus
        confidence = mapping_data.get('confidence', 0.0)
        if confidence > 0.85:
            score += 0.3
        elif confidence > 0.75:
            score += 0.15
        elif confidence > 0.65:
            score += 0.05
        
        # Constraints penalty is much more severe
        constraints = mapping_data.get('constraints', [])
        if isinstance(constraints, list):
            constraint_penalty = len(constraints) * 0.08  # Increased penalty
            score = max(0.0, score - constraint_penalty)
        
        return min(1.0, score)
    
    def _calculate_patent_potential(self, text: str, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative patent scoring"""
        
        score = 0.0
        
        # Require EXCEPTIONAL novelty for patent potential
        novelty_indicators = sum(1 for keyword in self.strong_novelty_keywords if keyword in text.lower())
        
        if novelty_indicators >= 2:
            score += 0.4
        elif novelty_indicators >= 1:
            score += 0.2
        
        # Cross-domain mapping must be genuinely novel
        if 'bio' in text.lower() and 'synthetic' in text.lower():
            # But only if it's actually innovative
            if novelty_indicators >= 1:
                score += 0.3
        
        # Require VERY high confidence for patent potential
        confidence = mapping_data.get('confidence', 0.0)
        if confidence > 0.9:
            score += 0.3
        elif confidence > 0.8:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_evidence_quality(self, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative evidence scoring"""
        
        score = 0.0
        
        # Mapping confidence must be EXCEPTIONAL
        confidence = mapping_data.get('confidence', 0.0)
        if confidence > 0.9:
            score += 0.5
        elif confidence > 0.8:
            score += 0.3
        elif confidence > 0.7:
            score += 0.15
        else:
            score += confidence * 0.1  # Much lower for mediocre confidence
        
        # Innovation potential must be VERY high
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        if innovation_potential > 0.9:
            score += 0.3
        elif innovation_potential > 0.8:
            score += 0.15
        else:
            score += innovation_potential * 0.1
        
        # Multiple patterns requirement is stricter
        source_patterns = mapping_data.get('source_patterns', [])
        if isinstance(source_patterns, list):
            if len(source_patterns) >= 10:
                score += 0.2
            elif len(source_patterns) >= 5:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_strategic_value(self, text: str, mapping_data: Dict) -> float:
        """CALIBRATED: Much more conservative strategic scoring"""
        
        score = 0.0
        
        # First-mover advantage requires EXCEPTIONAL novelty
        exceptional_novelty = sum(1 for word in ['breakthrough', 'first', 'unprecedented', 'revolutionary'] 
                                if word in text.lower())
        
        if exceptional_novelty >= 2:
            score += 0.4
        elif exceptional_novelty >= 1:
            score += 0.2
        
        # Innovation potential bonus only for truly exceptional cases
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        if innovation_potential > 0.95:
            score += 0.4
        elif innovation_potential > 0.85:
            score += 0.2
        
        # Biomimetic complexity bonus is much smaller
        if 'biomimetic' in text.lower() and exceptional_novelty >= 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _determine_quality_tier(self, overall_score: float) -> Tuple[str, str]:
        """CALIBRATED: Use calibrated thresholds for quality tiers"""
        
        if overall_score >= self.calibrated_thresholds['breakthrough']:
            return "BREAKTHROUGH", "IMMEDIATE DEVELOPMENT - Exceptional breakthrough with proven commercial potential"
        elif overall_score >= self.calibrated_thresholds['high_potential']:
            return "HIGH_POTENTIAL", "PRIORITY DEVELOPMENT - Strong evidence of commercial viability"
        elif overall_score >= self.calibrated_thresholds['promising']:
            return "PROMISING", "EVALUATE FURTHER - Promising but needs validation"
        elif overall_score >= self.calibrated_thresholds['moderate']:
            return "MODERATE", "MONITOR - Moderate potential, consider for future"
        elif overall_score >= self.calibrated_thresholds['low_priority']:
            return "LOW_PRIORITY", "ARCHIVE - Limited commercial potential"
        else:
            return "MINIMAL", "DISCARD - Insufficient evidence for development"
    
    def get_calibration_summary(self) -> Dict:
        """Get summary of calibration changes"""
        
        return {
            'calibration_type': 'Conservative Quality Scoring',
            'key_changes': [
                'Raised breakthrough threshold from 0.8 to 0.85',
                'Raised high_potential threshold from 0.7 to 0.75', 
                'Stricter novelty requirements (need strong evidence)',
                'Commercial relevance requires direct domain connection',
                'Feasibility requires proven demonstration evidence',
                'Patent potential needs exceptional novelty',
                'Evidence quality requires high confidence (>0.8)',
                'Strategic value needs revolutionary characteristics'
            ],
            'expected_impact': {
                'breakthrough_rate': '0.5-2% (vs previous 21%)',
                'high_potential_rate': '2-5% (vs previous 21%)',
                'commercial_rate': '5-15% (vs previous 98%)',
                'realistic_baseline': True
            },
            'thresholds': self.calibrated_thresholds
        }

def test_calibration():
    """Test calibrated vs original scoring"""
    
    print("🔧 CALIBRATED BREAKTHROUGH RANKER TEST")
    print("=" * 60)
    
    # Create sample mapping data  
    sample_mapping = {
        'source_papers': ['test_paper'],
        'target_domain': 'fastening_technology',
        'confidence': 0.70,
        'innovation_potential': 1.0,
        'description': 'Bio-inspired adhesion system',
        'source_patterns': ['pattern_1', 'pattern_2'],
        'target_applications': ['synthetic_fastening'],
        'key_innovations': ['biomimetic_design'],
        'testable_predictions': ['performance_validation'],
        'constraints': ['manufacturing_precision', 'material_selection'],
        'reasoning': 'Cross-domain mapping from biological adhesion to fastening'
    }
    
    # Test original vs calibrated
    original_ranker = BreakthroughRanker()
    calibrated_ranker = CalibratedBreakthroughRanker()
    
    original_breakthrough = original_ranker.rank_breakthrough(sample_mapping)
    calibrated_breakthrough = calibrated_ranker.rank_breakthrough(sample_mapping)
    
    print(f"📊 SCORING COMPARISON:")
    print(f"Original Score: {original_breakthrough.breakthrough_score.overall_score:.3f}")
    print(f"Calibrated Score: {calibrated_breakthrough.breakthrough_score.overall_score:.3f}")
    print(f"Original Tier: {original_breakthrough.breakthrough_score.quality_tier}")
    print(f"Calibrated Tier: {calibrated_breakthrough.breakthrough_score.quality_tier}")
    
    # Show calibration summary
    summary = calibrated_ranker.get_calibration_summary()
    print(f"\n🎯 CALIBRATION SUMMARY:")
    for change in summary['key_changes'][:5]:
        print(f"   • {change}")
    
    print(f"\n📈 EXPECTED IMPACT:")
    for metric, value in summary['expected_impact'].items():
        print(f"   {metric}: {value}")

if __name__ == "__main__":
    test_calibration()