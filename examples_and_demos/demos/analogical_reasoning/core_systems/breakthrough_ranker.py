#!/usr/bin/env python3
"""
Breakthrough Discovery Quality Ranking System
Ranks analogical mappings by commercial potential, scientific novelty, and feasibility
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

@dataclass
class BreakthroughScore:
    """Comprehensive scoring for breakthrough discoveries"""
    
    # Core Quality Metrics (0-1.0 scale)
    scientific_novelty: float = 0.0      # How novel is the cross-domain connection?
    commercial_potential: float = 0.0     # Market size and revenue potential
    technical_feasibility: float = 0.0    # Can this actually be built?
    patent_potential: float = 0.0         # IP protection and exclusivity
    
    # Evidence Quality Metrics (0-1.0 scale)
    source_credibility: float = 0.0       # Quality of source research
    mapping_confidence: float = 0.0       # Statistical confidence in analogy
    experimental_validation: float = 0.0   # How testable are the predictions?
    
    # Strategic Value Metrics (0-1.0 scale)
    competitive_advantage: float = 0.0     # First-mover advantage potential
    scalability: float = 0.0              # Can this scale to large markets?
    timeline_advantage: float = 0.0       # Speed to market advantage
    
    # Composite Scores
    overall_score: float = 0.0
    quality_tier: str = "Unranked"
    recommendation: str = ""

@dataclass
class RankedBreakthrough:
    """A breakthrough discovery with comprehensive ranking"""
    
    # Core Discovery Info
    discovery_id: str
    source_papers: List[str]
    target_domain: str
    discovery_description: str
    
    # Analogical Mapping Details
    source_patterns: List[str]
    target_applications: List[str]
    confidence_score: float
    innovation_potential: float
    
    # Quality Ranking
    breakthrough_score: BreakthroughScore
    
    # Commercial Assessment
    market_size_estimate: str
    revenue_potential: str
    development_cost: str
    time_to_market: str
    
    # Technical Details
    key_innovations: List[str]
    testable_predictions: List[str]
    manufacturing_requirements: List[str]
    
    # Risk Assessment
    technical_risks: List[str]
    market_risks: List[str]
    competitive_risks: List[str]

class BreakthroughRanker:
    """
    Comprehensive ranking system for breakthrough discoveries
    
    Evaluates analogical mappings across multiple dimensions to identify
    the highest-value opportunities for development and commercialization.
    """
    
    def __init__(self):
        # Commercial value indicators
        self.high_value_keywords = [
            'adhesion', 'fastening', 'attachment', 'bonding', 'grip',
            'efficiency', 'performance', 'optimization', 'enhancement',
            'manufacturing', 'scalable', 'cost', 'automated'
        ]
        
        # Technical novelty indicators  
        self.novelty_keywords = [
            'biomimetic', 'bio_inspired', 'novel', 'innovative', 'breakthrough',
            'unprecedented', 'first', 'unique', 'revolutionary', 'disruptive'
        ]
        
        # Feasibility indicators
        self.feasibility_keywords = [
            'proven', 'demonstrated', 'validated', 'tested', 'fabricated',
            'manufactured', 'scalable', 'practical', 'implementable'
        ]
        
        # Market size estimates for different domains
        self.market_sizes = {
            'fastening_technology': {'size': '15B', 'growth': '4.5%'},
            'adhesive_systems': {'size': '8.2B', 'growth': '5.1%'},
            'manufacturing_automation': {'size': '345B', 'growth': '7.8%'},
            'materials_engineering': {'size': '120B', 'growth': '6.2%'},
            'robotics_applications': {'size': '89B', 'growth': '12.3%'}
        }
    
    def rank_breakthrough(self, mapping_data: Dict) -> RankedBreakthrough:
        """Rank a single breakthrough discovery"""
        
        # Extract core information
        discovery_id = self._generate_discovery_id(mapping_data)
        
        # Calculate comprehensive scoring
        breakthrough_score = self._calculate_breakthrough_score(mapping_data)
        
        # Assess commercial potential
        commercial_assessment = self._assess_commercial_potential(mapping_data)
        
        # Evaluate technical feasibility
        technical_assessment = self._assess_technical_feasibility(mapping_data)
        
        # Risk analysis
        risk_assessment = self._assess_risks(mapping_data)
        
        return RankedBreakthrough(
            discovery_id=discovery_id,
            source_papers=mapping_data.get('source_papers', []),
            target_domain=mapping_data.get('target_domain', 'unknown'),
            discovery_description=self._generate_description(mapping_data),
            
            source_patterns=mapping_data.get('source_patterns', []),
            target_applications=mapping_data.get('target_applications', []),
            confidence_score=mapping_data.get('confidence', 0.0),
            innovation_potential=mapping_data.get('innovation_potential', 0.0),
            
            breakthrough_score=breakthrough_score,
            
            market_size_estimate=commercial_assessment['market_size'],
            revenue_potential=commercial_assessment['revenue_potential'],
            development_cost=commercial_assessment['development_cost'],
            time_to_market=commercial_assessment['time_to_market'],
            
            key_innovations=technical_assessment['innovations'],
            testable_predictions=technical_assessment['predictions'],
            manufacturing_requirements=technical_assessment['manufacturing'],
            
            technical_risks=risk_assessment['technical'],
            market_risks=risk_assessment['market'],
            competitive_risks=risk_assessment['competitive']
        )
    
    def _calculate_breakthrough_score(self, mapping_data: Dict) -> BreakthroughScore:
        """Calculate comprehensive breakthrough score"""
        
        # Extract text for analysis
        all_text = self._extract_all_text(mapping_data)
        
        # Scientific Novelty (0-1.0)
        novelty_score = self._calculate_novelty_score(all_text, mapping_data)
        
        # Commercial Potential (0-1.0)
        commercial_score = self._calculate_commercial_score(all_text, mapping_data)
        
        # Technical Feasibility (0-1.0)
        feasibility_score = self._calculate_feasibility_score(all_text, mapping_data)
        
        # Patent Potential (0-1.0)
        patent_score = self._calculate_patent_potential(all_text, mapping_data)
        
        # Evidence Quality (0-1.0)
        evidence_score = self._calculate_evidence_quality(mapping_data)
        
        # Strategic Value (0-1.0)
        strategic_score = self._calculate_strategic_value(all_text, mapping_data)
        
        # Calculate overall score (weighted average)
        weights = {
            'novelty': 0.20,
            'commercial': 0.25,
            'feasibility': 0.20,
            'patent': 0.15,
            'evidence': 0.10,
            'strategic': 0.10
        }
        
        overall_score = (
            novelty_score * weights['novelty'] +
            commercial_score * weights['commercial'] +
            feasibility_score * weights['feasibility'] +
            patent_score * weights['patent'] +
            evidence_score * weights['evidence'] +
            strategic_score * weights['strategic']
        )
        
        # Determine quality tier
        quality_tier, recommendation = self._determine_quality_tier(overall_score)
        
        return BreakthroughScore(
            scientific_novelty=novelty_score,
            commercial_potential=commercial_score,
            technical_feasibility=feasibility_score,
            patent_potential=patent_score,
            source_credibility=evidence_score,
            mapping_confidence=mapping_data.get('confidence', 0.0),
            experimental_validation=self._assess_testability(mapping_data),
            competitive_advantage=strategic_score,
            scalability=self._assess_scalability(all_text),
            timeline_advantage=self._assess_timeline_advantage(mapping_data),
            overall_score=overall_score,
            quality_tier=quality_tier,
            recommendation=recommendation
        )
    
    def _calculate_novelty_score(self, text: str, mapping_data: Dict) -> float:
        """Calculate scientific novelty score"""
        
        score = 0.0
        
        # Keyword-based novelty assessment
        novelty_matches = sum(1 for keyword in self.novelty_keywords if keyword in text.lower())
        score += min(0.4, novelty_matches * 0.1)
        
        # Cross-domain distance (biological → engineering = high novelty)
        if 'bio' in text.lower() and any(tech in text.lower() for tech in ['engineering', 'manufacturing', 'synthetic']):
            score += 0.3
        
        # Confidence in mapping (higher confidence = more novel if it's a distant mapping)
        confidence = mapping_data.get('confidence', 0.0)
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        
        # High confidence + high innovation = genuine novelty
        if confidence > 0.7 and innovation_potential > 0.8:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_commercial_score(self, text: str, mapping_data: Dict) -> float:
        """Calculate commercial potential score"""
        
        score = 0.0
        
        # Market-relevant keywords
        commercial_matches = sum(1 for keyword in self.high_value_keywords if keyword in text.lower())
        score += min(0.5, commercial_matches * 0.08)
        
        # Target domain market size
        target_domain = mapping_data.get('target_domain', '')
        if target_domain in self.market_sizes:
            market_info = self.market_sizes[target_domain]
            # Larger markets = higher score
            market_value = float(market_info['size'].replace('B', ''))
            if market_value > 50:
                score += 0.3
            elif market_value > 10:
                score += 0.2
            else:
                score += 0.1
        
        # Innovation potential indicates commercial disruption
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        score += innovation_potential * 0.2
        
        return min(1.0, score)
    
    def _calculate_feasibility_score(self, text: str, mapping_data: Dict) -> float:
        """Calculate technical feasibility score"""
        
        score = 0.0
        
        # Feasibility indicators
        feasibility_matches = sum(1 for keyword in self.feasibility_keywords if keyword in text.lower())
        score += min(0.4, feasibility_matches * 0.1)
        
        # Mapping confidence suggests technical viability
        confidence = mapping_data.get('confidence', 0.0)
        score += confidence * 0.3
        
        # Fewer constraints = higher feasibility
        constraints = mapping_data.get('constraints', [])
        if isinstance(constraints, list):
            constraint_penalty = len(constraints) * 0.05
            score += max(0.0, 0.3 - constraint_penalty)
        
        return min(1.0, score)
    
    def _calculate_patent_potential(self, text: str, mapping_data: Dict) -> float:
        """Calculate patent potential score"""
        
        score = 0.0
        
        # Novel cross-domain mappings have high patent potential
        if 'bio' in text.lower() and 'synthetic' in text.lower():
            score += 0.4
        
        # Specific technical innovations
        innovations = mapping_data.get('key_innovations', [])
        if isinstance(innovations, list) and len(innovations) > 0:
            score += min(0.3, len(innovations) * 0.1)
        
        # High confidence mappings are more patentable
        confidence = mapping_data.get('confidence', 0.0)
        if confidence > 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_evidence_quality(self, mapping_data: Dict) -> float:
        """Calculate evidence quality score"""
        
        score = 0.0
        
        # Mapping confidence
        score += mapping_data.get('confidence', 0.0) * 0.5
        
        # Innovation potential
        score += mapping_data.get('innovation_potential', 0.0) * 0.3
        
        # Multiple source patterns = stronger evidence
        source_patterns = mapping_data.get('source_patterns', [])
        if isinstance(source_patterns, list):
            score += min(0.2, len(source_patterns) * 0.05)
        
        return min(1.0, score)
    
    def _calculate_strategic_value(self, text: str, mapping_data: Dict) -> float:
        """Calculate strategic value score"""
        
        score = 0.0
        
        # First-mover advantage indicators
        if any(word in text.lower() for word in ['novel', 'first', 'unprecedented', 'breakthrough']):
            score += 0.4
        
        # High innovation potential = strategic advantage
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        score += innovation_potential * 0.4
        
        # Cross-domain complexity = competitive moats
        if 'biomimetic' in text.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_testability(self, mapping_data: Dict) -> float:
        """Assess how testable the predictions are"""
        
        score = 0.0
        
        # Check for testable predictions
        predictions = mapping_data.get('testable_predictions', [])
        if isinstance(predictions, list):
            score += min(0.6, len(predictions) * 0.2)
        
        # High confidence suggests testability
        confidence = mapping_data.get('confidence', 0.0)
        score += confidence * 0.4
        
        return min(1.0, score)
    
    def _assess_scalability(self, text: str) -> float:
        """Assess scalability potential"""
        
        score = 0.5  # Default moderate scalability
        
        if 'manufacturing' in text.lower():
            score += 0.3
        if 'scalable' in text.lower() or 'mass' in text.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_timeline_advantage(self, mapping_data: Dict) -> float:
        """Assess timeline to market advantage"""
        
        # Higher feasibility = faster timeline
        confidence = mapping_data.get('confidence', 0.0)
        constraints = len(mapping_data.get('constraints', []))
        
        # Fewer constraints and higher confidence = faster development
        score = confidence * 0.7 + max(0.0, (5 - constraints) * 0.06)
        
        return min(1.0, score)
    
    def _determine_quality_tier(self, overall_score: float) -> Tuple[str, str]:
        """Determine quality tier and recommendation"""
        
        if overall_score >= 0.8:
            return "BREAKTHROUGH", "IMMEDIATE DEVELOPMENT - Highest priority for R&D investment and patent filing"
        elif overall_score >= 0.7:
            return "HIGH_POTENTIAL", "PRIORITY DEVELOPMENT - Strong candidate for proof-of-concept development"
        elif overall_score >= 0.6:
            return "PROMISING", "EVALUATE FURTHER - Conduct detailed feasibility study before proceeding"
        elif overall_score >= 0.5:
            return "MODERATE", "MONITOR - Keep on watchlist for future development opportunities"
        elif overall_score >= 0.4:
            return "LOW_PRIORITY", "ARCHIVE - Low commercial potential, consider for research only"
        else:
            return "MINIMAL", "DISCARD - Insufficient evidence or potential for development"
    
    def _extract_all_text(self, mapping_data: Dict) -> str:
        """Extract all text from mapping data for analysis"""
        
        text_parts = []
        
        # Add various text fields
        for key in ['description', 'reasoning', 'source_patterns', 'target_applications', 'key_innovations']:
            value = mapping_data.get(key, '')
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, list):
                text_parts.extend([str(item) for item in value])
        
        return ' '.join(text_parts).lower()
    
    def _generate_discovery_id(self, mapping_data: Dict) -> str:
        """Generate unique discovery ID"""
        
        target = mapping_data.get('target_domain', 'unknown')
        confidence = mapping_data.get('confidence', 0.0)
        
        return f"{target}_{confidence:.2f}_{hash(str(mapping_data)) % 10000:04d}"
    
    def _generate_description(self, mapping_data: Dict) -> str:
        """Generate human-readable discovery description"""
        
        source_patterns = mapping_data.get('source_patterns', [])
        target_domain = mapping_data.get('target_domain', 'unknown domain')
        confidence = mapping_data.get('confidence', 0.0)
        
        if source_patterns:
            pattern_desc = f"patterns from {len(source_patterns)} source elements"
        else:
            pattern_desc = "cross-domain patterns"
        
        return f"Analogical mapping of {pattern_desc} to {target_domain} applications (confidence: {confidence:.2f})"
    
    def _assess_commercial_potential(self, mapping_data: Dict) -> Dict[str, str]:
        """Assess commercial potential"""
        
        target_domain = mapping_data.get('target_domain', '')
        confidence = mapping_data.get('confidence', 0.0)
        
        # Market size assessment
        if target_domain in self.market_sizes:
            market_info = self.market_sizes[target_domain]
            market_size = f"${market_info['size']} market growing at {market_info['growth']}/year"
        else:
            market_size = "Market size requires analysis"
        
        # Revenue potential based on confidence and innovation
        innovation_potential = mapping_data.get('innovation_potential', 0.0)
        if confidence > 0.7 and innovation_potential > 0.8:
            revenue_potential = "$10M-100M+ potential within 5 years"
        elif confidence > 0.6:
            revenue_potential = "$1M-10M potential within 3-5 years"
        else:
            revenue_potential = "$100K-1M potential, longer timeline"
        
        # Development cost estimate
        complexity = len(mapping_data.get('constraints', []))
        if complexity > 4:
            development_cost = "$1M-5M (high complexity)"
        elif complexity > 2:
            development_cost = "$500K-1M (moderate complexity)"
        else:
            development_cost = "$100K-500K (lower complexity)"
        
        # Time to market
        if confidence > 0.8:
            time_to_market = "1-2 years (high confidence)"
        elif confidence > 0.6:
            time_to_market = "2-3 years (moderate validation needed)"
        else:
            time_to_market = "3-5 years (significant development required)"
        
        return {
            'market_size': market_size,
            'revenue_potential': revenue_potential,
            'development_cost': development_cost,
            'time_to_market': time_to_market
        }
    
    def _assess_technical_feasibility(self, mapping_data: Dict) -> Dict[str, List[str]]:
        """Assess technical feasibility and requirements"""
        
        # Extract or generate technical details
        innovations = mapping_data.get('key_innovations', [])
        if not innovations:
            # Generate based on patterns
            innovations = ["biomimetic design approach", "cross-domain pattern implementation"]
        
        predictions = mapping_data.get('testable_predictions', [])
        if not predictions:
            predictions = ["performance improvement validation", "manufacturing feasibility confirmation"]
        
        manufacturing = mapping_data.get('manufacturing_requirements', [])
        if not manufacturing:
            manufacturing = ["specialized manufacturing process", "quality control systems"]
        
        return {
            'innovations': innovations,
            'predictions': predictions,
            'manufacturing': manufacturing
        }
    
    def _assess_risks(self, mapping_data: Dict) -> Dict[str, List[str]]:
        """Assess various risk categories"""
        
        confidence = mapping_data.get('confidence', 0.0)
        constraints = mapping_data.get('constraints', [])
        
        # Technical risks
        technical_risks = []
        if confidence < 0.7:
            technical_risks.append("Analogical mapping requires validation")
        if len(constraints) > 3:
            technical_risks.append("Multiple technical constraints to overcome")
        
        # Market risks
        market_risks = ["Market adoption uncertainty", "Competitive response"]
        
        # Competitive risks
        competitive_risks = ["Patent landscape analysis needed", "First-mover advantage timing"]
        
        return {
            'technical': technical_risks,
            'market': market_risks,
            'competitive': competitive_risks
        }

def analyze_batch_results(results_file: str) -> List[RankedBreakthrough]:
    """Analyze batch processing results and rank all breakthroughs"""
    
    ranker = BreakthroughRanker()
    ranked_breakthroughs = []
    
    # Load and process results
    # This would parse the actual batch results and extract mappings
    # For now, creating sample structure
    
    print("🔍 ANALYZING BREAKTHROUGH DISCOVERIES...")
    print("=" * 60)
    
    return ranked_breakthroughs

if __name__ == "__main__":
    # Example usage
    print("🏆 BREAKTHROUGH DISCOVERY RANKING SYSTEM")
    print("=" * 60)
    print("This system ranks analogical mappings by:")
    print("• Scientific novelty and innovation potential")
    print("• Commercial market opportunity and revenue potential") 
    print("• Technical feasibility and development timeline")
    print("• Patent potential and competitive advantage")
    print("• Evidence quality and experimental validation")
    print()
    print("Quality Tiers:")
    print("• BREAKTHROUGH (0.8+): Immediate development priority")
    print("• HIGH_POTENTIAL (0.7+): Priority development candidate")
    print("• PROMISING (0.6+): Evaluate further with feasibility study")
    print("• MODERATE (0.5+): Monitor for future opportunities")
    print("• LOW_PRIORITY (0.4+): Archive for research reference")
    print("• MINIMAL (<0.4): Discard - insufficient potential")