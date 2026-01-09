#!/usr/bin/env python3
"""
Cross-Domain Analogical Discovery Engine
Advanced pipeline enhancement for identifying breakthrough discoveries through 
cross-domain analogical pattern mapping.

This addresses the critical insight that true breakthroughs come from unexpected
connections between distant domains, not incremental advances within domains.
"""

import json
import time
import math
import itertools
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import numpy as np

@dataclass
class CrossDomainMapping:
    """Analogical mapping between discoveries from different domains"""
    source_discovery_id: str
    target_discovery_id: str
    source_domain: str
    target_domain: str
    analogical_strength: float
    mechanism_similarity: float
    functional_similarity: float
    structural_similarity: float
    novelty_score: float
    cross_domain_distance: float
    breakthrough_potential: float
    analogical_pattern: str

@dataclass
class CrossDomainBreakthrough:
    """Enhanced breakthrough discovery with cross-domain analogical foundation"""
    discovery_id: str
    primary_title: str
    source_domains: List[str]
    analogical_mappings: List[CrossDomainMapping]
    cross_domain_novelty: float
    analogical_confidence: float
    breakthrough_mechanism: str
    synthetic_potential: float
    domain_bridge_strength: float
    expected_value_multiplier: float

class CrossDomainAnalogicalEngine:
    """Advanced engine for cross-domain analogical discovery"""
    
    def __init__(self):
        self.setup_domain_distance_matrix()
        self.setup_analogical_patterns()
        self.setup_cross_domain_valuations()
        
    def setup_domain_distance_matrix(self):
        """Setup semantic distance matrix between scientific domains"""
        
        # Define domain relationships based on shared concepts/methods
        self.domain_relationships = {
            'biomolecular_engineering': {
                'biotechnology': 0.1,           # Very close
                'materials_science': 0.4,       # Bio-materials connection
                'nanotechnology': 0.3,          # Bio-nano interface
                'quantum_physics': 0.8,         # Very distant - high potential
                'artificial_intelligence': 0.6, # Bio-AI connection
                'energy_systems': 0.5,          # Bioenergy
                'photonics': 0.7,              # Bio-photonics
                'robotics': 0.6,               # Bio-inspired robotics
            },
            'quantum_physics': {
                'photonics': 0.2,              # Very close
                'semiconductor_physics': 0.3,  # Close
                'materials_science': 0.4,      # Quantum materials
                'artificial_intelligence': 0.5, # Quantum computing
                'biomolecular_engineering': 0.8, # Very distant - high potential
                'biotechnology': 0.8,          # Very distant - high potential
                'neuroscience': 0.7,           # Quantum brain theories
                'crystallography': 0.4,        # Quantum crystals
            },
            'artificial_intelligence': {
                'neuroscience': 0.2,           # Very close
                'robotics': 0.3,              # Close
                'computational_chemistry': 0.4, # AI for chemistry
                'biomolecular_engineering': 0.6, # AI for bio
                'quantum_physics': 0.5,        # Quantum AI
                'materials_science': 0.5,      # AI materials discovery
                'photonics': 0.6,             # Optical computing
            },
            'materials_science': {
                'nanotechnology': 0.2,         # Very close
                'crystallography': 0.3,        # Close
                'semiconductor_physics': 0.3,  # Close
                'quantum_physics': 0.4,        # Quantum materials
                'biomolecular_engineering': 0.4, # Bio-materials
                'energy_systems': 0.4,         # Energy materials
                'photonics': 0.5,             # Optical materials
            }
        }
        
        # Complete symmetric matrix
        all_domains = set()
        for domain, connections in self.domain_relationships.items():
            all_domains.add(domain)
            all_domains.update(connections.keys())
        
        # Add default distances for missing pairs
        for domain1 in all_domains:
            if domain1 not in self.domain_relationships:
                self.domain_relationships[domain1] = {}
            for domain2 in all_domains:
                if domain1 != domain2:
                    if domain2 not in self.domain_relationships[domain1]:
                        # Use reverse lookup or default
                        if domain2 in self.domain_relationships and domain1 in self.domain_relationships[domain2]:
                            self.domain_relationships[domain1][domain2] = self.domain_relationships[domain2][domain1]
                        else:
                            self.domain_relationships[domain1][domain2] = 0.6  # Default distance
    
    def setup_analogical_patterns(self):
        """Setup patterns for identifying analogical similarities"""
        
        self.analogical_patterns = {
            'structural_similarity': {
                'patterns': [
                    'network_topology', 'hierarchical_organization', 'symmetry_breaking',
                    'phase_transitions', 'self_organization', 'emergent_properties',
                    'scaling_laws', 'feedback_loops', 'critical_points'
                ],
                'weight': 0.3
            },
            'functional_similarity': {
                'patterns': [
                    'information_processing', 'energy_conversion', 'signal_transduction',
                    'pattern_recognition', 'optimization', 'adaptation',
                    'memory_storage', 'error_correction', 'self_repair'
                ],
                'weight': 0.4
            },
            'mechanism_similarity': {
                'patterns': [
                    'catalysis', 'resonance', 'interference', 'diffusion',
                    'amplification', 'filtering', 'switching', 'transport',
                    'binding', 'folding', 'assembly', 'disassembly'
                ],
                'weight': 0.3
            }
        }
    
    def setup_cross_domain_valuations(self):
        """Setup enhanced valuations for cross-domain discoveries"""
        
        # Value multipliers based on domain distance
        self.cross_domain_multipliers = {
            (0.0, 0.3): 1.1,     # very_close: Slight novelty bonus
            (0.3, 0.5): 1.5,     # close: Moderate novelty bonus
            (0.5, 0.7): 3.0,     # distant: High novelty bonus
            (0.7, 1.0): 10.0     # very_distant: Breakthrough potential
        }
        
        # Domain-specific analogical strengths
        self.domain_analogical_strength = {
            'biomolecular_engineering': 0.8,    # Rich in analogical patterns
            'quantum_physics': 0.9,             # Highly analogical field
            'artificial_intelligence': 0.7,     # Pattern-based field
            'materials_science': 0.6,           # Structure-focused
            'neuroscience': 0.8,               # Network-based
            'robotics': 0.7,                   # Bio-inspired
            'energy_systems': 0.5,             # Engineering-focused
            'photonics': 0.6,                  # Wave-based analogies
        }
    
    def extract_analogical_features(self, discovery: Dict) -> Dict[str, float]:
        """Extract features relevant for analogical mapping"""
        
        title = discovery.get('title', '').lower()
        domain = discovery.get('domain', '')
        
        features = {
            'structural_features': 0.0,
            'functional_features': 0.0,
            'mechanism_features': 0.0,
            'abstraction_level': 0.5,
            'analogical_richness': 0.5
        }
        
        # Analyze structural patterns
        structural_keywords = ['network', 'hierarchy', 'structure', 'organization', 'topology', 'symmetry']
        features['structural_features'] = sum(1 for kw in structural_keywords if kw in title) / len(structural_keywords)
        
        # Analyze functional patterns  
        functional_keywords = ['process', 'function', 'mechanism', 'pathway', 'system', 'operation']
        features['functional_features'] = sum(1 for kw in functional_keywords if kw in title) / len(functional_keywords)
        
        # Analyze mechanism patterns
        mechanism_keywords = ['binding', 'transport', 'assembly', 'conversion', 'catalysis', 'signaling']
        features['mechanism_features'] = sum(1 for kw in mechanism_keywords if kw in title) / len(mechanism_keywords)
        
        # Assess abstraction level (higher = more transferable)
        abstract_keywords = ['principle', 'theory', 'model', 'algorithm', 'method', 'approach']
        features['abstraction_level'] = min(1.0, 0.3 + sum(1 for kw in abstract_keywords if kw in title) * 0.15)
        
        # Domain-specific analogical richness
        features['analogical_richness'] = self.domain_analogical_strength.get(domain, 0.5)
        
        return features
    
    def calculate_analogical_similarity(self, discovery1: Dict, discovery2: Dict) -> CrossDomainMapping:
        """Calculate analogical similarity between discoveries from different domains"""
        
        domain1 = discovery1.get('domain', '')
        domain2 = discovery2.get('domain', '')
        
        # Skip if same domain
        if domain1 == domain2:
            return None
        
        # Get domain distance
        cross_domain_distance = self.domain_relationships.get(domain1, {}).get(domain2, 0.6)
        
        # Extract analogical features
        features1 = self.extract_analogical_features(discovery1)
        features2 = self.extract_analogical_features(discovery2)
        
        # Calculate similarity components
        structural_sim = 1.0 - abs(features1['structural_features'] - features2['structural_features'])
        functional_sim = 1.0 - abs(features1['functional_features'] - features2['functional_features'])
        mechanism_sim = 1.0 - abs(features1['mechanism_features'] - features2['mechanism_features'])
        
        # Weight similarities
        analogical_strength = (
            structural_sim * self.analogical_patterns['structural_similarity']['weight'] +
            functional_sim * self.analogical_patterns['functional_similarity']['weight'] +
            mechanism_sim * self.analogical_patterns['mechanism_similarity']['weight']
        )
        
        # Calculate novelty score (higher for distant domains with high similarity)
        novelty_score = cross_domain_distance * analogical_strength
        
        # Calculate breakthrough potential
        abstraction_bonus = (features1['abstraction_level'] + features2['abstraction_level']) / 2
        analogical_richness = (features1['analogical_richness'] + features2['analogical_richness']) / 2
        breakthrough_potential = novelty_score * abstraction_bonus * analogical_richness
        
        # Generate analogical pattern description
        analogical_pattern = self._generate_analogical_pattern_description(
            discovery1, discovery2, structural_sim, functional_sim, mechanism_sim
        )
        
        return CrossDomainMapping(
            source_discovery_id=discovery1.get('paper_id', 'unknown'),
            target_discovery_id=discovery2.get('paper_id', 'unknown'),
            source_domain=domain1,
            target_domain=domain2,
            analogical_strength=analogical_strength,
            mechanism_similarity=mechanism_sim,
            functional_similarity=functional_sim,
            structural_similarity=structural_sim,
            novelty_score=novelty_score,
            cross_domain_distance=cross_domain_distance,
            breakthrough_potential=breakthrough_potential,
            analogical_pattern=analogical_pattern
        )
    
    def _generate_analogical_pattern_description(self, discovery1: Dict, discovery2: Dict, 
                                               struct_sim: float, func_sim: float, mech_sim: float) -> str:
        """Generate human-readable description of analogical pattern"""
        
        domain1 = discovery1.get('domain', '').replace('_', ' ').title()
        domain2 = discovery2.get('domain', '').replace('_', ' ').title()
        
        # Determine primary similarity type
        if struct_sim > func_sim and struct_sim > mech_sim:
            pattern_type = "structural"
            pattern_desc = f"Similar organizational principles between {domain1} and {domain2}"
        elif func_sim > mech_sim:
            pattern_type = "functional"
            pattern_desc = f"Analogous functional mechanisms between {domain1} and {domain2}"
        else:
            pattern_type = "mechanistic"
            pattern_desc = f"Similar operational mechanisms between {domain1} and {domain2}"
        
        return f"{pattern_type.title()} analogy: {pattern_desc}"
    
    def identify_cross_domain_breakthroughs(self, discoveries: List[Dict], 
                                          min_breakthrough_potential: float = 0.3) -> List[CrossDomainBreakthrough]:
        """Identify breakthrough discoveries based on cross-domain analogical patterns"""
        
        print(f"🚀 IDENTIFYING CROSS-DOMAIN ANALOGICAL BREAKTHROUGHS")
        print("=" * 70)
        print(f"📊 Analyzing {len(discoveries)} discoveries for cross-domain patterns")
        
        cross_domain_mappings = []
        
        # Find all cross-domain analogical mappings
        for i, discovery1 in enumerate(discoveries):
            for j, discovery2 in enumerate(discoveries[i+1:], i+1):
                mapping = self.calculate_analogical_similarity(discovery1, discovery2)
                if mapping and mapping.breakthrough_potential >= min_breakthrough_potential:
                    cross_domain_mappings.append(mapping)
        
        print(f"   🎯 Found {len(cross_domain_mappings)} high-potential cross-domain mappings")
        
        # Group mappings into breakthrough opportunities
        breakthrough_clusters = self._cluster_analogical_mappings(cross_domain_mappings)
        
        cross_domain_breakthroughs = []
        for cluster_id, cluster_mappings in breakthrough_clusters.items():
            breakthrough = self._create_cross_domain_breakthrough(cluster_mappings, discoveries)
            if breakthrough:
                cross_domain_breakthroughs.append(breakthrough)
        
        print(f"   💎 Identified {len(cross_domain_breakthroughs)} cross-domain breakthrough opportunities")
        
        return cross_domain_breakthroughs
    
    def _cluster_analogical_mappings(self, mappings: List[CrossDomainMapping]) -> Dict[str, List[CrossDomainMapping]]:
        """Cluster analogical mappings into coherent breakthrough opportunities"""
        
        clusters = defaultdict(list)
        
        for mapping in mappings:
            # Create cluster key based on domain pair and pattern type
            domain_pair = tuple(sorted([mapping.source_domain, mapping.target_domain]))
            pattern_hash = hashlib.md5(mapping.analogical_pattern.encode()).hexdigest()[:8]
            cluster_key = f"{domain_pair[0]}_{domain_pair[1]}_{pattern_hash}"
            
            clusters[cluster_key].append(mapping)
        
        return dict(clusters)
    
    def _create_cross_domain_breakthrough(self, mappings: List[CrossDomainMapping], 
                                        discoveries: List[Dict]) -> Optional[CrossDomainBreakthrough]:
        """Create cross-domain breakthrough from cluster of mappings"""
        
        if not mappings:
            return None
        
        # Get representative mapping
        primary_mapping = max(mappings, key=lambda m: m.breakthrough_potential)
        
        # Extract involved domains
        source_domains = list(set([m.source_domain for m in mappings] + [m.target_domain for m in mappings]))
        
        # Calculate aggregate metrics
        avg_analogical_strength = sum(m.analogical_strength for m in mappings) / len(mappings)
        max_breakthrough_potential = max(m.breakthrough_potential for m in mappings)
        avg_cross_domain_distance = sum(m.cross_domain_distance for m in mappings) / len(mappings)
        
        # Calculate value multiplier based on cross-domain distance
        value_multiplier = 1.0
        for (min_dist, max_dist), multiplier in self.cross_domain_multipliers.items():
            if min_dist <= avg_cross_domain_distance < max_dist:
                value_multiplier = multiplier
                break
        
        # Generate synthetic breakthrough title
        domain_bridge = f"{primary_mapping.source_domain.replace('_', ' ').title()} → {primary_mapping.target_domain.replace('_', ' ').title()}"
        breakthrough_title = f"Cross-domain breakthrough: {domain_bridge} analogical synthesis"
        
        discovery_id = f"cross_domain_{primary_mapping.source_domain}_{primary_mapping.target_domain}_{hashlib.md5(breakthrough_title.encode()).hexdigest()[:8]}"
        
        return CrossDomainBreakthrough(
            discovery_id=discovery_id,
            primary_title=breakthrough_title,
            source_domains=source_domains,
            analogical_mappings=mappings,
            cross_domain_novelty=avg_cross_domain_distance,
            analogical_confidence=avg_analogical_strength,
            breakthrough_mechanism=primary_mapping.analogical_pattern,
            synthetic_potential=max_breakthrough_potential,
            domain_bridge_strength=len(mappings),
            expected_value_multiplier=value_multiplier
        )
    
    def calculate_cross_domain_value(self, breakthrough: CrossDomainBreakthrough, 
                                   base_discovery_value: float = 75.0) -> Dict[str, Any]:
        """Calculate enhanced valuation for cross-domain breakthrough"""
        
        # Base value enhancement
        enhanced_base_value = base_discovery_value * breakthrough.expected_value_multiplier
        
        # Cross-domain novelty bonus
        novelty_bonus = 1 + (breakthrough.cross_domain_novelty * 2)  # Up to 3x for very distant domains
        
        # Analogical confidence adjustment
        confidence_factor = 0.5 + (breakthrough.analogical_confidence * 0.5)  # 0.5 to 1.0 range
        
        # Domain bridge strength (number of supporting analogical mappings)
        bridge_factor = min(2.0, 1 + (breakthrough.domain_bridge_strength * 0.1))  # Up to 2x for strong bridges
        
        # Calculate final value
        final_value = enhanced_base_value * novelty_bonus * confidence_factor * bridge_factor
        
        # Determine investment category
        if final_value > 500:  # >$500M
            investment_category = "BREAKTHROUGH"
        elif final_value > 100:  # >$100M
            investment_category = "TRANSFORMATIVE"
        elif final_value > 50:   # >$50M
            investment_category = "SIGNIFICANT"
        else:
            investment_category = "VALUABLE"
        
        return {
            'base_value': base_discovery_value,
            'enhanced_base_value': enhanced_base_value,
            'novelty_bonus': novelty_bonus,
            'confidence_factor': confidence_factor,
            'bridge_factor': bridge_factor,
            'final_value': final_value,
            'investment_category': investment_category,
            'value_multiplier_total': final_value / base_discovery_value,
            'cross_domain_advantage': final_value / enhanced_base_value
        }
    
    def process_cross_domain_portfolio(self, discoveries: List[Dict]) -> Dict[str, Any]:
        """Process complete discovery portfolio for cross-domain opportunities"""
        
        print(f"🚀 CROSS-DOMAIN ANALOGICAL PORTFOLIO ANALYSIS")
        print("=" * 70)
        
        start_time = time.time()
        
        # Identify cross-domain breakthroughs
        cross_domain_breakthroughs = self.identify_cross_domain_breakthroughs(discoveries)
        
        # Calculate valuations
        breakthrough_valuations = []
        total_cross_domain_value = 0
        
        for breakthrough in cross_domain_breakthroughs:
            valuation = self.calculate_cross_domain_value(breakthrough)
            breakthrough_valuations.append({
                'breakthrough': asdict(breakthrough),
                'valuation': valuation
            })
            total_cross_domain_value += valuation['final_value']
        
        # Generate portfolio analysis
        portfolio_analysis = self._analyze_cross_domain_portfolio(cross_domain_breakthroughs, breakthrough_valuations)
        
        processing_time = time.time() - start_time
        
        results = {
            'processing_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_discoveries_analyzed': len(discoveries),
                'cross_domain_breakthroughs_identified': len(cross_domain_breakthroughs),
                'total_cross_domain_value': total_cross_domain_value,
                'processing_time_seconds': processing_time,
                'methodology': 'Cross-Domain Analogical Discovery Analysis'
            },
            'cross_domain_breakthroughs': breakthrough_valuations,
            'portfolio_analysis': portfolio_analysis
        }
        
        print(f"\n💎 CROSS-DOMAIN ANALYSIS COMPLETE!")
        print(f"   🎯 Cross-domain breakthroughs: {len(cross_domain_breakthroughs)}")
        print(f"   💰 Total cross-domain value: ${total_cross_domain_value:.1f}M")
        print(f"   📈 Average breakthrough value: ${total_cross_domain_value/max(1,len(cross_domain_breakthroughs)):.1f}M")
        print(f"   ⏱️ Processing time: {processing_time:.1f}s")
        
        return results
    
    def _analyze_cross_domain_portfolio(self, breakthroughs: List[CrossDomainBreakthrough], 
                                      valuations: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-domain breakthrough portfolio"""
        
        if not breakthroughs:
            return {
                'domain_bridge_analysis': {},
                'value_distribution': {},
                'analogical_pattern_analysis': {},
                'investment_categories': {},
                'top_opportunities': []
            }
        
        # Analyze domain bridges
        domain_bridges = defaultdict(list)
        for breakthrough in breakthroughs:
            bridge_key = " ↔ ".join(sorted(breakthrough.source_domains))
            domain_bridges[bridge_key].append(breakthrough)
        
        # Analyze value distribution
        values = [v['valuation']['final_value'] for v in valuations]
        value_distribution = {
            'total_value': sum(values),
            'average_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'median_value': sorted(values)[len(values)//2] if values else 0
        }
        
        # Analyze analogical patterns
        pattern_analysis = defaultdict(list)
        for breakthrough in breakthroughs:
            pattern_analysis[breakthrough.breakthrough_mechanism].append(breakthrough.synthetic_potential)
        
        # Analyze investment categories
        investment_categories = defaultdict(int)
        for valuation in valuations:
            investment_categories[valuation['valuation']['investment_category']] += 1
        
        # Top opportunities
        top_opportunities = sorted(valuations, key=lambda x: x['valuation']['final_value'], reverse=True)[:5]
        
        return {
            'domain_bridge_analysis': {
                bridge: {
                    'breakthrough_count': len(breakthroughs_list),
                    'total_value': sum(bt.synthetic_potential for bt in breakthroughs_list),
                    'avg_analogical_confidence': sum(bt.analogical_confidence for bt in breakthroughs_list) / len(breakthroughs_list)
                }
                for bridge, breakthroughs_list in domain_bridges.items()
            },
            'value_distribution': value_distribution,
            'analogical_pattern_analysis': {
                pattern: {
                    'count': len(potentials),
                    'avg_potential': sum(potentials) / len(potentials),
                    'max_potential': max(potentials)
                }
                for pattern, potentials in pattern_analysis.items()
            },
            'investment_categories': dict(investment_categories),
            'top_opportunities': [
                {
                    'title': opp['breakthrough']['primary_title'],
                    'domains': opp['breakthrough']['source_domains'],
                    'value': opp['valuation']['final_value'],
                    'category': opp['valuation']['investment_category'],
                    'value_multiplier': opp['valuation']['value_multiplier_total']
                }
                for opp in top_opportunities
            ]
        }

def main():
    """Test cross-domain analogical engine"""
    
    print(f"🚀 TESTING CROSS-DOMAIN ANALOGICAL DISCOVERY ENGINE")
    print("=" * 70)
    
    # Create test engine
    engine = CrossDomainAnalogicalEngine()
    
    # Load sample discoveries from multiple domains
    sample_discoveries = [
        {
            'paper_id': 'bio_1',
            'title': 'Engineering protein folding for enhanced stability',
            'domain': 'biomolecular_engineering'
        },
        {
            'paper_id': 'quantum_1',
            'title': 'Quantum coherence in crystalline systems',
            'domain': 'quantum_physics'
        },
        {
            'paper_id': 'ai_1',
            'title': 'Neural network optimization using feedback loops',
            'domain': 'artificial_intelligence'
        },
        {
            'paper_id': 'materials_1',
            'title': 'Self-organizing nanostructures with hierarchical organization',
            'domain': 'materials_science'
        }
    ]
    
    # Process cross-domain portfolio
    results = engine.process_cross_domain_portfolio(sample_discoveries)
    
    # Save results
    results_file = Path('cross_domain_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")
    print(f"\n✅ CROSS-DOMAIN ANALOGICAL ENGINE READY!")
    
    return results

if __name__ == "__main__":
    main()