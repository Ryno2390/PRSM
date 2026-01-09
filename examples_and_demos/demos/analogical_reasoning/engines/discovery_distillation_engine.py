#!/usr/bin/env python3
"""
Discovery Distillation and Risk-Adjusted Valuation Engine
Sophisticated system for deduplicating discoveries and applying realistic valuations

This addresses the critical flaws in naive "discovery count × uniform value" approach
by implementing clustering, domain-specific valuation, and risk adjustment.
"""

import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import re

@dataclass
class DistilledDiscovery:
    """A unique discovery after deduplication and clustering"""
    cluster_id: str
    primary_title: str
    concept_category: str
    mechanism_type: str
    application_domain: str
    supporting_papers: List[str]
    confidence_score: float
    uniqueness_score: float
    technical_maturity: str
    commercial_readiness: str
    time_horizon: str
    risk_level: str

@dataclass
class DomainValuation:
    """Domain-specific valuation parameters"""
    domain_name: str
    base_value_range: Tuple[float, float]  # (min, max) in millions
    success_probability: float
    time_to_market_years: float
    development_cost_multiplier: float
    market_size_factor: float
    competitive_intensity: float

@dataclass
class RiskAdjustedValue:
    """Risk-adjusted valuation for a discovery"""
    discovery_id: str
    base_value: float
    risk_adjustments: Dict[str, float]
    final_value: float
    confidence_level: str
    investment_category: str

class DiscoveryDistillationEngine:
    """Advanced engine for discovery deduplication and realistic valuation"""
    
    def __init__(self):
        self.setup_domain_valuations()
        self.setup_risk_frameworks()
        
    def setup_domain_valuations(self):
        """Setup realistic domain-specific valuation parameters"""
        
        self.domain_valuations = {
            'biomolecular_engineering': DomainValuation(
                domain_name='biomolecular_engineering',
                base_value_range=(50, 500),  # $50M-500M
                success_probability=0.15,    # 15% success rate
                time_to_market_years=8,      # 8 years development
                development_cost_multiplier=2.5,
                market_size_factor=1.2,
                competitive_intensity=0.7
            ),
            'materials_science': DomainValuation(
                domain_name='materials_science',
                base_value_range=(30, 300),  # $30M-300M
                success_probability=0.25,    # 25% success rate
                time_to_market_years=5,      # 5 years development
                development_cost_multiplier=1.8,
                market_size_factor=1.0,
                competitive_intensity=0.8
            ),
            'quantum_physics': DomainValuation(
                domain_name='quantum_physics',
                base_value_range=(100, 2000), # $100M-2B
                success_probability=0.08,     # 8% success rate
                time_to_market_years=12,     # 12 years development
                development_cost_multiplier=4.0,
                market_size_factor=2.0,
                competitive_intensity=0.5
            ),
            'nanotechnology': DomainValuation(
                domain_name='nanotechnology',
                base_value_range=(40, 400),  # $40M-400M
                success_probability=0.20,    # 20% success rate
                time_to_market_years=6,      # 6 years development
                development_cost_multiplier=2.0,
                market_size_factor=1.5,
                competitive_intensity=0.6
            ),
            'artificial_intelligence': DomainValuation(
                domain_name='artificial_intelligence',
                base_value_range=(10, 200),  # $10M-200M
                success_probability=0.35,    # 35% success rate
                time_to_market_years=2,      # 2 years development
                development_cost_multiplier=1.2,
                market_size_factor=1.8,
                competitive_intensity=0.9
            ),
            'energy_systems': DomainValuation(
                domain_name='energy_systems',
                base_value_range=(100, 1000), # $100M-1B
                success_probability=0.12,     # 12% success rate
                time_to_market_years=10,     # 10 years development
                development_cost_multiplier=3.0,
                market_size_factor=2.5,
                competitive_intensity=0.7
            ),
            'biotechnology': DomainValuation(
                domain_name='biotechnology',
                base_value_range=(75, 750),  # $75M-750M
                success_probability=0.18,    # 18% success rate
                time_to_market_years=9,      # 9 years development
                development_cost_multiplier=2.8,
                market_size_factor=1.4,
                competitive_intensity=0.6
            ),
            'photonics': DomainValuation(
                domain_name='photonics',
                base_value_range=(25, 250),  # $25M-250M
                success_probability=0.22,    # 22% success rate
                time_to_market_years=4,      # 4 years development
                development_cost_multiplier=1.5,
                market_size_factor=1.1,
                competitive_intensity=0.8
            )
        }
        
        # Default valuation for unknown domains
        self.default_valuation = DomainValuation(
            domain_name='generic',
            base_value_range=(20, 200),
            success_probability=0.20,
            time_to_market_years=5,
            development_cost_multiplier=2.0,
            market_size_factor=1.0,
            competitive_intensity=0.75
        )
    
    def setup_risk_frameworks(self):
        """Setup risk adjustment frameworks"""
        
        # Technical maturity risk adjustments
        self.technical_maturity_factors = {
            'lab_prototype': 0.3,      # 70% discount
            'proof_of_concept': 0.5,   # 50% discount
            'pilot_scale': 0.7,        # 30% discount
            'demonstration': 0.85,     # 15% discount
            'commercial_ready': 0.95   # 5% discount
        }
        
        # Commercial readiness risk adjustments
        self.commercial_readiness_factors = {
            'research_stage': 0.2,     # 80% discount
            'early_development': 0.4,  # 60% discount
            'market_validation': 0.6,  # 40% discount
            'scaling_ready': 0.8,      # 20% discount
            'market_ready': 0.9        # 10% discount
        }
        
        # Time horizon risk adjustments
        self.time_horizon_factors = {
            'immediate': 0.95,         # <1 year
            'near_term': 0.85,         # 1-3 years
            'medium_term': 0.65,       # 3-7 years
            'long_term': 0.45,         # 7-15 years
            'far_future': 0.25         # >15 years
        }
        
        # Risk level multipliers
        self.risk_level_factors = {
            'low': 0.9,
            'moderate': 0.7,
            'high': 0.5,
            'extreme': 0.3
        }
    
    def extract_discovery_features(self, discovery: Dict) -> Dict[str, str]:
        """Extract key features from discovery for clustering"""
        
        title = discovery.get('title', '').lower()
        domain = discovery.get('domain', 'unknown')
        
        # Extract mechanism type from title/description
        mechanism_patterns = {
            'protein_engineering': ['protein', 'enzyme', 'peptide', 'amino'],
            'molecular_assembly': ['assembly', 'self-assembly', 'organization'],
            'catalysis': ['catalyst', 'catalytic', 'reaction', 'chemical'],
            'materials_synthesis': ['synthesis', 'fabrication', 'manufacturing'],
            'energy_conversion': ['energy', 'conversion', 'efficiency', 'power'],
            'sensing_detection': ['sensor', 'detection', 'sensing', 'monitoring'],
            'computational_modeling': ['modeling', 'simulation', 'computational'],
            'surface_modification': ['surface', 'interface', 'coating'],
            'transport_delivery': ['transport', 'delivery', 'targeting'],
            'imaging_characterization': ['imaging', 'microscopy', 'characterization']
        }
        
        mechanism_type = 'other'
        for mech, keywords in mechanism_patterns.items():
            if any(keyword in title for keyword in keywords):
                mechanism_type = mech
                break
        
        # Extract application domain from title/description
        application_patterns = {
            'medical_therapeutic': ['medical', 'therapeutic', 'drug', 'treatment', 'therapy'],
            'manufacturing_industrial': ['manufacturing', 'industrial', 'production', 'fabrication'],
            'electronics_computing': ['electronic', 'computing', 'semiconductor', 'device'],
            'energy_storage': ['battery', 'storage', 'energy', 'fuel', 'solar'],
            'environmental': ['environmental', 'pollution', 'waste', 'clean'],
            'aerospace_defense': ['aerospace', 'defense', 'space', 'military'],
            'automotive': ['automotive', 'vehicle', 'transportation'],
            'consumer_products': ['consumer', 'commercial', 'product']
        }
        
        application_domain = 'research_tools'
        for app, keywords in application_patterns.items():
            if any(keyword in title for keyword in keywords):
                application_domain = app
                break
        
        # Extract concept category
        concept_patterns = {
            'precision_manufacturing': ['precision', 'atomic', 'nanoscale', 'accurate'],
            'efficiency_optimization': ['efficiency', 'optimization', 'improved', 'enhanced'],
            'novel_materials': ['novel', 'new', 'advanced', 'material'],
            'biological_systems': ['biological', 'bio', 'living', 'cell'],
            'smart_systems': ['smart', 'adaptive', 'responsive', 'intelligent'],
            'high_performance': ['high-performance', 'ultra', 'extreme', 'superior']
        }
        
        concept_category = 'general_advancement'
        for concept, keywords in concept_patterns.items():
            if any(keyword in title for keyword in keywords):
                concept_category = concept
                break
        
        return {
            'mechanism_type': mechanism_type,
            'application_domain': application_domain,
            'concept_category': concept_category,
            'domain': domain
        }
    
    def calculate_similarity_score(self, discovery1: Dict, discovery2: Dict) -> float:
        """Calculate similarity score between two discoveries"""
        
        features1 = self.extract_discovery_features(discovery1)
        features2 = self.extract_discovery_features(discovery2)
        
        # Weight different features
        weights = {
            'mechanism_type': 0.4,
            'application_domain': 0.3,
            'concept_category': 0.2,
            'domain': 0.1
        }
        
        similarity = 0.0
        for feature, weight in weights.items():
            if features1[feature] == features2[feature]:
                similarity += weight
        
        # Text similarity for titles
        title1_words = set(discovery1.get('title', '').lower().split())
        title2_words = set(discovery2.get('title', '').lower().split())
        
        if title1_words and title2_words:
            title_similarity = len(title1_words.intersection(title2_words)) / len(title1_words.union(title2_words))
            similarity += 0.2 * title_similarity
        
        return min(similarity, 1.0)
    
    def cluster_discoveries(self, discoveries: List[Dict], similarity_threshold: float = 0.7) -> List[List[Dict]]:
        """Cluster similar discoveries together"""
        
        print(f"🔍 CLUSTERING {len(discoveries)} DISCOVERIES")
        print("=" * 50)
        
        clusters = []
        processed = set()
        
        for i, discovery in enumerate(discoveries):
            if i in processed:
                continue
            
            # Start new cluster with this discovery
            cluster = [discovery]
            processed.add(i)
            
            # Find similar discoveries
            for j, other_discovery in enumerate(discoveries[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_similarity_score(discovery, other_discovery)
                
                if similarity >= similarity_threshold:
                    cluster.append(other_discovery)
                    processed.add(j)
            
            clusters.append(cluster)
        
        print(f"   📊 Original discoveries: {len(discoveries)}")
        print(f"   🎯 Unique clusters: {len(clusters)}")
        print(f"   📉 Deduplication ratio: {len(clusters)/len(discoveries):.1%}")
        
        return clusters
    
    def create_distilled_discovery(self, cluster: List[Dict]) -> DistilledDiscovery:
        """Create a distilled discovery from a cluster"""
        
        # Use the discovery with highest confidence as primary
        primary = max(cluster, key=lambda d: d.get('assessment', {}).get('success_probability', 0))
        
        # Extract features from primary discovery
        features = self.extract_discovery_features(primary)
        
        # Calculate cluster-level metrics
        confidence_scores = [d.get('assessment', {}).get('success_probability', 0.5) for d in cluster]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Uniqueness score based on cluster size (smaller = more unique)
        uniqueness_score = min(1.0, 2.0 / len(cluster))
        
        # Assess technical maturity and commercial readiness
        technical_maturity = self._assess_technical_maturity(primary)
        commercial_readiness = self._assess_commercial_readiness(primary)
        time_horizon = self._assess_time_horizon(primary, features)
        risk_level = self._assess_risk_level(primary, features)
        
        # Generate cluster ID
        cluster_id = hashlib.md5(
            f"{features['mechanism_type']}_{features['application_domain']}_{features['concept_category']}".encode()
        ).hexdigest()[:8]
        
        return DistilledDiscovery(
            cluster_id=cluster_id,
            primary_title=primary.get('title', 'Unknown'),
            concept_category=features['concept_category'],
            mechanism_type=features['mechanism_type'],
            application_domain=features['application_domain'],
            supporting_papers=[d.get('paper_id', 'unknown') for d in cluster],
            confidence_score=avg_confidence,
            uniqueness_score=uniqueness_score,
            technical_maturity=technical_maturity,
            commercial_readiness=commercial_readiness,
            time_horizon=time_horizon,
            risk_level=risk_level
        )
    
    def _assess_technical_maturity(self, discovery: Dict) -> str:
        """Assess technical maturity level"""
        
        title = discovery.get('title', '').lower()
        
        if any(word in title for word in ['commercial', 'production', 'scale']):
            return 'commercial_ready'
        elif any(word in title for word in ['demonstration', 'prototype', 'pilot']):
            return 'demonstration'
        elif any(word in title for word in ['proof', 'concept', 'feasibility']):
            return 'proof_of_concept'
        elif any(word in title for word in ['lab', 'laboratory', 'experimental']):
            return 'lab_prototype'
        else:
            return 'pilot_scale'  # Default middle ground
    
    def _assess_commercial_readiness(self, discovery: Dict) -> str:
        """Assess commercial readiness level"""
        
        title = discovery.get('title', '').lower()
        domain = discovery.get('domain', '')
        
        # AI/software tends to be more market-ready
        if domain == 'artificial_intelligence':
            return 'scaling_ready'
        
        if any(word in title for word in ['market', 'commercial', 'application']):
            return 'market_validation'
        elif any(word in title for word in ['development', 'engineering', 'design']):
            return 'early_development'
        elif any(word in title for word in ['research', 'study', 'investigation']):
            return 'research_stage'
        else:
            return 'early_development'  # Default
    
    def _assess_time_horizon(self, discovery: Dict, features: Dict) -> str:
        """Assess time horizon for commercialization"""
        
        domain = discovery.get('domain', '')
        mechanism = features['mechanism_type']
        application = features['application_domain']
        
        # Domain-based time horizons
        if domain == 'artificial_intelligence':
            return 'near_term'
        elif domain in ['quantum_physics', 'energy_systems']:
            return 'long_term'
        elif domain in ['biomolecular_engineering', 'biotechnology']:
            return 'long_term'
        elif application == 'medical_therapeutic':
            return 'long_term'
        elif application in ['electronics_computing', 'consumer_products']:
            return 'near_term'
        else:
            return 'medium_term'
    
    def _assess_risk_level(self, discovery: Dict, features: Dict) -> str:
        """Assess overall risk level"""
        
        domain = discovery.get('domain', '')
        mechanism = features['mechanism_type']
        application = features['application_domain']
        
        # High-risk domains/applications
        if domain in ['quantum_physics', 'energy_systems']:
            return 'high'
        elif application == 'medical_therapeutic':
            return 'high'
        elif domain == 'artificial_intelligence':
            return 'moderate'
        elif application in ['manufacturing_industrial', 'electronics_computing']:
            return 'moderate'
        else:
            return 'moderate'
    
    def calculate_risk_adjusted_value(self, distilled_discovery: DistilledDiscovery) -> RiskAdjustedValue:
        """Calculate risk-adjusted valuation for a distilled discovery"""
        
        # Get domain valuation parameters
        domain_val = self.domain_valuations.get(
            distilled_discovery.application_domain, 
            self.default_valuation
        )
        
        # Base value calculation
        base_value = (domain_val.base_value_range[0] + domain_val.base_value_range[1]) / 2
        
        # Apply various risk adjustments
        risk_adjustments = {}
        
        # Technical maturity adjustment
        tech_factor = self.technical_maturity_factors.get(distilled_discovery.technical_maturity, 0.5)
        risk_adjustments['technical_maturity'] = tech_factor
        
        # Commercial readiness adjustment
        commercial_factor = self.commercial_readiness_factors.get(distilled_discovery.commercial_readiness, 0.4)
        risk_adjustments['commercial_readiness'] = commercial_factor
        
        # Time horizon adjustment
        time_factor = self.time_horizon_factors.get(distilled_discovery.time_horizon, 0.6)
        risk_adjustments['time_horizon'] = time_factor
        
        # Risk level adjustment
        risk_factor = self.risk_level_factors.get(distilled_discovery.risk_level, 0.6)
        risk_adjustments['risk_level'] = risk_factor
        
        # Uniqueness bonus
        uniqueness_factor = 0.8 + (0.4 * distilled_discovery.uniqueness_score)
        risk_adjustments['uniqueness'] = uniqueness_factor
        
        # Confidence adjustment
        confidence_factor = 0.5 + (0.5 * distilled_discovery.confidence_score)
        risk_adjustments['confidence'] = confidence_factor
        
        # Market size factor from domain
        market_factor = domain_val.market_size_factor
        risk_adjustments['market_size'] = market_factor
        
        # Success probability from domain
        success_factor = domain_val.success_probability
        risk_adjustments['success_probability'] = success_factor
        
        # Calculate final value
        final_value = base_value
        for factor_name, factor_value in risk_adjustments.items():
            final_value *= factor_value
        
        # Determine confidence level
        if distilled_discovery.confidence_score > 0.8 and distilled_discovery.uniqueness_score > 0.7:
            confidence_level = "HIGH"
        elif distilled_discovery.confidence_score > 0.6 and distilled_discovery.uniqueness_score > 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Determine investment category
        if final_value > 100:  # >$100M
            investment_category = "BREAKTHROUGH"
        elif final_value > 20:  # >$20M
            investment_category = "SIGNIFICANT"
        elif final_value > 5:   # >$5M
            investment_category = "VALUABLE"
        else:
            investment_category = "INCREMENTAL"
        
        return RiskAdjustedValue(
            discovery_id=distilled_discovery.cluster_id,
            base_value=base_value,
            risk_adjustments=risk_adjustments,
            final_value=final_value,
            confidence_level=confidence_level,
            investment_category=investment_category
        )
    
    def process_discovery_portfolio(self, discoveries: List[Dict]) -> Dict:
        """Process complete discovery portfolio with distillation and valuation"""
        
        print(f"🚀 DISCOVERY DISTILLATION & RISK-ADJUSTED VALUATION")
        print("=" * 70)
        print(f"📊 Processing {len(discoveries)} raw discoveries")
        
        # Step 1: Cluster similar discoveries
        clusters = self.cluster_discoveries(discoveries)
        
        # Step 2: Create distilled discoveries
        print(f"\n🧪 CREATING DISTILLED DISCOVERIES")
        distilled_discoveries = []
        for i, cluster in enumerate(clusters):
            distilled = self.create_distilled_discovery(cluster)
            distilled_discoveries.append(distilled)
            
            if i < 5:  # Show first few examples
                print(f"   🎯 Cluster {i+1}: {distilled.concept_category} ({len(cluster)} papers)")
        
        # Step 3: Calculate risk-adjusted valuations
        print(f"\n💰 CALCULATING RISK-ADJUSTED VALUATIONS")
        valuations = []
        total_value = 0
        
        for distilled in distilled_discoveries:
            valuation = self.calculate_risk_adjusted_value(distilled)
            valuations.append(valuation)
            total_value += valuation.final_value
        
        # Step 4: Generate portfolio analysis
        portfolio_analysis = self._generate_portfolio_analysis(distilled_discoveries, valuations)
        
        results = {
            'processing_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'raw_discoveries': len(discoveries),
                'distilled_discoveries': len(distilled_discoveries),
                'deduplication_ratio': len(distilled_discoveries) / len(discoveries),
                'total_portfolio_value': total_value
            },
            'distilled_discoveries': [asdict(d) for d in distilled_discoveries],
            'risk_adjusted_valuations': [asdict(v) for v in valuations],
            'portfolio_analysis': portfolio_analysis
        }
        
        print(f"\n💎 DISTILLATION COMPLETE!")
        print(f"   📊 Raw discoveries: {len(discoveries)}")
        print(f"   🎯 Unique discoveries: {len(distilled_discoveries)}")
        print(f"   💰 Total portfolio value: ${total_value:.1f}M")
        print(f"   📈 Average value per discovery: ${total_value/len(distilled_discoveries):.1f}M")
        
        return results
    
    def _generate_portfolio_analysis(self, distilled_discoveries: List[DistilledDiscovery], 
                                   valuations: List[RiskAdjustedValue]) -> Dict:
        """Generate comprehensive portfolio analysis"""
        
        # Category breakdowns
        by_category = defaultdict(list)
        by_investment_tier = defaultdict(list)
        by_domain = defaultdict(list)
        by_risk = defaultdict(list)
        
        for discovery, valuation in zip(distilled_discoveries, valuations):
            by_category[discovery.concept_category].append(valuation.final_value)
            by_investment_tier[valuation.investment_category].append(valuation.final_value)
            by_domain[discovery.application_domain].append(valuation.final_value)
            by_risk[discovery.risk_level].append(valuation.final_value)
        
        # Calculate statistics for each breakdown
        def calc_stats(value_dict):
            return {
                category: {
                    'count': len(values),
                    'total_value': sum(values),
                    'avg_value': sum(values) / len(values),
                    'max_value': max(values),
                    'min_value': min(values)
                }
                for category, values in value_dict.items()
            }
        
        # Top discoveries
        top_discoveries = sorted(
            zip(distilled_discoveries, valuations),
            key=lambda x: x[1].final_value,
            reverse=True
        )[:10]
        
        return {
            'by_concept_category': calc_stats(by_category),
            'by_investment_tier': calc_stats(by_investment_tier),
            'by_application_domain': calc_stats(by_domain),
            'by_risk_level': calc_stats(by_risk),
            'top_10_discoveries': [
                {
                    'title': disc.primary_title,
                    'category': disc.concept_category,
                    'value': val.final_value,
                    'confidence': val.confidence_level,
                    'investment_tier': val.investment_category
                }
                for disc, val in top_discoveries
            ],
            'portfolio_metrics': {
                'total_discoveries': len(distilled_discoveries),
                'breakthrough_count': len([v for v in valuations if v.investment_category == 'BREAKTHROUGH']),
                'significant_count': len([v for v in valuations if v.investment_category == 'SIGNIFICANT']),
                'valuable_count': len([v for v in valuations if v.investment_category == 'VALUABLE']),
                'incremental_count': len([v for v in valuations if v.investment_category == 'INCREMENTAL']),
                'high_confidence_count': len([v for v in valuations if v.confidence_level == 'HIGH']),
                'total_value': sum(v.final_value for v in valuations),
                'avg_value_per_discovery': sum(v.final_value for v in valuations) / len(valuations)
            }
        }

def main():
    """Test the discovery distillation engine"""
    
    print(f"🚀 TESTING DISCOVERY DISTILLATION ENGINE")
    print("=" * 70)
    
    # Create test engine
    engine = DiscoveryDistillationEngine()
    
    # Load sample discoveries (would normally come from pipeline)
    sample_discoveries = [
        {
            'paper_id': 'test_1',
            'title': 'Engineering hemoglobin for enhanced efficiency in drug delivery',
            'domain': 'biomolecular_engineering',
            'assessment': {'success_probability': 0.7}
        },
        {
            'paper_id': 'test_2', 
            'title': 'Protein motor systems for precision manufacturing',
            'domain': 'biomolecular_engineering',
            'assessment': {'success_probability': 0.6}
        },
        {
            'paper_id': 'test_3',
            'title': 'Advanced materials with high thermal conductivity',
            'domain': 'materials_science',
            'assessment': {'success_probability': 0.8}
        }
    ]
    
    # Process portfolio
    results = engine.process_discovery_portfolio(sample_discoveries)
    
    # Save results
    results_file = Path('discovery_distillation_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")
    print(f"\n✅ DISCOVERY DISTILLATION ENGINE READY!")
    
    return results

if __name__ == "__main__":
    main()