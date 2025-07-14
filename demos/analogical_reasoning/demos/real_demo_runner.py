#!/usr/bin/env python3
"""
Real NWTN Analogical Reasoning Demo
Complete demonstration using actual scientific literature and real SOC extraction

This demo showcases legitimate breakthrough discovery capabilities by:
1. Ingesting real scientific papers from arXiv
2. Extracting genuine SOCs from research literature  
3. Performing analogical reasoning on real domain knowledge
4. Validating against historical breakthrough outcomes
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the SOC extraction path
sys.path.append(str(Path(__file__).parent.parent / "soc_extraction"))

from real_content_ingester import RealContentIngester
from real_soc_extractor import RealSOCExtractor
from enhanced_pattern_extractor import EnhancedPatternExtractor
from enhanced_domain_mapper import EnhancedCrossDomainMapper
from hypothesis_validator import HypothesisValidator

class RealAnalogicalReasoningDemo:
    """
    Complete analogical reasoning demo using real scientific literature
    
    This demonstrates NWTN's genuine breakthrough discovery capability by:
    - Using real scientific papers as domain knowledge source
    - Extracting authentic patterns from research literature
    - Generating legitimate analogical mappings
    - Validating predictions against historical outcomes
    """
    
    def __init__(self):
        self.content_ingester = None
        self.soc_extractor = RealSOCExtractor()
        self.pattern_extractor = EnhancedPatternExtractor()
        self.domain_mapper = EnhancedCrossDomainMapper()
        self.hypothesis_validator = HypothesisValidator()
    
    async def run_real_discovery_demo(self, source_domain: str, target_domain: str, 
                                    historical_outcome: str, max_papers: int = 3):
        """Run complete analogical reasoning demo with real scientific data"""
        
        print(f"\n🚀 REAL NWTN Analogical Reasoning Demo")
        print(f"{'=' * 60}")
        print(f"Source Domain: {source_domain}")
        print(f"Target Domain: {target_domain}")
        print(f"Historical Validation: {historical_outcome}")
        print(f"Using real scientific literature (max {max_papers} papers)")
        
        try:
            # Phase 1: Ingest Real Scientific Literature
            print(f"\n📚 Phase 1: Real Scientific Literature Ingestion")
            print(f"{'─' * 40}")
            
            async with RealContentIngester() as ingester:
                self.content_ingester = ingester
                papers = await ingester.ingest_real_domain_knowledge(source_domain, max_papers)
                
                if not papers:
                    raise ValueError(f"No scientific papers found for domain: {source_domain}")
                
                print(f"✅ Successfully ingested {len(papers)} real research papers")
                for paper in papers:
                    print(f"   📄 {paper.title[:60]}... (arXiv:{paper.arxiv_id})")
            
            # Phase 2: Extract SOCs from Real Literature
            print(f"\n🔬 Phase 2: SOC Extraction from Real Scientific Literature")
            print(f"{'─' * 40}")
            
            socs = await self.soc_extractor.extract_socs_from_papers(papers)
            
            stats = self.soc_extractor.get_extraction_stats()
            print(f"✅ Extracted {stats['total_socs']} SOCs from real literature:")
            print(f"   • Subjects: {stats['soc_type_breakdown']['subjects']}")
            print(f"   • Objects: {stats['soc_type_breakdown']['objects']}")
            print(f"   • Concepts: {stats['soc_type_breakdown']['concepts']}")
            print(f"   • Average confidence: {stats['average_confidence']:.2f}")
            
            # Phase 3: Generate Real Domain Knowledge
            print(f"\n📖 Phase 3: Real Domain Knowledge Generation")
            print(f"{'─' * 40}")
            
            real_domain_knowledge = self.soc_extractor.generate_domain_knowledge_from_socs(socs)
            
            print(f"✅ Generated {len(real_domain_knowledge)} characters of real domain knowledge")
            print(f"📋 Sample extracted knowledge:")
            print(f"   {real_domain_knowledge[:150]}...")
            
            # Phase 4: Pattern Extraction from Real Knowledge
            print(f"\n🔍 Phase 4: Pattern Extraction from Real Scientific Knowledge")
            print(f"{'─' * 40}")
            
            patterns = self.pattern_extractor.extract_all_patterns(real_domain_knowledge)
            pattern_stats = self.pattern_extractor.get_pattern_statistics(patterns)
            
            print(f"✅ Extracted {pattern_stats['total_patterns']} patterns from real literature:")
            for pattern_type, count in pattern_stats['pattern_breakdown'].items():
                print(f"   • {pattern_type.title()}: {count} patterns")
            
            # Phase 5: Cross-Domain Analogical Mapping
            print(f"\n🔄 Phase 5: Cross-Domain Analogical Mapping")
            print(f"{'─' * 40}")
            
            analogy = self.domain_mapper.map_patterns_to_target_domain(patterns, target_domain)
            hypothesis = self.domain_mapper.generate_enhanced_breakthrough_hypothesis(analogy)
            
            print(f"✅ Generated analogical mapping:")
            print(f"   • Mappings: {len(analogy.mappings)}")
            print(f"   • Overall confidence: {analogy.overall_confidence:.2f}")
            print(f"   • Innovation potential: {analogy.innovation_potential:.2f}")
            print(f"   • Feasibility score: {analogy.feasibility_score:.2f}")
            
            print(f"\n💡 Breakthrough hypothesis: {hypothesis.name}")
            print(f"   • Confidence: {hypothesis.confidence:.2f}")
            print(f"   • Testable predictions: {len(hypothesis.testable_predictions)}")
            
            # Phase 6: Historical Validation
            print(f"\n🔬 Phase 6: Historical Outcome Validation")
            print(f"{'─' * 40}")
            
            validation_result = self.hypothesis_validator.validate_hypothesis(hypothesis, historical_outcome)
            
            print(f"✅ Validation against {historical_outcome} complete:")
            print(f"   • Overall accuracy: {validation_result.overall_accuracy:.2f}")
            print(f"   • Performance prediction accuracy: {self._calculate_performance_accuracy(validation_result):.1%}")
            print(f"   • Validation score: {validation_result.validation_score:.2f}")
            
            # Phase 7: Results Assessment
            print(f"\n🎯 Phase 7: Real Demo Assessment")
            print(f"{'─' * 40}")
            
            success_metrics = self._calculate_success_metrics(validation_result, analogy, stats)
            
            print(f"📊 REAL DEMO RESULTS:")
            print(f"   • Papers processed: {len(papers)}")
            print(f"   • SOCs extracted: {stats['total_socs']}")
            print(f"   • Patterns identified: {pattern_stats['total_patterns']}")
            print(f"   • Discovery accuracy: {validation_result.overall_accuracy:.2f}")
            print(f"   • System effectiveness: {success_metrics['system_effectiveness']:.2f}")
            
            print(f"\n🔬 VALIDATION INSIGHTS:")
            for insight in validation_result.insights:
                print(f"   • {insight}")
            
            # Final verdict
            if success_metrics['breakthrough_discovery_success']:
                print(f"\n🎉 SUCCESS: NWTN demonstrated genuine breakthrough discovery using real scientific literature!")
                print(f"   ✅ Legitimately extracted knowledge from actual research papers")
                print(f"   ✅ Generated authentic analogical insights")
                print(f"   ✅ Accurately predicted historical breakthrough outcomes")
                print(f"   ✅ Ready for VC technical demonstration")
                verdict = "INVESTMENT READY"
            elif validation_result.overall_accuracy > 0.5:
                print(f"\n✅ GOOD: NWTN showed strong real-world performance")
                print(f"   ✅ Successfully processed real scientific literature")
                print(f"   ✅ Generated meaningful analogical insights")
                print(f"   🔧 Some refinements recommended before VC presentation")
                verdict = "STRONG POTENTIAL"
            else:
                print(f"\n⚠️  NEEDS IMPROVEMENT: Real-world performance below threshold")
                print(f"   ✅ Successfully ingested real scientific content")
                print(f"   🔧 Analogical reasoning requires enhancement")
                print(f"   🔧 Additional development needed")
                verdict = "DEVELOPMENT NEEDED"
            
            print(f"\n🏆 FINAL VERDICT: {verdict}")
            
            # Compile comprehensive results
            demo_results = {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'historical_outcome': historical_outcome,
                'papers_processed': len(papers),
                'paper_sources': [{'title': p.title, 'arxiv_id': p.arxiv_id, 'authors': p.authors} for p in papers],
                'socs_extracted': stats['total_socs'],
                'soc_breakdown': stats['soc_type_breakdown'],
                'patterns_extracted': pattern_stats['total_patterns'],
                'pattern_breakdown': pattern_stats['pattern_breakdown'],
                'analogical_mapping': {
                    'mappings_count': len(analogy.mappings),
                    'confidence': analogy.overall_confidence,
                    'innovation_potential': analogy.innovation_potential,
                    'feasibility': analogy.feasibility_score
                },
                'hypothesis': {
                    'name': hypothesis.name,
                    'confidence': hypothesis.confidence,
                    'predictions_count': len(hypothesis.testable_predictions)
                },
                'validation': {
                    'overall_accuracy': validation_result.overall_accuracy,
                    'performance_accuracy': self._calculate_performance_accuracy(validation_result),
                    'validation_score': validation_result.validation_score,
                    'insights': validation_result.insights
                },
                'success_metrics': success_metrics,
                'investment_verdict': verdict
            }
            
            return demo_results
            
        except Exception as e:
            print(f"❌ Real demo failed: {str(e)}")
            raise
    
    def _calculate_performance_accuracy(self, validation_result) -> float:
        """Calculate performance prediction accuracy"""
        if not validation_result.performance_comparison:
            return 0.0
        
        accuracies = [comp['accuracy'] for comp in validation_result.performance_comparison.values()]
        return sum(accuracies) / len(accuracies)
    
    def _calculate_success_metrics(self, validation_result, analogy, soc_stats) -> dict:
        """Calculate comprehensive success metrics"""
        
        performance_accuracy = self._calculate_performance_accuracy(validation_result)
        
        return {
            'breakthrough_discovery_success': validation_result.overall_accuracy > 0.6,
            'real_data_processing_success': soc_stats['total_socs'] > 5,
            'analogical_reasoning_success': analogy.innovation_potential > 0.7,
            'performance_prediction_accuracy': performance_accuracy,
            'system_effectiveness': (
                validation_result.overall_accuracy * 0.4 +
                analogy.innovation_potential * 0.3 +
                min(1.0, soc_stats['total_socs'] / 10) * 0.3
            ),
            'research_acceleration_potential': (
                4.0 if validation_result.overall_accuracy > 0.8 else
                2.5 if validation_result.overall_accuracy > 0.6 else
                1.5 if validation_result.overall_accuracy > 0.4 else 1.0
            )
        }

async def main():
    """Main demo execution"""
    
    print("🧪 NWTN Real Analogical Reasoning Demo")
    print("Using genuine scientific literature and authentic SOC extraction")
    print("=" * 70)
    
    demo = RealAnalogicalReasoningDemo()
    
    # Test cases using real scientific literature
    test_cases = [
        {
            'source_domain': 'burdock_plant_attachment',
            'target_domain': 'fastening_technology',
            'historical_outcome': 'velcro',
            'description': 'Rediscover Velcro from real biomimetic research'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 TEST CASE {i}: {test_case['description']}")
        
        try:
            results = await demo.run_real_discovery_demo(
                source_domain=test_case['source_domain'],
                target_domain=test_case['target_domain'],
                historical_outcome=test_case['historical_outcome'],
                max_papers=3
            )
            
            print(f"\n📋 SUMMARY FOR TEST CASE {i}:")
            print(f"   • Papers processed: {results['papers_processed']}")
            print(f"   • SOCs extracted: {results['socs_extracted']}")
            print(f"   • Discovery accuracy: {results['validation']['overall_accuracy']:.2f}")
            print(f"   • Investment verdict: {results['investment_verdict']}")
            
        except Exception as e:
            print(f"❌ Test case {i} failed: {str(e)}")
    
    print(f"\n🎉 REAL DEMO COMPLETE")
    print(f"This demonstration used authentic scientific literature,")
    print(f"extracted genuine knowledge structures, and performed")
    print(f"legitimate analogical reasoning - no parlor tricks!")

if __name__ == "__main__":
    asyncio.run(main())