#!/usr/bin/env python3
"""
NWTN Analogical Reasoning Demo Runner
Complete demonstration of NWTN's breakthrough discovery capabilities

This demo showcases NWTN's ability to systematically discover breakthrough innovations
through analogical reasoning, using the historical discovery of Velcro as validation.
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from pattern_extractor import PatternExtractor
from domain_mapper import CrossDomainMapper 
from hypothesis_validator import HypothesisValidator

class AnalogicalReasoningDemo:
    """Complete analogical reasoning demonstration"""
    
    def __init__(self):
        self.pattern_extractor = PatternExtractor()
        self.domain_mapper = CrossDomainMapper()
        self.hypothesis_validator = HypothesisValidator()
        
        # Demo test cases
        self.test_cases = self._load_test_cases()
    
    def run_demo(self, test_case: str = "velcro_discovery", 
                 show_detailed_output: bool = True) -> Dict[str, Any]:
        """Run complete analogical reasoning demo"""
        
        print(f"\n🚀 NWTN Analogical Reasoning Demo")
        print(f"{'=' * 60}")
        print(f"Test Case: {test_case}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if test_case not in self.test_cases:
            raise ValueError(f"Unknown test case: {test_case}")
        
        case_data = self.test_cases[test_case]
        start_time = time.time()
        
        try:
            # Phase 1: Pattern Extraction
            print(f"\n📋 Phase 1: Pattern Extraction from Source Domain")
            print(f"Source Domain: {case_data['source_domain']}")
            
            if show_detailed_output:
                print(f"Source Knowledge: {case_data['source_knowledge'][:100]}...")
            
            patterns = self.pattern_extractor.extract_all_patterns(
                case_data['source_knowledge']
            )
            
            pattern_stats = self.pattern_extractor.get_pattern_statistics(patterns)
            print(f"✅ Extracted {pattern_stats['total_patterns']} patterns")
            
            # Phase 2: Cross-Domain Mapping
            print(f"\n🔄 Phase 2: Cross-Domain Analogical Mapping")
            print(f"Target Domain: {case_data['target_domain']}")
            
            analogy = self.domain_mapper.map_patterns_to_target_domain(
                patterns, case_data['target_domain']
            )
            
            hypothesis = self.domain_mapper.generate_breakthrough_hypothesis(analogy)
            
            # Phase 3: Hypothesis Validation
            print(f"\n🔬 Phase 3: Hypothesis Validation")
            print(f"Validation Target: {case_data['historical_outcome']}")
            
            validation_result = self.hypothesis_validator.validate_hypothesis(
                hypothesis, case_data['historical_outcome']
            )
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            
            # Compile results
            demo_results = {
                'test_case': test_case,
                'execution_time': execution_time,
                'pattern_extraction': {
                    'patterns_found': pattern_stats['total_patterns'],
                    'pattern_breakdown': pattern_stats['pattern_breakdown'],
                    'confidence_distribution': pattern_stats['confidence_distribution']
                },
                'analogical_mapping': {
                    'mappings_generated': len(analogy.mappings),
                    'overall_confidence': analogy.overall_confidence,
                    'innovation_potential': analogy.innovation_potential,
                    'feasibility_score': analogy.feasibility_score
                },
                'breakthrough_hypothesis': {
                    'innovation_name': hypothesis.name,
                    'description': hypothesis.description,
                    'confidence': hypothesis.confidence,
                    'predicted_properties': hypothesis.predicted_properties,
                    'testable_predictions': len(hypothesis.testable_predictions)
                },
                'validation': {
                    'overall_accuracy': validation_result.overall_accuracy,
                    'validation_score': validation_result.validation_score,
                    'performance_comparison': validation_result.performance_comparison,
                    'insights': validation_result.insights
                },
                'success_metrics': self._calculate_success_metrics(validation_result, analogy)
            }
            
            # Display results
            if show_detailed_output:
                self._display_detailed_results(demo_results, validation_result)
            
            self._display_summary(demo_results)
            
            return demo_results
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
    
    def run_all_test_cases(self) -> Dict[str, Any]:
        """Run all available test cases"""
        
        print(f"\n🧪 Running All NWTN Analogical Reasoning Test Cases")
        print(f"{'=' * 70}")
        
        all_results = {}
        summary_stats = {
            'total_cases': len(self.test_cases),
            'successful_cases': 0,
            'average_accuracy': 0.0,
            'average_execution_time': 0.0
        }
        
        for test_case in self.test_cases.keys():
            print(f"\n{'─' * 40}")
            print(f"Running Test Case: {test_case}")
            
            try:
                results = self.run_demo(test_case, show_detailed_output=False)
                all_results[test_case] = results
                
                if results['validation']['overall_accuracy'] > 0.5:
                    summary_stats['successful_cases'] += 1
                
                summary_stats['average_accuracy'] += results['validation']['overall_accuracy']
                summary_stats['average_execution_time'] += results['execution_time']
                
            except Exception as e:
                print(f"❌ Test case {test_case} failed: {str(e)}")
                all_results[test_case] = {'error': str(e)}
        
        # Calculate averages
        if summary_stats['total_cases'] > 0:
            summary_stats['average_accuracy'] /= summary_stats['total_cases']
            summary_stats['average_execution_time'] /= summary_stats['total_cases']
        
        # Display comprehensive summary
        self._display_comprehensive_summary(all_results, summary_stats)
        
        return {
            'test_results': all_results,
            'summary_statistics': summary_stats
        }
    
    def _calculate_success_metrics(self, validation_result, analogy) -> Dict[str, Any]:
        """Calculate success metrics for investment validation"""
        
        # Key metrics for VC evaluation
        metrics = {
            'breakthrough_discovery_success': validation_result.overall_accuracy > 0.6,
            'performance_prediction_accuracy': validation_result.overall_accuracy,
            'innovation_potential_score': analogy.innovation_potential,
            'technical_feasibility_score': analogy.feasibility_score,
            'research_acceleration_factor': self._estimate_acceleration_factor(validation_result),
            'commercial_viability_indicator': validation_result.overall_accuracy > 0.7
        }
        
        # Overall NWTN system effectiveness
        metrics['system_effectiveness'] = (
            metrics['performance_prediction_accuracy'] * 0.4 +
            metrics['innovation_potential_score'] * 0.3 +
            metrics['technical_feasibility_score'] * 0.3
        )
        
        return metrics
    
    def _estimate_acceleration_factor(self, validation_result) -> float:
        """Estimate R&D acceleration factor based on accuracy"""
        
        # Higher accuracy suggests better guidance, thus more acceleration
        accuracy = validation_result.overall_accuracy
        
        if accuracy > 0.8:
            return 4.0  # 4x acceleration (20 years → 5 years)
        elif accuracy > 0.6:
            return 2.5  # 2.5x acceleration
        elif accuracy > 0.4:
            return 1.5  # 1.5x acceleration
        else:
            return 1.0  # No acceleration
    
    def _display_detailed_results(self, results: Dict[str, Any], 
                                validation_result) -> None:
        """Display detailed demo results"""
        
        print(f"\n📊 DETAILED RESULTS")
        print(f"{'─' * 40}")
        
        # Pattern extraction details
        print(f"\n🔍 Pattern Extraction:")
        for pattern_type, count in results['pattern_extraction']['pattern_breakdown'].items():
            print(f"  • {pattern_type.title()}: {count} patterns")
        
        # Analogical mapping details
        print(f"\n🔄 Analogical Mapping:")
        print(f"  • Mappings generated: {results['analogical_mapping']['mappings_generated']}")
        print(f"  • Overall confidence: {results['analogical_mapping']['overall_confidence']:.2f}")
        print(f"  • Innovation potential: {results['analogical_mapping']['innovation_potential']:.2f}")
        
        # Breakthrough hypothesis details
        print(f"\n💡 Breakthrough Hypothesis:")
        print(f"  • Innovation: {results['breakthrough_hypothesis']['innovation_name']}")
        print(f"  • Confidence: {results['breakthrough_hypothesis']['confidence']:.2f}")
        print(f"  • Testable predictions: {results['breakthrough_hypothesis']['testable_predictions']}")
        
        # Performance comparison
        print(f"\n📈 Performance Prediction Accuracy:")
        for prop, comparison in results['validation']['performance_comparison'].items():
            print(f"  • {prop}: {comparison['predicted']:.1f} vs {comparison['actual']:.1f} " +
                  f"(accuracy: {comparison['accuracy']:.2f})")
        
        # Key insights
        print(f"\n🔬 Validation Insights:")
        for insight in results['validation']['insights']:
            print(f"  • {insight}")
    
    def _display_summary(self, results: Dict[str, Any]) -> None:
        """Display demo summary"""
        
        print(f"\n🎯 DEMO SUMMARY")
        print(f"{'─' * 30}")
        
        success_metrics = results['success_metrics']
        
        print(f"Overall Accuracy: {results['validation']['overall_accuracy']:.2f}")
        print(f"System Effectiveness: {success_metrics['system_effectiveness']:.2f}")
        print(f"Estimated R&D Acceleration: {success_metrics['research_acceleration_factor']:.1f}x")
        print(f"Execution Time: {results['execution_time']:.1f} seconds")
        
        # Success assessment
        if success_metrics['breakthrough_discovery_success']:
            print(f"\n🎉 SUCCESS: NWTN demonstrated breakthrough discovery capability!")
            print(f"   Ready for VC demonstration and technical validation.")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Results show potential but require refinement.")
    
    def _display_comprehensive_summary(self, all_results: Dict[str, Any],
                                     summary_stats: Dict[str, Any]) -> None:
        """Display comprehensive summary of all test cases"""
        
        print(f"\n🏆 COMPREHENSIVE TEST RESULTS")
        print(f"{'=' * 50}")
        
        print(f"Total Test Cases: {summary_stats['total_cases']}")
        print(f"Successful Cases: {summary_stats['successful_cases']}")
        print(f"Success Rate: {summary_stats['successful_cases'] / summary_stats['total_cases']:.1%}")
        print(f"Average Accuracy: {summary_stats['average_accuracy']:.2f}")
        print(f"Average Execution Time: {summary_stats['average_execution_time']:.1f}s")
        
        print(f"\n📋 Individual Case Results:")
        for case_name, results in all_results.items():
            if 'error' in results:
                print(f"  ❌ {case_name}: FAILED")
            else:
                accuracy = results['validation']['overall_accuracy']
                status = "🎉" if accuracy > 0.7 else "✅" if accuracy > 0.5 else "⚠️"
                print(f"  {status} {case_name}: {accuracy:.2f} accuracy")
        
        # Investment readiness assessment
        if summary_stats['average_accuracy'] > 0.6 and summary_stats['successful_cases'] >= 2:
            print(f"\n🚀 INVESTMENT READY: NWTN demonstrates consistent analogical reasoning capability")
            print(f"   Recommended for VC presentation and technical due diligence.")
        else:
            print(f"\n🔧 DEVELOPMENT NEEDED: Additional refinement recommended before VC presentation.")
    
    def _load_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """Load test cases for analogical reasoning demos"""
        
        return {
            "velcro_discovery": {
                "source_domain": "burdock_plant_burr_attachment",
                "target_domain": "fastening_technology",
                "historical_outcome": "velcro",
                "source_knowledge": """
                Burdock plant seeds are covered with numerous small hooks that have curved tips.
                These microscopic hooks attach strongly to fabric fibers and animal fur.
                The hooks are made of a tough, flexible material that allows them to grip
                onto loop-like structures in fabric. The curved shape of each hook provides
                mechanical advantage, making attachment strong but reversible.
                When pulled with sufficient force, the hooks detach cleanly due to their
                flexibility. The high density of hooks distributes load across many
                attachment points, making the overall grip very strong.
                """
            },
            
            "biomimetic_flight": {
                "source_domain": "bird_wing_aerodynamics", 
                "target_domain": "mechanical_flight",
                "historical_outcome": "biomimetic_flight",
                "source_knowledge": """
                Birds achieve flight through curved wing surfaces that create lift.
                The cambered airfoil shape causes air to move faster over the top surface,
                creating lower pressure above the wing. Birds control flight through
                three axes of movement: pitch, roll, and yaw. Wing flexibility allows
                for fine control adjustments. The aspect ratio of wings affects
                efficiency, with longer wings providing better lift-to-drag ratios.
                """
            }
        }

def main():
    """Main demo execution"""
    
    parser = argparse.ArgumentParser(description='NWTN Analogical Reasoning Demo')
    parser.add_argument('--test-case', default='velcro_discovery',
                       help='Test case to run (default: velcro_discovery)')
    parser.add_argument('--all-cases', action='store_true',
                       help='Run all test cases')
    parser.add_argument('--save-results', 
                       help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output mode')
    
    args = parser.parse_args()
    
    demo = AnalogicalReasoningDemo()
    
    try:
        if args.all_cases:
            results = demo.run_all_test_cases()
        else:
            results = demo.run_demo(args.test_case, show_detailed_output=not args.quiet)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.save_results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo execution failed: {str(e)}")
        return None

if __name__ == "__main__":
    main()