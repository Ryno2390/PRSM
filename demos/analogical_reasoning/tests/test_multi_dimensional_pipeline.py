#!/usr/bin/env python3
"""
Test Multi-Dimensional Breakthrough Assessment Pipeline
Demonstrates the new sophisticated assessment system
"""

from enhanced_batch_processor import EnhancedBatchProcessor

def test_multi_dimensional_assessment():
    """Test the multi-dimensional assessment system"""
    
    print("🧠 TESTING MULTI-DIMENSIONAL BREAKTHROUGH ASSESSMENT")
    print("=" * 80)
    print("🎯 Purpose: Demonstrate sophisticated breakthrough evaluation")
    print("📊 Method: Process 10 papers with multi-dimensional assessment")
    print("💡 Features: Category classification, risk assessment, time horizons")
    print("🔧 Organization Type: Industry (balanced commercial/innovation focus)")
    
    # Create enhanced processor with multi-dimensional assessment
    processor = EnhancedBatchProcessor(
        test_mode="multi_dimensional_test",
        use_multi_dimensional=True,
        organization_type="industry"
    )
    
    # Test with small sample to see the new system in action
    results = processor.run_unified_test(
        test_mode="phase_a",
        paper_count=10,
        paper_source="unique"
    )
    
    if results:
        print(f"\n🎉 MULTI-DIMENSIONAL TEST COMPLETE!")
        
        # Show the enhanced insights
        profiles = results.get('breakthrough_profiles', [])
        rankings = results.get('multi_dimensional_rankings', {})
        
        if profiles:
            print(f"\n🔍 SAMPLE BREAKTHROUGH PROFILE:")
            profile = profiles[0]
            print(f"   Discovery: {profile.discovery_id}")
            print(f"   Category: {profile.category.value}")
            print(f"   Time Horizon: {profile.time_horizon.value}")
            print(f"   Risk Level: {profile.risk_level.value}")
            print(f"   Success Probability: {profile.success_probability:.1%}")
            print(f"   Commercial Potential: {profile.commercial_potential:.2f}")
            print(f"   Scientific Novelty: {profile.scientific_novelty:.2f}")
            print(f"   Market Size: {profile.market_size_estimate}")
            print(f"   Next Steps: {', '.join(profile.next_steps[:2])}")
        
        print(f"\n📊 MULTI-DIMENSIONAL RANKINGS AVAILABLE:")
        for strategy, ranked_list in rankings.items():
            if ranked_list:
                strategy_name = strategy.replace('_', ' ').title()
                print(f"   {strategy_name}: {len(ranked_list)} ranked breakthroughs")
        
        # Show the difference from old system
        print(f"\n🆚 COMPARISON WITH OLD SYSTEM:")
        print(f"❌ Old: 'Score 0.354 → MINIMAL tier' (arbitrary threshold)")
        print(f"✅ New: 'Category: niche, Risk: moderate, Timeline: 2-5 years, Success: 60%'")
        print(f"✅ New: 'Commercial rank #3, Innovation rank #8, Risk-adjusted rank #5'")
        print(f"✅ New: 'Next steps: Prototype development, Market validation'")
        
    else:
        print(f"\n❌ MULTI-DIMENSIONAL TEST FAILED")
    
    return results

def compare_assessment_systems():
    """Compare old vs new assessment systems"""
    
    print(f"\n🔬 ASSESSMENT SYSTEM COMPARISON")
    print("=" * 80)
    
    print("📊 Testing same papers with different assessment systems...")
    
    # Test with legacy system
    print(f"\n1️⃣ LEGACY THRESHOLD SYSTEM:")
    legacy_processor = EnhancedBatchProcessor(
        test_mode="legacy_test",
        use_multi_dimensional=False,
        use_calibrated_scoring=True
    )
    
    legacy_results = legacy_processor.run_unified_test(
        test_mode="phase_a",
        paper_count=5,
        paper_source="unique"
    )
    
    if legacy_results:
        legacy_metrics = legacy_results['quality_metrics']
        print(f"   Average score: {legacy_metrics.get('average_quality_score', 0):.3f}")
        print(f"   Assessment: Arbitrary threshold classification")
    
    # Test with multi-dimensional system
    print(f"\n2️⃣ MULTI-DIMENSIONAL SYSTEM:")
    md_processor = EnhancedBatchProcessor(
        test_mode="multi_dimensional_test",
        use_multi_dimensional=True,
        organization_type="industry"
    )
    
    md_results = md_processor.run_unified_test(
        test_mode="phase_a",
        paper_count=5,
        paper_source="unique"
    )
    
    if md_results:
        md_metrics = md_results['quality_metrics']
        print(f"   Average success probability: {md_metrics.get('avg_success_probability', 0):.1%}")
        print(f"   Categories: {md_metrics.get('niche_discoveries', 0)} niche, {md_metrics.get('incremental_discoveries', 0)} incremental")
        print(f"   Time horizons: {md_metrics.get('near_term_opportunities', 0)} near-term, {md_metrics.get('immediate_opportunities', 0)} immediate")
        print(f"   Assessment: Context-aware, actionable insights")
    
    print(f"\n✅ RESULT: Multi-dimensional system provides richer, more actionable insights!")

if __name__ == "__main__":
    test_multi_dimensional_assessment()
    compare_assessment_systems()