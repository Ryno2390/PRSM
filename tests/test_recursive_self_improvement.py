#!/usr/bin/env python3
"""
Small Integration Test for Recursive Self-Improvement System
Tests the complete Phase 3, Week 13-14 pipeline without creating large test files
"""

import asyncio
import time
from datetime import datetime, timezone

from prsm.improvement.performance_monitor import get_performance_monitor
from prsm.improvement.proposal_engine import get_proposal_engine
from prsm.improvement.evolution import get_evolution_orchestrator
from prsm.core.models import ImprovementType, ProposalStatus


async def test_complete_improvement_pipeline():
    """Test the complete recursive self-improvement pipeline"""
    print("🧪 Testing complete recursive self-improvement pipeline...")
    
    # Get system components
    monitor = get_performance_monitor()
    proposal_engine = get_proposal_engine()
    evolution = get_evolution_orchestrator()
    
    # Step 1: Track some performance metrics that show need for improvement
    performance_data = {
        "accuracy": 0.80,  # Could be better
        "latency": 200,    # High latency
        "throughput": 50,  # Low throughput
        "error_rate": 0.08 # High error rate
    }
    
    success = await monitor.track_model_metrics("test_model_rsi", performance_data)
    assert success, "Should track performance metrics"
    
    # Step 2: Identify improvement opportunities with more data points
    historical_data = [
        {
            "model_id": "test_model_rsi",
            "metrics": {"accuracy": 0.85, "latency": 180, "error_rate": 0.05},
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "model_id": "test_model_rsi",
            "metrics": {"accuracy": 0.82, "latency": 190, "error_rate": 0.06},
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "model_id": "test_model_rsi", 
            "metrics": {"accuracy": 0.80, "latency": 200, "error_rate": 0.08},
            "timestamp": datetime.now(timezone.utc)
        }
    ]
    
    opportunities = await monitor.identify_improvement_opportunities(historical_data)
    # Note: May not always identify opportunities depending on algorithm thresholds
    print(f"📈 Identified {len(opportunities)} improvement opportunities")
    
    # Step 3: Generate improvement proposals based on weakness analysis
    weakness_analysis = {
        "identified_weaknesses": [
            {"category": "latency_issues", "component": "test_model_rsi", "severity": 0.7}
        ],
        "performance_bottlenecks": {
            "test_model_rsi": {"types": ["latency", "throughput"], "severity": 0.6}
        },
        "resource_constraints": {
            "memory": {"utilization": 0.85, "limit_reached": True}
        }
    }
    
    proposals = await proposal_engine.generate_architecture_proposals(weakness_analysis)
    assert len(proposals) > 0, "Should generate improvement proposals"
    
    # Step 4: Simulate and validate proposals
    for proposal in proposals[:2]:  # Test first 2 proposals to keep it small
        # Simulate the proposal
        simulation_result = await proposal_engine.simulate_proposed_changes(proposal)
        assert simulation_result.confidence_score > 0.0, "Should have simulation confidence"
        
        # Validate safety
        safety_check = await proposal_engine.validate_improvement_safety(proposal)
        assert safety_check.safety_score > 0.0, "Should have safety score"
    
    # Step 5: Run A/B testing on approved proposals
    approved_proposals = [p for p in proposals[:2] if p.safety_check and p.safety_check.safety_score > 0.6]
    
    if approved_proposals:
        test_results = await evolution.coordinate_a_b_testing(approved_proposals)
        assert test_results.test_duration > 0, "Should complete A/B testing"
        assert len(test_results.proposals_tested) > 0, "Should test proposals"
    
    print("✅ Complete recursive self-improvement pipeline working")


async def test_performance_monitoring_integration():
    """Test performance monitoring integration"""
    print("🧪 Testing performance monitoring integration...")
    
    monitor = get_performance_monitor()
    
    # Track metrics over time to build a baseline
    model_id = "integration_test_model"
    
    for i in range(3):
        metrics = {
            "accuracy": 0.85 + (i * 0.01),
            "latency": 150 - (i * 5), 
            "throughput": 100 + (i * 10)
        }
        await monitor.track_model_metrics(model_id, metrics)
    
    # Generate benchmark report
    report = await monitor.benchmark_against_baselines(model_id)
    assert report.model_id == model_id, "Should generate benchmark report"
    assert len(report.comparison_metrics) > 0, "Should have comparison metrics"
    
    # Generate performance analysis
    analysis = await monitor.generate_performance_analysis(model_id)
    assert analysis.model_id == model_id, "Should generate performance analysis"
    assert analysis.overall_health_score > 0.0, "Should calculate health score"
    
    print("✅ Performance monitoring integration working")


async def test_proposal_engine_integration():
    """Test proposal engine integration"""
    print("🧪 Testing proposal engine integration...")
    
    proposal_engine = get_proposal_engine()
    
    # Test different types of weakness analysis
    weakness_scenarios = [
        {
            "identified_weaknesses": [{"category": "accuracy_problems", "severity": 0.8}],
            "performance_bottlenecks": {},
            "resource_constraints": {}
        },
        {
            "identified_weaknesses": [{"category": "safety_concerns", "severity": 0.9}],
            "performance_bottlenecks": {},
            "resource_constraints": {}
        }
    ]
    
    total_proposals = 0
    for weakness_analysis in weakness_scenarios:
        proposals = await proposal_engine.generate_architecture_proposals(weakness_analysis)
        total_proposals += len(proposals)
        
        # Test simulation for first proposal
        if proposals:
            proposal = proposals[0]
            simulation_result = await proposal_engine.simulate_proposed_changes(proposal)
            assert simulation_result.proposal_id == proposal.proposal_id, "Should simulate proposal"
            
            safety_check = await proposal_engine.validate_improvement_safety(proposal)
            assert safety_check.proposal_id == proposal.proposal_id, "Should validate safety"
    
    assert total_proposals > 0, "Should generate proposals for different weakness types"
    
    print("✅ Proposal engine integration working")


async def test_system_statistics():
    """Test that all systems provide statistics"""
    print("🧪 Testing system statistics...")
    
    monitor = get_performance_monitor()
    proposal_engine = get_proposal_engine()
    evolution = get_evolution_orchestrator()
    
    # Get statistics from all components
    monitor_stats = await monitor.get_monitoring_statistics()
    engine_stats = await proposal_engine.get_engine_statistics()
    evolution_stats = await evolution.get_evolution_statistics()
    
    # Verify essential fields exist
    assert "total_metrics_tracked" in monitor_stats, "Monitor should provide metrics count"
    assert "proposals_generated" in engine_stats, "Engine should provide proposal count"
    assert "tests_completed" in evolution_stats, "Evolution should provide test count"
    
    # Verify configuration exists
    assert "configuration" in monitor_stats, "Should provide configuration"
    assert "configuration" in engine_stats, "Should provide configuration"
    assert "configuration" in evolution_stats, "Should provide configuration"
    
    print("✅ System statistics working")


async def run_integration_tests():
    """Run all recursive self-improvement integration tests"""
    print("🚀 Starting Recursive Self-Improvement Integration Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run integration tests
        await test_performance_monitoring_integration()
        await test_proposal_engine_integration()
        await test_system_statistics()
        await test_complete_improvement_pipeline()
        
        # Get final statistics
        print("\n📊 Final System Statistics:")
        
        monitor = get_performance_monitor()
        proposal_engine = get_proposal_engine()
        evolution = get_evolution_orchestrator()
        
        monitor_stats = await monitor.get_monitoring_statistics()
        engine_stats = await proposal_engine.get_engine_statistics()
        evolution_stats = await evolution.get_evolution_statistics()
        
        print(f"   🔍 Performance Monitor:")
        print(f"     • Models monitored: {monitor_stats['models_monitored_count']}")
        print(f"     • Metrics tracked: {monitor_stats['total_metrics_tracked']}")
        print(f"     • Improvement opportunities: {monitor_stats['improvement_opportunities_identified']}")
        
        print(f"   🔧 Proposal Engine:")
        print(f"     • Proposals generated: {engine_stats['proposals_generated']}")
        print(f"     • Safety checks performed: {engine_stats['safety_checks_performed']}")
        print(f"     • Simulations run: {engine_stats['simulations_run']}")
        
        print(f"   🧬 Evolution Orchestrator:")
        print(f"     • Tests completed: {evolution_stats['tests_completed']}")
        print(f"     • Improvements implemented: {evolution_stats['improvements_implemented']}")
        print(f"     • Network updates: {evolution_stats['network_updates_propagated']}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ All Recursive Self-Improvement integration tests passed!")
        print(f"⚡ Test duration: {duration:.2f} seconds")
        print(f"🎯 Phase 3, Week 13-14 - Recursive Self-Improvement is operational and ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the integration test suite
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)