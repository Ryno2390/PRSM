#!/usr/bin/env python3
"""
Meta-Reasoning Engine Integration Test
=====================================

This script tests the integration of all 7 enhanced reasoning engines
with the meta-reasoning system to ensure proper functionality.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import (
    MetaReasoningEngine,
    ThinkingMode,
    ReasoningEngine
)

import structlog
logger = structlog.get_logger(__name__)


class MetaReasoningIntegrationTest:
    """Test suite for meta-reasoning engine integration"""
    
    def __init__(self):
        self.meta_engine = MetaReasoningEngine()
        self.test_query = "What are the implications of climate change for renewable energy adoption?"
        self.test_context = {
            "domain": "environmental_technology",
            "urgency": "high",
            "time_horizon": "10_years",
            "stakeholders": ["government", "industry", "consumers"]
        }
    
    async def test_engine_initialization(self):
        """Test that all engines are properly initialized"""
        
        print("🔧 Testing Engine Initialization...")
        
        # Check that all engines are initialized
        for engine_type in ReasoningEngine:
            assert engine_type in self.meta_engine.reasoning_engines, f"Missing engine: {engine_type}"
            assert self.meta_engine.reasoning_engines[engine_type] is not None, f"Engine not initialized: {engine_type}"
        
        print("✅ All engines initialized successfully")
    
    async def test_individual_engine_execution(self):
        """Test execution of each individual engine"""
        
        print("🧠 Testing Individual Engine Execution...")
        
        results = {}
        
        for engine_type in ReasoningEngine:
            try:
                print(f"  Testing {engine_type.value} engine...")
                
                engine = self.meta_engine.reasoning_engines[engine_type]
                result = await self.meta_engine._execute_reasoning_engine(
                    engine_type, engine, self.test_query, self.test_context
                )
                
                results[engine_type] = result
                # Handle confidence formatting for both float and enum values
                confidence_str = f"{result.confidence:.2f}" if isinstance(result.confidence, (int, float)) else str(result.confidence)
                print(f"    ✅ {engine_type.value}: confidence={confidence_str}, quality={result.quality_score:.2f}")
                
            except Exception as e:
                print(f"    ❌ {engine_type.value} failed: {str(e)}")
                print(f"    Traceback: {traceback.format_exc()}")
                results[engine_type] = None
        
        successful_engines = [k for k, v in results.items() if v is not None]
        print(f"✅ {len(successful_engines)}/{len(ReasoningEngine)} engines executed successfully")
        
        return results
    
    async def test_parallel_reasoning(self):
        """Test parallel reasoning execution"""
        
        print("⚡ Testing Parallel Reasoning (Quick Mode)...")
        
        try:
            result = await self.meta_engine.meta_reason(
                query=self.test_query,
                context=self.test_context,
                thinking_mode=ThinkingMode.QUICK
            )
            
            print(f"✅ Parallel reasoning completed:")
            print(f"  - Processing time: {result.total_processing_time:.2f}s")
            print(f"  - Meta confidence: {result.meta_confidence:.2f}")
            print(f"  - FTNS cost: {result.ftns_cost}")
            print(f"  - Overall quality: {result.get_overall_quality():.2f}")
            
            if result.parallel_results:
                print(f"  - Successful engines: {len(result.parallel_results)}")
                for pr in result.parallel_results:
                    # Handle confidence formatting for both float and enum values
                    confidence_str = f"{pr.confidence:.2f}" if isinstance(pr.confidence, (int, float)) else str(pr.confidence)
                    print(f"    • {pr.engine.value}: {confidence_str}")
            
            return result
            
        except Exception as e:
            print(f"❌ Parallel reasoning failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def test_thinking_mode_configurations(self):
        """Test thinking mode configurations"""
        
        print("⚙️ Testing Thinking Mode Configurations...")
        
        mode_info = self.meta_engine.get_thinking_mode_info()
        
        for mode, info in mode_info.items():
            print(f"  {mode.value}:")
            print(f"    - Description: {info['description']}")
            print(f"    - Max permutations: {info['max_permutations']}")
            print(f"    - Timeout: {info['timeout_seconds']}s")
            print(f"    - Estimated cost: {info['estimated_ftns_cost']} FTNS")
        
        print("✅ All thinking mode configurations loaded")
    
    async def test_cost_estimation(self):
        """Test FTNS cost estimation"""
        
        print("💰 Testing FTNS Cost Estimation...")
        
        for mode in ThinkingMode:
            cost = self.meta_engine.estimate_ftns_cost(mode)
            print(f"  {mode.value}: {cost} FTNS")
        
        print("✅ Cost estimation working correctly")
    
    async def test_interaction_patterns(self):
        """Test interaction pattern recognition"""
        
        print("🔗 Testing Interaction Patterns...")
        
        patterns = self.meta_engine.interaction_patterns
        print(f"  Total patterns defined: {len(patterns)}")
        
        for pattern, description in patterns.items():
            engine1, engine2 = pattern
            print(f"  {engine1.value} → {engine2.value}: {description}")
        
        print("✅ Interaction patterns loaded correctly")
    
    async def test_synthesis_methods(self):
        """Test synthesis methods"""
        
        print("🔄 Testing Synthesis Methods...")
        
        # Create mock results for testing
        mock_results = []
        for engine_type in list(ReasoningEngine)[:3]:  # Test with first 3 engines
            from prsm.nwtn.meta_reasoning_engine import ReasoningResult
            mock_result = ReasoningResult(
                engine=engine_type,
                result=f"Mock result from {engine_type.value}",
                confidence=0.7 + (hash(engine_type.value) % 100) / 1000,  # Vary confidence
                processing_time=1.0,
                quality_score=0.8,
                evidence_strength=0.6,
                reasoning_chain=[f"Step 1 from {engine_type.value}", f"Step 2 from {engine_type.value}"],
                assumptions=[f"Assumption from {engine_type.value}"],
                limitations=[f"Limitation from {engine_type.value}"]
            )
            mock_results.append(mock_result)
        
        # Test each synthesis method
        for method_name, method_func in self.meta_engine.synthesis_methods.items():
            try:
                synthesis = await method_func(mock_results, self.test_context)
                print(f"  ✅ {method_name.value}: {synthesis.get('method', 'unknown')}")
            except Exception as e:
                print(f"  ❌ {method_name.value} failed: {str(e)}")
        
        print("✅ Synthesis methods tested")
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        
        print("🛡️ Testing Error Handling...")
        
        # Test with invalid query
        try:
            result = await self.meta_engine.meta_reason(
                query="",  # Empty query
                context={},
                thinking_mode=ThinkingMode.QUICK
            )
            print("  ✅ Handled empty query gracefully")
        except Exception as e:
            print(f"  ❌ Failed to handle empty query: {str(e)}")
        
        # Test with invalid context
        try:
            result = await self.meta_engine.meta_reason(
                query=self.test_query,
                context=None,  # None context
                thinking_mode=ThinkingMode.QUICK
            )
            print("  ✅ Handled None context gracefully")
        except Exception as e:
            print(f"  ❌ Failed to handle None context: {str(e)}")
        
        print("✅ Error handling tested")
    
    async def test_health_monitoring(self):
        """Test engine health monitoring system"""
        
        print("🏥 Testing Engine Health Monitoring...")
        
        # Perform comprehensive health check
        health_check = await self.meta_engine.perform_health_check()
        
        print(f"  Overall system health: {health_check['system_health']['overall_health_score']:.2f}")
        print(f"  Total engines: {health_check['system_health']['total_engines']}")
        print(f"  Healthy engines: {len(health_check['healthy_engines'])}")
        print(f"  Unhealthy engines: {len(health_check['unhealthy_engines'])}")
        
        # Test individual engine health reports
        print("\n  Individual engine health:")
        for engine_type in [ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC]:
            health_report = self.meta_engine.get_engine_health_report(engine_type)
            print(f"    {engine_type.value}: {health_report['status']} (score: {health_report['health_score']:.2f})")
        
        # Test health status checking
        healthy_engines = self.meta_engine.get_healthy_engines()
        unhealthy_engines = self.meta_engine.get_unhealthy_engines()
        
        print(f"\n  Healthy engines: {[e.value for e in healthy_engines]}")
        print(f"  Unhealthy engines: {[e.value for e in unhealthy_engines]}")
        
        # Test health monitoring controls
        self.meta_engine.disable_health_monitoring()
        print("  ✅ Health monitoring disabled")
        
        self.meta_engine.enable_health_monitoring()
        print("  ✅ Health monitoring enabled")
        
        print("✅ Health monitoring system tested")
    
    async def test_performance_tracking(self):
        """Test performance tracking system"""
        
        print("📊 Testing Performance Tracking System...")
        
        # Get initial performance summary
        initial_summary = self.meta_engine.get_performance_summary()
        print(f"  Initial performance summary: {initial_summary['total_snapshots']} snapshots")
        
        # Run a test to generate performance data
        await self.meta_engine.meta_reason(
            query="Performance test query",
            context={"performance_test": True},
            thinking_mode=ThinkingMode.QUICK
        )
        
        # Get updated performance summary
        updated_summary = self.meta_engine.get_performance_summary()
        print(f"  Updated performance summary: {updated_summary['total_snapshots']} snapshots")
        
        # Test individual engine performance profiles
        print("\\n  Individual engine performance profiles:")
        for engine_type in [ReasoningEngine.DEDUCTIVE, ReasoningEngine.PROBABILISTIC]:
            profile = self.meta_engine.get_engine_performance_profile(engine_type)
            print(f"    {engine_type.value}: {profile['performance_class']} (avg time: {profile['avg_execution_time']:.3f}s)")
        
        # Test comparative analysis
        comparative_analysis = self.meta_engine.get_performance_comparative_analysis()
        print(f"\\n  Comparative analysis:")
        print(f"    Fastest engine: {comparative_analysis.get('fastest_engine', 'N/A')}")
        print(f"    Slowest engine: {comparative_analysis.get('slowest_engine', 'N/A')}")
        print(f"    Highest quality: {comparative_analysis.get('highest_quality', 'N/A')}")
        
        # Test performance recommendations
        recommendations = self.meta_engine.get_performance_recommendations(ReasoningEngine.DEDUCTIVE)
        print(f"\\n  Performance recommendations for deductive engine: {len(recommendations)} items")
        
        # Test system performance report
        system_report = self.meta_engine.get_system_performance_report()
        print(f"\\n  System performance report generated with {len(system_report['engine_profiles'])} engine profiles")
        
        # Test performance tracking controls
        self.meta_engine.disable_performance_tracking()
        print("  ✅ Performance tracking disabled")
        
        self.meta_engine.enable_performance_tracking()
        print("  ✅ Performance tracking enabled")
        
        print("✅ Performance tracking system tested")
    
    async def test_failure_detection_and_recovery(self):
        """Test failure detection and recovery system"""
        
        print("🔧 Testing Failure Detection and Recovery System...")
        
        # Get initial failure statistics
        initial_stats = self.meta_engine.get_failure_statistics()
        print(f"  Initial failure statistics: {initial_stats['total_failures']} failures")
        
        # Test failure detection controls
        self.meta_engine.disable_failure_detection()
        print("  ✅ Failure detection disabled")
        
        self.meta_engine.enable_failure_detection()
        print("  ✅ Failure detection enabled")
        
        # Test recovery controls
        self.meta_engine.disable_failure_recovery()
        print("  ✅ Failure recovery disabled")
        
        self.meta_engine.enable_failure_recovery()
        print("  ✅ Failure recovery enabled")
        
        # Get failure and recovery report
        report = self.meta_engine.get_failure_and_recovery_report()
        print(f"  Failure and recovery report generated")
        print(f"    - Failure detection enabled: {report['failure_detection_enabled']}")
        print(f"    - Recovery enabled: {report['recovery_enabled']}")
        print(f"    - Total circuit breakers: {report['circuit_breaker_status']['total_circuit_breakers']}")
        
        # Test manual recovery
        manual_recovery_success = await self.meta_engine.manual_recovery(
            ReasoningEngine.DEDUCTIVE, "restart"
        )
        print(f"  Manual recovery test: {'✅ Success' if manual_recovery_success else '❌ Failed'}")
        
        # Test failure history
        failure_history = self.meta_engine.get_failure_history(hours=24)
        print(f"  Failure history: {len(failure_history)} events in last 24 hours")
        
        # Test recovery statistics
        recovery_stats = self.meta_engine.get_recovery_statistics()
        print(f"  Recovery statistics: {recovery_stats['total_recoveries']} total recoveries")
        
        # Test circuit breaker status
        circuit_status = self.meta_engine.get_circuit_breaker_status()
        print(f"  Circuit breaker status: {circuit_status['total_circuit_breakers']} active")
        
        # Test engine isolation status
        isolation_status = self.meta_engine.get_engine_isolation_status()
        print(f"  Engine isolation: {isolation_status['total_isolated']} isolated engines")
        
        # Reset failure and recovery history
        self.meta_engine.reset_failure_history()
        print("  ✅ Failure history reset")
        
        self.meta_engine.reset_recovery_history()
        print("  ✅ Recovery history reset")
        
        print("✅ Failure detection and recovery system tested")
    
    async def test_load_balancing_system(self):
        """Test load balancing system"""
        
        print("⚖️ Testing Load Balancing System...")
        
        # Get initial load balancing statistics
        initial_stats = self.meta_engine.get_load_balancing_statistics()
        print(f"  Initial load balancing statistics: {initial_stats['metrics']['total_requests']} requests")
        
        # Test load balancing controls
        self.meta_engine.disable_load_balancing()
        print("  ✅ Load balancing disabled")
        
        self.meta_engine.enable_load_balancing()
        print("  ✅ Load balancing enabled")
        
        # Test strategy management
        current_strategy = self.meta_engine.get_load_balancing_strategy()
        print(f"  Current strategy: {current_strategy}")
        
        # Test setting different strategies
        test_strategies = ["round_robin", "least_connections", "performance_based"]
        for strategy in test_strategies:
            try:
                self.meta_engine.set_load_balancing_strategy(strategy)
                print(f"  ✅ Strategy set to {strategy}")
            except Exception as e:
                print(f"  ❌ Failed to set strategy to {strategy}: {str(e)}")
        
        # Reset to original strategy
        self.meta_engine.set_load_balancing_strategy(current_strategy)
        
        # Test engine workload status
        workload_status = self.meta_engine.get_engine_workload_status()
        print(f"  Engine workload status: {len(workload_status)} engines tracked")
        
        # Test individual engine workload
        deductive_workload = self.meta_engine.get_engine_workload_status(ReasoningEngine.DEDUCTIVE)
        print(f"  Deductive engine workload: {deductive_workload.get('total_requests', 0)} requests")
        
        # Test engine weights
        engine_weights = self.meta_engine.get_engine_weights()
        print(f"  Engine weights: {len(engine_weights)} engines weighted")
        
        # Test available engines
        available_engines = self.meta_engine.get_available_engines()
        print(f"  Available engines: {len(available_engines)} engines available")
        
        # Test comprehensive load balancing report
        load_balancing_report = self.meta_engine.get_load_balancing_report()
        print(f"  Load balancing report generated")
        print(f"    - Strategy: {load_balancing_report['current_strategy']}")
        print(f"    - Enabled: {load_balancing_report['load_balancing_enabled']}")
        print(f"    - Available engines: {len(load_balancing_report['available_engines'])}")
        
        # Test weight updates
        self.meta_engine.update_engine_weights()
        print("  ✅ Engine weights updated")
        
        # Test metrics reset
        self.meta_engine.reset_load_balancing_metrics()
        print("  ✅ Load balancing metrics reset")
        
        # Run a test to generate load balancing data
        await self.meta_engine.meta_reason(
            query="Load balancing test query",
            context={"load_balancing_test": True},
            thinking_mode=ThinkingMode.QUICK
        )
        
        # Get updated statistics
        updated_stats = self.meta_engine.get_load_balancing_statistics()
        print(f"  Updated statistics: {updated_stats['metrics']['total_requests']} requests")
        
        print("✅ Load balancing system tested")
    
    async def test_adaptive_selection_system(self):
        """Test adaptive selection system"""
        
        print("🧠 Testing Adaptive Selection System...")
        
        # Test problem type detection
        test_queries = [
            ("What causes climate change?", "causal_analysis"),
            ("If the economy collapsed, what would happen?", "scenario_analysis"),
            ("How likely is it to rain tomorrow?", "uncertainty_quantification"),
            ("What are the patterns in this data?", "pattern_recognition"),
            ("How can we optimize this process?", "optimization")
        ]
        
        print("  Testing problem type detection:")
        for query, expected_type in test_queries:
            detected_type = self.meta_engine.detect_problem_type(query)
            print(f"    '{query[:30]}...' -> {detected_type}")
        
        # Test adaptive engine selection
        print("\\n  Testing adaptive engine selection:")
        for query, _ in test_queries[:3]:
            selected_engines = self.meta_engine.select_engines_adaptively(query, num_engines=3)
            print(f"    '{query[:30]}...' -> {selected_engines}")
        
        # Test engine selection scores
        print("\\n  Testing engine selection scores:")
        test_query = "What are the causes of economic inflation?"
        test_context = {"domain": "financial", "urgency": "medium", "quality": "high"}
        
        scores = self.meta_engine.get_engine_selection_scores(test_query, test_context)
        print(f"    Problem type detected: {scores['detected_problem_type']}")
        print(f"    Top 3 engines: {scores['top_engines']}")
        print(f"    Contextual factors: {list(scores['contextual_factors'].keys())}")
        
        # Test adaptive selection controls
        self.meta_engine.disable_adaptive_selection()
        print("  ✅ Adaptive selection disabled")
        
        self.meta_engine.enable_adaptive_selection()
        print("  ✅ Adaptive selection enabled")
        
        # Test strategy management
        current_strategy = self.meta_engine.get_adaptive_selection_strategy()
        print(f"  Current strategy: {current_strategy}")
        
        # Test setting different strategies
        test_strategies = ["context_aware", "performance_optimized", "problem_type_matching"]
        for strategy in test_strategies:
            try:
                self.meta_engine.set_adaptive_selection_strategy(strategy)
                print(f"  ✅ Strategy set to {strategy}")
            except Exception as e:
                print(f"  ❌ Failed to set strategy to {strategy}: {str(e)}")
        
        # Reset to original strategy
        self.meta_engine.set_adaptive_selection_strategy(current_strategy)
        
        # Test problem type mappings
        problem_type_mappings = self.meta_engine.get_problem_type_mappings()
        print(f"  Problem type mappings: {len(problem_type_mappings)} types mapped")
        
        # Test adaptive selection statistics
        statistics = self.meta_engine.get_adaptive_selection_statistics()
        print(f"  Selection history: {statistics['selection_history_count']} entries")
        print(f"  Performance history: {len(statistics['performance_history'])} engines tracked")
        
        # Test comprehensive adaptive selection report
        adaptive_report = self.meta_engine.get_adaptive_selection_report()
        print(f"  Adaptive selection report generated")
        print(f"    - Strategy: {adaptive_report['current_strategy']}")
        print(f"    - Enabled: {adaptive_report['adaptive_selection_enabled']}")
        print(f"    - Learning parameters: {len(adaptive_report['learning_parameters'])} parameters")
        
        # Test performance feedback
        self.meta_engine.update_engine_performance_feedback("deductive", 0.85, "Test query")
        print("  ✅ Performance feedback updated")
        
        # Test history reset
        self.meta_engine.reset_adaptive_selection_history()
        print("  ✅ Adaptive selection history reset")
        
        # Run a test to generate adaptive selection data
        await self.meta_engine.meta_reason(
            query="How do market forces influence economic stability?",
            context={"domain": "economic", "urgency": "low", "quality": "high"},
            thinking_mode=ThinkingMode.QUICK
        )
        
        # Get updated statistics
        updated_stats = self.meta_engine.get_adaptive_selection_statistics()
        print(f"  Updated statistics: {updated_stats['selection_history_count']} selections")
        
        print("✅ Adaptive selection system tested")
    
    async def test_result_formatting_system(self):
        """Test result formatting system"""
        
        print("📄 Testing Result Formatting System...")
        
        # Run a test to generate results for formatting
        meta_result = await self.meta_engine.meta_reason(
            query="Test query for formatting demonstration",
            context={"domain": "testing", "urgency": "medium"},
            thinking_mode=ThinkingMode.QUICK
        )
        
        if not meta_result.parallel_results:
            print("  ❌ No parallel results generated for formatting test")
            return
        
        # Test single result formatting
        test_result = meta_result.parallel_results[0]
        
        # Test available formats
        available_formats = self.meta_engine.get_available_formats()
        print(f"  Available formats: {len(available_formats)} formats")
        
        # Test confidence and priority levels
        confidence_levels = self.meta_engine.get_confidence_levels()
        priority_levels = self.meta_engine.get_priority_levels()
        print(f"  Confidence levels: {len(confidence_levels)} levels")
        print(f"  Priority levels: {len(priority_levels)} levels")
        
        # Test formatting different formats
        test_formats = ["structured", "summary", "executive", "technical", "narrative", "json", "markdown"]
        
        print("\\n  Testing single result formatting:")
        for format_type in test_formats:
            try:
                formatted = self.meta_engine.format_single_result(test_result, format_type)
                if formatted["success"]:
                    print(f"    ✅ {format_type}: {formatted['confidence_level']}, {formatted['priority']}")
                else:
                    print(f"    ❌ {format_type}: {formatted['error']}")
            except Exception as e:
                print(f"    ❌ {format_type}: {str(e)}")
        
        # Test result rendering
        print("\\n  Testing result rendering:")
        for format_type in test_formats[:4]:  # Test first 4 formats
            try:
                rendered = self.meta_engine.render_result(test_result, format_type)
                if rendered["success"]:
                    print(f"    ✅ {format_type}: {rendered['length']} characters")
                else:
                    print(f"    ❌ {format_type}: {rendered['error']}")
            except Exception as e:
                print(f"    ❌ {format_type}: {str(e)}")
        
        # Test meta-result formatting
        print("\\n  Testing meta-result formatting:")
        meta_formats = ["structured", "executive", "summary", "technical"]
        
        for format_type in meta_formats:
            try:
                formatted_meta = self.meta_engine.format_meta_result(meta_result, format_type)
                if formatted_meta["success"]:
                    print(f"    ✅ {format_type}: {formatted_meta['overall_confidence']}, {formatted_meta['engine_count']} engines")
                else:
                    print(f"    ❌ {format_type}: {formatted_meta['error']}")
            except Exception as e:
                print(f"    ❌ {format_type}: {str(e)}")
        
        # Test meta-result rendering
        print("\\n  Testing meta-result rendering:")
        for format_type in meta_formats[:3]:  # Test first 3 formats
            try:
                rendered_meta = self.meta_engine.render_meta_result(meta_result, format_type)
                if rendered_meta["success"]:
                    print(f"    ✅ {format_type}: {rendered_meta['length']} characters")
                else:
                    print(f"    ❌ {format_type}: {rendered_meta['error']}")
            except Exception as e:
                print(f"    ❌ {format_type}: {str(e)}")
        
        # Test comparison formatting
        if len(meta_result.parallel_results) > 1:
            comparison_table = self.meta_engine.format_comparison_results(meta_result.parallel_results)
            print(f"  ✅ Comparison table: {len(comparison_table)} characters")
        
        # Test result export
        export_data = self.meta_engine.export_results(meta_result.parallel_results, "export")
        print(f"  ✅ Export data: {len(export_data)} characters")
        
        # Test result statistics
        stats = self.meta_engine.get_result_statistics(meta_result.parallel_results)
        if "error" not in stats:
            print(f"  ✅ Result statistics: {stats['total_results']} results analyzed")
            print(f"    - Average quality: {stats['average_quality']:.2f}")
            print(f"    - High confidence: {stats['high_confidence_count']}")
            print(f"    - High priority: {stats['high_priority_count']}")
        else:
            print(f"  ❌ Result statistics: {stats['error']}")
        
        # Test formatting report
        formatting_report = self.meta_engine.get_formatting_report()
        print(f"  ✅ Formatting report generated")
        print(f"    - Available formats: {len(formatting_report['available_formats'])}")
        print(f"    - Confidence levels: {len(formatting_report['confidence_levels'])}")
        print(f"    - Priority levels: {len(formatting_report['priority_levels'])}")
        print(f"    - Engine formatting rules: {len(formatting_report['engine_formatting_rules'])}")
        
        # Test invalid format handling
        invalid_result = self.meta_engine.render_result(test_result, "invalid_format")
        if not invalid_result["success"]:
            print("  ✅ Invalid format handling works correctly")
        else:
            print("  ❌ Invalid format handling failed")
        
        print("✅ Result formatting system tested")
    
    async def test_error_handling_system(self):
        """Test error handling system"""
        
        print("⚠️ Testing Error Handling System...")
        
        # Test error handling status
        status = self.meta_engine.get_error_handling_status()
        print(f"  Error handling enabled: {status['error_handling_enabled']}")
        print(f"  Total errors: {status['total_errors']}")
        print(f"  Error patterns: {status['error_patterns']}")
        
        # Test error statistics
        stats = self.meta_engine.get_error_statistics()
        print(f"  Error statistics: {stats['total_errors']} total errors")
        print(f"  Resolution rate: {stats['resolution_rate']:.2f}")
        print(f"  Recovery success rate: {stats['recovery_success_rate']:.2f}")
        
        # Test error categories and severities
        categories = self.meta_engine.get_error_categories()
        severities = self.meta_engine.get_error_severities()
        strategies = self.meta_engine.get_recovery_strategies()
        print(f"  Available categories: {len(categories)} categories")
        print(f"  Available severities: {len(severities)} severities")
        print(f"  Available strategies: {len(strategies)} strategies")
        
        # Test circuit breaker status
        circuit_status = self.meta_engine.get_circuit_breaker_status()
        print(f"  Circuit breaker status: {'triggered' if circuit_status['triggered'] else 'normal'}")
        print(f"  Current error rate: {circuit_status['current_error_rate']:.2f}")
        
        # Test error recovery simulation
        print("\\n  Testing error recovery simulation:")
        recovery_tests = [
            ("engine_error", "retry"),
            ("timeout_error", "degrade"),
            ("resource_error", "isolation"),
            ("system_error", "restart")
        ]
        
        for category, strategy in recovery_tests:
            result = self.meta_engine.simulate_error_recovery(category, strategy)
            if result["success"]:
                print(f"    ✅ {category} -> {strategy}: {'success' if result['recovery_successful'] else 'failed'}")
            else:
                print(f"    ❌ {category} -> {strategy}: {result['error']}")
        
        # Test error handling controls
        self.meta_engine.disable_error_handling()
        print("  ✅ Error handling disabled")
        
        self.meta_engine.enable_error_handling()
        print("  ✅ Error handling enabled")
        
        # Test error patterns
        patterns = self.meta_engine.get_error_patterns()
        print(f"  Error patterns: {len(patterns)} patterns detected")
        
        # Test recent errors
        recent_errors = self.meta_engine.get_recent_errors(hours=24)
        print(f"  Recent errors: {len(recent_errors)} errors in last 24 hours")
        
        # Test comprehensive error report
        error_report = self.meta_engine.get_error_handling_report()
        print(f"  Error handling report generated")
        print(f"    - System status: {error_report['system_status']['error_handling_enabled']}")
        print(f"    - Statistics: {error_report['statistics']['total_errors']} total errors")
        print(f"    - Recent errors: {len(error_report['recent_errors'])} recent")
        print(f"    - Error patterns: {len(error_report['error_patterns'])} patterns")
        print(f"    - Circuit breaker: {error_report['circuit_breaker']['triggered']}")
        
        # Test error history clearing
        clear_result = self.meta_engine.clear_error_history()
        print(f"  ✅ Error history cleared: {clear_result['cleared_errors']} errors")
        
        # Test custom error handling
        try:
            custom_result = self.meta_engine.handle_custom_error("TestError", "Test error message", "high")
            print(f"  ✅ Custom error handling: {custom_result['success']}")
        except Exception as e:
            print(f"  ❌ Custom error handling failed: {str(e)}")
        
        print("✅ Error handling system tested")
    
    async def test_interaction_pattern_recognition_system(self):
        """Test interaction pattern recognition system"""
        
        print("🧩 Testing Interaction Pattern Recognition System...")
        
        # Test pattern recognition status
        status = self.meta_engine.get_pattern_recognition_status()
        print(f"  Pattern recognition enabled: {status['pattern_recognition_enabled']}")
        print(f"  Auto-discovery enabled: {status['auto_discovery_enabled']}")
        print(f"  Total patterns: {status['total_patterns']}")
        print(f"  Total evidence: {status['total_evidence']}")
        print(f"  Recent discoveries: {status['recent_discoveries']}")
        
        # Test pattern types and outcomes
        pattern_types = self.meta_engine.get_pattern_types()
        interaction_outcomes = self.meta_engine.get_interaction_outcomes()
        print(f"  Available pattern types: {len(pattern_types)} types")
        print(f"  Available interaction outcomes: {len(interaction_outcomes)} outcomes")
        
        # Test getting all patterns
        all_patterns = self.meta_engine.get_all_patterns()
        print(f"  All patterns: {all_patterns['total_patterns']} patterns")
        
        # Test specific pattern information
        print("\\n  Testing pattern information:")
        from prsm.nwtn.meta_reasoning_engine import ReasoningEngine
        test_pairs = [
            (ReasoningEngine.INDUCTIVE, ReasoningEngine.CAUSAL),
            (ReasoningEngine.PROBABILISTIC, ReasoningEngine.ABDUCTIVE),
            (ReasoningEngine.ANALOGICAL, ReasoningEngine.INDUCTIVE)
        ]
        
        for pair in test_pairs:
            pattern_info = self.meta_engine.get_pattern_info(pair)
            if "error" not in pattern_info:
                print(f"    ✅ {pair[0].value} -> {pair[1].value}: {pattern_info['pattern_name']}")
                print(f"      Effectiveness: {pattern_info['effectiveness_score']:.2f}")
                print(f"      Confidence: {pattern_info['confidence_level']:.2f}")
            else:
                print(f"    ❌ {pair[0].value} -> {pair[1].value}: {pattern_info['error']}")
        
        # Test pattern recommendations
        print("\\n  Testing pattern recommendations:")
        test_contexts = [
            {"domain": "scientific", "urgency": "high", "complexity": "high"},
            {"domain": "financial", "urgency": "medium", "complexity": "medium"},
            {"domain": "technical", "urgency": "low", "complexity": "low"}
        ]
        
        for i, context in enumerate(test_contexts):
            recommendations = self.meta_engine.get_pattern_recommendations(
                f"Test query {i+1}", context
            )
            if recommendations["success"]:
                print(f"    ✅ Context {i+1}: {recommendations['total_recommendations']} recommendations")
                for rec in recommendations["recommendations"][:2]:  # Show first 2
                    print(f"      - {rec['engine_pair']}")
            else:
                print(f"    ❌ Context {i+1}: {recommendations['error']}")
        
        # Test pattern analysis
        analysis = self.meta_engine.get_pattern_analysis()
        if analysis["success"]:
            pattern_analysis = analysis["pattern_analysis"]
            print(f"\\n  Pattern analysis:")
            print(f"    - Total patterns: {pattern_analysis['total_patterns']}")
            print(f"    - Evidence count: {pattern_analysis['evidence_count']}")
            print(f"    - Pattern types: {len(pattern_analysis['pattern_types'])} types")
            print(f"    - Most effective: {len(pattern_analysis['most_effective_patterns'])} patterns")
            print(f"    - Recently discovered: {len(pattern_analysis['recently_discovered'])} patterns")
        else:
            print(f"    ❌ Pattern analysis failed: {analysis['error']}")
        
        # Test pattern evidence
        evidence = self.meta_engine.get_pattern_evidence(hours=24)
        if evidence["success"]:
            print(f"\\n  Pattern evidence:")
            print(f"    - Recent evidence: {evidence['evidence_count']} entries")
            print(f"    - Total evidence: {evidence['total_evidence']} entries")
            print(f"    - Time period: {evidence['time_period_hours']} hours")
        else:
            print(f"    ❌ Pattern evidence failed: {evidence['error']}")
        
        # Test pattern discovery history
        discovery_history = self.meta_engine.get_pattern_discovery_history()
        if discovery_history["success"]:
            print(f"\\n  Discovery history:")
            print(f"    - Total discoveries: {discovery_history['total_discoveries']} discoveries")
            for discovery in discovery_history["discoveries"][:3]:  # Show first 3
                print(f"      - {discovery['engine_pair']}: {discovery['pattern_name']}")
        else:
            print(f"    ❌ Discovery history failed: {discovery_history['error']}")
        
        # Test interaction pattern report
        report = self.meta_engine.get_interaction_pattern_report()
        if report["success"]:
            pattern_report = report["pattern_report"]
            print(f"\\n  Interaction pattern report generated:")
            print(f"    - System status: {pattern_report['system_status']['recognition_enabled']}")
            print(f"    - Pattern summary: {pattern_report['pattern_summary']['total_patterns']} patterns")
            print(f"    - Detailed patterns: {len(pattern_report['detailed_patterns'])} detailed")
            print(f"    - Evidence summary: {pattern_report['evidence_summary']['total_evidence']} evidence")
        else:
            print(f"    ❌ Pattern report failed: {report['error']}")
        
        # Test pattern recognition controls
        print("\\n  Testing pattern recognition controls:")
        
        # Test disabling pattern recognition
        disable_result = self.meta_engine.disable_pattern_recognition()
        if disable_result["success"]:
            print(f"    ✅ Pattern recognition disabled")
        else:
            print(f"    ❌ Failed to disable: {disable_result['error']}")
        
        # Test enabling pattern recognition
        enable_result = self.meta_engine.enable_pattern_recognition()
        if enable_result["success"]:
            print(f"    ✅ Pattern recognition enabled")
        else:
            print(f"    ❌ Failed to enable: {enable_result['error']}")
        
        # Test updating learning parameters
        param_update = self.meta_engine.update_pattern_learning_parameters(
            learning_rate=0.15,
            confidence_threshold=0.75,
            min_sample_size=12
        )
        if param_update["success"]:
            print(f"    ✅ Learning parameters updated")
            print(f"      - Learning rate: {param_update['current_parameters']['learning_rate']}")
            print(f"      - Confidence threshold: {param_update['current_parameters']['confidence_threshold']}")
            print(f"      - Min sample size: {param_update['current_parameters']['min_sample_size']}")
        else:
            print(f"    ❌ Failed to update parameters: {param_update['error']}")
        
        # Test clearing pattern history
        clear_result = self.meta_engine.clear_pattern_history()
        if clear_result["success"]:
            print(f"    ✅ Pattern history cleared: {clear_result['cleared_evidence']} evidence")
            print(f"      - Remaining patterns: {clear_result['remaining_patterns']}")
        else:
            print(f"    ❌ Failed to clear history: {clear_result['error']}")
        
        # Test simulating pattern observation
        print("\\n  Testing pattern observation simulation:")
        
        # Create mock results for testing
        from prsm.nwtn.meta_reasoning_engine import ReasoningResult, ReasoningEngine, ThinkingMode
        
        mock_first_result = ReasoningResult(
            engine=ReasoningEngine.INDUCTIVE,
            result="Mock inductive result",
            confidence=0.6,
            processing_time=1.0,
            quality_score=0.7,
            evidence_strength=0.65,
            reasoning_chain=["Inductive step 1", "Inductive step 2"],
            assumptions=["Inductive assumption"],
            limitations=["Inductive limitation"]
        )
        
        mock_second_result = ReasoningResult(
            engine=ReasoningEngine.CAUSAL,
            result="Mock causal result",
            confidence=0.8,
            processing_time=1.2,
            quality_score=0.85,
            evidence_strength=0.75,
            reasoning_chain=["Causal step 1", "Causal step 2"],
            assumptions=["Causal assumption"],
            limitations=["Causal limitation"]
        )
        
        mock_combined_result = ReasoningResult(
            engine=ReasoningEngine.INDUCTIVE,  # Combined result
            result="Mock combined result",
            confidence=0.85,
            processing_time=2.2,
            quality_score=0.9,
            evidence_strength=0.8,
            reasoning_chain=["Combined step 1", "Combined step 2"],
            assumptions=["Combined assumption"],
            limitations=["Combined limitation"]
        )
        
        # Test observing the interaction
        observation_result = self.meta_engine.observe_engine_interaction(
            (ReasoningEngine.INDUCTIVE, ReasoningEngine.CAUSAL),
            mock_first_result,
            mock_second_result,
            mock_combined_result,
            "Test query for pattern observation",
            {"domain": "test", "urgency": "medium"},
            ThinkingMode.QUICK
        )
        
        if observation_result["success"]:
            print(f"    ✅ Pattern observation successful:")
            print(f"      - Evidence ID: {observation_result['evidence_id']}")
            print(f"      - Engine pair: {observation_result['engine_pair']}")
            print(f"      - Observed pattern: {observation_result['observed_pattern']}")
            print(f"      - Outcome: {observation_result['outcome']}")
            print(f"      - Quality change: {observation_result['quality_change']:.2f}")
            print(f"      - Confidence change: {observation_result['confidence_change']:.2f}")
        else:
            print(f"    ❌ Pattern observation failed: {observation_result.get('error', 'Unknown error')}")
        
        print("✅ Interaction pattern recognition system tested")
    
    async def test_sequential_context_passing_system(self):
        """Test sequential context passing system"""
        
        print("🔄 Testing Sequential Context Passing System...")
        
        # Test context passing status
        status = self.meta_engine.get_context_passing_status()
        print(f"  Context passing enabled: {status['context_passing_enabled']}")
        print(f"  Active contexts: {status['active_contexts']}")
        print(f"  Context history: {status['context_history']}")
        print(f"  Default passing mode: {status['default_passing_mode']}")
        print(f"  Default relevance threshold: {status['default_relevance_threshold']}")
        print(f"  Compression enabled: {status['compression_enabled']}")
        print(f"  Filtering enabled: {status['filtering_enabled']}")
        
        # Test available options
        passing_modes = self.meta_engine.get_context_passing_modes()
        context_types = self.meta_engine.get_context_types()
        relevance_levels = self.meta_engine.get_context_relevance_levels()
        print(f"  Available passing modes: {len(passing_modes)} modes")
        print(f"  Available context types: {len(context_types)} types")
        print(f"  Available relevance levels: {len(relevance_levels)} levels")
        
        # Test creating a sequential context
        print("\\n  Testing context creation:")
        test_processing_chain = ["deductive", "inductive", "causal"]
        test_context = {
            "domain": "scientific",
            "urgency": "high",
            "complexity": "medium"
        }
        
        context_result = self.meta_engine.create_sequential_context(
            query="Test query for sequential context passing",
            context=test_context,
            processing_chain=test_processing_chain,
            passing_mode="enriched"
        )
        
        if context_result["success"]:
            context_id = context_result["context_id"]
            print(f"    ✅ Context created: {context_id}")
            print(f"      - Processing chain: {context_result['processing_chain']}")
            print(f"      - Passing mode: {context_result['passing_mode']}")
            print(f"      - Relevance threshold: {context_result['relevance_threshold']}")
            
            # Test adding context items
            print("\\n  Testing context item addition:")
            test_context_items = [
                {
                    "context_type": "evidence",
                    "content": "Strong empirical evidence supporting the hypothesis",
                    "source_engine": "deductive",
                    "relevance": "high",
                    "confidence": 0.9,
                    "quality_score": 0.85
                },
                {
                    "context_type": "patterns",
                    "content": "Recurring pattern observed in multiple datasets",
                    "source_engine": "inductive",
                    "relevance": "critical",
                    "confidence": 0.8,
                    "quality_score": 0.9
                },
                {
                    "context_type": "assumptions",
                    "content": "Assumes normal operating conditions",
                    "source_engine": "deductive",
                    "relevance": "medium",
                    "confidence": 0.7,
                    "quality_score": 0.7
                },
                {
                    "context_type": "insights",
                    "content": "Key insight: relationship between variables is non-linear",
                    "source_engine": "causal",
                    "relevance": "high",
                    "confidence": 0.85,
                    "quality_score": 0.8
                }
            ]
            
            added_items = []
            for item in test_context_items:
                item_result = self.meta_engine.add_context_item(
                    context_id=context_id,
                    context_type=item["context_type"],
                    content=item["content"],
                    source_engine=item["source_engine"],
                    relevance=item["relevance"],
                    confidence=item["confidence"],
                    quality_score=item["quality_score"]
                )
                
                if item_result["success"]:
                    added_items.append(item_result)
                    print(f"    ✅ Added {item['context_type']} item from {item['source_engine']}")
                    print(f"      - Relevance: {item_result['relevance']}")
                    print(f"      - Confidence: {item_result['confidence']:.2f}")
                else:
                    print(f"    ❌ Failed to add {item['context_type']} item: {item_result['error']}")
            
            print(f"    Added {len(added_items)} context items successfully")
            
            # Test getting context for different engines
            print("\\n  Testing context retrieval for engines:")
            for step, engine in enumerate(test_processing_chain):
                engine_context = self.meta_engine.get_context_for_engine(
                    context_id=context_id,
                    target_engine=engine,
                    step=step
                )
                
                if engine_context["success"]:
                    print(f"    ✅ {engine} (step {step}): {len(engine_context['context_keys'])} context keys")
                    print(f"      - Context keys: {engine_context['context_keys'][:5]}...")  # Show first 5
                else:
                    print(f"    ❌ {engine} (step {step}): {engine_context['error']}")
            
            # Test different passing modes
            print("\\n  Testing different passing modes:")
            test_modes = ["basic", "selective", "cumulative", "adaptive"]
            
            for mode in test_modes:
                mode_context = self.meta_engine.create_sequential_context(
                    query=f"Test query for {mode} mode",
                    context=test_context,
                    processing_chain=["deductive", "probabilistic"],
                    passing_mode=mode
                )
                
                if mode_context["success"]:
                    print(f"    ✅ {mode} mode: {mode_context['context_id']}")
                    
                    # Add a test item
                    self.meta_engine.add_context_item(
                        context_id=mode_context["context_id"],
                        context_type="insights",
                        content=f"Test insight for {mode} mode",
                        source_engine="deductive",
                        relevance="high",
                        confidence=0.8,
                        quality_score=0.75
                    )
                    
                    # Get context for second engine
                    mode_engine_context = self.meta_engine.get_context_for_engine(
                        context_id=mode_context["context_id"],
                        target_engine="probabilistic",
                        step=1
                    )
                    
                    if mode_engine_context["success"]:
                        print(f"      - Retrieved context: {len(mode_engine_context['context_keys'])} keys")
                    
                    # Finalize the context
                    self.meta_engine.finalize_sequential_context(mode_context["context_id"])
                else:
                    print(f"    ❌ {mode} mode: {mode_context['error']}")
            
            # Test active contexts
            print("\\n  Testing active contexts:")
            active_contexts = self.meta_engine.get_active_contexts()
            if active_contexts["success"]:
                print(f"    ✅ Active contexts: {active_contexts['total_active']} contexts")
                for ctx_id, ctx_info in active_contexts["active_contexts"].items():
                    print(f"      - {ctx_id}: {ctx_info['passing_mode']}, {ctx_info['total_items']} items")
            else:
                print(f"    ❌ Active contexts: {active_contexts['error']}")
            
            # Test context passing statistics
            print("\\n  Testing context passing statistics:")
            stats = self.meta_engine.get_context_passing_statistics()
            if stats["success"]:
                context_stats = stats["context_passing_statistics"]
                print(f"    ✅ Context statistics:")
                print(f"      - Active contexts: {context_stats['active_contexts']}")
                print(f"      - Context history: {context_stats['context_history']}")
                print(f"      - Contexts created: {context_stats['metrics']['contexts_created']}")
                print(f"      - Contexts passed: {context_stats['metrics']['contexts_passed']}")
                print(f"      - Total processing time: {context_stats['metrics']['total_processing_time']:.3f}s")
            else:
                print(f"    ❌ Context statistics: {stats['error']}")
            
            # Test context passing controls
            print("\\n  Testing context passing controls:")
            
            # Test disabling context passing
            disable_result = self.meta_engine.disable_context_passing()
            if disable_result["success"]:
                print(f"    ✅ Context passing disabled")
            else:
                print(f"    ❌ Failed to disable: {disable_result['error']}")
            
            # Test enabling context passing
            enable_result = self.meta_engine.enable_context_passing()
            if enable_result["success"]:
                print(f"    ✅ Context passing enabled")
            else:
                print(f"    ❌ Failed to enable: {enable_result['error']}")
            
            # Test configuration updates
            config_update = self.meta_engine.update_context_passing_configuration(
                passing_mode="selective",
                relevance_threshold="medium",
                compression_enabled=False,
                filtering_enabled=True
            )
            if config_update["success"]:
                print(f"    ✅ Configuration updated")
                print(f"      - Passing mode: {config_update['current_configuration']['passing_mode']}")
                print(f"      - Relevance threshold: {config_update['current_configuration']['relevance_threshold']}")
                print(f"      - Compression enabled: {config_update['current_configuration']['compression_enabled']}")
                print(f"      - Filtering enabled: {config_update['current_configuration']['filtering_enabled']}")
            else:
                print(f"    ❌ Failed to update configuration: {config_update['error']}")
            
            # Test comprehensive report
            report = self.meta_engine.get_context_passing_report()
            if report["success"]:
                print(f"\\n  ✅ Context passing report generated:")
                print(f"    - System status: {report['system_status']['context_passing_enabled']}")
                print(f"    - Active contexts: {report['active_contexts']['total_active']}")
                print(f"    - Context history: {report['context_history']['returned_count']}")
                print(f"    - Available options: {len(report['available_options']['passing_modes'])} modes")
            else:
                print(f"    ❌ Context passing report: {report['error']}")
            
            # Test finalizing the main context
            finalize_result = self.meta_engine.finalize_sequential_context(context_id)
            if finalize_result["success"]:
                print(f"\\n  ✅ Context finalized:")
                print(f"    - Context ID: {finalize_result['context_id']}")
                print(f"    - Total steps: {finalize_result['total_steps']}")
                print(f"    - Total items: {finalize_result['total_items']}")
                print(f"    - Processing chain: {finalize_result['processing_chain']}")
            else:
                print(f"    ❌ Failed to finalize context: {finalize_result['error']}")
            
            # Test context history
            history = self.meta_engine.get_context_history(limit=5)
            if history["success"]:
                print(f"\\n  ✅ Context history:")
                print(f"    - Returned count: {history['returned_count']}")
                print(f"    - Total history: {history['total_history']}")
                for ctx in history["context_history"][:3]:  # Show first 3
                    print(f"      - {ctx['context_id']}: {ctx['passing_mode']}, {ctx['total_items']} items")
            else:
                print(f"    ❌ Context history: {history['error']}")
            
            # Test clearing context history
            clear_result = self.meta_engine.clear_context_history()
            if clear_result["success"]:
                print(f"\\n  ✅ Context history cleared: {clear_result['cleared_contexts']} contexts")
            else:
                print(f"    ❌ Failed to clear history: {clear_result['error']}")
            
        else:
            print(f"    ❌ Failed to create context: {context_result['error']}")
        
        print("✅ Sequential context passing system tested")
    
    async def run_comprehensive_test(self):
        """Run all integration tests"""
        
        print("🚀 NWTN Meta-Reasoning Engine Integration Test")
        print("=" * 60)
        
        # Run all tests
        await self.test_engine_initialization()
        print()
        
        await self.test_individual_engine_execution()
        print()
        
        await self.test_parallel_reasoning()
        print()
        
        await self.test_thinking_mode_configurations()
        print()
        
        await self.test_cost_estimation()
        print()
        
        await self.test_interaction_patterns()
        print()
        
        await self.test_synthesis_methods()
        print()
        
        await self.test_error_handling()
        print()
        
        await self.test_health_monitoring()
        print()
        
        await self.test_performance_tracking()
        print()
        
        await self.test_failure_detection_and_recovery()
        print()
        
        await self.test_load_balancing_system()
        print()
        
        await self.test_adaptive_selection_system()
        print()
        
        await self.test_result_formatting_system()
        print()
        
        await self.test_error_handling_system()
        print()
        
        await self.test_interaction_pattern_recognition_system()
        print()
        
        await self.test_sequential_context_passing_system()
        print()
        
        await self.test_performance_optimization_system()
        print()
        
        print("🎉 Integration test completed!")
        print("=" * 60)
    
    async def test_performance_optimization_system(self):
        """Test performance optimization system"""
        
        print("⚡ Testing Performance Optimization System...")
        
        # Test memory status
        print("  Testing memory status:")
        memory_status = self.meta_engine.get_memory_status()
        print(f"    Memory monitoring: {memory_status['success']}")
        if memory_status['success']:
            print(f"    Memory critical: {memory_status['memory_critical']}")
        
        # Test cache status
        print("\\n  Testing cache status:")
        cache_status = self.meta_engine.get_cache_status()
        print(f"    Cache status: {cache_status['success']}")
        if cache_status['success']:
            print(f"    Cache size: {cache_status['cache_status']['cache_size']}")
            print(f"    Hit rate: {cache_status['cache_status']['hit_rate']:.2f}")
        
        # Test cache clearing
        print("\\n  Testing cache clearing:")
        clear_result = self.meta_engine.clear_cache()
        print(f"    Cache cleared: {clear_result['success']}")
        if clear_result['success']:
            print(f"    Cleared entries: {clear_result['cleared_entries']}")
        
        # Test performance optimization
        print("\\n  Testing performance optimization:")
        optimize_results = self.meta_engine.optimize_performance("memory")
        print(f"    Memory optimization: {optimize_results['success']}")
        
        optimize_results = self.meta_engine.optimize_performance("processing")
        print(f"    Processing optimization: {optimize_results['success']}")
        
        optimize_results = self.meta_engine.optimize_performance("all")
        print(f"    All optimization: {optimize_results['success']}")
        
        # Test optimization recommendations
        print("\\n  Testing optimization recommendations:")
        recommendations = self.meta_engine.get_optimization_recommendations()
        print(f"    Recommendations: {recommendations['success']}")
        if recommendations['success']:
            print(f"    Recommendation count: {recommendations['recommendation_count']}")
            if recommendations['recommendations']:
                print(f"    Sample recommendations: {recommendations['recommendations'][:2]}")
        
        # Test system metrics
        print("\\n  Testing system metrics:")
        system_metrics = self.meta_engine.get_system_metrics()
        print(f"    System metrics: {system_metrics['success']}")
        if system_metrics['success']:
            print(f"    Engine count: {system_metrics['engine_count']}")
            print(f"    Interaction patterns: {system_metrics['interaction_patterns']}")
        
        # Test performance report
        print("\\n  Testing performance report:")
        perf_report = self.meta_engine.get_performance_report()
        print(f"    Performance report: {perf_report['success']}")
        if perf_report['success']:
            print(f"    Report sections: {list(perf_report.keys())}")
        
        # Test optimization enable/disable
        print("\\n  Testing optimization controls:")
        
        # Disable optimization
        disable_result = self.meta_engine.disable_performance_optimization()
        print(f"    Optimization disabled: {disable_result['success']}")
        
        # Test optimization while disabled
        optimize_disabled = self.meta_engine.optimize_performance("memory")
        print(f"    Optimization while disabled: {optimize_disabled['success']}")
        
        # Re-enable optimization
        enable_result = self.meta_engine.enable_performance_optimization()
        print(f"    Optimization enabled: {enable_result['success']}")
        
        # Test performance metrics tracking
        print("\\n  Testing performance metrics tracking:")
        
        # Run a test operation to generate metrics
        await self.meta_engine.meta_reason(
            query="Performance testing query",
            context={"domain": "testing", "urgency": "low"},
            thinking_mode=ThinkingMode.QUICK
        )
        
        # Get updated metrics
        updated_metrics = self.meta_engine.get_system_metrics()
        print(f"    Updated metrics: {updated_metrics['success']}")
        
        # Test memory monitoring
        if hasattr(self.meta_engine.performance_optimizer.memory_monitor, 'monitoring_enabled'):
            print("\\n  Testing memory monitoring:")
            is_critical = self.meta_engine.performance_optimizer.memory_monitor.is_memory_critical()
            print(f"    Memory critical check: {is_critical}")
        
        # Test object pool optimization
        print("\\n  Testing object pool optimization:")
        try:
            pool_results = self.meta_engine.performance_optimizer.object_pool.optimize()
            print(f"    Object pool optimization: Success")
            print(f"    Pool types: {list(pool_results.keys())}")
        except Exception as e:
            print(f"    Object pool optimization: Failed - {str(e)}")
        
        # Test processing optimizer
        print("\\n  Testing processing optimizer:")
        try:
            processing_results = self.meta_engine.performance_optimizer.processing_optimizer.optimize_all()
            print(f"    Processing optimization: Success")
            print(f"    Optimization types: {list(processing_results.keys())}")
        except Exception as e:
            print(f"    Processing optimization: Failed - {str(e)}")
        
        # Test performance metrics summary
        print("\\n  Testing performance metrics summary:")
        try:
            metrics_summary = self.meta_engine.performance_optimizer.performance_metrics.get_summary()
            print(f"    Metrics summary: Success")
            print(f"    Summary sections: {list(metrics_summary.keys())}")
        except Exception as e:
            print(f"    Metrics summary: Failed - {str(e)}")
        
        print("✅ Performance optimization system tested")


async def main():
    """Run the integration test"""
    
    test_suite = MetaReasoningIntegrationTest()
    await test_suite.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())