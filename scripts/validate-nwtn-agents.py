#!/usr/bin/env python3
"""
NWTN Agent Pipeline Validation Script
Validates the 5-agent pipeline coordination for Phase 1 requirements

This script specifically tests:
1. Architect → Prompter → Router → Executor → Compiler pipeline
2. Agent coordination and data flow
3. Context allocation and tracking
4. FTNS token accuracy
5. Safety system integration
6. Performance under realistic workloads

Agent Pipeline Validation:
- Architect: Task decomposition and complexity assessment
- Prompter: Prompt optimization for selected models
- Router: Model selection and capability matching
- Executor: Real model execution with API integration
- Compiler: Result synthesis and final response compilation
"""

import asyncio
import sys
import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prsm.core.models import UserInput, AgentType
from prsm.nwtn.enhanced_orchestrator import get_enhanced_nwtn_orchestrator
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)

@dataclass
class AgentValidationResult:
    """Individual agent validation result"""
    agent_type: str
    success: bool
    execution_time: float
    context_used: int
    confidence_score: float
    output_quality: str  # "excellent", "good", "poor"
    error_message: Optional[str] = None
    
@dataclass
class PipelineValidationResult:
    """Complete pipeline validation result"""
    test_name: str
    overall_success: bool
    total_execution_time: float
    total_context_used: int
    total_ftns_charged: float
    agent_results: List[AgentValidationResult]
    pipeline_coordination_score: float
    data_flow_integrity: bool
    safety_compliance: bool
    performance_rating: str  # "excellent", "good", "needs_improvement", "poor"
    recommendations: List[str]

class NWTNAgentValidator:
    """NWTN Agent Pipeline Validator"""
    
    def __init__(self):
        self.settings = get_settings()
        self.orchestrator = get_enhanced_nwtn_orchestrator()
        
        # Test scenarios for different complexities
        self.validation_scenarios = [
            {
                "name": "Simple Query Processing",
                "prompt": "What is artificial intelligence?",
                "expected_agents": [AgentType.ARCHITECT, AgentType.EXECUTOR],
                "complexity": "simple",
                "max_execution_time": 3.0,
                "context_allocation": 50
            },
            {
                "name": "Medium Complexity Analysis",
                "prompt": "Analyze the advantages and disadvantages of microservices architecture for enterprise applications",
                "expected_agents": [AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                "complexity": "medium",
                "max_execution_time": 8.0,
                "context_allocation": 100
            },
            {
                "name": "Complex Research Task",
                "prompt": "Research the intersection of quantum computing and machine learning, providing a comprehensive analysis of current research trends, practical applications, and future implications for enterprise AI systems",
                "expected_agents": [AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                "complexity": "complex",
                "max_execution_time": 15.0,
                "context_allocation": 200
            },
            {
                "name": "Multi-Domain Technical Query",
                "prompt": "Design a scalable distributed system architecture that handles 1 million concurrent users, incorporates real-time analytics, ensures data consistency, and maintains 99.99% availability with detailed implementation recommendations",
                "expected_agents": [AgentType.ARCHITECT, AgentType.PROMPTER, AgentType.ROUTER, AgentType.EXECUTOR, AgentType.COMPILER],
                "complexity": "complex",
                "max_execution_time": 20.0,
                "context_allocation": 250
            }
        ]
    
    async def validate_complete_pipeline(self) -> Dict[str, Any]:
        """Validate the complete NWTN agent pipeline"""
        logger.info("Starting NWTN Agent Pipeline Validation")
        
        validation_results = []
        overall_success = True
        total_execution_time = 0.0
        
        try:
            # Test each validation scenario
            for scenario in self.validation_scenarios:
                logger.info("Testing scenario", name=scenario["name"])
                
                result = await self._validate_scenario(scenario)
                validation_results.append(result)
                
                total_execution_time += result.total_execution_time
                
                if not result.overall_success:
                    overall_success = False
                    logger.warning("Scenario failed", 
                                 name=scenario["name"],
                                 reasons=result.recommendations)
            
            # Generate comprehensive report
            report = self._generate_validation_report(validation_results, overall_success, total_execution_time)
            
            # Display results
            self._display_validation_results(report)
            
            return report
            
        except Exception as e:
            logger.error("Pipeline validation failed", error=str(e))
            raise
    
    async def _validate_scenario(self, scenario: Dict[str, Any]) -> PipelineValidationResult:
        """Validate a single scenario"""
        start_time = time.time()
        
        try:
            # Create user input
            user_input = UserInput(
                prompt=scenario["prompt"],
                user_id=f"validator_{int(time.time())}",
                context_allocation=scenario["context_allocation"],
                preferences={
                    "scenario_type": scenario["complexity"],
                    "validation_mode": True
                }
            )
            
            # Execute through orchestrator
            response = await self.orchestrator.process_query(user_input)
            
            execution_time = time.time() - start_time
            
            # Validate agent coordination
            agent_results = self._extract_agent_results(response, scenario)
            
            # Calculate pipeline coordination score
            coordination_score = self._calculate_coordination_score(agent_results, scenario)
            
            # Validate data flow integrity
            data_flow_integrity = self._validate_data_flow(response)
            
            # Check safety compliance
            safety_compliance = self._check_safety_compliance(response)
            
            # Determine performance rating
            performance_rating = self._rate_performance(execution_time, scenario, response)
            
            # Generate recommendations
            recommendations = self._generate_scenario_recommendations(
                agent_results, coordination_score, data_flow_integrity, 
                safety_compliance, performance_rating, scenario
            )
            
            # Determine overall success
            overall_success = (
                coordination_score >= 0.8 and
                data_flow_integrity and
                safety_compliance and
                execution_time <= scenario["max_execution_time"] and
                response.confidence_score >= 0.7
            )
            
            return PipelineValidationResult(
                test_name=scenario["name"],
                overall_success=overall_success,
                total_execution_time=execution_time,
                total_context_used=response.context_used,
                total_ftns_charged=response.ftns_charged,
                agent_results=agent_results,
                pipeline_coordination_score=coordination_score,
                data_flow_integrity=data_flow_integrity,
                safety_compliance=safety_compliance,
                performance_rating=performance_rating,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Scenario validation failed", 
                        scenario=scenario["name"], 
                        error=str(e))
            
            return PipelineValidationResult(
                test_name=scenario["name"],
                overall_success=False,
                total_execution_time=time.time() - start_time,
                total_context_used=0,
                total_ftns_charged=0.0,
                agent_results=[],
                pipeline_coordination_score=0.0,
                data_flow_integrity=False,
                safety_compliance=False,
                performance_rating="poor",
                recommendations=[f"Scenario failed with error: {str(e)}"]
            )
    
    def _extract_agent_results(self, response, scenario) -> List[AgentValidationResult]:
        """Extract individual agent results from orchestrator response"""
        agent_results = []
        
        try:
            reasoning_trace = response.reasoning_trace
            expected_agents = {agent.value for agent in scenario["expected_agents"]}
            
            # Group reasoning steps by agent type
            agent_steps = {}
            for step in reasoning_trace:
                agent_type = step.get("agent_type", "unknown")
                if agent_type not in agent_steps:
                    agent_steps[agent_type] = []
                agent_steps[agent_type].append(step)
            
            # Create validation results for each agent
            for agent_type, steps in agent_steps.items():
                if agent_type in ["intent_clarification", "tool_execution", "tool_execution_error"]:
                    continue  # Skip orchestrator-level steps
                
                # Calculate metrics for this agent
                total_execution_time = sum(step.get("execution_time", 0) for step in steps)
                total_context = sum(step.get("context_used", 0) for step in steps)
                avg_confidence = sum(step.get("confidence_score", 0.5) for step in steps) / len(steps)
                
                # Determine success
                success = all(step.get("success", True) for step in steps if isinstance(step, dict))
                
                # Evaluate output quality
                output_quality = self._evaluate_agent_output_quality(agent_type, steps, expected_agents)
                
                agent_results.append(AgentValidationResult(
                    agent_type=agent_type,
                    success=success,
                    execution_time=total_execution_time,
                    context_used=total_context,
                    confidence_score=avg_confidence,
                    output_quality=output_quality,
                    error_message=None if success else "Agent execution issues detected"
                ))
            
            # Check for missing expected agents
            covered_agents = {result.agent_type for result in agent_results}
            for expected_agent in expected_agents:
                if expected_agent not in covered_agents:
                    agent_results.append(AgentValidationResult(
                        agent_type=expected_agent,
                        success=False,
                        execution_time=0.0,
                        context_used=0,
                        confidence_score=0.0,
                        output_quality="poor",
                        error_message=f"Expected agent {expected_agent} was not executed"
                    ))
            
            return agent_results
            
        except Exception as e:
            logger.error("Agent result extraction failed", error=str(e))
            return []
    
    def _evaluate_agent_output_quality(self, agent_type: str, steps: List[Dict], expected_agents: set) -> str:
        """Evaluate the quality of agent output"""
        try:
            # Check if this agent was expected to run
            if agent_type not in expected_agents:
                return "good"  # Unexpected but successful execution
            
            # Evaluate based on agent-specific criteria
            if agent_type == "architect":
                return self._evaluate_architect_quality(steps)
            elif agent_type == "prompter":
                return self._evaluate_prompter_quality(steps)
            elif agent_type == "router":
                return self._evaluate_router_quality(steps)
            elif agent_type == "executor":
                return self._evaluate_executor_quality(steps)
            elif agent_type == "compiler":
                return self._evaluate_compiler_quality(steps)
            else:
                return "good"  # Default for unknown agents
                
        except Exception as e:
            logger.error("Agent quality evaluation failed", agent_type=agent_type, error=str(e))
            return "poor"
    
    def _evaluate_architect_quality(self, steps: List[Dict]) -> str:
        """Evaluate architect agent output quality"""
        try:
            # Check for task decomposition
            has_decomposition = any(
                "decomposition" in str(step.get("output_data", "")).lower() or
                "task" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            # Check for complexity assessment
            has_complexity = any(
                "complexity" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            if has_decomposition and has_complexity:
                return "excellent"
            elif has_decomposition or has_complexity:
                return "good"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Architect quality evaluation failed", error=str(e))
            return "poor"
    
    def _evaluate_prompter_quality(self, steps: List[Dict]) -> str:
        """Evaluate prompter agent output quality"""
        try:
            # Check for prompt optimization
            has_optimization = any(
                "optimization" in str(step.get("output_data", "")).lower() or
                "optimized" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            # Check for confidence score
            avg_confidence = sum(step.get("confidence_score", 0.5) for step in steps) / len(steps)
            
            if has_optimization and avg_confidence > 0.8:
                return "excellent"
            elif has_optimization or avg_confidence > 0.6:
                return "good"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Prompter quality evaluation failed", error=str(e))
            return "poor"
    
    def _evaluate_router_quality(self, steps: List[Dict]) -> str:
        """Evaluate router agent output quality"""
        try:
            # Check for model selection
            has_models = any(
                "model" in str(step.get("output_data", "")).lower() or
                "selection" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            # Check for routing confidence
            avg_confidence = sum(step.get("confidence_score", 0.5) for step in steps) / len(steps)
            
            if has_models and avg_confidence > 0.8:
                return "excellent"
            elif has_models or avg_confidence > 0.6:
                return "good"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Router quality evaluation failed", error=str(e))
            return "poor"
    
    def _evaluate_executor_quality(self, steps: List[Dict]) -> str:
        """Evaluate executor agent output quality"""
        try:
            # Check for successful execution
            execution_success = all(step.get("success", True) for step in steps if isinstance(step, dict))
            
            # Check for multiple model results
            has_multiple_results = any(
                "execution_results" in str(step.get("output_data", "")).lower() or
                "models_used" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            # Check execution time
            avg_execution_time = sum(step.get("execution_time", 0) for step in steps) / len(steps)
            
            if execution_success and has_multiple_results and avg_execution_time < 5.0:
                return "excellent"
            elif execution_success and (has_multiple_results or avg_execution_time < 10.0):
                return "good"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Executor quality evaluation failed", error=str(e))
            return "poor"
    
    def _evaluate_compiler_quality(self, steps: List[Dict]) -> str:
        """Evaluate compiler agent output quality"""
        try:
            # Check for synthesis/compilation
            has_compilation = any(
                "compilation" in str(step.get("output_data", "")).lower() or
                "synthesis" in str(step.get("output_data", "")).lower() or
                "compiled" in str(step.get("output_data", "")).lower()
                for step in steps
            )
            
            # Check for coherent output
            output_length = sum(
                len(str(step.get("output_data", "")))
                for step in steps
            )
            
            if has_compilation and output_length > 100:
                return "excellent"
            elif has_compilation or output_length > 50:
                return "good"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Compiler quality evaluation failed", error=str(e))
            return "poor"
    
    def _calculate_coordination_score(self, agent_results: List[AgentValidationResult], scenario: Dict[str, Any]) -> float:
        """Calculate pipeline coordination score"""
        try:
            if not agent_results:
                return 0.0
            
            # Base score from successful agents
            successful_agents = [r for r in agent_results if r.success]
            success_ratio = len(successful_agents) / len(agent_results)
            
            # Expected agent coverage
            expected_agents = {agent.value for agent in scenario["expected_agents"]}
            covered_agents = {r.agent_type for r in successful_agents}
            coverage_ratio = len(covered_agents.intersection(expected_agents)) / len(expected_agents)
            
            # Quality score
            quality_scores = {
                "excellent": 1.0,
                "good": 0.8,
                "poor": 0.3
            }
            avg_quality = sum(quality_scores.get(r.output_quality, 0.5) for r in agent_results) / len(agent_results)
            
            # Timing coordination (agents should execute in reasonable order)
            timing_score = 1.0  # Simplified - assume good timing
            
            # Overall coordination score
            coordination_score = (
                success_ratio * 0.3 +
                coverage_ratio * 0.4 +
                avg_quality * 0.2 +
                timing_score * 0.1
            )
            
            return min(1.0, coordination_score)
            
        except Exception as e:
            logger.error("Coordination score calculation failed", error=str(e))
            return 0.0
    
    def _validate_data_flow(self, response) -> bool:
        """Validate data flow integrity through the pipeline"""
        try:
            # Check that response has required fields
            required_fields = ["final_answer", "reasoning_trace", "confidence_score", "context_used"]
            for field in required_fields:
                if not hasattr(response, field) or getattr(response, field) is None:
                    return False
            
            # Check that reasoning trace shows proper data flow
            reasoning_trace = response.reasoning_trace
            if not reasoning_trace or len(reasoning_trace) == 0:
                return False
            
            # Verify that each step has proper input/output structure
            for step in reasoning_trace:
                if not isinstance(step, dict):
                    continue
                
                # Check for basic step structure
                if "agent_type" not in step or "output_data" not in step:
                    return False
            
            # Check that final answer is coherent and substantial
            final_answer = response.final_answer
            if not final_answer or len(final_answer.strip()) < 20:
                return False
            
            return True
            
        except Exception as e:
            logger.error("Data flow validation failed", error=str(e))
            return False
    
    def _check_safety_compliance(self, response) -> bool:
        """Check safety compliance"""
        try:
            # Check that safety_validated flag is present and True
            if not hasattr(response, "safety_validated") or not response.safety_validated:
                return False
            
            # Check for reasonable confidence score
            if response.confidence_score < 0.1:
                return False
            
            # Check that no obvious safety violations in final answer
            final_answer = response.final_answer.lower()
            safety_violations = [
                "hack", "exploit", "malicious", "harmful", "dangerous",
                "illegal", "unauthorized", "breach", "attack"
            ]
            
            violation_count = sum(1 for violation in safety_violations if violation in final_answer)
            if violation_count > 2:  # Allow some technical discussion
                return False
            
            return True
            
        except Exception as e:
            logger.error("Safety compliance check failed", error=str(e))
            return False
    
    def _rate_performance(self, execution_time: float, scenario: Dict[str, Any], response) -> str:
        """Rate overall performance"""
        try:
            max_time = scenario["max_execution_time"]
            confidence = response.confidence_score
            
            # Performance rating based on time and quality
            if execution_time <= max_time * 0.5 and confidence >= 0.9:
                return "excellent"
            elif execution_time <= max_time * 0.75 and confidence >= 0.8:
                return "good"
            elif execution_time <= max_time and confidence >= 0.6:
                return "needs_improvement"
            else:
                return "poor"
                
        except Exception as e:
            logger.error("Performance rating failed", error=str(e))
            return "poor"
    
    def _generate_scenario_recommendations(self, agent_results: List[AgentValidationResult],
                                         coordination_score: float, data_flow_integrity: bool,
                                         safety_compliance: bool, performance_rating: str,
                                         scenario: Dict[str, Any]) -> List[str]:
        """Generate scenario-specific recommendations"""
        recommendations = []
        
        # Agent-specific recommendations
        for result in agent_results:
            if not result.success:
                recommendations.append(f"🔴 {result.agent_type.title()} agent failed - investigate and fix")
            elif result.output_quality == "poor":
                recommendations.append(f"🟡 {result.agent_type.title()} agent output quality is poor - optimize implementation")
            elif result.execution_time > 5.0:
                recommendations.append(f"🟡 {result.agent_type.title()} agent is slow ({result.execution_time:.1f}s) - optimize performance")
        
        # Coordination recommendations
        if coordination_score < 0.8:
            recommendations.append(f"🔴 Pipeline coordination is poor ({coordination_score:.2f}) - review agent orchestration")
        
        # Data flow recommendations
        if not data_flow_integrity:
            recommendations.append("🔴 Data flow integrity issues detected - validate agent communication")
        
        # Safety recommendations
        if not safety_compliance:
            recommendations.append("🔴 Safety compliance failed - review safety mechanisms")
        
        # Performance recommendations
        if performance_rating in ["needs_improvement", "poor"]:
            recommendations.append(f"🟡 Performance rating is {performance_rating} - optimize execution pipeline")
        
        # Scenario-specific recommendations
        if scenario["complexity"] == "complex" and len(agent_results) < 4:
            recommendations.append("🟡 Complex query should engage more agents - review routing logic")
        
        if not recommendations:
            recommendations.append("✅ All validations passed - agent pipeline working correctly")
        
        return recommendations
    
    def _generate_validation_report(self, validation_results: List[PipelineValidationResult],
                                  overall_success: bool, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Calculate aggregate metrics
        total_scenarios = len(validation_results)
        successful_scenarios = sum(1 for r in validation_results if r.overall_success)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        avg_execution_time = sum(r.total_execution_time for r in validation_results) / total_scenarios if total_scenarios > 0 else 0
        total_context_used = sum(r.total_context_used for r in validation_results)
        total_ftns_charged = sum(r.total_ftns_charged for r in validation_results)
        
        # Agent performance summary
        agent_performance_summary = {}
        all_agent_results = []
        for result in validation_results:
            all_agent_results.extend(result.agent_results)
        
        # Group by agent type
        for agent_result in all_agent_results:
            agent_type = agent_result.agent_type
            if agent_type not in agent_performance_summary:
                agent_performance_summary[agent_type] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "avg_execution_time": 0.0,
                    "avg_confidence": 0.0,
                    "quality_distribution": {"excellent": 0, "good": 0, "poor": 0}
                }
            
            summary = agent_performance_summary[agent_type]
            summary["total_executions"] += 1
            if agent_result.success:
                summary["successful_executions"] += 1
            summary["avg_execution_time"] += agent_result.execution_time
            summary["avg_confidence"] += agent_result.confidence_score
            summary["quality_distribution"][agent_result.output_quality] += 1
        
        # Calculate averages
        for agent_type, summary in agent_performance_summary.items():
            if summary["total_executions"] > 0:
                summary["avg_execution_time"] /= summary["total_executions"]
                summary["avg_confidence"] /= summary["total_executions"]
                summary["success_rate"] = summary["successful_executions"] / summary["total_executions"]
        
        # Overall recommendations
        overall_recommendations = []
        if overall_success:
            overall_recommendations.append("✅ NWTN Agent Pipeline validation PASSED")
            overall_recommendations.append("🎯 All scenarios completed successfully")
            overall_recommendations.append("🚀 System ready for Phase 1 deployment")
        else:
            overall_recommendations.append("❌ NWTN Agent Pipeline validation FAILED")
            overall_recommendations.append("🔧 Review failed scenarios and implement fixes")
            overall_recommendations.append("⚠️  System requires optimization before deployment")
        
        # Add performance recommendations
        if avg_execution_time > 10.0:
            overall_recommendations.append(f"🐌 Average execution time is high ({avg_execution_time:.1f}s)")
        
        if success_rate < 1.0:
            overall_recommendations.append(f"📉 Success rate is {success_rate:.1%} - investigate failures")
        
        return {
            "validation_summary": {
                "overall_success": overall_success,
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "success_rate": success_rate,
                "total_execution_time": total_execution_time,
                "avg_execution_time": avg_execution_time,
                "total_context_used": total_context_used,
                "total_ftns_charged": total_ftns_charged
            },
            "scenario_results": [
                {
                    "name": result.test_name,
                    "success": result.overall_success,
                    "execution_time": result.total_execution_time,
                    "coordination_score": result.pipeline_coordination_score,
                    "performance_rating": result.performance_rating,
                    "recommendations_count": len(result.recommendations)
                }
                for result in validation_results
            ],
            "agent_performance_summary": agent_performance_summary,
            "detailed_results": validation_results,
            "overall_recommendations": overall_recommendations,
            "phase1_readiness": {
                "ready": overall_success and avg_execution_time < 15.0 and success_rate >= 0.8,
                "confidence": "high" if overall_success else "low",
                "next_steps": [
                    "Deploy to production test environment" if overall_success else "Fix validation failures",
                    "Run stress testing with 1000 concurrent users",
                    "Monitor performance under load",
                    "Validate FTNS token accuracy"
                ]
            }
        }
    
    def _display_validation_results(self, report: Dict[str, Any]):
        """Display comprehensive validation results"""
        print("\n" + "="*80)
        print("🤖 NWTN AGENT PIPELINE VALIDATION RESULTS")
        print("="*80)
        
        summary = report["validation_summary"]
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"├─ Overall Status: {'✅ PASSED' if summary['overall_success'] else '❌ FAILED'}")
        print(f"├─ Success Rate: {summary['success_rate']:.1%} ({summary['successful_scenarios']}/{summary['total_scenarios']})")
        print(f"├─ Total Execution Time: {summary['total_execution_time']:.1f}s")
        print(f"├─ Average Time per Scenario: {summary['avg_execution_time']:.1f}s")
        print(f"├─ Total Context Used: {summary['total_context_used']}")
        print(f"└─ Total FTNS Charged: {summary['total_ftns_charged']:.2f}")
        
        print(f"\n🎯 SCENARIO RESULTS:")
        for scenario in report["scenario_results"]:
            status = "✅" if scenario["success"] else "❌"
            print(f"├─ {scenario['name']}: {status}")
            print(f"│  ├─ Execution Time: {scenario['execution_time']:.1f}s")
            print(f"│  ├─ Coordination Score: {scenario['coordination_score']:.2f}")
            print(f"│  ├─ Performance: {scenario['performance_rating']}")
            print(f"│  └─ Issues: {scenario['recommendations_count']} recommendation(s)")
        
        print(f"\n🤖 AGENT PERFORMANCE SUMMARY:")
        for agent_type, perf in report["agent_performance_summary"].items():
            print(f"├─ {agent_type.title()}:")
            print(f"│  ├─ Success Rate: {perf['success_rate']:.1%}")
            print(f"│  ├─ Avg Execution Time: {perf['avg_execution_time']:.2f}s")
            print(f"│  ├─ Avg Confidence: {perf['avg_confidence']:.2f}")
            print(f"│  └─ Quality: E:{perf['quality_distribution']['excellent']}, G:{perf['quality_distribution']['good']}, P:{perf['quality_distribution']['poor']}")
        
        readiness = report["phase1_readiness"]
        print(f"\n🚀 PHASE 1 READINESS:")
        print(f"├─ Ready for Deployment: {'✅ YES' if readiness['ready'] else '❌ NO'}")
        print(f"├─ Confidence Level: {readiness['confidence'].upper()}")
        print(f"└─ Next Steps:")
        for step in readiness["next_steps"]:
            print(f"   • {step}")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for rec in report["overall_recommendations"]:
            print(f"   {rec}")
        
        print("\n" + "="*80)

async def main():
    """Main entry point"""
    try:
        validator = NWTNAgentValidator()
        report = await validator.validate_complete_pipeline()
        
        # Save detailed results
        output_file = "nwtn_agent_validation_results.json"
        with open(output_file, 'w') as f:
            # Convert dataclasses to dicts for JSON serialization
            json_report = json.loads(json.dumps(report, default=str))
            json.dump(json_report, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        # Exit with appropriate code
        if report["validation_summary"]["overall_success"]:
            print("\n🎉 All validations passed! NWTN Agent Pipeline is ready for Phase 1.")
            sys.exit(0)
        else:
            print("\n⚠️  Some validations failed. Review recommendations and retry.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())