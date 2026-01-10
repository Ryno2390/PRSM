"""
PRSM Evolution Orchestrator
Coordinates A/B testing, validates improvements, and propagates updates to the network
"""

import asyncio
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from prsm.core.config import settings
from prsm.core.models import (
    ImprovementProposal, ProposalStatus, TestResults, UpdatePackage,
    PerformanceMetric, MetricType
)
from prsm.core.safety.monitor import SafetyMonitor
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork
from ..federation.consensus import DistributedConsensus
from prsm.economy.tokenomics.ftns_service import get_ftns_service
from prsm.compute.distillation.orchestrator import get_distillation_orchestrator
from prsm.compute.distillation.models import DistillationRequest, ModelSize, OptimizationTarget


# === Evolution Configuration ===

# A/B Testing settings
AB_TEST_DURATION_HOURS = float(getattr(settings, "PRSM_AB_TEST_DURATION", 48.0))
MIN_SAMPLE_SIZE = int(getattr(settings, "PRSM_MIN_AB_SAMPLE_SIZE", 100))
SIGNIFICANCE_THRESHOLD = float(getattr(settings, "PRSM_SIGNIFICANCE_THRESHOLD", 0.05))
MINIMUM_IMPROVEMENT_THRESHOLD = float(getattr(settings, "PRSM_MIN_IMPROVEMENT", 0.02))  # 2%

# Implementation settings
ROLLOUT_PERCENTAGE_STEPS = [10, 25, 50, 75, 100]
MONITORING_GRACE_PERIOD_HOURS = float(getattr(settings, "PRSM_MONITORING_GRACE", 2.0))
ROLLBACK_TRIGGER_THRESHOLD = float(getattr(settings, "PRSM_ROLLBACK_THRESHOLD", 0.1))  # 10% degradation

# Network propagation settings
CONSENSUS_TIMEOUT_SECONDS = float(getattr(settings, "PRSM_CONSENSUS_TIMEOUT", 300.0))
NETWORK_UPDATE_BATCH_SIZE = int(getattr(settings, "PRSM_UPDATE_BATCH_SIZE", 10))
PROPAGATION_RETRY_ATTEMPTS = int(getattr(settings, "PRSM_PROPAGATION_RETRIES", 3))


class EvolutionOrchestrator:
    """
    Orchestrates the evolution and improvement of PRSM systems through
    controlled A/B testing, validation, and network-wide propagation
    """
    
    def __init__(self):
        # Component integration
        self.safety_monitor = SafetyMonitor()
        self.circuit_breaker = CircuitBreakerNetwork()
        self.consensus = DistributedConsensus()
        
        # Evolution tracking
        self.active_tests: Dict[UUID, Dict[str, Any]] = {}
        self.completed_tests: Dict[UUID, TestResults] = {}
        self.implemented_improvements: Dict[UUID, Dict[str, Any]] = {}
        self.network_updates: Dict[UUID, UpdatePackage] = {}
        
        # Performance tracking
        self.evolution_stats = {
            "tests_initiated": 0,
            "tests_completed": 0,
            "improvements_implemented": 0,
            "improvements_rolled_back": 0,
            "network_updates_propagated": 0,
            "consensus_validations": 0,
            "total_improvement_value": 0.0
        }
        
        # Test configuration
        self.test_configurations = {
            "performance": {
                "metrics": ["latency", "throughput", "resource_usage"],
                "duration_multiplier": 1.0,
                "sample_size_multiplier": 1.0
            },
            "accuracy": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "duration_multiplier": 1.5,
                "sample_size_multiplier": 2.0
            },
            "safety": {
                "metrics": ["safety_score", "threat_detection", "false_positive_rate"],
                "duration_multiplier": 2.0,
                "sample_size_multiplier": 3.0
            }
        }
        
        # Synchronization
        self._tests_lock = asyncio.Lock()
        self._implementations_lock = asyncio.Lock()
        
        print("🧬 EvolutionOrchestrator initialized")
    
    
    async def coordinate_a_b_testing(self, proposals: List[ImprovementProposal]) -> TestResults:
        """
        Coordinate A/B testing for improvement proposals
        
        Args:
            proposals: List of improvement proposals to test
            
        Returns:
            Comprehensive test results for all proposals
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Validate proposals for testing
            testable_proposals = await self._validate_proposals_for_testing(proposals)
            
            if not testable_proposals:
                raise ValueError("No proposals are suitable for A/B testing")
            
            # Design test configuration
            test_config = await self._design_test_configuration(testable_proposals)
            
            # Initialize test environment
            test_environment = await self._initialize_test_environment(test_config)
            
            # Execute parallel tests
            test_results = await self._execute_parallel_tests(testable_proposals, test_environment)
            
            # Analyze results
            analysis_results = await self._analyze_test_results(test_results)
            
            # Generate recommendations
            recommendations = await self._generate_test_recommendations(analysis_results)
            
            # Calculate statistical significance
            statistical_analysis = await self._perform_statistical_analysis(test_results)
            
            # Create comprehensive test results
            end_time = datetime.now(timezone.utc)
            test_duration = (end_time - start_time).total_seconds()
            
            comprehensive_results = TestResults(
                test_id=uuid4(),
                proposals_tested=[p.proposal_id for p in testable_proposals],
                test_duration=test_duration / 3600.0,  # Convert seconds to hours
                sample_size=test_config.get("min_sample_size", MIN_SAMPLE_SIZE),
                control_group_performance=test_environment.get("control_baseline", {}),
                treatment_group_performance={},  # Will be simplified
                statistical_significance={},  # Will be simplified
                confidence_interval={}
            )
            
            # Store test results
            async with self._tests_lock:
                for proposal in testable_proposals:
                    self.completed_tests[proposal.proposal_id] = comprehensive_results
            
            # Update stats
            self.evolution_stats["tests_completed"] += 1
            
            print(f"🧪 A/B testing completed for {len(testable_proposals)} proposals in {test_duration:.1f}s")
            
            return comprehensive_results
            
        except Exception as e:
            print(f"❌ Error coordinating A/B testing: {str(e)}")
            return TestResults(
                test_id=uuid4(),
                proposals_tested=[p.proposal_id for p in proposals],
                test_duration=0.0,
                sample_size=0,
                control_group_performance={},
                treatment_group_performance={},
                statistical_significance={},
                confidence_interval={}
            )
    
    
    async def implement_validated_improvements(self, approved_proposals: List[ImprovementProposal]) -> Dict[str, Any]:
        """
        Implement validated improvements with gradual rollout and monitoring
        
        Args:
            approved_proposals: List of approved improvement proposals
            
        Returns:
            Implementation results and status
        """
        try:
            implementation_results = {
                "successful_implementations": [],
                "failed_implementations": [],
                "rollback_implementations": [],
                "total_value_added": 0.0,
                "implementation_summary": {}
            }
            
            # Sort proposals by priority and safety score
            sorted_proposals = sorted(
                approved_proposals, 
                key=lambda p: (p.priority_score, p.safety_check.safety_score if p.safety_check else 0.5),
                reverse=True
            )
            
            # Implement each proposal with gradual rollout
            for proposal in sorted_proposals:
                implementation_result = await self._implement_single_proposal(proposal)
                
                if implementation_result["success"]:
                    implementation_results["successful_implementations"].append(proposal.proposal_id)
                    implementation_results["total_value_added"] += implementation_result.get("value_added", 0.0)
                elif implementation_result.get("rolled_back"):
                    implementation_results["rollback_implementations"].append(proposal.proposal_id)
                else:
                    implementation_results["failed_implementations"].append(proposal.proposal_id)
                
                # Store detailed results
                implementation_results["implementation_summary"][str(proposal.proposal_id)] = implementation_result
            
            # Update global stats
            self.evolution_stats["improvements_implemented"] += len(implementation_results["successful_implementations"])
            self.evolution_stats["improvements_rolled_back"] += len(implementation_results["rollback_implementations"])
            self.evolution_stats["total_improvement_value"] += implementation_results["total_value_added"]
            
            print(f"✅ Implemented {len(implementation_results['successful_implementations'])}/{len(approved_proposals)} improvements")
            
            return implementation_results
            
        except Exception as e:
            print(f"❌ Error implementing improvements: {str(e)}")
            return {
                "successful_implementations": [],
                "failed_implementations": [str(p.proposal_id) for p in approved_proposals],
                "rollback_implementations": [],
                "total_value_added": 0.0,
                "implementation_summary": {"error": str(e)}
            }
    
    
    async def propagate_updates_to_network(self, update_package: UpdatePackage) -> Dict[str, Any]:
        """
        Propagate approved updates to the distributed network
        
        Args:
            update_package: Package containing updates to propagate
            
        Returns:
            Propagation results and network status
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Validate update package
            validation_result = await self._validate_update_package(update_package)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid update package: {validation_result['reason']}")
            
            # Prepare network update
            network_nodes = await self._discover_network_nodes()
            
            # Achieve consensus on the update
            consensus_result = await self._achieve_update_consensus(update_package, network_nodes)
            
            if not consensus_result["consensus_achieved"]:
                raise ValueError("Failed to achieve network consensus for update")
            
            # Propagate updates in batches
            propagation_results = await self._propagate_updates_in_batches(
                update_package, network_nodes, consensus_result
            )
            
            # Monitor propagation status
            monitoring_results = await self._monitor_propagation_health(
                update_package, propagation_results
            )
            
            # Handle any propagation failures
            failure_handling = await self._handle_propagation_failures(
                propagation_results, monitoring_results
            )
            
            # Calculate propagation success rate
            total_nodes = len(network_nodes)
            successful_nodes = len(propagation_results.get("successful_nodes", []))
            success_rate = successful_nodes / max(1, total_nodes)
            
            # Create propagation results
            end_time = datetime.now(timezone.utc)
            propagation_duration = (end_time - start_time).total_seconds()
            
            results = {
                "update_id": update_package.update_id,
                "propagation_duration": propagation_duration,
                "total_nodes": total_nodes,
                "successful_nodes": successful_nodes,
                "success_rate": success_rate,
                "consensus_details": consensus_result,
                "propagation_details": propagation_results,
                "monitoring_results": monitoring_results,
                "failure_handling": failure_handling
            }
            
            # Store network update
            async with self._implementations_lock:
                self.network_updates[update_package.update_id] = update_package
            
            # Update stats
            self.evolution_stats["network_updates_propagated"] += 1
            self.evolution_stats["consensus_validations"] += 1
            
            print(f"🌐 Propagated update {update_package.update_id} to {successful_nodes}/{total_nodes} nodes ({success_rate:.1%})")
            
            return results
            
        except Exception as e:
            print(f"❌ Error propagating updates to network: {str(e)}")
            return {
                "update_id": update_package.update_id,
                "propagation_duration": 0.0,
                "total_nodes": 0,
                "successful_nodes": 0,
                "success_rate": 0.0,
                "error": str(e)
            }
    
    
    async def get_active_tests(self) -> Dict[UUID, Dict[str, Any]]:
        """Get currently active A/B tests"""
        async with self._tests_lock:
            return dict(self.active_tests)
    
    
    async def get_test_results(self, test_id: Optional[UUID] = None) -> Dict[UUID, TestResults]:
        """Get test results (all or specific test)"""
        async with self._tests_lock:
            if test_id:
                return {test_id: self.completed_tests.get(test_id)} if test_id in self.completed_tests else {}
            return dict(self.completed_tests)
    
    
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get current evolution statistics"""
        async with self._tests_lock, self._implementations_lock:
            return {
                **self.evolution_stats,
                "active_tests_count": len(self.active_tests),
                "completed_tests_count": len(self.completed_tests),
                "implemented_improvements_count": len(self.implemented_improvements),
                "network_updates_count": len(self.network_updates),
                "average_improvement_value": (
                    self.evolution_stats["total_improvement_value"] / 
                    max(1, self.evolution_stats["improvements_implemented"])
                ),
                "configuration": {
                    "ab_test_duration_hours": AB_TEST_DURATION_HOURS,
                    "min_sample_size": MIN_SAMPLE_SIZE,
                    "significance_threshold": SIGNIFICANCE_THRESHOLD,
                    "minimum_improvement": MINIMUM_IMPROVEMENT_THRESHOLD
                }
            }
    
    
    # === Private Helper Methods ===
    
    async def _validate_proposals_for_testing(self, proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
        """Validate proposals for A/B testing readiness"""
        testable_proposals = []
        
        for proposal in proposals:
            # Check if proposal has safety validation
            if not proposal.safety_check:
                print(f"⚠️ Proposal {proposal.proposal_id} lacks safety validation")
                continue
            
            # Check safety score
            if proposal.safety_check.safety_score < 0.6:
                print(f"⚠️ Proposal {proposal.proposal_id} has low safety score: {proposal.safety_check.safety_score}")
                continue
            
            # Check if simulation results exist
            if not proposal.simulation_result:
                print(f"⚠️ Proposal {proposal.proposal_id} lacks simulation results")
                continue
            
            # Check simulation confidence
            if proposal.simulation_result.confidence_score < 0.5:
                print(f"⚠️ Proposal {proposal.proposal_id} has low simulation confidence: {proposal.simulation_result.confidence_score}")
                continue
            
            testable_proposals.append(proposal)
        
        return testable_proposals
    
    
    async def _design_test_configuration(self, proposals: List[ImprovementProposal]) -> Dict[str, Any]:
        """Design optimal test configuration for proposals"""
        # Determine test category based on proposal types
        proposal_types = [p.improvement_type for p in proposals]
        
        # Select appropriate test configuration
        if any("safety" in str(ptype).lower() for ptype in proposal_types):
            config_type = "safety"
        elif any("accuracy" in str(ptype).lower() or "training" in str(ptype).lower() for ptype in proposal_types):
            config_type = "accuracy"
        else:
            config_type = "performance"
        
        base_config = self.test_configurations[config_type]
        
        return {
            "test_type": config_type,
            "metrics_to_track": base_config["metrics"],
            "test_duration_hours": AB_TEST_DURATION_HOURS * base_config["duration_multiplier"],
            "min_sample_size": int(MIN_SAMPLE_SIZE * base_config["sample_size_multiplier"]),
            "control_group_percentage": 50,
            "treatment_group_percentage": 50,
            "significance_threshold": SIGNIFICANCE_THRESHOLD,
            "monitoring_frequency_minutes": 15
        }
    
    
    async def _initialize_test_environment(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize A/B test environment"""
        # Create baseline performance snapshot
        control_baseline = {}
        for metric in test_config["metrics_to_track"]:
            # Simulate baseline performance
            if metric == "latency":
                control_baseline[metric] = random.uniform(100, 300)
            elif metric == "throughput":
                control_baseline[metric] = random.uniform(50, 150)
            elif metric == "accuracy":
                control_baseline[metric] = random.uniform(0.8, 0.95)
            elif metric == "resource_usage":
                control_baseline[metric] = random.uniform(0.4, 0.7)
            elif metric == "safety_score":
                control_baseline[metric] = random.uniform(0.7, 0.9)
            else:
                control_baseline[metric] = random.uniform(0.5, 1.0)
        
        return {
            "control_baseline": control_baseline,
            "test_start_time": datetime.now(timezone.utc),
            "monitoring_checkpoints": [],
            "safety_monitors_active": True
        }
    
    
    async def _execute_parallel_tests(self, proposals: List[ImprovementProposal], 
                                    test_environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel A/B tests for multiple proposals"""
        test_results = {}
        
        # Simulate test execution for each proposal
        for proposal in proposals:
            # Simulate treatment group performance
            treatment_performance = {}
            control_baseline = test_environment["control_baseline"]
            
            for metric, baseline_value in control_baseline.items():
                # Apply expected benefits from proposal
                if metric in proposal.expected_benefits:
                    benefit = proposal.expected_benefits[metric]
                    # Add realistic variance
                    variance = random.uniform(-0.1, 0.1)
                    if metric in ["latency", "error_rate", "resource_usage"]:
                        # Lower is better
                        improvement_factor = 1 + benefit + variance
                        treatment_value = baseline_value * improvement_factor
                    else:
                        # Higher is better
                        improvement_factor = 1 + benefit + variance
                        treatment_value = baseline_value * improvement_factor
                else:
                    # No significant change
                    treatment_value = baseline_value * random.uniform(0.98, 1.02)
                
                treatment_performance[metric] = treatment_value
            
            # Simulate sample collection
            sample_size = random.randint(MIN_SAMPLE_SIZE, MIN_SAMPLE_SIZE * 3)
            
            test_results[str(proposal.proposal_id)] = {
                "control_performance": control_baseline,
                "treatment_performance": treatment_performance,
                "sample_size": sample_size,
                "test_duration_actual": AB_TEST_DURATION_HOURS * random.uniform(0.9, 1.1),
                "data_quality_score": random.uniform(0.8, 1.0)
            }
        
        return test_results
    
    
    async def _analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze A/B test results for statistical significance"""
        analysis = {
            "treatment_performance": {},
            "performance_improvements": {},
            "statistical_tests": {},
            "effect_sizes": {}
        }
        
        for proposal_id, results in test_results.items():
            control = results["control_performance"]
            treatment = results["treatment_performance"]
            
            # Calculate improvements
            improvements = {}
            effect_sizes = {}
            
            for metric in control.keys():
                control_val = control[metric]
                treatment_val = treatment[metric]
                
                if metric in ["latency", "error_rate", "resource_usage"]:
                    # Lower is better
                    improvement = (control_val - treatment_val) / control_val
                else:
                    # Higher is better
                    improvement = (treatment_val - control_val) / control_val
                
                improvements[metric] = improvement
                # Simulate effect size calculation
                effect_sizes[metric] = abs(improvement) / random.uniform(0.1, 0.3)
            
            analysis["treatment_performance"][proposal_id] = treatment
            analysis["performance_improvements"][proposal_id] = improvements
            analysis["effect_sizes"][proposal_id] = effect_sizes
        
        return analysis
    
    
    async def _generate_test_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test analysis"""
        recommendations = []
        
        improvements = analysis_results.get("performance_improvements", {})
        effect_sizes = analysis_results.get("effect_sizes", {})
        
        for proposal_id, metrics in improvements.items():
            # Check for significant improvements
            significant_improvements = []
            for metric, improvement in metrics.items():
                if improvement > MINIMUM_IMPROVEMENT_THRESHOLD:
                    effect_size = effect_sizes.get(proposal_id, {}).get(metric, 0)
                    if effect_size > 0.2:  # Medium effect size
                        significant_improvements.append(f"{metric}: +{improvement:.1%}")
            
            if significant_improvements:
                recommendations.append(
                    f"Proposal {proposal_id}: Recommend implementation - significant improvements in {', '.join(significant_improvements)}"
                )
            else:
                recommendations.append(
                    f"Proposal {proposal_id}: Insufficient evidence for implementation"
                )
        
        return recommendations
    
    
    async def _perform_statistical_analysis(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on test results"""
        statistical_analysis = {
            "significance": {},
            "confidence_intervals": {},
            "power_analysis": {}
        }
        
        for proposal_id, results in test_results.items():
            # Simulate statistical significance testing
            sample_size = results["sample_size"]
            data_quality = results["data_quality_score"]
            
            # Calculate p-values for each metric
            significance = {}
            confidence_intervals = {}
            
            for metric in results["control_performance"].keys():
                # Simulate p-value calculation
                # Higher sample size and data quality = more likely to be significant
                p_value = random.uniform(0.01, 0.15) * (1 - data_quality) * (100 / sample_size)
                significance[metric] = p_value < SIGNIFICANCE_THRESHOLD
                
                # Simulate confidence intervals
                control_val = results["control_performance"][metric]
                treatment_val = results["treatment_performance"][metric]
                
                # Simple confidence interval simulation
                margin_of_error = abs(treatment_val - control_val) * 0.1
                ci_lower = (treatment_val - control_val) - margin_of_error
                ci_upper = (treatment_val - control_val) + margin_of_error
                confidence_intervals[metric] = (ci_lower, ci_upper)
            
            statistical_analysis["significance"][proposal_id] = significance
            statistical_analysis["confidence_intervals"][proposal_id] = confidence_intervals
        
        return statistical_analysis
    
    
    async def _implement_single_proposal(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """Implement a single proposal with gradual rollout"""
        implementation_result = {
            "success": False,
            "rolled_back": False,
            "value_added": 0.0,
            "rollout_stages": [],
            "monitoring_data": [],
            "final_status": "pending"
        }
        
        try:
            # === NEW: Autonomous Distillation Trigger ===
            # If this is an economic efficiency regression fix, trigger distillation
            if proposal.improvement_type == "architecture" and \
               proposal.supporting_data.get("recommendation") == "ssm_distillation":
                
                print(f"🚀 Autonomously triggering distillation for {proposal.target_component} due to economic regression")
                
                # Create distillation request based on the identified model
                # We assume the target_component format is 'model_id:metric' or just 'model_id'
                model_id = proposal.target_component.split(':')[0]
                
                request = DistillationRequest(
                    user_id="PRSM_RSI_SYSTEM",
                    teacher_model=model_id,
                    domain="scientific_reasoning", # Default, ideally from metadata
                    target_size="small",
                    optimization_target="efficiency",
                    target_architecture="ssm",
                    budget_ftns=1000 # RSI has its own photon budget
                )
                
                distiller = get_distillation_orchestrator()
                # Create distillation job (don't await completion here, it's a background process)
                asyncio.create_task(distiller.create_distillation(request))
                
                implementation_result["autonomous_job_triggered"] = True
            
            # Gradual rollout implementation
            for rollout_percentage in ROLLOUT_PERCENTAGE_STEPS:
                stage_result = await self._execute_rollout_stage(proposal, rollout_percentage)
                implementation_result["rollout_stages"].append(stage_result)
                
                # Monitor performance after each stage
                monitoring_data = await self._monitor_rollout_performance(proposal, rollout_percentage)
                implementation_result["monitoring_data"].append(monitoring_data)
                
                # Check if rollback is needed
                if monitoring_data.get("performance_degradation", 0) > ROLLBACK_TRIGGER_THRESHOLD:
                    await self._execute_rollback(proposal, rollout_percentage)
                    implementation_result["rolled_back"] = True
                    implementation_result["final_status"] = "rolled_back"
                    return implementation_result
                
                # Brief pause between stages
                await asyncio.sleep(1)  # Simulate staged rollout timing
            
            # Implementation successful
            implementation_result["success"] = True
            implementation_result["final_status"] = "implemented"
            
            # Calculate value added
            if proposal.simulation_result:
                predicted_changes = proposal.simulation_result.predicted_performance_change
                value_added = sum(abs(change) for change in predicted_changes.values()) / len(predicted_changes)
                implementation_result["value_added"] = value_added
            
            # Update proposal status
            proposal.status = ProposalStatus.COMPLETED
            
            # Store implementation
            async with self._implementations_lock:
                self.implemented_improvements[proposal.proposal_id] = {
                    "proposal": proposal,
                    "implementation_time": datetime.now(timezone.utc),
                    "result": implementation_result
                }
            
        except Exception as e:
            implementation_result["final_status"] = "failed"
            implementation_result["error"] = str(e)
        
        return implementation_result
    
    
    async def _execute_rollout_stage(self, proposal: ImprovementProposal, percentage: int) -> Dict[str, Any]:
        """Execute a single rollout stage"""
        # Simulate rollout execution
        await asyncio.sleep(0.5)  # Simulate deployment time
        
        return {
            "percentage": percentage,
            "execution_time": datetime.now(timezone.utc),
            "success": True,
            "affected_systems": [proposal.target_component],
            "deployment_method": "blue_green" if percentage == 100 else "canary"
        }
    
    
    async def _monitor_rollout_performance(self, proposal: ImprovementProposal, percentage: int) -> Dict[str, Any]:
        """Monitor performance during rollout stage"""
        # Simulate performance monitoring
        base_degradation = random.uniform(0.0, 0.05)  # Max 5% degradation
        
        # Safety enhancements are less likely to cause degradation
        if proposal.improvement_type.value == "safety_enhancement":
            degradation_factor = 0.5
        else:
            degradation_factor = 1.0
        
        performance_degradation = base_degradation * degradation_factor
        
        return {
            "rollout_percentage": percentage,
            "monitoring_duration": MONITORING_GRACE_PERIOD_HOURS,
            "performance_degradation": performance_degradation,
            "error_rate_change": random.uniform(-0.01, 0.02),
            "resource_usage_change": random.uniform(-0.05, 0.1),
            "safety_score_change": random.uniform(-0.02, 0.05)
        }
    
    
    async def _execute_rollback(self, proposal: ImprovementProposal, rollout_percentage: int) -> bool:
        """Execute rollback for a proposal"""
        try:
            # Simulate rollback execution
            await asyncio.sleep(0.3)  # Simulate rollback time
            
            print(f"🔄 Rolling back proposal {proposal.proposal_id} at {rollout_percentage}% rollout")
            
            # Update proposal status
            proposal.status = ProposalStatus.REJECTED
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to rollback proposal {proposal.proposal_id}: {str(e)}")
            return False
    
    
    async def _validate_update_package(self, update_package: UpdatePackage) -> Dict[str, Any]:
        """Validate update package for network propagation"""
        validation_result = {
            "valid": True,
            "reason": "",
            "checks_performed": []
        }
        
        # Check package integrity
        if not update_package.update_id:
            validation_result["valid"] = False
            validation_result["reason"] = "Missing update ID"
            return validation_result
        
        validation_result["checks_performed"].append("update_id_present")
        
        # Check content validity
        if not update_package.update_content:
            validation_result["valid"] = False
            validation_result["reason"] = "Empty update content"
            return validation_result
        
        validation_result["checks_performed"].append("content_present")
        
        # Check safety requirements
        safety_validation = await self.safety_monitor.validate_model_output(
            update_package.update_content,
            ["no_malicious_code", "data_integrity"]
        )
        
        if not safety_validation.is_safe:
            validation_result["valid"] = False
            validation_result["reason"] = "Failed safety validation"
            return validation_result
        
        validation_result["checks_performed"].append("safety_validation")
        
        return validation_result
    
    
    async def _discover_network_nodes(self) -> List[Dict[str, Any]]:
        """Discover available network nodes for update propagation"""
        # Simulate network node discovery
        num_nodes = random.randint(5, 15)
        nodes = []
        
        for i in range(num_nodes):
            node = {
                "node_id": f"node_{i:03d}",
                "ip_address": f"192.168.1.{i+10}",
                "port": 8080 + i,
                "node_type": random.choice(["validator", "compute", "storage"]),
                "capacity": random.uniform(0.5, 1.0),
                "reliability_score": random.uniform(0.7, 1.0),
                "last_seen": datetime.now(timezone.utc) - timedelta(minutes=random.randint(1, 60))
            }
            nodes.append(node)
        
        return nodes
    
    
    async def _achieve_update_consensus(self, update_package: UpdatePackage, 
                                      network_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Achieve consensus for network update"""
        try:
            # Prepare consensus data
            consensus_data = {
                "update_id": str(update_package.update_id),
                "update_hash": hash(str(update_package.update_content)),
                "node_count": len(network_nodes),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Simulate consensus achievement
            consensus_result = await self.consensus.achieve_result_consensus(
                [consensus_data for _ in range(len(network_nodes))]
            )
            
            return {
                "consensus_achieved": consensus_result is not None,
                "consensus_data": consensus_result,
                "participating_nodes": len(network_nodes),
                "consensus_duration": random.uniform(30, 180),  # seconds
                "consensus_score": random.uniform(0.8, 1.0)
            }
            
        except Exception as e:
            return {
                "consensus_achieved": False,
                "error": str(e),
                "participating_nodes": 0,
                "consensus_duration": 0.0,
                "consensus_score": 0.0
            }
    
    
    async def _propagate_updates_in_batches(self, update_package: UpdatePackage,
                                          network_nodes: List[Dict[str, Any]],
                                          consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate updates to network nodes in batches"""
        successful_nodes = []
        failed_nodes = []
        
        # Split nodes into batches
        for i in range(0, len(network_nodes), NETWORK_UPDATE_BATCH_SIZE):
            batch = network_nodes[i:i + NETWORK_UPDATE_BATCH_SIZE]
            
            # Process batch
            for node in batch:
                success = await self._propagate_to_single_node(update_package, node)
                if success:
                    successful_nodes.append(node["node_id"])
                else:
                    failed_nodes.append(node["node_id"])
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        return {
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
            "total_batches": len(network_nodes) // NETWORK_UPDATE_BATCH_SIZE + 1,
            "propagation_method": "batch_sequential"
        }
    
    
    async def _propagate_to_single_node(self, update_package: UpdatePackage, 
                                      node: Dict[str, Any]) -> bool:
        """Propagate update to a single node"""
        try:
            # Simulate network communication
            success_probability = node.get("reliability_score", 0.8)
            return random.random() < success_probability
            
        except Exception:
            return False
    
    
    async def _monitor_propagation_health(self, update_package: UpdatePackage,
                                        propagation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor health of update propagation"""
        successful_count = len(propagation_results.get("successful_nodes", []))
        failed_count = len(propagation_results.get("failed_nodes", []))
        total_count = successful_count + failed_count
        
        success_rate = successful_count / max(1, total_count)
        
        return {
            "overall_health": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "critical",
            "success_rate": success_rate,
            "successful_propagations": successful_count,
            "failed_propagations": failed_count,
            "monitoring_duration": random.uniform(60, 300),  # seconds
            "health_checks_performed": ["connectivity", "resource_usage", "error_rates"]
        }
    
    
    async def _handle_propagation_failures(self, propagation_results: Dict[str, Any],
                                         monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Handle any propagation failures"""
        failed_nodes = propagation_results.get("failed_nodes", [])
        
        if not failed_nodes:
            return {"failures_handled": 0, "retry_attempts": 0, "recovery_actions": []}
        
        # Attempt retries for failed nodes
        retry_results = []
        for node_id in failed_nodes[:5]:  # Limit retries
            # Simulate retry attempt
            retry_success = random.random() < 0.6  # 60% success rate on retry
            retry_results.append({
                "node_id": node_id,
                "retry_successful": retry_success
            })
        
        return {
            "failures_handled": len(failed_nodes),
            "retry_attempts": len(retry_results),
            "successful_retries": sum(1 for r in retry_results if r["retry_successful"]),
            "recovery_actions": ["node_retry", "health_check", "fallback_routing"]
        }


# === Global Evolution Orchestrator Instance ===

_evolution_orchestrator_instance: Optional[EvolutionOrchestrator] = None

def get_evolution_orchestrator() -> EvolutionOrchestrator:
    """Get or create the global evolution orchestrator instance"""
    global _evolution_orchestrator_instance
    if _evolution_orchestrator_instance is None:
        _evolution_orchestrator_instance = EvolutionOrchestrator()
    return _evolution_orchestrator_instance