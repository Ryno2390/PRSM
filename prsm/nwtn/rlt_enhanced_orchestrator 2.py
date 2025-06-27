"""
RLT-Enhanced NWTN Orchestrator

Extends the Enhanced NWTN Orchestrator with RLT (Reinforcement Learning Teachers)
integration for optimal teacher selection, student capability assessment, and
teaching effectiveness evaluation.

Key RLT Enhancements:
- RLT teacher selection and coordination
- Student capability assessment and tracking
- Teaching effectiveness evaluation with dense rewards
- Progressive difficulty adaptation
- Quality-driven teacher routing
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4

import structlog

from .enhanced_orchestrator import EnhancedNWTNOrchestrator
from ..agents.routers.rlt_enhanced_router import RLTEnhancedRouter, RLTTeacherSelection
from ..teachers.seal_rlt_enhanced_teacher import SEALRLTEnhancedTeacher, SEALRLTConfig, SEALRLTTeachingSession
from ..teachers.rlt.dense_reward_trainer import RLTTrainingConfig
from ..teachers.rlt.student_comprehension_evaluator import ComprehensionMetrics, EvaluationConfig
from ..teachers.rlt.quality_monitor import QualityMetrics, MonitoringConfig
from ..core.models import (
    UserInput, PRSMSession, ClarifiedPrompt, PRSMResponse,
    ReasoningStep, AgentType, TaskStatus, ArchitectTask
)

logger = structlog.get_logger(__name__)


class StudentCapabilityProfile:
    """Student capability profile for adaptive teaching"""
    
    def __init__(
        self,
        student_id: str,
        domain_capabilities: Dict[str, float] = None,
        learning_style: str = "adaptive",
        comprehension_history: List[float] = None,
        difficulty_progression: List[float] = None
    ):
        self.student_id = student_id
        self.domain_capabilities = domain_capabilities or {}
        self.learning_style = learning_style
        self.comprehension_history = comprehension_history or []
        self.difficulty_progression = difficulty_progression or []
        self.last_updated = datetime.now(timezone.utc)
        
        # Adaptive learning metrics
        self.learning_velocity = 0.0
        self.optimal_challenge_level = 0.6
        self.mastery_threshold = 0.8
        self.improvement_rate = 0.0
    
    def update_capability(self, domain: str, new_score: float):
        """Update capability for specific domain"""
        self.domain_capabilities[domain] = new_score
        self.last_updated = datetime.now(timezone.utc)
        
        # Calculate improvement rate
        if len(self.comprehension_history) > 1:
            recent_scores = self.comprehension_history[-5:]
            older_scores = self.comprehension_history[-10:-5] if len(self.comprehension_history) >= 10 else []
            
            if older_scores:
                self.improvement_rate = np.mean(recent_scores) - np.mean(older_scores)
    
    def get_capability(self, domain: str) -> float:
        """Get capability for domain with fallback to general capability"""
        if domain in self.domain_capabilities:
            return self.domain_capabilities[domain]
        
        # Calculate general capability from all domains
        if self.domain_capabilities:
            return np.mean(list(self.domain_capabilities.values()))
        
        return 0.5  # Default capability


class RLTTeachingEvaluation:
    """Comprehensive teaching effectiveness evaluation"""
    
    def __init__(
        self,
        session_id: UUID,
        teacher_id: str,
        student_profile: StudentCapabilityProfile,
        question: str,
        solution: str,
        explanation: str
    ):
        self.session_id = session_id
        self.teacher_id = teacher_id
        self.student_profile = student_profile
        self.question = question
        self.solution = solution
        self.explanation = explanation
        self.timestamp = datetime.now(timezone.utc)
        
        # Evaluation metrics
        self.dense_rewards: Dict[str, float] = {}
        self.comprehension_metrics: Optional[ComprehensionMetrics] = None
        self.quality_metrics: Optional[QualityMetrics] = None
        self.effectiveness_score: float = 0.0
        self.student_improvement: float = 0.0
        self.adaptation_success: float = 0.0


class RLTEnhancedOrchestrator(EnhancedNWTNOrchestrator):
    """
    RLT-Enhanced NWTN Orchestrator
    
    Extends the Enhanced Orchestrator with RLT capabilities:
    - RLT teacher selection and coordination
    - Student capability assessment and tracking
    - Teaching effectiveness evaluation with dense rewards
    - Progressive difficulty adaptation
    - Quality-driven teacher routing
    
    Integration with Enhanced Orchestrator:
    - Inherits all production-ready features (database, safety, FTNS)
    - Adds RLT-specific teacher coordination
    - Provides student-teacher matching optimization
    - Enables teaching effectiveness tracking
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Replace router with RLT-enhanced version
        self.rlt_router = RLTEnhancedRouter(agent_id="rlt_router_001")
        
        # RLT-specific components
        self.rlt_teacher_coordinator = None  # Initialized on first use
        self.student_profiles: Dict[str, StudentCapabilityProfile] = {}
        self.teaching_evaluations: List[RLTTeachingEvaluation] = []
        self.active_teaching_sessions: Dict[str, SEALRLTTeachingSession] = {}
        
        # RLT configuration
        self.rlt_config = SEALRLTConfig(
            seal_enabled=True,
            rlt_enabled=True,
            quality_threshold=0.75,
            comprehensive_logging=True
        )
        
        # Enhanced performance tracking
        self.rlt_performance_stats = {
            "total_teaching_sessions": 0,
            "successful_explanations": 0,
            "avg_student_improvement": 0.0,
            "avg_teaching_effectiveness": 0.0,
            "total_dense_rewards": 0.0,
            "teacher_utilization": {},
            "domain_coverage": {},
            "adaptation_success_rate": 0.0
        }
        
        logger.info("RLT-Enhanced NWTN Orchestrator initialized with teacher coordination")
    
    async def process_teaching_query(
        self, 
        user_input: UserInput,
        question: str,
        solution: str,
        learning_objectives: List[str] = None
    ) -> PRSMResponse:
        """
        Process teaching query with RLT teacher coordination
        
        Enhanced teaching flow:
        1. Assess student capability and learning profile
        2. Select optimal RLT teacher for content and student
        3. Coordinate teaching session with dense reward optimization
        4. Evaluate teaching effectiveness and student comprehension
        5. Update student profile and teacher performance metrics
        
        Args:
            user_input: Standard user input with context
            question: Question to be explained
            solution: Solution to be taught
            learning_objectives: Specific learning objectives
            
        Returns:
            Enhanced PRSM response with teaching evaluation
        """
        start_time = time.time()
        session = None
        teaching_session = None
        
        try:
            logger.info("Processing RLT teaching query",
                       user_id=user_input.user_id,
                       question_length=len(question),
                       solution_length=len(solution))
            
            # Step 1: Create enhanced session with teaching context
            session = await self._create_teaching_session(user_input, question, solution, learning_objectives)
            
            # Step 2: Assess student capability
            student_profile = await self._assess_student_capability(
                user_input.user_id, question, solution, session
            )
            
            # Step 3: Select optimal RLT teacher
            teacher_selection = await self._select_rlt_teacher(
                question, solution, student_profile, session
            )
            
            # Step 4: Create and coordinate teaching session
            teaching_session = await self._create_rlt_teaching_session(
                teacher_selection, student_profile, question, solution, learning_objectives
            )
            
            # Step 5: Execute teaching with dense reward optimization
            teaching_result = await self._execute_rlt_teaching(
                teaching_session, teacher_selection, student_profile
            )
            
            # Step 6: Evaluate teaching effectiveness
            evaluation = await self._evaluate_teaching_effectiveness(
                teaching_session, teaching_result, student_profile
            )
            
            # Step 7: Update student profile and teacher metrics
            await self._update_learning_profiles(student_profile, evaluation, teaching_result)
            
            # Step 8: Create enhanced response with teaching evaluation
            response = await self._create_teaching_response(
                session, teaching_result, evaluation, time.time() - start_time
            )
            
            # Step 9: Finalize teaching session
            await self._finalize_teaching_session(teaching_session, evaluation)
            
            return response
            
        except Exception as e:
            logger.error("RLT teaching query processing failed",
                        session_id=session.session_id if session else "unknown",
                        error=str(e))
            
            if session:
                await self._handle_teaching_error(session, teaching_session, e, time.time() - start_time)
            
            raise
    
    async def select_rlt_teacher(
        self, 
        task_complexity: float, 
        student_capability: float,
        domain: str = "general",
        learning_objectives: List[str] = None
    ) -> RLTTeacherSelection:
        """
        Select optimal RLT teacher for given complexity and student capability
        
        Args:
            task_complexity: Complexity of the teaching task (0.0-1.0)
            student_capability: Current student capability level (0.0-1.0)
            domain: Subject domain
            learning_objectives: Specific learning objectives
            
        Returns:
            Complete teacher selection with optimization details
        """
        try:
            logger.info("Selecting RLT teacher",
                       complexity=task_complexity,
                       capability=student_capability,
                       domain=domain)
            
            # Use RLT router for teacher discovery and selection
            mock_question = f"Teach {domain} concepts at {task_complexity:.1f} difficulty"
            mock_solution = f"Content for {student_capability:.1f} capability student"
            
            teacher_selection = await self.rlt_router.route_to_optimal_teacher(
                question=mock_question,
                solution=mock_solution,
                student_model=f"student_{hash(str(student_capability))%1000:03d}",
                student_capability=student_capability
            )
            
            # Enhance with learning objectives
            if learning_objectives:
                teacher_selection.learning_objectives = learning_objectives
            
            # Update performance tracking
            if teacher_selection.selected_rlt_teacher:
                teacher_id = teacher_selection.selected_rlt_teacher.model_id
                if teacher_id not in self.rlt_performance_stats["teacher_utilization"]:
                    self.rlt_performance_stats["teacher_utilization"][teacher_id] = 0
                self.rlt_performance_stats["teacher_utilization"][teacher_id] += 1
            
            # Update domain coverage
            if domain not in self.rlt_performance_stats["domain_coverage"]:
                self.rlt_performance_stats["domain_coverage"][domain] = 0
            self.rlt_performance_stats["domain_coverage"][domain] += 1
            
            logger.info("RLT teacher selected",
                       teacher_id=teacher_selection.selected_rlt_teacher.model_id if teacher_selection.selected_rlt_teacher else None,
                       quality_confidence=teacher_selection.quality_confidence,
                       predicted_improvement=teacher_selection.predicted_improvement)
            
            return teacher_selection
            
        except Exception as e:
            logger.error("RLT teacher selection failed", error=str(e))
            raise
    
    async def coordinate_rlt_training_pipeline(
        self, 
        questions: List[str], 
        solutions: List[str],
        student_id: str = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Coordinate RLT training pipeline for multiple question-solution pairs
        
        Args:
            questions: List of questions to be taught
            solutions: Corresponding solutions
            student_id: Optional student identifier for personalization
            domain: Subject domain
            
        Returns:
            Complete training pipeline results with effectiveness metrics
        """
        start_time = time.time()
        
        try:
            logger.info("Coordinating RLT training pipeline",
                       question_count=len(questions),
                       student_id=student_id,
                       domain=domain)
            
            if len(questions) != len(solutions):
                raise ValueError("Questions and solutions must have equal length")
            
            # Get or create student profile
            if student_id:
                student_profile = await self._get_or_create_student_profile(student_id, domain)
            else:
                # Create anonymous student profile
                student_profile = StudentCapabilityProfile(
                    student_id=f"anonymous_{uuid4().hex[:8]}",
                    domain_capabilities={domain: 0.5}
                )
            
            # Initialize training results
            training_results = {
                "total_questions": len(questions),
                "successful_teachings": 0,
                "failed_teachings": 0,
                "teaching_evaluations": [],
                "student_improvement": 0.0,
                "avg_effectiveness": 0.0,
                "total_dense_rewards": 0.0,
                "domain_mastery_progression": [],
                "teacher_performance": {}
            }
            
            # Process each question-solution pair
            for i, (question, solution) in enumerate(zip(questions, solutions)):
                try:
                    logger.info("Processing training item",
                               item=i+1,
                               total=len(questions),
                               student_capability=student_profile.get_capability(domain))
                    
                    # Select teacher for this specific content
                    complexity = await self._assess_content_complexity(question, solution)
                    teacher_selection = await self.select_rlt_teacher(
                        complexity, 
                        student_profile.get_capability(domain),
                        domain
                    )
                    
                    # Create teaching session
                    teaching_session = await self._create_rlt_teaching_session(
                        teacher_selection, student_profile, question, solution, []
                    )
                    
                    # Execute teaching
                    teaching_result = await self._execute_rlt_teaching(
                        teaching_session, teacher_selection, student_profile
                    )
                    
                    # Evaluate effectiveness
                    evaluation = await self._evaluate_teaching_effectiveness(
                        teaching_session, teaching_result, student_profile
                    )
                    
                    # Update results
                    if teaching_result.get("success", False):
                        training_results["successful_teachings"] += 1
                        training_results["teaching_evaluations"].append(evaluation)
                        
                        # Update student profile
                        await self._update_learning_profiles(student_profile, evaluation, teaching_result)
                        
                        # Track domain mastery progression
                        current_mastery = student_profile.get_capability(domain)
                        training_results["domain_mastery_progression"].append(current_mastery)
                        
                        # Track teacher performance
                        teacher_id = teacher_selection.selected_rlt_teacher.model_id if teacher_selection.selected_rlt_teacher else "unknown"
                        if teacher_id not in training_results["teacher_performance"]:
                            training_results["teacher_performance"][teacher_id] = {
                                "teachings": 0,
                                "effectiveness_scores": [],
                                "dense_rewards": []
                            }
                        
                        training_results["teacher_performance"][teacher_id]["teachings"] += 1
                        training_results["teacher_performance"][teacher_id]["effectiveness_scores"].append(
                            evaluation.effectiveness_score
                        )
                        training_results["teacher_performance"][teacher_id]["dense_rewards"].append(
                            sum(evaluation.dense_rewards.values()) if evaluation.dense_rewards else 0.0
                        )
                    else:
                        training_results["failed_teachings"] += 1
                    
                    # Finalize teaching session
                    await self._finalize_teaching_session(teaching_session, evaluation)
                    
                except Exception as item_error:
                    logger.error("Training item failed",
                               item=i+1,
                               error=str(item_error))
                    training_results["failed_teachings"] += 1
            
            # Calculate final metrics
            if training_results["teaching_evaluations"]:
                training_results["avg_effectiveness"] = np.mean([
                    eval.effectiveness_score for eval in training_results["teaching_evaluations"]
                ])
                training_results["total_dense_rewards"] = sum([
                    sum(eval.dense_rewards.values()) if eval.dense_rewards else 0.0
                    for eval in training_results["teaching_evaluations"]
                ])
                
                # Calculate student improvement
                if len(training_results["domain_mastery_progression"]) > 1:
                    initial_mastery = training_results["domain_mastery_progression"][0]
                    final_mastery = training_results["domain_mastery_progression"][-1]
                    training_results["student_improvement"] = final_mastery - initial_mastery
            
            # Update global performance stats
            self.rlt_performance_stats["total_teaching_sessions"] += training_results["successful_teachings"]
            self.rlt_performance_stats["successful_explanations"] += training_results["successful_teachings"]
            
            if training_results["avg_effectiveness"] > 0:
                current_avg = self.rlt_performance_stats["avg_teaching_effectiveness"]
                total_sessions = self.rlt_performance_stats["total_teaching_sessions"]
                self.rlt_performance_stats["avg_teaching_effectiveness"] = (
                    (current_avg * (total_sessions - training_results["successful_teachings"]) + 
                     training_results["avg_effectiveness"] * training_results["successful_teachings"]) / 
                    max(total_sessions, 1)
                )
            
            training_time = time.time() - start_time
            
            logger.info("RLT training pipeline completed",
                       total_questions=training_results["total_questions"],
                       successful=training_results["successful_teachings"],
                       avg_effectiveness=training_results["avg_effectiveness"],
                       student_improvement=training_results["student_improvement"],
                       training_time=training_time)
            
            return {
                **training_results,
                "training_time": training_time,
                "success_rate": training_results["successful_teachings"] / len(questions),
                "student_profile": {
                    "student_id": student_profile.student_id,
                    "final_capabilities": student_profile.domain_capabilities,
                    "improvement_rate": student_profile.improvement_rate,
                    "learning_velocity": student_profile.learning_velocity
                }
            }
            
        except Exception as e:
            logger.error("RLT training pipeline coordination failed", error=str(e))
            raise
    
    async def evaluate_teaching_effectiveness(
        self, 
        teacher_outputs: List[Dict[str, Any]], 
        student_progress: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate overall teaching effectiveness across multiple outputs
        
        Args:
            teacher_outputs: List of teacher output data
            student_progress: Student progress metrics
            
        Returns:
            Comprehensive effectiveness evaluation
        """
        try:
            logger.info("Evaluating teaching effectiveness",
                       teacher_outputs_count=len(teacher_outputs))
            
            if not teacher_outputs:
                return {"error": "No teacher outputs provided"}
            
            effectiveness_metrics = {
                "total_evaluations": len(teacher_outputs),
                "successful_explanations": 0,
                "avg_explanation_quality": 0.0,
                "avg_student_comprehension": 0.0,
                "avg_dense_rewards": 0.0,
                "teacher_consistency": 0.0,
                "adaptation_effectiveness": 0.0,
                "overall_effectiveness": 0.0,
                "individual_evaluations": []
            }
            
            quality_scores = []
            comprehension_scores = []
            dense_reward_totals = []
            
            # Evaluate each teacher output
            for i, output in enumerate(teacher_outputs):
                try:
                    # Extract evaluation metrics from output
                    evaluation_data = output.get("evaluation", {})
                    
                    if evaluation_data:
                        quality_score = evaluation_data.get("explanation_quality", 0.0)
                        comprehension_score = evaluation_data.get("student_comprehension", 0.0)
                        dense_rewards = evaluation_data.get("dense_rewards", {})
                        
                        quality_scores.append(quality_score)
                        comprehension_scores.append(comprehension_score)
                        dense_reward_totals.append(sum(dense_rewards.values()) if dense_rewards else 0.0)
                        
                        effectiveness_metrics["individual_evaluations"].append({
                            "output_index": i,
                            "quality_score": quality_score,
                            "comprehension_score": comprehension_score,
                            "dense_rewards_total": dense_reward_totals[-1],
                            "effectiveness": (quality_score + comprehension_score) / 2
                        })
                        
                        if quality_score > 0.6 and comprehension_score > 0.6:
                            effectiveness_metrics["successful_explanations"] += 1
                    
                except Exception as eval_error:
                    logger.warning("Individual evaluation failed",
                                 output_index=i,
                                 error=str(eval_error))
            
            # Calculate aggregate metrics
            if quality_scores:
                effectiveness_metrics["avg_explanation_quality"] = np.mean(quality_scores)
                effectiveness_metrics["teacher_consistency"] = 1.0 - np.std(quality_scores)
            
            if comprehension_scores:
                effectiveness_metrics["avg_student_comprehension"] = np.mean(comprehension_scores)
            
            if dense_reward_totals:
                effectiveness_metrics["avg_dense_rewards"] = np.mean(dense_reward_totals)
            
            # Calculate adaptation effectiveness from student progress
            if student_progress:
                initial_capability = student_progress.get("initial_capability", 0.5)
                final_capability = student_progress.get("final_capability", 0.5)
                improvement = final_capability - initial_capability
                
                effectiveness_metrics["adaptation_effectiveness"] = max(0.0, min(1.0, improvement * 2))
                effectiveness_metrics["student_improvement"] = improvement
            
            # Calculate overall effectiveness
            effectiveness_components = [
                effectiveness_metrics["avg_explanation_quality"] * 0.3,
                effectiveness_metrics["avg_student_comprehension"] * 0.3,
                effectiveness_metrics["teacher_consistency"] * 0.2,
                effectiveness_metrics["adaptation_effectiveness"] * 0.2
            ]
            
            effectiveness_metrics["overall_effectiveness"] = sum(effectiveness_components)
            
            # Success rate
            effectiveness_metrics["success_rate"] = (
                effectiveness_metrics["successful_explanations"] / 
                max(effectiveness_metrics["total_evaluations"], 1)
            )
            
            logger.info("Teaching effectiveness evaluation completed",
                       overall_effectiveness=effectiveness_metrics["overall_effectiveness"],
                       success_rate=effectiveness_metrics["success_rate"],
                       avg_quality=effectiveness_metrics["avg_explanation_quality"])
            
            return effectiveness_metrics
            
        except Exception as e:
            logger.error("Teaching effectiveness evaluation failed", error=str(e))
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _create_teaching_session(
        self, 
        user_input: UserInput, 
        question: str, 
        solution: str, 
        learning_objectives: List[str]
    ) -> PRSMSession:
        """Create enhanced session for teaching"""
        session = await self._create_persistent_session(user_input)
        
        # Add teaching-specific metadata
        session.metadata.update({
            "session_type": "rlt_teaching",
            "question": question[:200],  # Truncated for storage
            "solution": solution[:200],
            "learning_objectives": learning_objectives or [],
            "teaching_mode": "rlt_enhanced"
        })
        
        return session
    
    async def _assess_student_capability(
        self, 
        user_id: str, 
        question: str, 
        solution: str, 
        session: PRSMSession
    ) -> StudentCapabilityProfile:
        """Assess or retrieve student capability profile"""
        
        # Get existing profile or create new one
        profile = await self._get_or_create_student_profile(user_id)
        
        # Assess content to determine domain
        domain = await self._extract_domain_from_content(question, solution)
        complexity = await self._assess_content_complexity(question, solution)
        
        # If no existing capability for this domain, estimate from content complexity
        if domain not in profile.domain_capabilities:
            # Conservative estimate - start slightly below content complexity
            estimated_capability = max(0.1, complexity - 0.2)
            profile.domain_capabilities[domain] = estimated_capability
            
            logger.info("Estimated student capability for new domain",
                       user_id=user_id,
                       domain=domain,
                       estimated_capability=estimated_capability,
                       content_complexity=complexity)
        
        return profile
    
    async def _get_or_create_student_profile(self, user_id: str, domain: str = None) -> StudentCapabilityProfile:
        """Get existing student profile or create new one"""
        
        if user_id in self.student_profiles:
            return self.student_profiles[user_id]
        
        # Create new profile
        profile = StudentCapabilityProfile(
            student_id=user_id,
            domain_capabilities={domain: 0.5} if domain else {},
            learning_style="adaptive"
        )
        
        self.student_profiles[user_id] = profile
        
        logger.info("Created new student profile",
                   user_id=user_id,
                   initial_domains=list(profile.domain_capabilities.keys()))
        
        return profile
    
    async def _select_rlt_teacher(
        self, 
        question: str, 
        solution: str, 
        student_profile: StudentCapabilityProfile, 
        session: PRSMSession
    ) -> RLTTeacherSelection:
        """Select optimal RLT teacher for specific content and student"""
        
        domain = await self._extract_domain_from_content(question, solution)
        student_capability = student_profile.get_capability(domain)
        
        teacher_selection = await self.rlt_router.route_to_optimal_teacher(
            question=question,
            solution=solution,
            student_model=student_profile.student_id,
            student_capability=student_capability
        )
        
        # Store reasoning for teacher selection
        await self.database_service.create_reasoning_step(
            session_id=session.session_id,
            step_data={
                "agent_type": "rlt_teacher_selection",
                "agent_id": "rlt_orchestrator",
                "input_data": {
                    "domain": domain,
                    "student_capability": student_capability,
                    "question_complexity": await self._assess_content_complexity(question, solution)
                },
                "output_data": {
                    "selected_teacher": teacher_selection.selected_rlt_teacher.model_id if teacher_selection.selected_rlt_teacher else None,
                    "quality_confidence": teacher_selection.quality_confidence,
                    "predicted_improvement": teacher_selection.predicted_improvement
                },
                "execution_time": 0.1,
                "confidence_score": teacher_selection.quality_confidence
            }
        )
        
        return teacher_selection
    
    async def _create_rlt_teaching_session(
        self, 
        teacher_selection: RLTTeacherSelection, 
        student_profile: StudentCapabilityProfile, 
        question: str, 
        solution: str, 
        learning_objectives: List[str]
    ) -> SEALRLTTeachingSession:
        """Create RLT teaching session"""
        
        if not teacher_selection.selected_rlt_teacher:
            raise ValueError("No teacher selected for RLT teaching session")
        
        # Initialize RLT teacher coordinator if needed
        if not self.rlt_teacher_coordinator:
            self.rlt_teacher_coordinator = SEALRLTEnhancedTeacher(
                teacher_model=None,  # Mock teacher model
                config=self.rlt_config
            )
        
        # Create teaching session
        domain = await self._extract_domain_from_content(question, solution)
        
        teaching_session = await self.rlt_teacher_coordinator.create_enhanced_teaching_session(
            student_id=student_profile.student_id,
            domain=domain,
            learning_objectives=learning_objectives,
            enable_hybrid_mode=True
        )
        
        # Store in active sessions
        self.active_teaching_sessions[str(teaching_session.session_id)] = teaching_session
        
        logger.info("Created RLT teaching session",
                   teaching_session_id=str(teaching_session.session_id),
                   student_id=student_profile.student_id,
                   teacher_id=teacher_selection.selected_rlt_teacher.model_id,
                   domain=domain)
        
        return teaching_session
    
    async def _execute_rlt_teaching(
        self, 
        teaching_session: SEALRLTTeachingSession, 
        teacher_selection: RLTTeacherSelection, 
        student_profile: StudentCapabilityProfile
    ) -> Dict[str, Any]:
        """Execute RLT teaching with dense reward optimization"""
        
        try:
            # Extract question-solution pair from teaching session
            if teaching_session.question_solution_pairs:
                qa_pair = teaching_session.question_solution_pairs[0]
                question = qa_pair["question"]
                solution = qa_pair["solution"]
            else:
                # Use mock data for demonstration
                question = "Sample teaching question"
                solution = "Sample solution"
            
            # Generate explanation with RLT methodology
            explanation, metrics = await self.rlt_teacher_coordinator.generate_rlt_explanation_with_seal_enhancement(
                question=question,
                solution=solution,
                session=teaching_session,
                use_seal_adaptation=self.rlt_config.seal_enabled
            )
            
            # Extract dense rewards
            dense_rewards = metrics.get("rlt_rewards", {})
            comprehension_metrics = metrics.get("comprehension_metrics", {})
            quality_metrics = metrics.get("quality_metrics", {})
            
            teaching_result = {
                "success": True,
                "explanation": explanation,
                "dense_rewards": dense_rewards,
                "comprehension_metrics": comprehension_metrics,
                "quality_metrics": quality_metrics,
                "teacher_id": teacher_selection.selected_rlt_teacher.model_id,
                "session_id": str(teaching_session.session_id),
                "execution_time": 1.5,  # Mock execution time
                "seal_enhancement": metrics.get("seal_enhancement", {}),
                "methodology": "rlt_enhanced"
            }
            
            logger.info("RLT teaching executed successfully",
                       session_id=str(teaching_session.session_id),
                       explanation_length=len(explanation),
                       dense_rewards=dense_rewards,
                       comprehension_score=comprehension_metrics.get("overall_comprehension", 0.0))
            
            return teaching_result
            
        except Exception as e:
            logger.error("RLT teaching execution failed",
                        session_id=str(teaching_session.session_id),
                        error=str(e))
            
            return {
                "success": False,
                "error": str(e),
                "teacher_id": teacher_selection.selected_rlt_teacher.model_id,
                "session_id": str(teaching_session.session_id)
            }
    
    async def _evaluate_teaching_effectiveness(
        self, 
        teaching_session: SEALRLTTeachingSession, 
        teaching_result: Dict[str, Any], 
        student_profile: StudentCapabilityProfile
    ) -> RLTTeachingEvaluation:
        """Evaluate teaching effectiveness with RLT metrics"""
        
        question = teaching_result.get("question", "")
        solution = teaching_result.get("solution", "")
        explanation = teaching_result.get("explanation", "")
        teacher_id = teaching_result.get("teacher_id", "unknown")
        
        evaluation = RLTTeachingEvaluation(
            session_id=teaching_session.session_id,
            teacher_id=teacher_id,
            student_profile=student_profile,
            question=question,
            solution=solution,
            explanation=explanation
        )
        
        # Extract metrics from teaching result
        evaluation.dense_rewards = teaching_result.get("dense_rewards", {})
        
        # Calculate effectiveness score
        quality_score = teaching_result.get("quality_metrics", {}).get("overall_quality", 0.0)
        comprehension_score = teaching_result.get("comprehension_metrics", {}).get("overall_comprehension", 0.0)
        dense_reward_total = sum(evaluation.dense_rewards.values()) if evaluation.dense_rewards else 0.0
        
        # Normalize dense rewards to 0-1 scale
        normalized_rewards = min(1.0, dense_reward_total / 2.0)  # Assume max reward is 2.0
        
        evaluation.effectiveness_score = (
            quality_score * 0.4 +
            comprehension_score * 0.4 +
            normalized_rewards * 0.2
        )
        
        # Calculate student improvement (mock for now)
        domain = await self._extract_domain_from_content(question, solution)
        current_capability = student_profile.get_capability(domain)
        improvement_potential = min(0.2, (1.0 - current_capability) * 0.3)  # Up to 20% improvement
        evaluation.student_improvement = improvement_potential * evaluation.effectiveness_score
        
        # Calculate adaptation success
        content_complexity = await self._assess_content_complexity(question, solution)
        capability_match = 1.0 - abs(content_complexity - current_capability)
        evaluation.adaptation_success = capability_match * evaluation.effectiveness_score
        
        logger.info("Teaching effectiveness evaluated",
                   session_id=str(teaching_session.session_id),
                   effectiveness_score=evaluation.effectiveness_score,
                   student_improvement=evaluation.student_improvement,
                   adaptation_success=evaluation.adaptation_success)
        
        return evaluation
    
    async def _update_learning_profiles(
        self, 
        student_profile: StudentCapabilityProfile, 
        evaluation: RLTTeachingEvaluation, 
        teaching_result: Dict[str, Any]
    ):
        """Update student learning profile based on teaching evaluation"""
        
        # Extract domain from evaluation
        domain = await self._extract_domain_from_content(evaluation.question, evaluation.solution)
        
        # Update comprehension history
        comprehension_score = teaching_result.get("comprehension_metrics", {}).get("overall_comprehension", 0.0)
        student_profile.comprehension_history.append(comprehension_score)
        
        # Update domain capability with improvement
        current_capability = student_profile.get_capability(domain)
        new_capability = min(1.0, current_capability + evaluation.student_improvement)
        student_profile.update_capability(domain, new_capability)
        
        # Update difficulty progression
        content_complexity = await self._assess_content_complexity(evaluation.question, evaluation.solution)
        student_profile.difficulty_progression.append(content_complexity)
        
        # Calculate learning velocity
        if len(student_profile.comprehension_history) >= 3:
            recent_scores = student_profile.comprehension_history[-3:]
            student_profile.learning_velocity = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        # Update optimal challenge level based on performance
        if evaluation.effectiveness_score > 0.8:
            # Student is doing well, can handle slightly more difficulty
            student_profile.optimal_challenge_level = min(1.0, student_profile.optimal_challenge_level + 0.05)
        elif evaluation.effectiveness_score < 0.6:
            # Student struggling, reduce difficulty
            student_profile.optimal_challenge_level = max(0.1, student_profile.optimal_challenge_level - 0.05)
        
        logger.info("Student learning profile updated",
                   student_id=student_profile.student_id,
                   domain=domain,
                   new_capability=new_capability,
                   improvement=evaluation.student_improvement,
                   learning_velocity=student_profile.learning_velocity)
    
    async def _create_teaching_response(
        self, 
        session: PRSMSession, 
        teaching_result: Dict[str, Any], 
        evaluation: RLTTeachingEvaluation, 
        execution_time: float
    ) -> PRSMResponse:
        """Create enhanced PRSM response with teaching evaluation"""
        
        # Create reasoning trace with teaching steps
        reasoning_steps = [
            {
                "step_id": "teacher_selection",
                "agent_type": "rlt_teacher_selection",
                "task": "Select optimal RLT teacher",
                "result": {
                    "teacher_id": evaluation.teacher_id,
                    "effectiveness_prediction": evaluation.effectiveness_score
                },
                "execution_time": 0.1
            },
            {
                "step_id": "rlt_teaching",
                "agent_type": "rlt_explanation_generation",
                "task": "Generate explanation with dense rewards",
                "result": teaching_result,
                "execution_time": teaching_result.get("execution_time", 1.5)
            },
            {
                "step_id": "effectiveness_evaluation",
                "agent_type": "teaching_evaluation",
                "task": "Evaluate teaching effectiveness",
                "result": {
                    "effectiveness_score": evaluation.effectiveness_score,
                    "student_improvement": evaluation.student_improvement,
                    "dense_rewards": evaluation.dense_rewards,
                    "adaptation_success": evaluation.adaptation_success
                },
                "execution_time": 0.2
            }
        ]
        
        # Calculate final answer
        explanation = teaching_result.get("explanation", "")
        if not explanation:
            explanation = "The RLT-enhanced teaching system processed your query using advanced teacher-student coordination."
        
        # Add teaching effectiveness summary
        effectiveness_summary = f"\n\n📊 Teaching Effectiveness Analysis:\n"
        effectiveness_summary += f"• Overall Effectiveness: {evaluation.effectiveness_score:.1%}\n"
        effectiveness_summary += f"• Student Improvement: +{evaluation.student_improvement:.1%}\n"
        effectiveness_summary += f"• Teacher Adaptation: {evaluation.adaptation_success:.1%}\n"
        
        if evaluation.dense_rewards:
            effectiveness_summary += f"• Dense Rewards: {sum(evaluation.dense_rewards.values()):.2f}\n"
        
        final_answer = explanation + effectiveness_summary
        
        response = PRSMResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            final_answer=final_answer,
            reasoning_trace=reasoning_steps,
            confidence_score=evaluation.effectiveness_score,
            context_used=50,  # Mock context usage
            ftns_charged=25.0,  # Mock FTNS charge
            sources=[f"rlt_teacher_{evaluation.teacher_id}", "seal_rlt_enhanced_orchestrator"],
            safety_validated=True,
            metadata={
                "orchestrator": "rlt_enhanced_nwtn",
                "teaching_methodology": "rlt_enhanced",
                "teacher_id": evaluation.teacher_id,
                "teaching_session_id": str(evaluation.session_id),
                "effectiveness_score": evaluation.effectiveness_score,
                "student_improvement": evaluation.student_improvement,
                "dense_rewards_total": sum(evaluation.dense_rewards.values()) if evaluation.dense_rewards else 0.0,
                "execution_time": execution_time
            }
        )
        
        return response
    
    async def _finalize_teaching_session(
        self, 
        teaching_session: SEALRLTTeachingSession, 
        evaluation: RLTTeachingEvaluation
    ):
        """Finalize teaching session and update metrics"""
        
        try:
            # Finalize the SEAL-RLT teaching session
            if self.rlt_teacher_coordinator:
                session_results = await self.rlt_teacher_coordinator.finalize_session(
                    str(teaching_session.session_id)
                )
                
                logger.info("SEAL-RLT teaching session finalized",
                           session_id=str(teaching_session.session_id),
                           final_performance=session_results.get("session_summary", {}).get("hybrid_performance", 0.0))
            
            # Store evaluation in history
            self.teaching_evaluations.append(evaluation)
            
            # Update global performance statistics
            self.rlt_performance_stats["total_teaching_sessions"] += 1
            if evaluation.effectiveness_score > 0.6:
                self.rlt_performance_stats["successful_explanations"] += 1
            
            # Update average metrics
            current_avg_effectiveness = self.rlt_performance_stats["avg_teaching_effectiveness"]
            total_sessions = self.rlt_performance_stats["total_teaching_sessions"]
            
            self.rlt_performance_stats["avg_teaching_effectiveness"] = (
                (current_avg_effectiveness * (total_sessions - 1) + evaluation.effectiveness_score) / total_sessions
            )
            
            current_avg_improvement = self.rlt_performance_stats["avg_student_improvement"]
            self.rlt_performance_stats["avg_student_improvement"] = (
                (current_avg_improvement * (total_sessions - 1) + evaluation.student_improvement) / total_sessions
            )
            
            # Update dense rewards total
            dense_reward_total = sum(evaluation.dense_rewards.values()) if evaluation.dense_rewards else 0.0
            self.rlt_performance_stats["total_dense_rewards"] += dense_reward_total
            
            # Calculate adaptation success rate
            adaptation_successes = sum(1 for eval in self.teaching_evaluations if eval.adaptation_success > 0.7)
            self.rlt_performance_stats["adaptation_success_rate"] = (
                adaptation_successes / len(self.teaching_evaluations)
            )
            
            # Remove from active sessions
            if str(teaching_session.session_id) in self.active_teaching_sessions:
                del self.active_teaching_sessions[str(teaching_session.session_id)]
            
            logger.info("Teaching session finalized successfully",
                       session_id=str(teaching_session.session_id),
                       evaluation_effectiveness=evaluation.effectiveness_score,
                       global_avg_effectiveness=self.rlt_performance_stats["avg_teaching_effectiveness"])
            
        except Exception as e:
            logger.error("Teaching session finalization failed",
                        session_id=str(teaching_session.session_id),
                        error=str(e))
    
    async def _handle_teaching_error(
        self, 
        session: PRSMSession, 
        teaching_session: Optional[SEALRLTTeachingSession], 
        error: Exception, 
        execution_time: float
    ):
        """Handle teaching-specific errors"""
        
        # Use parent error handling
        await self._handle_enhanced_error(session, error, execution_time)
        
        # Additional teaching-specific error handling
        try:
            # Create teaching-specific safety flag
            await self.database_service.create_safety_flag(
                session_id=session.session_id,
                flag_data={
                    "level": "medium",
                    "category": "teaching_failure",
                    "description": f"RLT teaching session failed: {str(error)}",
                    "triggered_by": "rlt_enhanced_orchestrator",
                    "metadata": {
                        "teaching_session_id": str(teaching_session.session_id) if teaching_session else None,
                        "execution_time": execution_time,
                        "error_type": type(error).__name__
                    }
                }
            )
            
            # Clean up teaching session if it exists
            if teaching_session and str(teaching_session.session_id) in self.active_teaching_sessions:
                del self.active_teaching_sessions[str(teaching_session.session_id)]
            
            logger.error("RLT teaching error handling completed",
                        session_id=session.session_id,
                        teaching_session_id=str(teaching_session.session_id) if teaching_session else None,
                        error_type=type(error).__name__)
            
        except Exception as handling_error:
            logger.error("Teaching error handling failed",
                        session_id=session.session_id,
                        original_error=str(error),
                        handling_error=str(handling_error))
    
    async def _extract_domain_from_content(self, question: str, solution: str) -> str:
        """Extract domain from question and solution content"""
        content = f"{question} {solution}".lower()
        
        domain_keywords = {
            "mathematics": ["math", "equation", "theorem", "calculus", "algebra", "geometry", "derivative", "integral"],
            "physics": ["physics", "force", "energy", "momentum", "quantum", "electromagnetic", "newton"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "bond", "atom", "ion"],
            "biology": ["biology", "cell", "protein", "gene", "organism", "evolution", "dna"],
            "computer_science": ["algorithm", "programming", "computer", "software", "code", "data structure"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content for keyword in keywords):
                return domain
        
        return "general"
    
    async def _assess_content_complexity(self, question: str, solution: str) -> float:
        """Assess complexity of question-solution pair"""
        complexity_score = 0.5  # Base complexity
        
        # Length factors
        question_length = len(question.split())
        solution_length = len(solution.split())
        
        if question_length > 50 or solution_length > 100:
            complexity_score += 0.2
        
        # Complexity indicators
        complex_terms = ["advanced", "complex", "difficult", "challenging", "prove", "derive", "optimize"]
        content = f"{question} {solution}".lower()
        
        complexity_count = sum(1 for term in complex_terms if term in content)
        complexity_score += min(0.3, complexity_count * 0.1)
        
        # Mathematical complexity
        math_indicators = ["∫", "∑", "∂", "lim", "theorem", "proof", "≥", "≤", "∞"]
        math_count = sum(1 for indicator in math_indicators if indicator in content)
        complexity_score += min(0.2, math_count * 0.05)
        
        return min(1.0, complexity_score)
    
    # Public analytics methods
    
    async def get_rlt_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive RLT performance analytics"""
        
        base_analytics = self.get_tool_usage_analytics()
        
        # Calculate advanced metrics
        total_evaluations = len(self.teaching_evaluations)
        if total_evaluations > 0:
            effectiveness_scores = [eval.effectiveness_score for eval in self.teaching_evaluations]
            improvement_scores = [eval.student_improvement for eval in self.teaching_evaluations]
            
            effectiveness_trend = "improving" if len(effectiveness_scores) >= 5 and \
                np.mean(effectiveness_scores[-3:]) > np.mean(effectiveness_scores[-6:-3]) else "stable"
        else:
            effectiveness_trend = "no_data"
        
        rlt_analytics = {
            "teaching_performance": {
                **self.rlt_performance_stats,
                "effectiveness_trend": effectiveness_trend,
                "total_evaluations": total_evaluations
            },
            "student_profiles": {
                "total_students": len(self.student_profiles),
                "avg_capabilities": {
                    domain: np.mean([profile.domain_capabilities.get(domain, 0.5) 
                                   for profile in self.student_profiles.values() 
                                   if domain in profile.domain_capabilities])
                    for domain in ["mathematics", "physics", "chemistry", "biology", "computer_science"]
                    if any(domain in profile.domain_capabilities for profile in self.student_profiles.values())
                }
            },
            "rlt_router_analytics": await self.rlt_router.get_rlt_routing_analytics(),
            "active_sessions": len(self.active_teaching_sessions)
        }
        
        return {**base_analytics, "rlt_enhanced_metrics": rlt_analytics}


# Factory function
def create_rlt_enhanced_orchestrator(**kwargs) -> RLTEnhancedOrchestrator:
    """Create RLT-enhanced NWTN orchestrator"""
    return RLTEnhancedOrchestrator(**kwargs)


# Global instance
rlt_enhanced_orchestrator = None

def get_rlt_enhanced_orchestrator() -> RLTEnhancedOrchestrator:
    """Get or create global RLT-enhanced orchestrator instance"""
    global rlt_enhanced_orchestrator
    if rlt_enhanced_orchestrator is None:
        rlt_enhanced_orchestrator = RLTEnhancedOrchestrator()
    return rlt_enhanced_orchestrator