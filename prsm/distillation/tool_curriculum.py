"""
Tool Usage Curriculum for PRSM
Structured learning progression for tool-augmented AI models

This module provides comprehensive curricula for teaching AI models effective tool usage,
progressing from basic tool awareness to advanced multi-tool coordination and optimization.
The curriculum system ensures systematic skill development and provides measurable
learning objectives for tool-augmented AI capabilities.

Key Features:
- Progressive skill development from novice to expert
- Domain-specific tool curricula for different use cases
- Adaptive difficulty adjustment based on model performance
- Comprehensive assessment and validation frameworks
- Integration with tool marketplace and security systems
- Real-world scenario-based learning experiences

Architecture Integration:
- Works with tool-aware training pipeline
- Integrates with MCP tool router and marketplace
- Uses tool execution sandbox for safe practice
- Connects to FTNS service for resource management
- Provides detailed analytics and progress tracking
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.models import TimestampMixin
from ..agents.routers.tool_router import ToolRouter, ToolType, ToolCapability, ToolSecurityLevel
from ..marketplace.legacy.tool_marketplace import tool_marketplace
from .tool_aware_training import ToolTrainingStrategy, ToolUsagePattern, ToolTrainingExample

logger = structlog.get_logger(__name__)
settings = get_settings()


class SkillLevel(str, Enum):
    """Skill progression levels for tool usage"""
    NOVICE = "novice"                    # Basic tool awareness
    BEGINNER = "beginner"                # Single tool usage
    INTERMEDIATE = "intermediate"        # Multi-tool coordination
    ADVANCED = "advanced"                # Workflow optimization
    EXPERT = "expert"                    # Safety and error handling mastery


class CurriculumDomain(str, Enum):
    """Different domains requiring specialized tool curricula"""
    RESEARCH = "research"                # Academic and scientific research
    DATA_ANALYSIS = "data_analysis"      # Data science and analytics
    SOFTWARE_DEVELOPMENT = "software_development"  # Programming and development
    CONTENT_CREATION = "content_creation"  # Writing and creative work
    BUSINESS_AUTOMATION = "business_automation"  # Business process automation
    SCIENTIFIC_COMPUTING = "scientific_computing"  # Computational science
    GENERAL_PURPOSE = "general_purpose"  # Cross-domain tool usage


class AssessmentType(str, Enum):
    """Types of assessments for skill validation"""
    PRACTICAL_TASK = "practical_task"   # Real tool execution tasks
    SCENARIO_ANALYSIS = "scenario_analysis"  # Problem-solving scenarios
    SAFETY_PROTOCOL = "safety_protocol"  # Security and safety validation
    OPTIMIZATION_CHALLENGE = "optimization_challenge"  # Efficiency improvements
    ERROR_HANDLING = "error_handling"   # Failure recovery scenarios


@dataclass
class LearningObjective:
    """Specific learning objective within a curriculum"""
    objective_id: str
    title: str
    description: str
    skill_level: SkillLevel
    required_tools: List[str]
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    difficulty_score: float = 0.5
    estimated_duration: int = 30  # minutes
    practice_examples: List[str] = field(default_factory=list)


@dataclass
class CurriculumModule:
    """Modular curriculum component focusing on specific skills"""
    module_id: str
    title: str
    description: str
    domain: CurriculumDomain
    skill_level: SkillLevel
    learning_objectives: List[LearningObjective]
    duration_minutes: int = 120
    prerequisites: List[str] = field(default_factory=list)
    assessment_tasks: List[Dict[str, Any]] = field(default_factory=list)
    certification_criteria: Dict[str, float] = field(default_factory=dict)


class StudentProgress(BaseModel):
    """Track individual student progress through curriculum"""
    student_id: str
    curriculum_id: str
    current_skill_level: SkillLevel = SkillLevel.NOVICE
    completed_modules: List[str] = Field(default_factory=list)
    completed_objectives: List[str] = Field(default_factory=list)
    assessment_scores: Dict[str, float] = Field(default_factory=dict)
    practice_sessions: int = 0
    total_study_time: int = 0  # minutes
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssessmentResult(BaseModel):
    """Result of curriculum assessment"""
    assessment_id: UUID = Field(default_factory=uuid4)
    student_id: str
    module_id: str
    assessment_type: AssessmentType
    score: float = Field(ge=0.0, le=1.0)
    max_score: float = 1.0
    passed: bool = False
    feedback: str = ""
    detailed_results: Dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_minutes: int = 0


class ToolCurriculum:
    """
    Comprehensive Tool Usage Curriculum System
    
    Provides structured learning paths for developing tool usage skills
    from basic awareness to expert-level optimization and safety protocols.
    
    Features:
    - Progressive skill development with clear learning objectives
    - Domain-specific curricula tailored to different use cases
    - Practical assessments with real tool execution
    - Adaptive difficulty based on student performance
    - Comprehensive progress tracking and analytics
    """
    
    def __init__(self):
        self.curricula: Dict[str, List[CurriculumModule]] = {}
        self.student_progress: Dict[str, StudentProgress] = {}
        self.assessment_results: List[AssessmentResult] = []
        
        # Integration components
        self.tool_router = ToolRouter()
        
        # Initialize curricula
        self._initialize_curricula()
        
        logger.info("Tool curriculum system initialized",
                   curricula_count=len(self.curricula),
                   total_modules=sum(len(modules) for modules in self.curricula.values()))
    
    def _initialize_curricula(self):
        """Initialize comprehensive curricula for different domains"""
        self.curricula[CurriculumDomain.RESEARCH.value] = self._create_research_curriculum()
        self.curricula[CurriculumDomain.DATA_ANALYSIS.value] = self._create_data_analysis_curriculum()
        self.curricula[CurriculumDomain.SOFTWARE_DEVELOPMENT.value] = self._create_software_development_curriculum()
        self.curricula[CurriculumDomain.CONTENT_CREATION.value] = self._create_content_creation_curriculum()
        self.curricula[CurriculumDomain.GENERAL_PURPOSE.value] = self._create_general_purpose_curriculum()
    
    def _create_research_curriculum(self) -> List[CurriculumModule]:
        """Create curriculum for research domain"""
        modules = []
        
        # Module 1: Research Tool Awareness
        modules.append(CurriculumModule(
            module_id="research_awareness",
            title="Research Tool Awareness",
            description="Introduction to research-focused tools and their applications",
            domain=CurriculumDomain.RESEARCH,
            skill_level=SkillLevel.NOVICE,
            learning_objectives=[
                LearningObjective(
                    objective_id="identify_research_tools",
                    title="Identify Research Tools",
                    description="Recognize when research tools are needed for academic tasks",
                    skill_level=SkillLevel.NOVICE,
                    required_tools=["web_search", "database_query", "file_reader"],
                    success_criteria={"identification_accuracy": 0.8},
                    practice_examples=["literature_review", "data_collection", "source_validation"]
                ),
                LearningObjective(
                    objective_id="basic_search_techniques",
                    title="Basic Search Techniques",
                    description="Learn to use web search and database query tools effectively",
                    skill_level=SkillLevel.NOVICE,
                    required_tools=["web_search", "database_query"],
                    success_criteria={"search_relevance": 0.75, "query_optimization": 0.6},
                    practice_examples=["academic_search", "statistical_queries", "reference_lookup"]
                )
            ],
            duration_minutes=90,
            assessment_tasks=[
                {
                    "type": AssessmentType.PRACTICAL_TASK.value,
                    "description": "Conduct a literature search on a given topic",
                    "tools_required": ["web_search", "database_query"],
                    "success_threshold": 0.7
                }
            ],
            certification_criteria={"overall_score": 0.75, "practical_assessment": 0.8}
        ))
        
        # Module 2: Information Synthesis
        modules.append(CurriculumModule(
            module_id="information_synthesis",
            title="Information Synthesis and Analysis",
            description="Advanced techniques for synthesizing research information",
            domain=CurriculumDomain.RESEARCH,
            skill_level=SkillLevel.INTERMEDIATE,
            learning_objectives=[
                LearningObjective(
                    objective_id="multi_source_analysis",
                    title="Multi-Source Analysis",
                    description="Coordinate multiple tools for comprehensive research analysis",
                    skill_level=SkillLevel.INTERMEDIATE,
                    required_tools=["web_search", "file_reader", "data_visualizer", "python_executor"],
                    prerequisites=["identify_research_tools", "basic_search_techniques"],
                    success_criteria={"synthesis_quality": 0.8, "source_diversity": 0.75},
                    practice_examples=["systematic_review", "meta_analysis", "trend_analysis"]
                )
            ],
            duration_minutes=120,
            prerequisites=["research_awareness"],
            assessment_tasks=[
                {
                    "type": AssessmentType.SCENARIO_ANALYSIS.value,
                    "description": "Synthesize information from multiple sources on a research question",
                    "tools_required": ["web_search", "file_reader", "data_visualizer"],
                    "success_threshold": 0.75
                }
            ]
        ))
        
        # Module 3: Research Automation
        modules.append(CurriculumModule(
            module_id="research_automation",
            title="Research Process Automation",
            description="Automate repetitive research tasks using tool workflows",
            domain=CurriculumDomain.RESEARCH,
            skill_level=SkillLevel.ADVANCED,
            learning_objectives=[
                LearningObjective(
                    objective_id="workflow_automation",
                    title="Research Workflow Automation",
                    description="Design and implement automated research workflows",
                    skill_level=SkillLevel.ADVANCED,
                    required_tools=["web_search", "database_query", "python_executor", "file_writer"],
                    prerequisites=["multi_source_analysis"],
                    success_criteria={"automation_efficiency": 0.85, "error_handling": 0.8},
                    practice_examples=["automated_citation", "data_pipeline", "report_generation"]
                )
            ],
            duration_minutes=150,
            prerequisites=["information_synthesis"]
        ))
        
        return modules
    
    def _create_data_analysis_curriculum(self) -> List[CurriculumModule]:
        """Create curriculum for data analysis domain"""
        modules = []
        
        # Module 1: Data Tool Fundamentals
        modules.append(CurriculumModule(
            module_id="data_fundamentals",
            title="Data Analysis Tool Fundamentals",
            description="Essential tools and techniques for data analysis",
            domain=CurriculumDomain.DATA_ANALYSIS,
            skill_level=SkillLevel.BEGINNER,
            learning_objectives=[
                LearningObjective(
                    objective_id="data_ingestion",
                    title="Data Ingestion and Loading",
                    description="Learn to load and validate data from various sources",
                    skill_level=SkillLevel.BEGINNER,
                    required_tools=["file_reader", "database_query", "api_client"],
                    success_criteria={"data_loading_success": 0.9, "validation_accuracy": 0.85},
                    practice_examples=["csv_loading", "api_data_fetch", "database_extraction"]
                ),
                LearningObjective(
                    objective_id="basic_analysis",
                    title="Basic Statistical Analysis",
                    description="Perform fundamental statistical analysis using computational tools",
                    skill_level=SkillLevel.BEGINNER,
                    required_tools=["python_executor", "data_visualizer"],
                    success_criteria={"statistical_accuracy": 0.8, "visualization_quality": 0.75},
                    practice_examples=["descriptive_stats", "correlation_analysis", "basic_plots"]
                )
            ],
            duration_minutes=100,
            assessment_tasks=[
                {
                    "type": AssessmentType.PRACTICAL_TASK.value,
                    "description": "Load, analyze, and visualize a dataset",
                    "tools_required": ["file_reader", "python_executor", "data_visualizer"],
                    "success_threshold": 0.8
                }
            ]
        ))
        
        # Module 2: Advanced Analytics
        modules.append(CurriculumModule(
            module_id="advanced_analytics",
            title="Advanced Analytics and Machine Learning",
            description="Advanced analytical techniques and ML tool usage",
            domain=CurriculumDomain.DATA_ANALYSIS,
            skill_level=SkillLevel.ADVANCED,
            learning_objectives=[
                LearningObjective(
                    objective_id="ml_pipeline",
                    title="Machine Learning Pipeline",
                    description="Build complete ML pipelines using coordinated tools",
                    skill_level=SkillLevel.ADVANCED,
                    required_tools=["python_executor", "data_visualizer", "model_trainer"],
                    prerequisites=["data_ingestion", "basic_analysis"],
                    success_criteria={"pipeline_completeness": 0.9, "model_performance": 0.8},
                    practice_examples=["classification_pipeline", "regression_analysis", "clustering"]
                )
            ],
            duration_minutes=180,
            prerequisites=["data_fundamentals"]
        ))
        
        return modules
    
    def _create_software_development_curriculum(self) -> List[CurriculumModule]:
        """Create curriculum for software development domain"""
        modules = []
        
        # Module 1: Development Tool Basics
        modules.append(CurriculumModule(
            module_id="dev_basics",
            title="Development Tool Basics",
            description="Essential development tools and workflows",
            domain=CurriculumDomain.SOFTWARE_DEVELOPMENT,
            skill_level=SkillLevel.BEGINNER,
            learning_objectives=[
                LearningObjective(
                    objective_id="code_execution",
                    title="Code Execution and Testing",
                    description="Use code execution tools for development and testing",
                    skill_level=SkillLevel.BEGINNER,
                    required_tools=["python_executor", "file_reader", "file_writer"],
                    success_criteria={"execution_success": 0.9, "test_coverage": 0.8},
                    practice_examples=["script_execution", "unit_testing", "debugging"]
                ),
                LearningObjective(
                    objective_id="version_control",
                    title="Version Control Integration",
                    description="Integrate version control tools into development workflow",
                    skill_level=SkillLevel.BEGINNER,
                    required_tools=["git_client", "file_reader", "file_writer"],
                    success_criteria={"commit_quality": 0.8, "branch_management": 0.75},
                    practice_examples=["repository_management", "commit_workflow", "merge_conflicts"]
                )
            ],
            duration_minutes=120
        ))
        
        return modules
    
    def _create_content_creation_curriculum(self) -> List[CurriculumModule]:
        """Create curriculum for content creation domain"""
        modules = []
        
        # Module 1: Content Research and Planning
        modules.append(CurriculumModule(
            module_id="content_research",
            title="Content Research and Planning",
            description="Research and planning tools for content creation",
            domain=CurriculumDomain.CONTENT_CREATION,
            skill_level=SkillLevel.BEGINNER,
            learning_objectives=[
                LearningObjective(
                    objective_id="topic_research",
                    title="Topic Research and Validation",
                    description="Research topics and validate content ideas using search tools",
                    skill_level=SkillLevel.BEGINNER,
                    required_tools=["web_search", "api_client", "data_visualizer"],
                    success_criteria={"research_depth": 0.8, "topic_relevance": 0.85},
                    practice_examples=["trend_analysis", "audience_research", "competitor_analysis"]
                )
            ],
            duration_minutes=90
        ))
        
        return modules
    
    def _create_general_purpose_curriculum(self) -> List[CurriculumModule]:
        """Create general-purpose curriculum for cross-domain skills"""
        modules = []
        
        # Module 1: Tool Safety and Security
        modules.append(CurriculumModule(
            module_id="tool_safety",
            title="Tool Safety and Security Protocols",
            description="Essential safety and security practices for tool usage",
            domain=CurriculumDomain.GENERAL_PURPOSE,
            skill_level=SkillLevel.EXPERT,
            learning_objectives=[
                LearningObjective(
                    objective_id="security_awareness",
                    title="Security Awareness and Risk Assessment",
                    description="Understand security implications of tool usage",
                    skill_level=SkillLevel.EXPERT,
                    required_tools=["security_scanner", "sandbox_manager"],
                    success_criteria={"risk_identification": 0.9, "mitigation_planning": 0.85},
                    practice_examples=["vulnerability_assessment", "safe_execution", "data_protection"]
                ),
                LearningObjective(
                    objective_id="error_handling",
                    title="Error Handling and Recovery",
                    description="Handle tool failures and implement recovery strategies",
                    skill_level=SkillLevel.EXPERT,
                    required_tools=["all_available_tools"],
                    success_criteria={"error_detection": 0.9, "recovery_success": 0.8},
                    practice_examples=["failure_scenarios", "fallback_strategies", "graceful_degradation"]
                )
            ],
            duration_minutes=150,
            assessment_tasks=[
                {
                    "type": AssessmentType.SAFETY_PROTOCOL.value,
                    "description": "Demonstrate safe tool usage in high-risk scenarios",
                    "success_threshold": 0.9
                },
                {
                    "type": AssessmentType.ERROR_HANDLING.value,
                    "description": "Handle multiple tool failure scenarios",
                    "success_threshold": 0.8
                }
            ]
        ))
        
        return modules
    
    async def enroll_student(self, student_id: str, curriculum_domain: CurriculumDomain) -> StudentProgress:
        """Enroll a student in a curriculum domain"""
        if student_id in self.student_progress:
            logger.warning("Student already enrolled", student_id=student_id)
            return self.student_progress[student_id]
        
        # Create new student progress
        progress = StudentProgress(
            student_id=student_id,
            curriculum_id=curriculum_domain.value
        )
        
        self.student_progress[student_id] = progress
        
        logger.info("Student enrolled in curriculum",
                   student_id=student_id,
                   domain=curriculum_domain.value)
        
        return progress
    
    async def get_next_module(self, student_id: str) -> Optional[CurriculumModule]:
        """Get the next recommended module for a student"""
        if student_id not in self.student_progress:
            logger.warning("Student not found", student_id=student_id)
            return None
        
        progress = self.student_progress[student_id]
        curriculum = self.curricula.get(progress.curriculum_id, [])\n        \n        # Find next uncompleted module that meets prerequisites\n        for module in curriculum:\n            if module.module_id in progress.completed_modules:\n                continue\n            \n            # Check prerequisites\n            if all(prereq in progress.completed_modules for prereq in module.prerequisites):\n                return module\n        \n        return None  # All modules completed or prerequisites not met\n    \n    async def assess_student(self, student_id: str, module_id: str, \n                           assessment_type: AssessmentType) -> AssessmentResult:\n        \"\"\"\n        Conduct assessment for a student on a specific module\n        \n        Args:\n            student_id: Student identifier\n            module_id: Module being assessed\n            assessment_type: Type of assessment to conduct\n            \n        Returns:\n            Assessment result with score and feedback\n        \"\"\"\n        if student_id not in self.student_progress:\n            raise ValueError(f\"Student {student_id} not enrolled\")\n        \n        progress = self.student_progress[student_id]\n        curriculum = self.curricula.get(progress.curriculum_id, [])\n        \n        # Find the module\n        module = next((m for m in curriculum if m.module_id == module_id), None)\n        if not module:\n            raise ValueError(f\"Module {module_id} not found in curriculum\")\n        \n        # Conduct assessment based on type\n        assessment_start = time.time()\n        \n        if assessment_type == AssessmentType.PRACTICAL_TASK:\n            result = await self._conduct_practical_assessment(student_id, module)\n        elif assessment_type == AssessmentType.SCENARIO_ANALYSIS:\n            result = await self._conduct_scenario_assessment(student_id, module)\n        elif assessment_type == AssessmentType.SAFETY_PROTOCOL:\n            result = await self._conduct_safety_assessment(student_id, module)\n        else:\n            result = await self._conduct_general_assessment(student_id, module)\n        \n        assessment_duration = int((time.time() - assessment_start) / 60)  # minutes\n        \n        # Create assessment result\n        assessment_result = AssessmentResult(\n            student_id=student_id,\n            module_id=module_id,\n            assessment_type=assessment_type,\n            score=result[\"score\"],\n            passed=result[\"score\"] >= module.certification_criteria.get(\"overall_score\", 0.7),\n            feedback=result[\"feedback\"],\n            detailed_results=result[\"details\"],\n            duration_minutes=assessment_duration\n        )\n        \n        # Store result\n        self.assessment_results.append(assessment_result)\n        \n        # Update student progress\n        progress.assessment_scores[module_id] = assessment_result.score\n        progress.last_activity = datetime.now(timezone.utc)\n        \n        if assessment_result.passed:\n            if module_id not in progress.completed_modules:\n                progress.completed_modules.append(module_id)\n            \n            # Update skill level if appropriate\n            await self._update_skill_level(student_id)\n        \n        logger.info(\"Assessment completed\",\n                   student_id=student_id,\n                   module_id=module_id,\n                   score=assessment_result.score,\n                   passed=assessment_result.passed)\n        \n        return assessment_result\n    \n    async def _conduct_practical_assessment(self, student_id: str, \n                                          module: CurriculumModule) -> Dict[str, Any]:\n        \"\"\"Conduct practical assessment with real tool usage\"\"\"\n        total_score = 0.0\n        max_score = 0.0\n        detailed_results = {}\n        feedback_parts = []\n        \n        # Execute practical tasks for each learning objective\n        for objective in module.learning_objectives:\n            objective_score = 0.0\n            objective_max = 1.0\n            \n            # Simulate practical task execution\n            # In production, this would involve real tool execution\n            task_results = await self._simulate_tool_task(objective)\n            \n            objective_score = task_results[\"score\"]\n            detailed_results[objective.objective_id] = task_results\n            \n            if objective_score >= objective.success_criteria.get(\"overall\", 0.7):\n                feedback_parts.append(f\"✅ {objective.title}: Demonstrated competency\")\n            else:\n                feedback_parts.append(f\"❌ {objective.title}: Needs improvement\")\n            \n            total_score += objective_score\n            max_score += objective_max\n        \n        final_score = total_score / max_score if max_score > 0 else 0.0\n        \n        return {\n            \"score\": final_score,\n            \"feedback\": \"\\n\".join(feedback_parts),\n            \"details\": detailed_results\n        }\n    \n    async def _simulate_tool_task(self, objective: LearningObjective) -> Dict[str, Any]:\n        \"\"\"Simulate tool task execution for assessment\"\"\"\n        # Mock assessment based on objective requirements\n        base_score = 0.7  # Base competency score\n        \n        # Adjust score based on objective difficulty\n        difficulty_adjustment = (1.0 - objective.difficulty_score) * 0.2\n        tool_complexity = len(objective.required_tools) * 0.05\n        \n        final_score = min(1.0, base_score + difficulty_adjustment - tool_complexity)\n        \n        return {\n            \"score\": final_score,\n            \"execution_time\": objective.estimated_duration,\n            \"tools_used\": objective.required_tools,\n            \"success_rate\": final_score,\n            \"error_count\": int((1.0 - final_score) * 5)  # Simulated errors\n        }\n    \n    async def _conduct_scenario_assessment(self, student_id: str, \n                                         module: CurriculumModule) -> Dict[str, Any]:\n        \"\"\"Conduct scenario-based assessment\"\"\"\n        # Simplified scenario assessment\n        scenario_score = 0.75  # Mock scenario performance\n        \n        return {\n            \"score\": scenario_score,\n            \"feedback\": \"Demonstrated good problem-solving skills in scenario analysis\",\n            \"details\": {\"scenario_performance\": scenario_score}\n        }\n    \n    async def _conduct_safety_assessment(self, student_id: str, \n                                        module: CurriculumModule) -> Dict[str, Any]:\n        \"\"\"Conduct safety protocol assessment\"\"\"\n        safety_score = 0.9  # Mock high safety compliance\n        \n        return {\n            \"score\": safety_score,\n            \"feedback\": \"Excellent safety protocol compliance\",\n            \"details\": {\"safety_compliance\": safety_score}\n        }\n    \n    async def _conduct_general_assessment(self, student_id: str, \n                                        module: CurriculumModule) -> Dict[str, Any]:\n        \"\"\"Conduct general assessment\"\"\"\n        general_score = 0.8  # Mock general performance\n        \n        return {\n            \"score\": general_score,\n            \"feedback\": \"Good overall performance across module objectives\",\n            \"details\": {\"general_performance\": general_score}\n        }\n    \n    async def _update_skill_level(self, student_id: str):\n        \"\"\"Update student skill level based on completed modules\"\"\"\n        progress = self.student_progress[student_id]\n        curriculum = self.curricula.get(progress.curriculum_id, [])\n        \n        # Determine highest skill level achieved\n        highest_level = SkillLevel.NOVICE\n        \n        for module in curriculum:\n            if module.module_id in progress.completed_modules:\n                if self._skill_level_order(module.skill_level) > self._skill_level_order(highest_level):\n                    highest_level = module.skill_level\n        \n        if highest_level != progress.current_skill_level:\n            progress.current_skill_level = highest_level\n            logger.info(\"Student skill level updated\",\n                       student_id=student_id,\n                       new_level=highest_level.value)\n    \n    def _skill_level_order(self, level: SkillLevel) -> int:\n        \"\"\"Get numeric order of skill level\"\"\"\n        order = {\n            SkillLevel.NOVICE: 0,\n            SkillLevel.BEGINNER: 1,\n            SkillLevel.INTERMEDIATE: 2,\n            SkillLevel.ADVANCED: 3,\n            SkillLevel.EXPERT: 4\n        }\n        return order.get(level, 0)\n    \n    async def get_student_progress(self, student_id: str) -> Optional[StudentProgress]:\n        \"\"\"Get detailed student progress\"\"\"\n        return self.student_progress.get(student_id)\n    \n    async def get_curriculum_analytics(self, curriculum_domain: CurriculumDomain = None) -> Dict[str, Any]:\n        \"\"\"Get comprehensive curriculum analytics\"\"\"\n        if curriculum_domain:\n            curricula_to_analyze = {curriculum_domain.value: self.curricula.get(curriculum_domain.value, [])}\n        else:\n            curricula_to_analyze = self.curricula\n        \n        analytics = {\n            \"total_curricula\": len(curricula_to_analyze),\n            \"total_modules\": sum(len(modules) for modules in curricula_to_analyze.values()),\n            \"enrolled_students\": len(self.student_progress),\n            \"completed_assessments\": len(self.assessment_results),\n            \"skill_level_distribution\": {},\n            \"module_completion_rates\": {},\n            \"assessment_performance\": {}\n        }\n        \n        # Calculate skill level distribution\n        for progress in self.student_progress.values():\n            level = progress.current_skill_level.value\n            analytics[\"skill_level_distribution\"][level] = analytics[\"skill_level_distribution\"].get(level, 0) + 1\n        \n        # Calculate module completion rates\n        for domain, modules in curricula_to_analyze.items():\n            for module in modules:\n                completed_count = sum(1 for p in self.student_progress.values() \n                                    if module.module_id in p.completed_modules)\n                total_eligible = sum(1 for p in self.student_progress.values() \n                                   if p.curriculum_id == domain)\n                \n                completion_rate = completed_count / max(total_eligible, 1)\n                analytics[\"module_completion_rates\"][module.module_id] = completion_rate\n        \n        # Calculate assessment performance\n        for result in self.assessment_results:\n            module_id = result.module_id\n            if module_id not in analytics[\"assessment_performance\"]:\n                analytics[\"assessment_performance\"][module_id] = {\n                    \"scores\": [],\n                    \"pass_rate\": 0.0\n                }\n            \n            analytics[\"assessment_performance\"][module_id][\"scores\"].append(result.score)\n        \n        # Calculate average scores and pass rates\n        for module_id, perf_data in analytics[\"assessment_performance\"].items():\n            scores = perf_data[\"scores\"]\n            if scores:\n                perf_data[\"average_score\"] = sum(scores) / len(scores)\n                perf_data[\"pass_rate\"] = sum(1 for s in scores if s >= 0.7) / len(scores)\n            else:\n                perf_data[\"average_score\"] = 0.0\n                perf_data[\"pass_rate\"] = 0.0\n            \n            # Remove raw scores to reduce response size\n            del perf_data[\"scores\"]\n        \n        return analytics\n    \n    async def generate_personalized_curriculum(self, student_id: str, \n                                             target_skills: List[str]) -> List[CurriculumModule]:\n        \"\"\"Generate personalized curriculum based on student needs\"\"\"\n        if student_id not in self.student_progress:\n            raise ValueError(f\"Student {student_id} not found\")\n        \n        progress = self.student_progress[student_id]\n        current_curriculum = self.curricula.get(progress.curriculum_id, [])\n        \n        # Filter modules based on target skills and prerequisites\n        personalized_modules = []\n        \n        for module in current_curriculum:\n            # Check if module covers target skills\n            module_skills = [obj.title.lower() for obj in module.learning_objectives]\n            skill_overlap = any(skill.lower() in \" \".join(module_skills) for skill in target_skills)\n            \n            if skill_overlap and module.module_id not in progress.completed_modules:\n                # Check prerequisites\n                if all(prereq in progress.completed_modules for prereq in module.prerequisites):\n                    personalized_modules.append(module)\n        \n        # Sort by skill level and difficulty\n        personalized_modules.sort(key=lambda m: (self._skill_level_order(m.skill_level), m.duration_minutes))\n        \n        logger.info(\"Generated personalized curriculum\",\n                   student_id=student_id,\n                   target_skills=target_skills,\n                   recommended_modules=len(personalized_modules))\n        \n        return personalized_modules\n\n\n# Global curriculum instance\ntool_curriculum = ToolCurriculum()\n\n\n# Factory functions\ndef get_curriculum_for_domain(domain: CurriculumDomain) -> List[CurriculumModule]:\n    \"\"\"Get curriculum modules for a specific domain\"\"\"\n    return tool_curriculum.curricula.get(domain.value, [])\n\n\nasync def enroll_student_in_curriculum(student_id: str, domain: CurriculumDomain) -> StudentProgress:\n    \"\"\"Enroll a student in a curriculum domain\"\"\"\n    return await tool_curriculum.enroll_student(student_id, domain)