"""
Hierarchical Architect AI
Enhanced from Co-Lab's decomposition.py for multi-level recursive decomposition
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import UUID, uuid4
import asyncio
import structlog
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from prsm.core.models import ArchitectTask, TaskHierarchy, TaskStatus, AgentType
from prsm.core.config import get_settings
from prsm.agents.base import BaseAgent


logger = structlog.get_logger(__name__)
settings = get_settings()


class DecompositionStrategy(str, Enum):
    """Strategies for task decomposition"""
    SEQUENTIAL = "sequential"  # Tasks that must be done in order
    PARALLEL = "parallel"     # Tasks that can be done simultaneously
    HYBRID = "hybrid"         # Mix of sequential and parallel tasks
    CONTEXTUAL = "contextual" # Strategy based on task context


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TaskAnalytics:
    """Analytics for task decomposition"""
    word_count: int
    domain_indicators: List[str]
    complexity_factors: Dict[str, float]
    estimated_duration: float
    resource_requirements: List[str]
    parallelization_potential: float


class HierarchicalArchitect(BaseAgent):
    """
    Hierarchical Architect AI for recursive task decomposition
    
    Extends Co-Lab's decomposition concept to support:
    - Multi-level decomposition hierarchy
    - Task complexity assessment
    - Dependency identification
    - Recursive refinement
    """
    
    def __init__(self, level: int = 1, max_depth: int = None, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id, agent_type=AgentType.ARCHITECT)
        self.level = level
        self.max_depth = max_depth or settings.max_decomposition_depth
        
        # Enhanced planning capabilities
        self.decomposition_cache = {}  # Cache for similar tasks
        self.task_analytics = {}       # Analytics for processed tasks
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.complexity_weights = self._initialize_complexity_weights()
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> TaskHierarchy:
        """
        Process input data by decomposing it into a task hierarchy
        
        Args:
            input_data: Task description string or ArchitectTask
            context: Optional context with session_id, etc.
            
        Returns:
            TaskHierarchy: Decomposed task structure
        """
        # Extract task information
        if isinstance(input_data, str):
            task_description = input_data
            session_id = context.get("session_id") if context else uuid4()
        elif isinstance(input_data, dict):
            task_description = input_data.get("task", str(input_data))
            session_id = input_data.get("session_id", uuid4())
        else:
            task_description = str(input_data)
            session_id = context.get("session_id") if context else uuid4()
        
        # Perform recursive decomposition
        return await self.recursive_decompose(
            task=task_description,
            session_id=session_id,
            parent_task_id=None,
            current_depth=0
        )
        
    async def recursive_decompose(
        self, 
        task: str, 
        session_id: UUID,
        parent_task_id: Optional[UUID] = None,
        current_depth: int = 0
    ) -> TaskHierarchy:
        """
        Recursively decompose a task into a hierarchy of subtasks
        
        Args:
            task: Task description to decompose
            session_id: Current session ID
            parent_task_id: Parent task ID if this is a subtask
            current_depth: Current decomposition depth
            
        Returns:
            TaskHierarchy: Complete task decomposition
        """
        # Ensure task is a string
        if not isinstance(task, str):
            task = str(task)
        
        logger.info("Starting recursive decomposition", 
                   hierarchy_level=self.level, depth=current_depth, task_length=len(task))
        
        complexity = await self.assess_complexity(task)
        
        # Create root task
        root_task = ArchitectTask(
            session_id=session_id,
            parent_task_id=parent_task_id,
            level=current_depth,
            instruction=task,
            complexity_score=complexity
        )
        
        hierarchy = TaskHierarchy(
            root_task=root_task,
            max_depth=self.max_depth
        )
        
        # Decide if further decomposition is needed
        if await self._should_decompose(root_task, current_depth):
            subtasks = await self._decompose_task(root_task)
            
            # Process each subtask recursively
            for subtask_desc in subtasks:
                subtask_hierarchy = await self.recursive_decompose(
                    task=subtask_desc,
                    session_id=session_id,
                    parent_task_id=root_task.task_id,
                    current_depth=current_depth + 1
                )
                
                # Merge subtask hierarchy
                hierarchy.subtasks.update(subtask_hierarchy.subtasks)
                hierarchy.dependencies.update(subtask_hierarchy.dependencies)
        
        hierarchy.total_tasks = len(hierarchy.subtasks) + 1  # +1 for root
        
        logger.info("Decomposition complete", 
                   total_tasks=hierarchy.total_tasks, 
                   max_depth_reached=current_depth)
        
        return hierarchy
    
    async def assess_complexity(self, task: str, use_advanced_analysis: bool = True) -> float:
        """
        Enhanced complexity assessment with advanced algorithms
        
        Args:
            task: Task description
            use_advanced_analysis: Whether to use advanced NLP analysis
            
        Returns:
            float: Complexity score (0.0 = simple, 1.0 = very complex)
        """
        task_str = str(task) if not isinstance(task, str) else task
        
        # Generate analytics for the task
        analytics = await self._analyze_task(task_str)
        
        # Multi-dimensional complexity assessment
        complexity_factors = {
            'length': self._assess_length_complexity(analytics.word_count),
            'domain': self._assess_domain_complexity(analytics.domain_indicators),
            'process': self._assess_process_complexity(task_str),
            'cognitive': self._assess_cognitive_complexity(task_str),
            'resource': self._assess_resource_complexity(analytics.resource_requirements),
            'temporal': self._assess_temporal_complexity(task_str)
        }
        
        # Weighted combination based on learned weights
        total_complexity = sum(
            complexity_factors[factor] * self.complexity_weights[factor]
            for factor in complexity_factors
        )
        
        # Store analytics for future reference
        task_hash = hashlib.md5(task_str.encode()).hexdigest()[:8]
        analytics.complexity_factors = complexity_factors
        self.task_analytics[task_hash] = analytics
        
        return min(total_complexity, 1.0)
    
    async def identify_dependencies(self, subtasks: List[str], strategy: DecompositionStrategy = DecompositionStrategy.CONTEXTUAL) -> Dict[str, List[str]]:
        """
        Enhanced dependency identification with semantic analysis
        
        Args:
            subtasks: List of subtask descriptions
            strategy: Decomposition strategy to use
            
        Returns:
            Dict mapping task indices to their dependencies
        """
        dependencies = defaultdict(list)
        
        # Multi-layered dependency analysis
        for i, task in enumerate(subtasks):
            task_lower = task.lower()
            
            for j, other_task in enumerate(subtasks):
                if i != j:
                    dependency_score = await self._calculate_dependency_score(task, other_task)
                    
                    if dependency_score > 0.7:  # Strong dependency
                        dependencies[str(i)].append(str(j))
                    elif dependency_score > 0.4 and strategy == DecompositionStrategy.SEQUENTIAL:
                        dependencies[str(i)].append(str(j))  # Weaker dependency in sequential mode
        
        # Apply strategy-specific dependency rules
        if strategy == DecompositionStrategy.PARALLEL:
            dependencies = self._optimize_for_parallelization(dependencies, subtasks)
        elif strategy == DecompositionStrategy.HYBRID:
            dependencies = self._create_hybrid_dependencies(dependencies, subtasks)
        
        return dict(dependencies)
    
    async def get_task_priorities(self, subtasks: List[str]) -> Dict[str, TaskPriority]:
        """Assign priorities to subtasks based on importance and dependencies"""
        priorities = {}
        
        for i, task in enumerate(subtasks):
            task_lower = task.lower()
            
            # Critical priority indicators
            if any(indicator in task_lower for indicator in ['critical', 'essential', 'required', 'must']):
                priorities[str(i)] = TaskPriority.CRITICAL
            # High priority indicators
            elif any(indicator in task_lower for indicator in ['important', 'priority', 'urgent', 'key']):
                priorities[str(i)] = TaskPriority.HIGH
            # Low priority indicators
            elif any(indicator in task_lower for indicator in ['optional', 'nice to have', 'if time']):
                priorities[str(i)] = TaskPriority.LOW
            else:
                priorities[str(i)] = TaskPriority.MEDIUM
        
        return priorities
    
    async def estimate_task_completion_time(self, subtasks: List[str]) -> Dict[str, float]:
        """Estimate completion time for each subtask"""
        completion_times = {}
        
        for i, task in enumerate(subtasks):
            analytics = await self._analyze_task(task)
            completion_times[str(i)] = analytics.estimated_duration
        
        return completion_times
    
    async def _should_decompose(self, task: ArchitectTask, depth: int) -> bool:
        """
        Enhanced decomposition decision with multiple criteria
        
        Args:
            task: Task to evaluate
            depth: Current decomposition depth
            
        Returns:
            bool: True if task should be decomposed further
        """
        # Don't decompose if we've reached max depth
        if depth >= self.max_depth:
            return False
        
        # Get task analytics
        task_hash = hashlib.md5(task.instruction.encode()).hexdigest()[:8]
        analytics = self.task_analytics.get(task_hash)
        
        # Multiple decomposition criteria
        criteria = {
            'complexity_threshold': task.complexity_score >= 0.3,
            'length_threshold': len(task.instruction) >= 50,
            'multi_concept': await self._has_multiple_concepts(task.instruction),
            'parallelization_potential': analytics and analytics.parallelization_potential > 0.5,
            'domain_complexity': self._requires_domain_expertise(task.instruction)
        }
        
        # Weighted decision based on criteria
        decomposition_score = sum([
            criteria['complexity_threshold'] * 0.3,
            criteria['length_threshold'] * 0.2,
            criteria['multi_concept'] * 0.25,
            criteria['parallelization_potential'] * 0.15,
            criteria['domain_complexity'] * 0.1
        ])
        
        # Decompose if score exceeds threshold
        return decomposition_score > 0.5
    
    async def _decompose_task(self, task: ArchitectTask) -> List[str]:
        """
        Enhanced task decomposition with strategy selection
        
        Args:
            task: Task to decompose
            
        Returns:
            List of subtask descriptions
        """
        logger.info("Decomposing task", task_id=task.task_id, 
                   complexity=task.complexity_score)
        
        # Select optimal decomposition strategy
        strategy = await self._select_decomposition_strategy(task)
        instruction = task.instruction
        
        # Use different decomposition approaches based on complexity and strategy
        if task.complexity_score > 0.7 and strategy == DecompositionStrategy.CONTEXTUAL:
            subtasks = await self._llm_based_decomposition(instruction)
        elif strategy == DecompositionStrategy.PARALLEL:
            subtasks = await self._parallel_decomposition(instruction)
        elif strategy == DecompositionStrategy.SEQUENTIAL:
            subtasks = await self._sequential_decomposition(instruction)
        else:
            subtasks = await self._rule_based_decomposition(instruction)
        
        # Optimize subtasks based on strategy
        return await self._optimize_subtasks(subtasks, strategy)
    
    async def _llm_based_decomposition(self, instruction: str) -> List[str]:
        """Use LLM to decompose complex tasks intelligently"""
        try:
            from prsm.agents.executors.model_executor import ModelExecutor
            
            # Create decomposition prompt
            decomposition_prompt = f"""
Break down the following complex task into 3-5 manageable subtasks. Each subtask should be:
- Specific and actionable
- Logically ordered (earlier tasks enable later ones)
- Focused on a single aspect of the overall goal

Task to decompose: {instruction}

Provide subtasks in this format:
1. [First subtask]
2. [Second subtask]
3. [Third subtask]
etc.

Focus on logical workflow and dependencies.
"""
            
            # Execute LLM decomposition
            executor = ModelExecutor()
            execution_request = {
                "task": decomposition_prompt,
                "models": ["gpt-3.5-turbo"],  # Use efficient model for decomposition
                "parallel": False
            }
            
            results = await executor.process(execution_request)
            
            if results and len(results) > 0 and results[0].success:
                response = results[0].result.get("content", "")
                subtasks = self._parse_llm_subtasks(response)
                if subtasks:
                    return subtasks
            
            # Fallback to rule-based if LLM fails
            logger.warning("LLM decomposition failed, falling back to rule-based")
            return await self._rule_based_decomposition(instruction)
            
        except Exception as e:
            logger.error(f"LLM decomposition error: {e}")
            return await self._rule_based_decomposition(instruction)
    
    async def _rule_based_decomposition(self, instruction: str) -> List[str]:
        """Rule-based task decomposition as fallback"""
        if "analyze" in instruction.lower():
            return [
                f"Gather relevant data for: {instruction}",
                f"Process and clean data for: {instruction}",
                f"Perform analysis of: {instruction}",
                f"Summarize findings for: {instruction}"
            ]
        elif "research" in instruction.lower():
            return [
                f"Define research scope for: {instruction}",
                f"Collect relevant sources for: {instruction}",
                f"Synthesize information for: {instruction}",
                f"Draw conclusions for: {instruction}"
            ]
        elif "implement" in instruction.lower() or "build" in instruction.lower():
            return [
                f"Design architecture for: {instruction}",
                f"Implement core functionality for: {instruction}",
                f"Test and validate: {instruction}",
                f"Document and finalize: {instruction}"
            ]
        elif "compare" in instruction.lower() or "evaluate" in instruction.lower():
            return [
                f"Identify comparison criteria for: {instruction}",
                f"Collect comparison data for: {instruction}",
                f"Perform comparative analysis for: {instruction}",
                f"Summarize evaluation results for: {instruction}"
            ]
        else:
            # Generic decomposition
            return [
                f"Plan approach for: {instruction}",
                f"Execute main work for: {instruction}",
                f"Review and refine: {instruction}"
            ]
    
    def _parse_llm_subtasks(self, response: str) -> List[str]:
        """Parse subtasks from LLM response"""
        subtasks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered list items
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                task = line
                if '. ' in task:
                    task = task.split('. ', 1)[1]
                elif '- ' in task:
                    task = task.split('- ', 1)[1]
                
                # Remove brackets if present
                task = task.strip('[]')
                
                if task and len(task) > 10:  # Valid subtask
                    subtasks.append(task)
        
        return subtasks if len(subtasks) >= 2 else []
    
    def _initialize_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific knowledge base"""
        return {
            'technical': {
                'keywords': ['algorithm', 'optimization', 'architecture', 'implementation', 'testing'],
                'complexity_multiplier': 1.3,
                'decomposition_depth': 4
            },
            'research': {
                'keywords': ['analyze', 'investigate', 'study', 'compare', 'evaluate'],
                'complexity_multiplier': 1.2,
                'decomposition_depth': 3
            },
            'creative': {
                'keywords': ['design', 'create', 'generate', 'brainstorm', 'innovate'],
                'complexity_multiplier': 1.1,
                'decomposition_depth': 3
            },
            'analytical': {
                'keywords': ['calculate', 'measure', 'quantify', 'assess', 'determine'],
                'complexity_multiplier': 1.4,
                'decomposition_depth': 4
            }
        }
    
    def _initialize_complexity_weights(self) -> Dict[str, float]:
        """Initialize weights for complexity factors"""
        return {
            'length': 0.15,
            'domain': 0.25,
            'process': 0.20,
            'cognitive': 0.15,
            'resource': 0.15,
            'temporal': 0.10
        }
    
    async def _analyze_task(self, task: str) -> TaskAnalytics:
        """Perform comprehensive task analysis"""
        words = task.split()
        word_count = len(words)
        
        # Identify domain indicators
        domain_indicators = []
        for domain, info in self.domain_knowledge.items():
            if any(keyword in task.lower() for keyword in info['keywords']):
                domain_indicators.append(domain)
        
        # Estimate resource requirements
        resource_requirements = self._identify_resource_requirements(task)
        
        # Calculate parallelization potential
        parallelization_potential = self._calculate_parallelization_potential(task)
        
        # Estimate duration (simplified heuristic)
        estimated_duration = self._estimate_task_duration(task, word_count)
        
        return TaskAnalytics(
            word_count=word_count,
            domain_indicators=domain_indicators,
            complexity_factors={},
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements,
            parallelization_potential=parallelization_potential
        )
    
    def _assess_length_complexity(self, word_count: int) -> float:
        """Assess complexity based on task length"""
        if word_count < 10:
            return 0.1
        elif word_count < 30:
            return 0.3
        elif word_count < 100:
            return 0.6
        else:
            return 0.9
    
    def _assess_domain_complexity(self, domain_indicators: List[str]) -> float:
        """Assess complexity based on domain indicators"""
        if not domain_indicators:
            return 0.2
        
        complexity = 0.0
        for domain in domain_indicators:
            if domain in self.domain_knowledge:
                complexity += self.domain_knowledge[domain]['complexity_multiplier'] * 0.2
        
        return min(complexity, 1.0)
    
    def _assess_process_complexity(self, task: str) -> float:
        """Assess complexity based on process indicators"""
        task_lower = task.lower()
        
        # Sequential process indicators
        sequential_indicators = ['first', 'then', 'next', 'finally', 'after', 'before']
        sequential_count = sum(1 for indicator in sequential_indicators if indicator in task_lower)
        
        # Parallel process indicators
        parallel_indicators = ['simultaneously', 'concurrently', 'in parallel', 'at the same time']
        parallel_count = sum(1 for indicator in parallel_indicators if indicator in task_lower)
        
        # Complex process indicators
        complex_indicators = ['coordinate', 'orchestrate', 'synchronize', 'integrate']
        complex_count = sum(1 for indicator in complex_indicators if indicator in task_lower)
        
        return min((sequential_count * 0.1) + (parallel_count * 0.15) + (complex_count * 0.2), 1.0)
    
    def _assess_cognitive_complexity(self, task: str) -> float:
        """Assess cognitive complexity requirements"""
        task_lower = task.lower()
        
        cognitive_indicators = {
            'high': ['analyze', 'synthesize', 'evaluate', 'design', 'create', 'innovate'],
            'medium': ['compare', 'contrast', 'explain', 'interpret', 'summarize'],
            'low': ['list', 'identify', 'describe', 'recall', 'repeat']
        }
        
        high_count = sum(1 for indicator in cognitive_indicators['high'] if indicator in task_lower)
        medium_count = sum(1 for indicator in cognitive_indicators['medium'] if indicator in task_lower)
        low_count = sum(1 for indicator in cognitive_indicators['low'] if indicator in task_lower)
        
        return min((high_count * 0.3) + (medium_count * 0.2) + (low_count * 0.1), 1.0)
    
    def _assess_resource_complexity(self, resource_requirements: List[str]) -> float:
        """Assess complexity based on resource requirements"""
        if not resource_requirements:
            return 0.1
        
        resource_weights = {
            'compute': 0.3,
            'memory': 0.25,
            'network': 0.2,
            'storage': 0.15,
            'external_api': 0.4,
            'database': 0.35
        }
        
        complexity = sum(resource_weights.get(resource, 0.1) for resource in resource_requirements)
        return min(complexity, 1.0)
    
    def _assess_temporal_complexity(self, task: str) -> float:
        """Assess complexity based on temporal requirements"""
        task_lower = task.lower()
        
        temporal_indicators = {
            'urgent': ['urgent', 'asap', 'immediately', 'now'],
            'scheduled': ['schedule', 'timeline', 'deadline', 'by'],
            'long_term': ['long-term', 'extended', 'ongoing', 'continuous']
        }
        
        if any(indicator in task_lower for indicator in temporal_indicators['urgent']):
            return 0.8
        elif any(indicator in task_lower for indicator in temporal_indicators['scheduled']):
            return 0.6
        elif any(indicator in task_lower for indicator in temporal_indicators['long_term']):
            return 0.4
        else:
            return 0.2
    
    def _identify_resource_requirements(self, task: str) -> List[str]:
        """Identify resource requirements from task description"""
        task_lower = task.lower()
        resources = []
        
        resource_indicators = {
            'compute': ['calculate', 'process', 'analyze', 'compute', 'algorithm'],
            'memory': ['large', 'big', 'massive', 'extensive', 'store'],
            'network': ['api', 'request', 'download', 'upload', 'fetch'],
            'storage': ['save', 'store', 'database', 'file', 'data'],
            'external_api': ['api', 'service', 'external', 'third-party'],
            'database': ['database', 'query', 'sql', 'table', 'record']
        }
        
        for resource, indicators in resource_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                resources.append(resource)
        
        return resources
    
    def _calculate_parallelization_potential(self, task: str) -> float:
        """Calculate how much the task can benefit from parallelization"""
        task_lower = task.lower()
        
        # Parallel-friendly indicators
        parallel_friendly = ['batch', 'multiple', 'each', 'all', 'list of', 'set of']
        parallel_score = sum(0.2 for indicator in parallel_friendly if indicator in task_lower)
        
        # Sequential requirement indicators (reduce parallelization potential)
        sequential_required = ['first', 'then', 'after', 'before', 'depends on']
        sequential_penalty = sum(0.15 for indicator in sequential_required if indicator in task_lower)
        
        return max(0.0, min(1.0, parallel_score - sequential_penalty))
    
    def _estimate_task_duration(self, task: str, word_count: int) -> float:
        """Estimate task duration in hours (simplified heuristic)"""
        base_duration = word_count * 0.1  # 0.1 hours per word as base
        
        # Adjust based on task complexity indicators
        task_lower = task.lower()
        
        if any(indicator in task_lower for indicator in ['research', 'analyze', 'investigate']):
            base_duration *= 2.0
        elif any(indicator in task_lower for indicator in ['implement', 'build', 'develop']):
            base_duration *= 1.5
        elif any(indicator in task_lower for indicator in ['test', 'validate', 'verify']):
            base_duration *= 1.2
        
        return min(base_duration, 40.0)  # Cap at 40 hours


    async def _calculate_dependency_score(self, task1: str, task2: str) -> float:
        """Calculate dependency score between two tasks"""
        task1_lower = task1.lower()
        task2_lower = task2.lower()
        
        # Explicit dependency indicators
        explicit_score = 0.0
        dependency_keywords = ['after', 'once', 'when', 'following', 'using', 'based on']
        for keyword in dependency_keywords:
            if keyword in task1_lower:
                # Check if task2 concepts appear after the dependency keyword
                task2_words = set(task2_lower.split())
                task1_words_after_keyword = task1_lower.split(keyword, 1)
                if len(task1_words_after_keyword) > 1:
                    remaining_words = set(task1_words_after_keyword[1].split())
                    overlap = len(task2_words.intersection(remaining_words))
                    if overlap >= 2:
                        explicit_score += 0.5
        
        # Semantic similarity (simplified)
        task1_words = set(task1_lower.split())
        task2_words = set(task2_lower.split())
        common_words = task1_words.intersection(task2_words)
        semantic_score = len(common_words) / max(len(task1_words), len(task2_words))
        
        # Output-input relationships
        output_input_score = 0.0
        output_indicators = ['result', 'output', 'data', 'information', 'report']
        input_indicators = ['input', 'data', 'information', 'based on']
        
        if any(indicator in task2_lower for indicator in output_indicators):
            if any(indicator in task1_lower for indicator in input_indicators):
                output_input_score = 0.3
        
        return min(explicit_score + semantic_score * 0.5 + output_input_score, 1.0)
    
    def _optimize_for_parallelization(self, dependencies: Dict[str, List[str]], subtasks: List[str]) -> Dict[str, List[str]]:
        """Optimize dependency graph for maximum parallelization"""
        # Remove weak dependencies to enable more parallelization
        optimized = {}
        for task_id, deps in dependencies.items():
            # Keep only strong dependencies for parallel execution
            strong_deps = []
            for dep in deps:
                dep_task = subtasks[int(dep)]
                current_task = subtasks[int(task_id)]
                if asyncio.run(self._calculate_dependency_score(current_task, dep_task)) > 0.7:
                    strong_deps.append(dep)
            optimized[task_id] = strong_deps
        return optimized
    
    def _create_hybrid_dependencies(self, dependencies: Dict[str, List[str]], subtasks: List[str]) -> Dict[str, List[str]]:
        """Create hybrid dependency structure balancing sequence and parallelism"""
        # Group tasks into sequential chains and parallel clusters
        return dependencies  # Simplified implementation
    
    async def _has_multiple_concepts(self, task: str) -> bool:
        """Check if task contains multiple distinct concepts"""
        # Simple heuristic: look for conjunctions and multiple domain indicators
        conjunctions = ['and', 'then', 'also', 'additionally', 'furthermore']
        conjunction_count = sum(1 for conj in conjunctions if conj in task.lower())
        
        # Check for multiple domain indicators
        domain_count = 0
        for domain_info in self.domain_knowledge.values():
            if any(keyword in task.lower() for keyword in domain_info['keywords']):
                domain_count += 1
        
        return conjunction_count >= 2 or domain_count >= 2
    
    def _requires_domain_expertise(self, task: str) -> bool:
        """Check if task requires specialized domain expertise"""
        specialized_domains = ['quantum', 'blockchain', 'cryptography', 'bioinformatics', 'genomics']
        return any(domain in task.lower() for domain in specialized_domains)
    
    async def _select_decomposition_strategy(self, task: ArchitectTask) -> DecompositionStrategy:
        """Select optimal decomposition strategy for the task"""
        task_hash = hashlib.md5(task.instruction.encode()).hexdigest()[:8]
        analytics = self.task_analytics.get(task_hash)
        
        if not analytics:
            return DecompositionStrategy.CONTEXTUAL
        
        # Strategy selection based on task characteristics
        if analytics.parallelization_potential > 0.7:
            return DecompositionStrategy.PARALLEL
        elif any(indicator in task.instruction.lower() for indicator in ['step by step', 'sequential', 'order']):
            return DecompositionStrategy.SEQUENTIAL
        elif analytics.parallelization_potential > 0.3:
            return DecompositionStrategy.HYBRID
        else:
            return DecompositionStrategy.CONTEXTUAL
    
    async def _parallel_decomposition(self, instruction: str) -> List[str]:
        """Decompose task optimized for parallel execution"""
        # Identify parallelizable components
        if 'analyze' in instruction.lower():
            return [
                f"Collect data sources for: {instruction}",
                f"Process data subset A for: {instruction}",
                f"Process data subset B for: {instruction}",
                f"Merge analysis results for: {instruction}"
            ]
        elif 'test' in instruction.lower():
            return [
                f"Prepare test environment for: {instruction}",
                f"Execute test suite A for: {instruction}",
                f"Execute test suite B for: {instruction}",
                f"Compile test results for: {instruction}"
            ]
        else:
            return await self._rule_based_decomposition(instruction)
    
    async def _sequential_decomposition(self, instruction: str) -> List[str]:
        """Decompose task optimized for sequential execution"""
        # Create clear sequential dependencies
        if 'implement' in instruction.lower():
            return [
                f"Design architecture for: {instruction}",
                f"Implement core components for: {instruction}",
                f"Integrate components for: {instruction}",
                f"Test and validate: {instruction}",
                f"Deploy and document: {instruction}"
            ]
        else:
            return await self._rule_based_decomposition(instruction)
    
    async def _optimize_subtasks(self, subtasks: List[str], strategy: DecompositionStrategy) -> List[str]:
        """Optimize subtasks based on selected strategy"""
        if strategy == DecompositionStrategy.PARALLEL:
            # Ensure subtasks are independent and can run in parallel
            optimized = []
            for task in subtasks:
                if not any(dep_word in task.lower() for dep_word in ['after', 'using', 'based on']):
                    optimized.append(task)
                else:
                    # Remove dependency language for parallel execution
                    cleaned_task = re.sub(r'\b(after|using|based on)\b.*', '', task).strip()
                    if cleaned_task:
                        optimized.append(cleaned_task)
            return optimized
        
        return subtasks


# Factory function for creating architects at different levels
def create_architect(level: int = 1) -> HierarchicalArchitect:
    """Create a hierarchical architect for a specific level"""
    return HierarchicalArchitect(level=level)