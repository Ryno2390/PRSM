"""
Hierarchical Architect AI
Enhanced from Co-Lab's decomposition.py for multi-level recursive decomposition
"""

from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
import asyncio
import structlog

from prsm.core.models import ArchitectTask, TaskHierarchy, TaskStatus, AgentType
from prsm.core.config import get_settings
from prsm.agents.base import BaseAgent


logger = structlog.get_logger(__name__)
settings = get_settings()


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
        logger.info("Starting recursive decomposition", 
                   level=self.level, depth=current_depth, task_length=len(task))
        
        # Create root task
        root_task = ArchitectTask(
            session_id=session_id,
            parent_task_id=parent_task_id,
            level=current_depth,
            instruction=task,
            complexity_score=await self.assess_complexity(task)
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
    
    async def assess_complexity(self, task: str) -> float:
        """
        Assess task complexity on a scale of 0.0 to 1.0
        
        Args:
            task: Task description
            
        Returns:
            float: Complexity score (0.0 = simple, 1.0 = very complex)
        """
        # TODO: Implement LLM-based complexity assessment
        # Factors to consider:
        # - Task length and detail
        # - Domain expertise required
        # - Number of concepts involved
        # - Interdisciplinary nature
        # - Research depth required
        
        # Placeholder: base complexity on task length
        base_complexity = min(len(task) / 1000, 1.0)
        
        # TODO: Add semantic complexity analysis
        
        return base_complexity
    
    async def identify_dependencies(self, subtasks: List[str]) -> Dict[str, List[str]]:
        """
        Identify dependencies between subtasks
        
        Args:
            subtasks: List of subtask descriptions
            
        Returns:
            Dict mapping task indices to their dependencies
        """
        # TODO: Implement dependency analysis
        # This should:
        # 1. Analyze task descriptions for dependencies
        # 2. Identify which tasks must complete before others
        # 3. Build dependency graph
        # 4. Detect circular dependencies
        
        dependencies = {}
        
        # Placeholder: assume sequential dependencies
        for i, task in enumerate(subtasks):
            if i > 0:
                dependencies[str(i)] = [str(i-1)]
            else:
                dependencies[str(i)] = []
        
        return dependencies
    
    async def _should_decompose(self, task: ArchitectTask, depth: int) -> bool:
        """
        Determine if a task should be further decomposed
        
        Args:
            task: Task to evaluate
            depth: Current decomposition depth
            
        Returns:
            bool: True if task should be decomposed further
        """
        # Don't decompose if we've reached max depth
        if depth >= self.max_depth:
            return False
        
        # Don't decompose simple tasks
        if task.complexity_score < 0.3:
            return False
        
        # Don't decompose very short tasks
        if len(task.instruction) < 50:
            return False
        
        # TODO: Add more sophisticated decomposition criteria
        # - Check if task contains multiple concepts
        # - Analyze if task can benefit from parallelization
        # - Consider available specialist models
        
        return True
    
    async def _decompose_task(self, task: ArchitectTask) -> List[str]:
        """
        Decompose a single task into subtasks
        
        Args:
            task: Task to decompose
            
        Returns:
            List of subtask descriptions
        """
        # TODO: Implement LLM-based task decomposition
        # This should use prompt engineering to break down complex tasks
        # into manageable subtasks while preserving intent and context
        
        logger.info("Decomposing task", task_id=task.task_id, 
                   complexity=task.complexity_score)
        
        # Placeholder: simple rule-based decomposition
        instruction = task.instruction
        
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
        else:
            # Generic decomposition
            return [
                f"Plan approach for: {instruction}",
                f"Execute main work for: {instruction}",
                f"Review and refine: {instruction}"
            ]


# Factory function for creating architects at different levels
def create_architect(level: int = 1) -> HierarchicalArchitect:
    """Create a hierarchical architect for a specific level"""
    return HierarchicalArchitect(level=level)