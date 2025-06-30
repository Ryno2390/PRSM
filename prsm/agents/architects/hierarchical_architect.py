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
    
    async def assess_complexity(self, task: str) -> float:
        """
        Assess task complexity on a scale of 0.0 to 1.0
        
        Args:
            task: Task description
            
        Returns:
            float: Complexity score (0.0 = simple, 1.0 = very complex)
        """
        # Heuristic-based complexity assessment considering multiple factors
        task_str = str(task) if not isinstance(task, str) else task
        task_lower = task_str.lower()
        
        # Base complexity from length (longer tasks often more complex)
        length_factor = min(len(task_str) / 2000, 0.4)  # Cap at 40% of total score
        
        # Technical domain complexity indicators
        domain_complexity = 0.0
        technical_domains = ['quantum', 'neural', 'blockchain', 'cryptography', 'genomics', 'bioinformatics']
        advanced_domains = ['machine learning', 'artificial intelligence', 'distributed systems', 'optimization']
        
        for domain in technical_domains:
            if domain in task_lower:
                domain_complexity += 0.2
        for domain in advanced_domains:
            if domain in task_lower:
                domain_complexity += 0.15
                
        domain_complexity = min(domain_complexity, 0.3)  # Cap at 30%
        
        # Multi-step process indicators
        process_complexity = 0.0
        process_indicators = ['analyze and', 'first', 'then', 'finally', 'step by step', 'multiple', 'coordinate']
        for indicator in process_indicators:
            if indicator in task_lower:
                process_complexity += 0.05
        process_complexity = min(process_complexity, 0.2)  # Cap at 20%
        
        # Research depth indicators
        research_complexity = 0.0
        research_indicators = ['research', 'investigate', 'comprehensive', 'thorough', 'detailed analysis']
        for indicator in research_indicators:
            if indicator in task_lower:
                research_complexity += 0.02
        research_complexity = min(research_complexity, 0.1)  # Cap at 10%
        
        # Combine all factors
        total_complexity = length_factor + domain_complexity + process_complexity + research_complexity
        return min(total_complexity, 1.0)
    
    async def identify_dependencies(self, subtasks: List[str]) -> Dict[str, List[str]]:
        """
        Identify dependencies between subtasks
        
        Args:
            subtasks: List of subtask descriptions
            
        Returns:
            Dict mapping task indices to their dependencies
        """
        # Analyze task descriptions for dependencies using keyword detection
        dependencies = {}
        
        for i, task in enumerate(subtasks):
            task_deps = []
            task_lower = task.lower()
            
            # Look for explicit references to other tasks
            for j, other_task in enumerate(subtasks):
                if i != j:
                    other_lower = other_task.lower()
                    
                    # Check for explicit ordering keywords
                    if any(keyword in task_lower for keyword in ['after', 'once', 'when', 'following']):
                        # Look for concepts from the other task
                        other_words = set(other_lower.split())
                        task_words = set(task_lower.split())
                        common_words = other_words.intersection(task_words)
                        
                        # If significant overlap in keywords, likely dependency
                        if len(common_words) >= 2:
                            task_deps.append(str(j))
                    
                    # Check for data flow dependencies (output -> input)
                    output_keywords = ['results from', 'output of', 'data from', 'using the']
                    if any(keyword in task_lower for keyword in output_keywords):
                        if any(word in task_lower for word in other_lower.split() if len(word) > 3):
                            task_deps.append(str(j))
            
            dependencies[str(i)] = task_deps
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
        logger.info("Decomposing task", task_id=task.task_id, 
                   complexity=task.complexity_score)
        
        # Use LLM-based task decomposition for complex tasks
        instruction = task.instruction
        
        if task.complexity_score > 0.7:
            return await self._llm_based_decomposition(instruction)
        else:
            return await self._rule_based_decomposition(instruction)
    
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


# Factory function for creating architects at different levels
def create_architect(level: int = 1) -> HierarchicalArchitect:
    """Create a hierarchical architect for a specific level"""
    return HierarchicalArchitect(level=level)