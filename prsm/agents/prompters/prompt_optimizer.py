"""
PRSM Prompt Optimizer
Advanced prompt optimization and enhancement for domain-specific AI reasoning
"""

import re
import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from prsm.agents.base import BaseAgent
from prsm.core.config import get_settings
from prsm.core.models import AgentType, SafetyLevel

logger = structlog.get_logger(__name__)
settings = get_settings()


class DomainType(str, Enum):
    """Scientific domain types for specialized prompt optimization"""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"
    COMPUTER_SCIENCE = "computer_science"
    MEDICINE = "medicine"
    ENGINEERING = "engineering"
    PSYCHOLOGY = "psychology"
    ECONOMICS = "economics"
    GENERAL_SCIENCE = "general_science"


class PromptType(str, Enum):
    """Types of prompts for different reasoning tasks"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EXPLANATORY = "explanatory"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    DECOMPOSITION = "decomposition"


class OptimizationStrategy(str, Enum):
    """Prompt optimization strategies"""
    CLARITY_ENHANCEMENT = "clarity_enhancement"
    DOMAIN_SPECIALIZATION = "domain_specialization"
    SAFETY_ALIGNMENT = "safety_alignment"
    CONTEXT_ENRICHMENT = "context_enrichment"
    REASONING_GUIDANCE = "reasoning_guidance"


class OptimizedPrompt(BaseModel):
    """Optimized prompt with metadata"""
    original_prompt: str
    optimized_prompt: str
    domain: DomainType
    prompt_type: PromptType
    strategies_applied: List[OptimizationStrategy]
    confidence_score: float = Field(ge=0.0, le=1.0)
    safety_validated: bool = False
    reasoning_guidance: Optional[str] = None
    context_templates: List[str] = Field(default_factory=list)
    optimization_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DomainStrategy(BaseModel):
    """Domain-specific optimization strategy"""
    domain: DomainType
    keywords: List[str]
    reasoning_patterns: List[str]
    safety_considerations: List[str]
    context_templates: List[str]
    enhancement_rules: Dict[str, str]


class PromptOptimizer(BaseAgent):
    """
    Advanced prompt optimization agent for PRSM
    
    Optimizes prompts for:
    - Domain-specific terminology and reasoning patterns
    - Safety alignment with PRSM guidelines
    - Context enrichment for better understanding
    - Reasoning guidance for systematic thinking
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id, AgentType.PROMPTER)
        self.domain_strategies = self._initialize_domain_strategies()
        self.safety_guidelines = self._initialize_safety_guidelines()
        self.context_templates = self._initialize_context_templates()
        
        logger.info("PromptOptimizer initialized",
                   agent_id=self.agent_id,
                   domains=len(self.domain_strategies),
                   safety_rules=len(self.safety_guidelines))
    
    def _initialize_domain_strategies(self) -> Dict[DomainType, DomainStrategy]:
        """Initialize domain-specific optimization strategies"""
        strategies = {}
        
        # Physics domain strategy
        strategies[DomainType.PHYSICS] = DomainStrategy(
            domain=DomainType.PHYSICS,
            keywords=["energy", "momentum", "force", "field", "quantum", "relativity", "thermodynamics"],
            reasoning_patterns=[
                "First principles analysis",
                "Dimensional analysis",
                "Conservation laws application",
                "Symmetry considerations"
            ],
            safety_considerations=[
                "Avoid speculation beyond established physics",
                "Clearly distinguish theoretical from experimental results",
                "Consider measurement uncertainties"
            ],
            context_templates=[
                "Physics Problem Analysis: {problem}",
                "Given the physical system: {system}, analyze using {method}",
                "Consider the following physical principles: {principles}"
            ],
            enhancement_rules={
                "units": "Always specify units for physical quantities",
                "assumptions": "State all assumptions clearly",
                "approximations": "Justify any approximations used"
            }
        )
        
        # Chemistry domain strategy
        strategies[DomainType.CHEMISTRY] = DomainStrategy(
            domain=DomainType.CHEMISTRY,
            keywords=["reaction", "molecule", "bond", "catalyst", "equilibrium", "thermodynamics", "kinetics"],
            reasoning_patterns=[
                "Mechanism analysis",
                "Thermodynamic feasibility",
                "Kinetic considerations",
                "Structure-function relationships"
            ],
            safety_considerations=[
                "Consider chemical safety and toxicity",
                "Verify reaction conditions and hazards",
                "Include proper safety protocols"
            ],
            context_templates=[
                "Chemical Analysis: {compound} in {conditions}",
                "Reaction mechanism for: {reaction}",
                "Consider the following chemical principles: {principles}"
            ],
            enhancement_rules={
                "nomenclature": "Use IUPAC nomenclature where appropriate",
                "conditions": "Specify reaction conditions (temperature, pressure, solvent)",
                "safety": "Include relevant safety considerations"
            }
        )
        
        # Biology domain strategy
        strategies[DomainType.BIOLOGY] = DomainStrategy(
            domain=DomainType.BIOLOGY,
            keywords=["protein", "gene", "cell", "organism", "evolution", "metabolism", "ecology"],
            reasoning_patterns=[
                "Structure-function analysis",
                "Evolutionary perspective",
                "Systems biology approach",
                "Experimental design considerations"
            ],
            safety_considerations=[
                "Consider ethical implications",
                "Respect for living organisms",
                "Biosafety protocols"
            ],
            context_templates=[
                "Biological System Analysis: {system} in {organism}",
                "Evolutionary analysis of: {trait}",
                "Consider the following biological principles: {principles}"
            ],
            enhancement_rules={
                "taxonomy": "Use proper taxonomic nomenclature",
                "evidence": "Base conclusions on experimental evidence",
                "ethics": "Consider ethical implications of biological research"
            }
        )
        
        # Mathematics domain strategy
        strategies[DomainType.MATHEMATICS] = DomainStrategy(
            domain=DomainType.MATHEMATICS,
            keywords=["theorem", "proof", "function", "equation", "algebra", "calculus", "topology"],
            reasoning_patterns=[
                "Logical deduction",
                "Mathematical induction",
                "Proof by contradiction",
                "Constructive proofs"
            ],
            safety_considerations=[
                "Verify mathematical rigor",
                "Check for logical consistency",
                "Validate assumptions"
            ],
            context_templates=[
                "Mathematical Problem: {problem}",
                "Prove or disprove: {statement}",
                "Consider the mathematical framework: {framework}"
            ],
            enhancement_rules={
                "rigor": "Maintain mathematical rigor",
                "notation": "Use standard mathematical notation",
                "assumptions": "State all assumptions and constraints"
            }
        )
        
        # Computer Science domain strategy
        strategies[DomainType.COMPUTER_SCIENCE] = DomainStrategy(
            domain=DomainType.COMPUTER_SCIENCE,
            keywords=["algorithm", "complexity", "data structure", "software", "security", "AI", "networks"],
            reasoning_patterns=[
                "Algorithmic thinking",
                "Complexity analysis",
                "System design principles",
                "Security considerations"
            ],
            safety_considerations=[
                "Consider security implications",
                "Validate input/output specifications",
                "Consider scalability and performance"
            ],
            context_templates=[
                "Algorithm Design: {problem} with constraints {constraints}",
                "System Analysis: {system} for {requirements}",
                "Consider computational complexity: {analysis}"
            ],
            enhancement_rules={
                "complexity": "Include time and space complexity analysis",
                "security": "Consider security implications",
                "testing": "Include testing and validation strategies"
            }
        )
        
        return strategies
    
    def _initialize_safety_guidelines(self) -> List[str]:
        """Initialize safety guidelines for prompt enhancement"""
        return [
            "Ensure factual accuracy and avoid misinformation",
            "Respect ethical boundaries in research",
            "Consider potential misuse of information",
            "Promote responsible scientific practices",
            "Encourage peer review and validation",
            "Respect intellectual property rights",
            "Consider environmental and social impact",
            "Maintain transparency in methodology",
            "Acknowledge limitations and uncertainties",
            "Promote inclusive and diverse perspectives"
        ]
    
    def _initialize_context_templates(self) -> Dict[PromptType, List[str]]:
        """Initialize context templates for different prompt types"""
        return {
            PromptType.ANALYTICAL: [
                "Analyze the following {subject} by breaking it down into components:",
                "Examine {topic} from multiple perspectives:",
                "Systematically evaluate {problem} considering:"
            ],
            PromptType.CREATIVE: [
                "Generate innovative approaches to {challenge}:",
                "Imagine novel solutions for {problem}:",
                "Explore creative possibilities for {scenario}:"
            ],
            PromptType.EXPLANATORY: [
                "Explain {concept} in clear, accessible terms:",
                "Describe the mechanism behind {phenomenon}:",
                "Clarify the relationship between {elements}:"
            ],
            PromptType.SYNTHESIS: [
                "Synthesize insights from {sources} to:",
                "Combine knowledge from {domains} to address:",
                "Integrate findings from {studies} to:"
            ],
            PromptType.EVALUATION: [
                "Evaluate the validity of {claim} based on:",
                "Assess the quality of {work} using criteria:",
                "Compare and contrast {options} considering:"
            ],
            PromptType.DECOMPOSITION: [
                "Break down {complex_problem} into manageable parts:",
                "Decompose {system} into its fundamental components:",
                "Analyze {process} step by step:"
            ]
        }
    
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> OptimizedPrompt:
        """Process prompt optimization request"""
        if not isinstance(input_data, (str, dict)):
            raise ValueError("Input must be a string prompt or optimization request dict")
        
        # Extract prompt and parameters
        if isinstance(input_data, str):
            prompt = input_data
            domain = DomainType.GENERAL_SCIENCE
            prompt_type = PromptType.ANALYTICAL
        else:
            prompt = input_data.get("prompt", "")
            domain = DomainType(input_data.get("domain", "general_science"))
            prompt_type = PromptType(input_data.get("prompt_type", "analytical"))
        
        logger.info("Optimizing prompt",
                   prompt_length=len(prompt),
                   domain=domain.value,
                   prompt_type=prompt_type.value)
        
        # Apply optimization strategies
        optimized_prompt = await self._optimize_prompt(prompt, domain, prompt_type)
        
        return optimized_prompt
    
    async def _optimize_prompt(self, prompt: str, domain: DomainType, 
                              prompt_type: PromptType) -> OptimizedPrompt:
        """Apply comprehensive prompt optimization"""
        strategies_applied = []
        optimization_metadata = {}
        
        # Start with original prompt
        optimized = prompt
        
        # 1. Domain specialization
        optimized, domain_metadata = await self._apply_domain_specialization(
            optimized, domain
        )
        strategies_applied.append(OptimizationStrategy.DOMAIN_SPECIALIZATION)
        optimization_metadata["domain_specialization"] = domain_metadata
        
        # 2. Clarity enhancement
        optimized, clarity_metadata = await self._enhance_clarity(optimized)
        strategies_applied.append(OptimizationStrategy.CLARITY_ENHANCEMENT)
        optimization_metadata["clarity_enhancement"] = clarity_metadata
        
        # 3. Context enrichment
        optimized, context_metadata = await self._enrich_context(
            optimized, prompt_type
        )
        strategies_applied.append(OptimizationStrategy.CONTEXT_ENRICHMENT)
        optimization_metadata["context_enrichment"] = context_metadata
        
        # 4. Reasoning guidance
        optimized, reasoning_metadata = await self._add_reasoning_guidance(
            optimized, domain
        )
        strategies_applied.append(OptimizationStrategy.REASONING_GUIDANCE)
        optimization_metadata["reasoning_guidance"] = reasoning_metadata
        
        # 5. Safety alignment
        optimized, safety_metadata = await self._align_with_safety(optimized)
        strategies_applied.append(OptimizationStrategy.SAFETY_ALIGNMENT)
        optimization_metadata["safety_alignment"] = safety_metadata
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            prompt, optimized, strategies_applied
        )
        
        # Generate context templates
        context_templates = self._generate_context_templates(domain, prompt_type)
        
        return OptimizedPrompt(
            original_prompt=prompt,
            optimized_prompt=optimized,
            domain=domain,
            prompt_type=prompt_type,
            strategies_applied=strategies_applied,
            confidence_score=confidence_score,
            safety_validated=safety_metadata.get("validated", False),
            reasoning_guidance=reasoning_metadata.get("guidance"),
            context_templates=context_templates,
            optimization_metadata=optimization_metadata
        )
    
    async def _apply_domain_specialization(self, prompt: str, 
                                         domain: DomainType) -> Tuple[str, Dict[str, Any]]:
        """Apply domain-specific terminology and patterns"""
        if domain not in self.domain_strategies:
            return prompt, {"applied": False, "reason": "Domain strategy not found"}
        
        strategy = self.domain_strategies[domain]
        optimized = prompt
        changes = []
        
        # Enhance with domain-specific keywords
        for keyword in strategy.keywords:
            if keyword.lower() in prompt.lower() and keyword not in optimized:
                # Context-aware keyword enhancement
                pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                if pattern.search(optimized):
                    changes.append(f"Enhanced keyword: {keyword}")
        
        # Apply enhancement rules
        for rule_type, rule_text in strategy.enhancement_rules.items():
            if rule_type == "units" and domain == DomainType.PHYSICS:
                if any(unit_word in optimized.lower() for unit_word in ["meter", "second", "gram", "joule"]):
                    optimized += f"\n\nNote: {rule_text}"
                    changes.append(f"Added rule: {rule_type}")
            elif rule_type == "safety" and domain == DomainType.CHEMISTRY:
                if "reaction" in optimized.lower():
                    optimized += f"\n\nSafety consideration: {rule_text}"
                    changes.append(f"Added rule: {rule_type}")
        
        return optimized, {
            "applied": True,
            "domain": domain.value,
            "changes": changes,
            "keywords_found": len([k for k in strategy.keywords if k.lower() in prompt.lower()])
        }
    
    async def _enhance_clarity(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Enhance prompt clarity and structure"""
        optimized = prompt
        improvements = []
        
        # Add structure if missing
        if "?" not in prompt and not prompt.strip().endswith(":"):
            optimized += "\n\nPlease provide a detailed analysis."
            improvements.append("Added explicit request")
        
        # Improve vague language
        vague_patterns = {
            r'\bstuff\b': 'material/concepts',
            r'\bthing\b': 'element/aspect',
            r'\bkind of\b': 'type of',
            r'\bsort of\b': 'type of'
        }
        
        for pattern, replacement in vague_patterns.items():
            if re.search(pattern, optimized, re.IGNORECASE):
                optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
                improvements.append(f"Replaced vague term: {pattern}")
        
        # Add specificity prompts
        if len(prompt.split()) < 10:  # Short prompts need more detail
            optimized += "\n\nConsider providing specific examples and detailed reasoning."
            improvements.append("Added specificity request")
        
        return optimized, {
            "applied": True,
            "improvements": improvements,
            "clarity_score": min(1.0, len(improvements) * 0.2 + 0.6)
        }
    
    async def _enrich_context(self, prompt: str, 
                            prompt_type: PromptType) -> Tuple[str, Dict[str, Any]]:
        """Enrich prompt with contextual information"""
        optimized = prompt
        context_additions = []
        
        # Add type-specific context
        if prompt_type in self.context_templates:
            templates = self.context_templates[prompt_type]
            
            # Select appropriate template based on prompt content
            for template in templates:
                if any(keyword in prompt.lower() for keyword in ["analyze", "examine", "evaluate"]):
                    if "Consider the following aspects:" not in optimized:
                        optimized += "\n\nConsider the following aspects:\n- Methodological approach\n- Evidence quality\n- Limitations and assumptions\n- Broader implications"
                        context_additions.append("Added analytical framework")
                    break
        
        # Add methodological guidance
        if prompt_type == PromptType.ANALYTICAL:
            if "methodology" not in optimized.lower():
                optimized += "\n\nDescribe your analytical methodology clearly."
                context_additions.append("Added methodology request")
        
        return optimized, {
            "applied": True,
            "prompt_type": prompt_type.value,
            "context_additions": context_additions
        }
    
    async def _add_reasoning_guidance(self, prompt: str, 
                                    domain: DomainType) -> Tuple[str, Dict[str, Any]]:
        """Add reasoning guidance specific to domain"""
        if domain not in self.domain_strategies:
            return prompt, {"applied": False}
        
        strategy = self.domain_strategies[domain]
        optimized = prompt
        guidance_added = []
        
        # Add domain-specific reasoning patterns
        reasoning_guidance = f"\n\nReasoning approach for {domain.value}:\n"
        for i, pattern in enumerate(strategy.reasoning_patterns[:3], 1):
            reasoning_guidance += f"{i}. {pattern}\n"
        
        optimized += reasoning_guidance
        guidance_added.append("Domain-specific reasoning patterns")
        
        # Add systematic thinking prompts
        optimized += "\nProvide step-by-step reasoning and justify each conclusion."
        guidance_added.append("Systematic thinking prompt")
        
        return optimized, {
            "applied": True,
            "guidance": reasoning_guidance,
            "patterns_added": len(strategy.reasoning_patterns[:3]),
            "guidance_components": guidance_added
        }
    
    async def _align_with_safety(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Align prompt with safety guidelines"""
        optimized = prompt
        safety_additions = []
        
        # Add safety reminders
        safety_note = "\n\nSafety reminders:\n- Ensure factual accuracy\n- Consider ethical implications\n- Acknowledge limitations\n- Promote responsible practices"
        
        optimized += safety_note
        safety_additions.append("General safety guidelines")
        
        # Check for potentially unsafe content
        unsafe_patterns = ["harmful", "dangerous", "illegal", "unethical"]
        safety_flags = []
        
        for pattern in unsafe_patterns:
            if pattern in prompt.lower():
                safety_flags.append(pattern)
        
        validated = len(safety_flags) == 0
        
        return optimized, {
            "applied": True,
            "validated": validated,
            "safety_flags": safety_flags,
            "safety_additions": safety_additions
        }
    
    def _calculate_confidence_score(self, original: str, optimized: str, 
                                  strategies: List[OptimizationStrategy]) -> float:
        """Calculate confidence score for optimization"""
        base_score = 0.5
        
        # Length improvement
        length_improvement = len(optimized) / max(len(original), 1)
        length_score = min(0.3, length_improvement * 0.1)
        
        # Strategy diversity
        strategy_score = len(strategies) * 0.1
        
        # Structure improvement (has questions, lists, etc.)
        structure_indicators = ["?", ":", "-", "1.", "2.", "3."]
        structure_score = sum(0.02 for indicator in structure_indicators 
                            if indicator in optimized and indicator not in original)
        
        confidence = base_score + length_score + strategy_score + structure_score
        return min(1.0, confidence)
    
    def _generate_context_templates(self, domain: DomainType, 
                                  prompt_type: PromptType) -> List[str]:
        """Generate context templates for the given domain and type"""
        templates = []
        
        # Domain-specific templates
        if domain in self.domain_strategies:
            strategy = self.domain_strategies[domain]
            templates.extend(strategy.context_templates[:2])
        
        # Type-specific templates
        if prompt_type in self.context_templates:
            templates.extend(self.context_templates[prompt_type][:2])
        
        return templates[:3]  # Limit to 3 templates
    
    async def optimize_for_domain(self, base_prompt: str, domain: str) -> str:
        """Optimize prompt for specific domain (public interface)"""
        try:
            domain_enum = DomainType(domain.lower())
        except ValueError:
            domain_enum = DomainType.GENERAL_SCIENCE
        
        result = await self.process({
            "prompt": base_prompt,
            "domain": domain_enum.value,
            "prompt_type": "analytical"
        })
        
        return result.optimized_prompt
    
    async def enhance_alignment(self, prompt: str, safety_guidelines: List[str]) -> str:
        """Enhance prompt alignment with safety guidelines (public interface)"""
        # Custom safety guidelines override defaults
        original_guidelines = self.safety_guidelines
        self.safety_guidelines = safety_guidelines
        
        try:
            result = await self.process(prompt)
            return result.optimized_prompt
        finally:
            # Restore original guidelines
            self.safety_guidelines = original_guidelines
    
    async def generate_context_templates(self, task_type: str) -> List[str]:
        """Generate context templates for task type (public interface)"""
        try:
            prompt_type = PromptType(task_type.lower())
        except ValueError:
            prompt_type = PromptType.ANALYTICAL
        
        return self._generate_context_templates(DomainType.GENERAL_SCIENCE, prompt_type)