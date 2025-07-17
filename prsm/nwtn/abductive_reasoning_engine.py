#!/usr/bin/env python3
"""
NWTN Abductive Reasoning Engine
Inference to the best explanation for observations

This module implements NWTN's abductive reasoning capabilities, which allow the system to:
1. Generate hypotheses to explain observations
2. Evaluate explanations based on criteria like simplicity, scope, and plausibility
3. Select the best explanation from competing hypotheses
4. Handle incomplete information and uncertainty

Abductive reasoning operates from observations to the most plausible explanations,
providing the foundation for diagnostic reasoning, scientific discovery, and
problem-solving under uncertainty.

Key Concepts:
- Hypothesis generation from observations
- Explanation evaluation and ranking
- Inference to the best explanation (IBE)
- Diagnostic reasoning patterns
- Scientific hypothesis formation
- Explanatory coherence assessment

Usage:
    from prsm.nwtn.abductive_reasoning_engine import AbductiveReasoningEngine
    
    engine = AbductiveReasoningEngine()
    result = await engine.generate_best_explanation(observations, context)
"""

import asyncio
import json
import math
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
from collections import defaultdict, Counter

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel
from prsm.nwtn.world_model_engine import WorldModelEngine
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class ExplanationType(str, Enum):
    """Types of explanations"""
    CAUSAL = "causal"               # Cause-and-effect explanation
    MECHANISTIC = "mechanistic"     # How something works
    FUNCTIONAL = "functional"       # What purpose something serves
    STRUCTURAL = "structural"       # How something is organized
    TELEOLOGICAL = "teleological"   # Why something exists/happens
    STATISTICAL = "statistical"     # Based on probability/statistics
    INTENTIONAL = "intentional"     # Based on goals/intentions


class HypothesisStatus(str, Enum):
    """Status of a hypothesis"""
    GENERATED = "generated"         # Newly generated
    EVALUATED = "evaluated"         # Evaluated for plausibility
    TESTED = "tested"              # Tested against evidence
    CONFIRMED = "confirmed"         # Supported by evidence
    REFUTED = "refuted"            # Contradicted by evidence
    SUPERSEDED = "superseded"       # Replaced by better explanation


class ExplanationCriteria(str, Enum):
    """Criteria for evaluating explanations"""
    SIMPLICITY = "simplicity"       # Occam's razor - simpler is better
    SCOPE = "scope"                # Explains more phenomena
    PLAUSIBILITY = "plausibility"   # Consistent with known facts
    COHERENCE = "coherence"         # Internal logical consistency
    TESTABILITY = "testability"     # Can generate testable predictions
    NOVELTY = "novelty"            # Provides new insights
    PRECISION = "precision"         # Specific and detailed
    GENERALITY = "generality"       # Applies broadly


@dataclass
class Evidence:
    """A piece of evidence or observation"""
    
    id: str
    description: str
    
    # Evidence properties
    evidence_type: str = "observation"  # "observation", "fact", "measurement", "testimony"
    reliability: float = 1.0
    relevance: float = 1.0
    
    # Context
    domain: str = "general"
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    supports: List[str] = field(default_factory=list)  # Hypothesis IDs this supports
    contradicts: List[str] = field(default_factory=list)  # Hypothesis IDs this contradicts
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Hypothesis:
    """A hypothesis that explains observations"""
    
    id: str
    statement: str
    explanation_type: ExplanationType
    
    # Hypothesis properties
    scope: List[str] = field(default_factory=list)  # What phenomena it explains
    assumptions: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    
    # Evaluation scores
    simplicity_score: float = 0.5
    scope_score: float = 0.5
    plausibility_score: float = 0.5
    coherence_score: float = 0.5
    testability_score: float = 0.5
    overall_score: float = 0.5
    
    # Evidence relations
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    
    # Status and confidence
    status: HypothesisStatus = HypothesisStatus.GENERATED
    confidence: float = 0.5
    
    # Metadata
    generated_by: str = "abductive_engine"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_overall_score(self) -> float:
        """Calculate overall explanatory score"""
        
        # Weighted combination of criteria
        weights = {
            'simplicity': 0.2,
            'scope': 0.25,
            'plausibility': 0.25,
            'coherence': 0.15,
            'testability': 0.15
        }
        
        score = (
            weights['simplicity'] * self.simplicity_score +
            weights['scope'] * self.scope_score +
            weights['plausibility'] * self.plausibility_score +
            weights['coherence'] * self.coherence_score +
            weights['testability'] * self.testability_score
        )
        
        # Adjust for evidence support
        if self.supporting_evidence:
            evidence_support = len(self.supporting_evidence) / max(len(self.supporting_evidence) + len(self.contradicting_evidence), 1)
            score *= (0.5 + 0.5 * evidence_support)
        
        self.overall_score = score
        return score


@dataclass
class AbductiveExplanation:
    """The result of abductive reasoning - the best explanation"""
    
    id: str
    query: str
    observations: List[Evidence]
    
    # Best explanation
    best_hypothesis: Hypothesis
    alternative_hypotheses: List[Hypothesis] = field(default_factory=list)
    
    # Reasoning process
    generation_method: str = "systematic"
    evaluation_criteria: List[ExplanationCriteria] = field(default_factory=list)
    
    # Confidence and uncertainty
    explanation_confidence: float = 0.5
    uncertainty_sources: List[str] = field(default_factory=list)
    
    # Implications
    implications: List[str] = field(default_factory=list)
    testable_predictions: List[str] = field(default_factory=list)
    
    # Limitations
    limitations: List[str] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AbductiveReasoningEngine:
    """
    Engine for abductive reasoning - inference to the best explanation
    
    This system enables NWTN to generate and evaluate hypotheses that explain
    observations, selecting the most plausible explanation from alternatives.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="abductive_reasoning_engine")
        self.world_model = WorldModelEngine()
        
        # Storage for hypotheses and explanations
        self.generated_hypotheses: List[Hypothesis] = []
        self.explanations: List[AbductiveExplanation] = []
        
        # Configuration
        self.max_hypotheses = 10
        self.min_plausibility_threshold = 0.3
        self.explanation_criteria = [
            ExplanationCriteria.SIMPLICITY,
            ExplanationCriteria.SCOPE,
            ExplanationCriteria.PLAUSIBILITY,
            ExplanationCriteria.COHERENCE,
            ExplanationCriteria.TESTABILITY
        ]
        
        # Knowledge base of common explanation patterns
        self.explanation_patterns = self._initialize_explanation_patterns()
        
        logger.info("Initialized Abductive Reasoning Engine")
    
    async def generate_best_explanation(
        self, 
        observations: List[str], 
        context: Dict[str, Any] = None
    ) -> AbductiveExplanation:
        """
        Generate the best explanation for given observations using abductive reasoning
        
        Args:
            observations: List of observation statements
            context: Additional context for reasoning
            
        Returns:
            AbductiveExplanation: Best explanation with alternatives and confidence
        """
        
        logger.info(
            "Starting abductive reasoning",
            observation_count=len(observations)
        )
        
        # Step 1: Parse observations into evidence
        evidence_list = await self._parse_observations_to_evidence(observations, context)
        
        # Step 2: Generate candidate hypotheses
        candidate_hypotheses = await self._generate_candidate_hypotheses(evidence_list, context)
        
        # Step 3: Evaluate hypotheses against criteria
        evaluated_hypotheses = await self._evaluate_hypotheses(candidate_hypotheses, evidence_list)
        
        # Step 4: Select best explanation
        best_explanation = await self._select_best_explanation(evaluated_hypotheses, evidence_list, context)
        
        # Step 5: Generate implications and predictions
        enhanced_explanation = await self._enhance_explanation(best_explanation)
        
        # Step 6: Store results
        self.generated_hypotheses.extend(evaluated_hypotheses)
        self.explanations.append(enhanced_explanation)
        
        logger.info(
            "Abductive reasoning complete",
            hypotheses_generated=len(candidate_hypotheses),
            best_explanation=best_explanation.best_hypothesis.statement,
            confidence=best_explanation.explanation_confidence
        )
        
        return enhanced_explanation
    
    async def _parse_observations_to_evidence(
        self, 
        observations: List[str], 
        context: Dict[str, Any] = None
    ) -> List[Evidence]:
        """Parse observation statements into structured evidence"""
        
        evidence_list = []
        
        for i, obs_text in enumerate(observations):
            # Determine evidence type
            evidence_type = await self._classify_evidence_type(obs_text)
            
            # Assess reliability
            reliability = await self._assess_evidence_reliability(obs_text)
            
            # Determine relevance
            relevance = await self._assess_evidence_relevance(obs_text, context)
            
            # Determine domain
            domain = await self._determine_evidence_domain(obs_text, context)
            
            # Create evidence object
            evidence = Evidence(
                id=f"evidence_{i+1}",
                description=obs_text,
                evidence_type=evidence_type,
                reliability=reliability,
                relevance=relevance,
                domain=domain,
                source=context.get("source", "unknown") if context else "unknown"
            )
            
            evidence_list.append(evidence)
        
        return evidence_list
    
    async def _classify_evidence_type(self, observation: str) -> str:
        """Classify the type of evidence"""
        
        obs_lower = str(observation).lower()
        
        # Direct observation indicators
        if any(indicator in obs_lower for indicator in ["observed", "saw", "noticed", "witnessed"]):
            return "observation"
        
        # Measurement indicators
        if any(indicator in obs_lower for indicator in ["measured", "recorded", "data shows", "results indicate"]):
            return "measurement"
        
        # Testimony indicators
        if any(indicator in obs_lower for indicator in ["reported", "stated", "claimed", "testified"]):
            return "testimony"
        
        # Fact indicators
        if any(indicator in obs_lower for indicator in ["known that", "established", "proven", "fact"]):
            return "fact"
        
        return "observation"  # Default
    
    async def _assess_evidence_reliability(self, observation: str) -> float:
        """Assess the reliability of evidence"""
        
        obs_lower = str(observation).lower()
        
        # High reliability indicators
        high_reliability = ["measured", "recorded", "data", "scientific", "peer-reviewed", "established"]
        
        # Medium reliability indicators
        medium_reliability = ["observed", "documented", "reported", "consistent"]
        
        # Low reliability indicators
        low_reliability = ["rumored", "alleged", "claimed", "suspected", "possibly", "might"]
        
        # Uncertainty indicators
        uncertainty = ["maybe", "perhaps", "seems", "appears", "unclear", "uncertain"]
        
        # Calculate reliability score
        reliability = 0.7  # Default
        
        if any(indicator in obs_lower for indicator in high_reliability):
            reliability += 0.2
        elif any(indicator in obs_lower for indicator in medium_reliability):
            reliability += 0.1
        
        if any(indicator in obs_lower for indicator in low_reliability):
            reliability -= 0.2
        
        if any(indicator in obs_lower for indicator in uncertainty):
            reliability -= 0.1
        
        return max(0.1, min(1.0, reliability))
    
    async def _assess_evidence_relevance(self, observation: str, context: Dict[str, Any] = None) -> float:
        """Assess the relevance of evidence to the query"""
        
        # If no context, assume moderate relevance
        if not context or "query" not in context:
            return 0.7
        
        query = str(context["query"]).lower()
        obs_lower = str(observation).lower()
        
        # Simple relevance based on keyword overlap
        query_words = set(query.split())
        obs_words = set(obs_lower.split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        query_words -= stop_words
        obs_words -= stop_words
        
        if query_words and obs_words:
            overlap = len(query_words & obs_words)
            relevance = overlap / len(query_words)
            return max(0.1, min(1.0, relevance))
        
        return 0.5  # Default moderate relevance
    
    async def _determine_evidence_domain(self, observation: str, context: Dict[str, Any] = None) -> str:
        """Determine the domain of evidence"""
        
        # Check context first
        if context and "domain" in context:
            return context["domain"]
        
        # Domain classification based on keywords
        domain_keywords = {
            "medical": ["patient", "symptom", "diagnosis", "treatment", "disease", "health", "medicine"],
            "criminal": ["crime", "suspect", "evidence", "witness", "investigation", "police", "legal"],
            "scientific": ["experiment", "hypothesis", "data", "research", "study", "analysis", "theory"],
            "technical": ["system", "component", "failure", "error", "malfunction", "technical", "engineering"],
            "social": ["behavior", "society", "culture", "group", "social", "interaction", "community"],
            "economic": ["market", "financial", "economic", "business", "trade", "money", "cost"],
            "environmental": ["climate", "weather", "environmental", "ecology", "nature", "pollution"]
        }
        
        obs_lower = str(observation).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in obs_lower for keyword in keywords):
                return domain
        
        return "general"
    
    async def _generate_candidate_hypotheses(
        self, 
        evidence_list: List[Evidence], 
        context: Dict[str, Any] = None
    ) -> List[Hypothesis]:
        """Generate candidate hypotheses to explain the evidence"""
        
        candidate_hypotheses = []
        
        # Use multiple hypothesis generation strategies
        
        # 1. Pattern-based generation
        pattern_hypotheses = await self._generate_pattern_based_hypotheses(evidence_list)
        candidate_hypotheses.extend(pattern_hypotheses)
        
        # 2. Causal hypotheses
        causal_hypotheses = await self._generate_causal_hypotheses(evidence_list)
        candidate_hypotheses.extend(causal_hypotheses)
        
        # 3. AI-assisted generation
        ai_hypotheses = await self._generate_ai_assisted_hypotheses(evidence_list, context)
        candidate_hypotheses.extend(ai_hypotheses)
        
        # 4. Domain-specific hypotheses
        domain_hypotheses = await self._generate_domain_specific_hypotheses(evidence_list)
        candidate_hypotheses.extend(domain_hypotheses)
        
        # Remove duplicates and limit to max_hypotheses
        unique_hypotheses = self._remove_duplicate_hypotheses(candidate_hypotheses)
        
        return unique_hypotheses[:self.max_hypotheses]
    
    async def _generate_pattern_based_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses based on known explanation patterns"""
        
        hypotheses = []
        
        for pattern in self.explanation_patterns:
            # Check if pattern matches evidence
            if await self._pattern_matches_evidence(pattern, evidence_list):
                # Generate hypothesis from pattern
                hypothesis = await self._instantiate_pattern_hypothesis(pattern, evidence_list)
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_causal_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate causal hypotheses to explain evidence"""
        
        hypotheses = []
        
        # Look for potential cause-effect relationships
        for evidence in evidence_list:
            # Extract potential causes and effects
            causes = await self._extract_potential_causes(evidence)
            effects = await self._extract_potential_effects(evidence)
            
            # Generate causal hypotheses
            for cause in causes:
                for effect in effects:
                    if cause != effect:
                        hypothesis = Hypothesis(
                            id=str(uuid4()),
                            statement=f"{cause} causes {effect}",
                            explanation_type=ExplanationType.CAUSAL,
                            scope=[evidence.id],
                            assumptions=[f"Causal relationship exists between {cause} and {effect}"],
                            predictions=[f"If {cause} occurs, then {effect} should follow"]
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_ai_assisted_hypotheses(
        self, 
        evidence_list: List[Evidence], 
        context: Dict[str, Any] = None
    ) -> List[Hypothesis]:
        """Generate hypotheses using AI assistance"""
        
        # Compile evidence descriptions
        evidence_descriptions = [f"- {evidence.description}" for evidence in evidence_list]
        evidence_text = "\n".join(evidence_descriptions)
        
        # Generate hypotheses prompt
        hypothesis_prompt = f"""
        Given the following observations, generate 3-5 plausible hypotheses that could explain them:
        
        Observations:
        {evidence_text}
        
        For each hypothesis, provide:
        1. A clear explanatory statement
        2. The type of explanation (causal, mechanistic, functional, etc.)
        3. Key assumptions
        4. Testable predictions
        
        Focus on generating diverse explanations that cover different aspects of the observations.
        """
        
        try:
            response = await self.model_executor.execute_request(
                prompt=hypothesis_prompt,
                model_name="gpt-4",
                temperature=0.7
            )
            
            # Parse response into hypotheses
            hypotheses = await self._parse_ai_hypotheses(response, evidence_list)
            return hypotheses
            
        except Exception as e:
            logger.error("Error generating AI-assisted hypotheses", error=str(e))
            return []
    
    async def _generate_domain_specific_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses specific to evidence domains"""
        
        hypotheses = []
        
        # Group evidence by domain
        domain_evidence = defaultdict(list)
        for evidence in evidence_list:
            domain_evidence[evidence.domain].append(evidence)
        
        # Generate domain-specific hypotheses
        for domain, domain_obs in domain_evidence.items():
            domain_hypotheses = await self._generate_for_domain(domain, domain_obs)
            hypotheses.extend(domain_hypotheses)
        
        return hypotheses
    
    async def _generate_for_domain(self, domain: str, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate hypotheses for a specific domain"""
        
        if domain == "medical":
            return await self._generate_medical_hypotheses(evidence_list)
        elif domain == "criminal":
            return await self._generate_criminal_hypotheses(evidence_list)
        elif domain == "scientific":
            return await self._generate_scientific_hypotheses(evidence_list)
        elif domain == "technical":
            return await self._generate_technical_hypotheses(evidence_list)
        else:
            return []
    
    async def _generate_medical_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate medical diagnostic hypotheses"""
        
        hypotheses = []
        
        # Extract symptoms and findings
        symptoms = []
        findings = []
        
        for evidence in evidence_list:
            if any(word in str(evidence.description).lower() for word in ["symptom", "complains", "reports", "feels"]):
                symptoms.append(evidence.description)
            elif any(word in str(evidence.description).lower() for word in ["test", "finding", "result", "shows"]):
                findings.append(evidence.description)
        
        # Generate diagnostic hypotheses
        if symptoms or findings:
            # Common medical hypothesis patterns
            hypothesis_patterns = [
                "Infectious disease causing systemic symptoms",
                "Inflammatory condition affecting multiple systems",
                "Metabolic disorder with characteristic presentation",
                "Autoimmune condition with diverse manifestations",
                "Medication side effect or interaction"
            ]
            
            for i, pattern in enumerate(hypothesis_patterns):
                hypothesis = Hypothesis(
                    id=f"medical_hyp_{i+1}",
                    statement=pattern,
                    explanation_type=ExplanationType.CAUSAL,
                    scope=[evidence.id for evidence in evidence_list],
                    assumptions=[f"Medical pattern: {pattern}"],
                    predictions=["Further diagnostic tests will support diagnosis"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_criminal_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate criminal investigation hypotheses"""
        
        hypotheses = []
        
        # Common criminal hypothesis patterns
        hypothesis_patterns = [
            "Planned crime with specific motive",
            "Crime of opportunity with circumstantial evidence",
            "Inside job with knowledge of security",
            "Random crime with no specific target",
            "Staged crime scene to mislead investigation"
        ]
        
        for i, pattern in enumerate(hypothesis_patterns):
            hypothesis = Hypothesis(
                id=f"criminal_hyp_{i+1}",
                statement=pattern,
                explanation_type=ExplanationType.INTENTIONAL,
                scope=[evidence.id for evidence in evidence_list],
                assumptions=[f"Criminal pattern: {pattern}"],
                predictions=["Additional evidence will support or refute hypothesis"]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_scientific_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate scientific hypotheses"""
        
        hypotheses = []
        
        # Scientific hypothesis patterns
        hypothesis_patterns = [
            "Underlying mechanism explains observed phenomena",
            "External factor influences system behavior",
            "Measurement error affects observed results",
            "Unknown variable confounds results",
            "Systematic bias in data collection"
        ]
        
        for i, pattern in enumerate(hypothesis_patterns):
            hypothesis = Hypothesis(
                id=f"scientific_hyp_{i+1}",
                statement=pattern,
                explanation_type=ExplanationType.MECHANISTIC,
                scope=[evidence.id for evidence in evidence_list],
                assumptions=[f"Scientific principle: {pattern}"],
                predictions=["Controlled experiments will test hypothesis"]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_technical_hypotheses(self, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Generate technical/engineering hypotheses"""
        
        hypotheses = []
        
        # Technical hypothesis patterns
        hypothesis_patterns = [
            "Component failure causes system malfunction",
            "Software bug produces unexpected behavior",
            "Configuration error affects system performance",
            "Hardware compatibility issue causes problems",
            "Environmental factor impacts system operation"
        ]
        
        for i, pattern in enumerate(hypothesis_patterns):
            hypothesis = Hypothesis(
                id=f"technical_hyp_{i+1}",
                statement=pattern,
                explanation_type=ExplanationType.CAUSAL,
                scope=[evidence.id for evidence in evidence_list],
                assumptions=[f"Technical principle: {pattern}"],
                predictions=["System diagnostics will reveal root cause"]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _pattern_matches_evidence(self, pattern: Dict[str, Any], evidence_list: List[Evidence]) -> bool:
        """Check if an explanation pattern matches the evidence"""
        
        # Simple pattern matching based on keywords
        pattern_keywords = pattern.get("keywords", [])
        
        for evidence in evidence_list:
            evidence_words = str(evidence.description).lower().split()
            if any(keyword in evidence_words for keyword in pattern_keywords):
                return True
        
        return False
    
    async def _instantiate_pattern_hypothesis(self, pattern: Dict[str, Any], evidence_list: List[Evidence]) -> Optional[Hypothesis]:
        """Create a hypothesis from a pattern"""
        
        hypothesis = Hypothesis(
            id=str(uuid4()),
            statement=pattern["statement"],
            explanation_type=ExplanationType(pattern["type"]),
            scope=[evidence.id for evidence in evidence_list],
            assumptions=pattern.get("assumptions", []),
            predictions=pattern.get("predictions", [])
        )
        
        return hypothesis
    
    async def _extract_potential_causes(self, evidence: Evidence) -> List[str]:
        """Extract potential causes from evidence"""
        
        causes = []
        
        # Look for causal indicators
        causal_patterns = [
            r"because of (.+)",
            r"due to (.+)",
            r"caused by (.+)",
            r"resulting from (.+)",
            r"triggered by (.+)",
            r"following (.+)"
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, evidence.description, re.IGNORECASE)
            causes.extend(matches)
        
        return causes
    
    async def _extract_potential_effects(self, evidence: Evidence) -> List[str]:
        """Extract potential effects from evidence"""
        
        effects = []
        
        # Look for effect indicators
        effect_patterns = [
            r"leads to (.+)",
            r"causes (.+)",
            r"results in (.+)",
            r"produces (.+)",
            r"triggers (.+)",
            r"followed by (.+)"
        ]
        
        for pattern in effect_patterns:
            matches = re.findall(pattern, evidence.description, re.IGNORECASE)
            effects.extend(matches)
        
        return effects
    
    async def _parse_ai_hypotheses(self, response: str, evidence_list: List[Evidence]) -> List[Hypothesis]:
        """Parse AI-generated hypotheses from response"""
        
        hypotheses = []
        
        # Simple parsing - in production would use more sophisticated NLP
        lines = response.split('\n')
        
        current_hypothesis = None
        current_statement = ""
        
        for line in lines:
            line = line.strip()
            
            # Look for hypothesis statements
            if line.startswith("Hypothesis") or line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                if current_hypothesis:
                    hypotheses.append(current_hypothesis)
                
                # Extract statement
                if ":" in line:
                    current_statement = line.split(":", 1)[1].strip()
                else:
                    current_statement = line
                
                current_hypothesis = Hypothesis(
                    id=str(uuid4()),
                    statement=current_statement,
                    explanation_type=ExplanationType.MECHANISTIC,  # Default
                    scope=[evidence.id for evidence in evidence_list]
                )
            
            # Parse type
            elif line.startswith("Type:") and current_hypothesis:
                type_str = str(line.split(":", 1)[1].strip()).lower()
                for exp_type in ExplanationType:
                    if exp_type.value in type_str:
                        current_hypothesis.explanation_type = exp_type
                        break
            
            # Parse assumptions
            elif line.startswith("Assumptions:") and current_hypothesis:
                assumptions_str = line.split(":", 1)[1].strip()
                current_hypothesis.assumptions = [assumptions_str]
            
            # Parse predictions
            elif line.startswith("Predictions:") and current_hypothesis:
                predictions_str = line.split(":", 1)[1].strip()
                current_hypothesis.predictions = [predictions_str]
        
        # Add final hypothesis
        if current_hypothesis:
            hypotheses.append(current_hypothesis)
        
        return hypotheses
    
    def _remove_duplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Remove duplicate hypotheses"""
        
        unique_hypotheses = []
        seen_statements = set()
        
        for hypothesis in hypotheses:
            if hypothesis.statement not in seen_statements:
                unique_hypotheses.append(hypothesis)
                seen_statements.add(hypothesis.statement)
        
        return unique_hypotheses
    
    async def _evaluate_hypotheses(
        self, 
        hypotheses: List[Hypothesis], 
        evidence_list: List[Evidence]
    ) -> List[Hypothesis]:
        """Evaluate hypotheses against explanation criteria"""
        
        evaluated_hypotheses = []
        
        for hypothesis in hypotheses:
            # Evaluate against each criterion
            hypothesis.simplicity_score = await self._evaluate_simplicity(hypothesis, evidence_list)
            hypothesis.scope_score = await self._evaluate_scope(hypothesis, evidence_list)
            hypothesis.plausibility_score = await self._evaluate_plausibility(hypothesis, evidence_list)
            hypothesis.coherence_score = await self._evaluate_coherence(hypothesis, evidence_list)
            hypothesis.testability_score = await self._evaluate_testability(hypothesis, evidence_list)
            
            # Calculate overall score
            hypothesis.calculate_overall_score()
            
            # Map evidence support
            await self._map_evidence_support(hypothesis, evidence_list)
            
            # Update status
            if hypothesis.overall_score >= 0.7:
                hypothesis.status = HypothesisStatus.CONFIRMED
            elif hypothesis.overall_score >= 0.5:
                hypothesis.status = HypothesisStatus.EVALUATED
            else:
                hypothesis.status = HypothesisStatus.GENERATED
            
            # Set confidence
            hypothesis.confidence = hypothesis.overall_score
            
            evaluated_hypotheses.append(hypothesis)
        
        return evaluated_hypotheses
    
    async def _evaluate_simplicity(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> float:
        """Evaluate simplicity of hypothesis (Occam's razor)"""
        
        # Simple heuristic: fewer assumptions = higher simplicity
        assumption_penalty = len(hypothesis.assumptions) * 0.1
        
        # Word count penalty for complex statements
        word_count = len(hypothesis.statement.split())
        complexity_penalty = max(0, (word_count - 10) * 0.01)
        
        simplicity = 1.0 - assumption_penalty - complexity_penalty
        
        return max(0.0, min(1.0, simplicity))
    
    async def _evaluate_scope(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> float:
        """Evaluate scope of hypothesis (how much it explains)"""
        
        # Scope based on how much evidence it explains
        explained_evidence = len(hypothesis.scope)
        total_evidence = len(evidence_list)
        
        if total_evidence > 0:
            scope_score = explained_evidence / total_evidence
        else:
            scope_score = 0.0
        
        return max(0.0, min(1.0, scope_score))
    
    async def _evaluate_plausibility(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> float:
        """Evaluate plausibility of hypothesis"""
        
        # Plausibility based on consistency with evidence
        supporting_evidence = len(hypothesis.supporting_evidence)
        contradicting_evidence = len(hypothesis.contradicting_evidence)
        total_evidence = supporting_evidence + contradicting_evidence
        
        if total_evidence > 0:
            plausibility = supporting_evidence / total_evidence
        else:
            plausibility = 0.5  # Neutral if no evidence
        
        # Adjust for evidence reliability
        if hypothesis.supporting_evidence:
            avg_reliability = sum(evidence.reliability for evidence in hypothesis.supporting_evidence) / len(hypothesis.supporting_evidence)
            plausibility *= avg_reliability
        
        return max(0.0, min(1.0, plausibility))
    
    async def _evaluate_coherence(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> float:
        """Evaluate internal coherence of hypothesis"""
        
        # Simple coherence check based on logical consistency
        # In a full implementation, would check for logical contradictions
        
        coherence = 0.8  # Default high coherence
        
        # Check for contradictory assumptions
        assumptions = [str(assumption).lower() for assumption in hypothesis.assumptions]
        
        # Look for contradictory terms
        contradictory_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("cause", "prevent"),
            ("enable", "disable")
        ]
        
        for term1, term2 in contradictory_pairs:
            if any(term1 in assumption for assumption in assumptions) and any(term2 in assumption for assumption in assumptions):
                coherence -= 0.2
        
        return max(0.0, min(1.0, coherence))
    
    async def _evaluate_testability(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> float:
        """Evaluate testability of hypothesis"""
        
        # Testability based on number of testable predictions
        testable_predictions = len(hypothesis.predictions)
        
        # Base testability on predictions
        if testable_predictions > 0:
            testability = min(1.0, testable_predictions * 0.3)
        else:
            testability = 0.1
        
        # Bonus for specific, measurable predictions
        specific_indicators = ["measure", "test", "observe", "detect", "quantify"]
        for prediction in hypothesis.predictions:
            if any(indicator in str(prediction).lower() for indicator in specific_indicators):
                testability += 0.1
        
        return max(0.0, min(1.0, testability))
    
    async def _map_evidence_support(self, hypothesis: Hypothesis, evidence_list: List[Evidence]) -> None:
        """Map which evidence supports or contradicts hypothesis"""
        
        for evidence in evidence_list:
            # Simple support/contradiction detection
            support_score = await self._calculate_evidence_support(hypothesis, evidence)
            
            if support_score > 0.6:
                hypothesis.supporting_evidence.append(evidence)
                evidence.supports.append(hypothesis.id)
            elif support_score < 0.4:
                hypothesis.contradicting_evidence.append(evidence)
                evidence.contradicts.append(hypothesis.id)
    
    async def _calculate_evidence_support(self, hypothesis: Hypothesis, evidence: Evidence) -> float:
        """Calculate how much evidence supports hypothesis"""
        
        # Simple keyword overlap method
        hyp_words = set(str(hypothesis.statement).lower().split())
        evidence_words = set(str(evidence.description).lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        hyp_words -= stop_words
        evidence_words -= stop_words
        
        if hyp_words and evidence_words:
            overlap = len(hyp_words & evidence_words)
            support = overlap / len(hyp_words)
            return max(0.0, min(1.0, support))
        
        return 0.5  # Neutral if no words to compare
    
    async def _select_best_explanation(
        self, 
        hypotheses: List[Hypothesis], 
        evidence_list: List[Evidence], 
        context: Dict[str, Any] = None
    ) -> AbductiveExplanation:
        """Select the best explanation from evaluated hypotheses"""
        
        if not hypotheses:
            # Create default explanation if no hypotheses
            default_hypothesis = Hypothesis(
                id="default",
                statement="No plausible explanation found",
                explanation_type=ExplanationType.MECHANISTIC,
                overall_score=0.1,
                confidence=0.1
            )
            
            return AbductiveExplanation(
                id=str(uuid4()),
                query=context.get("query", "Unknown query") if context else "Unknown query",
                observations=evidence_list,
                best_hypothesis=default_hypothesis,
                explanation_confidence=0.1
            )
        
        # Sort hypotheses by overall score
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.overall_score, reverse=True)
        
        # Select best hypothesis
        best_hypothesis = sorted_hypotheses[0]
        
        # Alternative hypotheses
        alternative_hypotheses = sorted_hypotheses[1:min(4, len(sorted_hypotheses))]
        
        # Calculate explanation confidence
        confidence = best_hypothesis.overall_score
        
        # Adjust confidence based on gap to alternatives
        if alternative_hypotheses:
            gap_to_next = best_hypothesis.overall_score - alternative_hypotheses[0].overall_score
            confidence += gap_to_next * 0.2  # Bonus for clear winner
        
        # Create explanation
        explanation = AbductiveExplanation(
            id=str(uuid4()),
            query=context.get("query", "Unknown query") if context else "Unknown query",
            observations=evidence_list,
            best_hypothesis=best_hypothesis,
            alternative_hypotheses=alternative_hypotheses,
            evaluation_criteria=self.explanation_criteria,
            explanation_confidence=min(1.0, confidence)
        )
        
        return explanation
    
    async def _enhance_explanation(self, explanation: AbductiveExplanation) -> AbductiveExplanation:
        """Enhance explanation with implications and predictions"""
        
        # Generate implications
        explanation.implications = await self._generate_implications(explanation.best_hypothesis)
        
        # Generate testable predictions
        explanation.testable_predictions = await self._generate_testable_predictions(explanation.best_hypothesis)
        
        # Identify limitations
        explanation.limitations = await self._identify_explanation_limitations(explanation)
        
        # Identify alternative interpretations
        explanation.alternative_interpretations = await self._identify_alternative_interpretations(explanation)
        
        # Identify uncertainty sources
        explanation.uncertainty_sources = await self._identify_uncertainty_sources(explanation)
        
        return explanation
    
    async def _generate_implications(self, hypothesis: Hypothesis) -> List[str]:
        """Generate implications of the hypothesis"""
        
        implications = []
        
        # Direct implications from predictions
        implications.extend(hypothesis.predictions)
        
        # Generate additional implications based on hypothesis type
        if hypothesis.explanation_type == ExplanationType.CAUSAL:
            implications.append("This causal relationship should be reproducible under similar conditions")
            implications.append("Intervening on the cause should affect the outcome")
        
        elif hypothesis.explanation_type == ExplanationType.MECHANISTIC:
            implications.append("The underlying mechanism should be observable or measurable")
            implications.append("Similar systems should exhibit similar mechanisms")
        
        elif hypothesis.explanation_type == ExplanationType.FUNCTIONAL:
            implications.append("The function should be necessary for the observed outcome")
            implications.append("Alternative functions should not produce the same outcome")
        
        return implications
    
    async def _generate_testable_predictions(self, hypothesis: Hypothesis) -> List[str]:
        """Generate testable predictions from hypothesis"""
        
        predictions = []
        
        # Include existing predictions
        predictions.extend(hypothesis.predictions)
        
        # Generate additional testable predictions
        predictions.append(f"If {hypothesis.statement} is true, then specific observable consequences should follow")
        predictions.append(f"Alternative explanations should be less well-supported by evidence")
        
        # Domain-specific predictions
        if any(domain in str(hypothesis.statement).lower() for domain in ["medical", "health", "disease"]):
            predictions.append("Additional symptoms or test results should be consistent with diagnosis")
        
        elif any(domain in str(hypothesis.statement).lower() for domain in ["technical", "system", "failure"]):
            predictions.append("System diagnostics should reveal the hypothesized problem")
        
        return predictions
    
    async def _identify_explanation_limitations(self, explanation: AbductiveExplanation) -> List[str]:
        """Identify limitations of the explanation"""
        
        limitations = []
        
        # Confidence-based limitations
        if explanation.explanation_confidence < 0.7:
            limitations.append("Explanation has moderate confidence - alternative explanations possible")
        
        # Evidence-based limitations
        if len(explanation.observations) < 3:
            limitations.append("Limited evidence base - more observations needed")
        
        # Hypothesis-based limitations
        if explanation.best_hypothesis.testability_score < 0.5:
            limitations.append("Hypothesis has limited testability")
        
        if explanation.best_hypothesis.scope_score < 0.5:
            limitations.append("Explanation covers only a subset of observations")
        
        # Alternative hypotheses
        if explanation.alternative_hypotheses:
            strong_alternatives = [h for h in explanation.alternative_hypotheses if h.overall_score > 0.6]
            if strong_alternatives:
                limitations.append(f"{len(strong_alternatives)} strong alternative explanations exist")
        
        return limitations
    
    async def _identify_alternative_interpretations(self, explanation: AbductiveExplanation) -> List[str]:
        """Identify alternative interpretations"""
        
        alternatives = []
        
        # From alternative hypotheses
        for alt_hypothesis in explanation.alternative_hypotheses:
            if alt_hypothesis.overall_score > 0.5:
                alternatives.append(alt_hypothesis.statement)
        
        # General alternatives
        alternatives.append("Observations may be due to chance or coincidence")
        alternatives.append("Multiple factors may be contributing to observations")
        alternatives.append("Observations may be incomplete or biased")
        
        return alternatives
    
    async def _identify_uncertainty_sources(self, explanation: AbductiveExplanation) -> List[str]:
        """Identify sources of uncertainty"""
        
        uncertainties = []
        
        # Evidence-based uncertainties
        unreliable_evidence = [obs for obs in explanation.observations if obs.reliability < 0.7]
        if unreliable_evidence:
            uncertainties.append(f"{len(unreliable_evidence)} observations have low reliability")
        
        # Hypothesis-based uncertainties
        if explanation.best_hypothesis.plausibility_score < 0.7:
            uncertainties.append("Hypothesis plausibility is moderate")
        
        # Prediction uncertainties
        if not explanation.testable_predictions:
            uncertainties.append("Limited ability to test explanation")
        
        return uncertainties
    
    def _initialize_explanation_patterns(self) -> List[Dict[str, Any]]:
        """Initialize common explanation patterns"""
        
        patterns = [
            {
                "name": "Common Cause",
                "statement": "A common underlying cause explains multiple observations",
                "type": "causal",
                "keywords": ["multiple", "symptoms", "effects", "related"],
                "assumptions": ["Single cause can have multiple effects"],
                "predictions": ["Other effects of the cause should be observable"]
            },
            {
                "name": "Mechanism Failure",
                "statement": "A system mechanism has failed causing observed problems",
                "type": "mechanistic",
                "keywords": ["failure", "malfunction", "broken", "error"],
                "assumptions": ["System has identifiable mechanisms"],
                "predictions": ["Mechanism repair should resolve problems"]
            },
            {
                "name": "External Interference",
                "statement": "External factors interfere with normal operation",
                "type": "causal",
                "keywords": ["external", "interference", "disruption", "environmental"],
                "assumptions": ["Normal operation is disrupted by external factors"],
                "predictions": ["Removing external factors should restore normal operation"]
            },
            {
                "name": "Progressive Deterioration",
                "statement": "Gradual deterioration explains worsening conditions",
                "type": "mechanistic",
                "keywords": ["gradual", "progressive", "worsening", "deterioration"],
                "assumptions": ["Conditions worsen over time"],
                "predictions": ["Deterioration should continue without intervention"]
            }
        ]
        
        return patterns
    
    def get_abductive_stats(self) -> Dict[str, Any]:
        """Get statistics about abductive reasoning usage"""
        
        return {
            "total_hypotheses": len(self.generated_hypotheses),
            "total_explanations": len(self.explanations),
            "hypothesis_types": {et.value: sum(1 for h in self.generated_hypotheses if h.explanation_type == et) for et in ExplanationType},
            "hypothesis_statuses": {hs.value: sum(1 for h in self.generated_hypotheses if h.status == hs) for hs in HypothesisStatus},
            "average_confidence": sum(h.confidence for h in self.generated_hypotheses) / max(len(self.generated_hypotheses), 1),
            "average_explanation_confidence": sum(e.explanation_confidence for e in self.explanations) / max(len(self.explanations), 1),
            "criteria_weights": {
                "simplicity": 0.2,
                "scope": 0.25,
                "plausibility": 0.25,
                "coherence": 0.15,
                "testability": 0.15
            }
        }