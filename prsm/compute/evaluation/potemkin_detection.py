#!/usr/bin/env python3
"""
Potemkin Understanding Detection Framework for NWTN
Based on "Potemkin Understanding in Large Language Models" by Mancoridis et al.

This module implements detection and prevention mechanisms to ensure NWTN's 
hybrid architecture demonstrates genuine understanding rather than "Potemkin understanding" 
- the illusion of understanding driven by answers irreconcilable with human reasoning.

Key Concepts from the Paper:
1. Keystones: Minimal sets of questions that humans can only answer correctly with true understanding
2. Potemkins: When models answer keystones correctly but fail to apply concepts coherently
3. Incoherence: Internal inconsistency in concept representations

NWTN Protection Strategy:
- Systematic keystone testing across reasoning domains
- Coherence validation between System 1 and System 2 outputs
- Cross-validation between explanation and application
- World model consistency checking
- Multi-perspective validation through agent teams

Usage:
    from prsm.compute.evaluation.potemkin_detection import PotemkinDetector
    
    detector = PotemkinDetector()
    result = await detector.evaluate_understanding(nwtn_agent, concept, domain)
    is_genuine = detector.is_genuine_understanding(result)
"""

import asyncio
import json
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
import numpy as np
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

from prsm.compute.nwtn.hybrid_architecture import HybridNWTNEngine, SOC, ConfidenceLevel
from prsm.compute.nwtn.hybrid_integration import HybridNWTNManager

logger = structlog.get_logger(__name__)


class ConceptDomain(str, Enum):
    """Domains for testing conceptual understanding"""
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    LOGIC = "logic"
    LANGUAGE = "language"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"


class KeystoneType(str, Enum):
    """Types of keystone questions"""
    DEFINITION = "definition"
    APPLICATION = "application"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EXPLANATION = "explanation"
    PREDICTION = "prediction"
    ANALYSIS = "analysis"


class PotemkinType(str, Enum):
    """Types of Potemkin understanding failures"""
    EXPLANATION_APPLICATION_GAP = "explanation_application_gap"
    INTERNAL_INCOHERENCE = "internal_incoherence"
    CLASSIFICATION_GENERATION_GAP = "classification_generation_gap"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CONTEXT_DEPENDENT_FAILURE = "context_dependent_failure"


@dataclass
class KeystoneQuestion:
    """A question that tests true understanding of a concept"""
    question_id: str
    concept: str
    domain: ConceptDomain
    keystone_type: KeystoneType
    question_text: str
    correct_answer: Any
    explanation: str
    human_failure_modes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "concept": self.concept,
            "domain": self.domain.value,
            "keystone_type": self.keystone_type.value,
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "human_failure_modes": self.human_failure_modes
        }


@dataclass
class PotemkinTestResult:
    """Result of testing for Potemkin understanding"""
    test_id: str
    concept: str
    domain: ConceptDomain
    potemkin_type: PotemkinType
    keystone_performance: float
    application_performance: float
    coherence_score: float
    is_potemkin: bool
    evidence: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "concept": self.concept,
            "domain": self.domain.value,
            "potemkin_type": self.potemkin_type.value,
            "keystone_performance": self.keystone_performance,
            "application_performance": self.application_performance,
            "coherence_score": self.coherence_score,
            "is_potemkin": self.is_potemkin,
            "evidence": self.evidence,
            "reasoning_trace": self.reasoning_trace,
            "timestamp": self.timestamp.isoformat()
        }


class PotemkinDetector:
    """
    Comprehensive Potemkin understanding detection for NWTN
    
    This class implements the testing framework from the paper to ensure
    NWTN demonstrates genuine understanding rather than sophisticated mimicry.
    """
    
    def __init__(self):
        self.keystones = self._initialize_keystone_library()
        self.coherence_tests = self._initialize_coherence_tests()
        self.test_results: List[PotemkinTestResult] = []
        
    def _initialize_keystone_library(self) -> Dict[str, List[KeystoneQuestion]]:
        """Initialize comprehensive keystone question library"""
        
        keystones = {}
        
        # Chemistry keystones
        keystones["chemistry"] = [
            KeystoneQuestion(
                question_id="chem_lewis_def",
                concept="lewis_structure",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.DEFINITION,
                question_text="What is a Lewis structure and what does it represent?",
                correct_answer="A Lewis structure is a diagram showing the bonding between atoms and lone pairs of electrons in a molecule, representing the valence electron distribution.",
                explanation="Lewis structures are fundamental to understanding molecular bonding",
                human_failure_modes=["confusing with molecular formulas", "ignoring lone pairs", "incorrect electron counting"]
            ),
            KeystoneQuestion(
                question_id="chem_lewis_app",
                concept="lewis_structure",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.APPLICATION,
                question_text="Draw the Lewis structure for water (H2O)",
                correct_answer="O with two H atoms bonded, O has two lone pairs",
                explanation="Application test for Lewis structure understanding",
                human_failure_modes=["incorrect bonding", "wrong electron count", "missing lone pairs"]
            ),
            KeystoneQuestion(
                question_id="chem_equilibrium_def",
                concept="chemical_equilibrium",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.DEFINITION,
                question_text="What is chemical equilibrium and what determines the equilibrium position?",
                correct_answer="Chemical equilibrium occurs when forward and reverse reaction rates are equal, determined by thermodynamics (ΔG = 0) and described by the equilibrium constant.",
                explanation="Equilibrium is fundamental to chemical reaction understanding",
                human_failure_modes=["confusing with equal concentrations", "ignoring thermodynamics", "misunderstanding kinetics vs thermodynamics"]
            ),
            KeystoneQuestion(
                question_id="chem_equilibrium_app",
                concept="chemical_equilibrium",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.PREDICTION,
                question_text="If the temperature is increased for an exothermic reaction at equilibrium, what happens to the equilibrium position?",
                correct_answer="The equilibrium shifts left (toward reactants) according to Le Chatelier's principle, as the system counteracts the temperature increase.",
                explanation="Tests understanding of equilibrium response to perturbations",
                human_failure_modes=["confusing endo/exothermic effects", "ignoring Le Chatelier's principle", "focusing on kinetics instead of thermodynamics"]
            ),
            KeystoneQuestion(
                question_id="chem_catalyst_def",
                concept="catalysis",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.DEFINITION,
                question_text="What is a catalyst and how does it affect chemical reactions?",
                correct_answer="A catalyst increases reaction rate by providing an alternative pathway with lower activation energy, without changing the thermodynamics or equilibrium position.",
                explanation="Catalysis is crucial for understanding reaction kinetics",
                human_failure_modes=["thinking catalysts change equilibrium", "confusing with reactants", "misunderstanding energy diagrams"]
            ),
            KeystoneQuestion(
                question_id="chem_catalyst_app",
                concept="catalysis",
                domain=ConceptDomain.CHEMISTRY,
                keystone_type=KeystoneType.ANALYSIS,
                question_text="Explain why a catalyst doesn't change the equilibrium constant of a reaction",
                correct_answer="Catalysts only change kinetics (pathway), not thermodynamics. The equilibrium constant depends only on ΔG°, which is unchanged by catalysts.",
                explanation="Tests deep understanding of thermodynamics vs kinetics",
                human_failure_modes=["confusing kinetics with thermodynamics", "thinking faster reaction changes equilibrium", "misunderstanding energy relationships"]
            )
        ]
        
        # Physics keystones
        keystones["physics"] = [
            KeystoneQuestion(
                question_id="phys_energy_def",
                concept="conservation_of_energy",
                domain=ConceptDomain.PHYSICS,
                keystone_type=KeystoneType.DEFINITION,
                question_text="State the law of conservation of energy and explain what it means",
                correct_answer="Energy cannot be created or destroyed, only transformed from one form to another. The total energy of an isolated system remains constant.",
                explanation="Energy conservation is fundamental to physics",
                human_failure_modes=["thinking energy can be lost", "confusing with conservation of mass", "ignoring system boundaries"]
            ),
            KeystoneQuestion(
                question_id="phys_energy_app",
                concept="conservation_of_energy",
                domain=ConceptDomain.PHYSICS,
                keystone_type=KeystoneType.APPLICATION,
                question_text="A ball is thrown upward. Describe the energy transformations from launch to maximum height",
                correct_answer="Kinetic energy is converted to gravitational potential energy. At maximum height, all kinetic energy has become potential energy.",
                explanation="Tests application of energy conservation principles",
                human_failure_modes=["ignoring air resistance effects", "incorrect energy identification", "thinking energy disappears"]
            ),
            KeystoneQuestion(
                question_id="phys_force_def",
                concept="newtons_laws",
                domain=ConceptDomain.PHYSICS,
                keystone_type=KeystoneType.DEFINITION,
                question_text="State Newton's second law and explain its significance",
                correct_answer="F = ma. The net force on an object equals its mass times acceleration. This relates cause (force) to effect (acceleration).",
                explanation="Newton's laws are fundamental to mechanics",
                human_failure_modes=["confusing force with velocity", "ignoring net force concept", "misunderstanding vector nature"]
            ),
            KeystoneQuestion(
                question_id="phys_force_app",
                concept="newtons_laws",
                domain=ConceptDomain.PHYSICS,
                keystone_type=KeystoneType.PREDICTION,
                question_text="If you push a box with 10N force and friction provides 6N opposing force, what is the net force?",
                correct_answer="4N in the direction of pushing. Net force = applied force - friction = 10N - 6N = 4N.",
                explanation="Tests understanding of net force calculation",
                human_failure_modes=["adding forces incorrectly", "ignoring friction", "confusing individual forces with net force"]
            )
        ]
        
        # Logic keystones
        keystones["logic"] = [
            KeystoneQuestion(
                question_id="logic_syllogism_def",
                concept="logical_syllogism",
                domain=ConceptDomain.LOGIC,
                keystone_type=KeystoneType.DEFINITION,
                question_text="What is a logical syllogism and what makes it valid?",
                correct_answer="A syllogism is a logical argument with two premises and a conclusion. It's valid if the conclusion necessarily follows from the premises, regardless of truth value.",
                explanation="Syllogisms test logical reasoning ability",
                human_failure_modes=["confusing validity with truth", "ignoring logical structure", "adding unstated assumptions"]
            ),
            KeystoneQuestion(
                question_id="logic_syllogism_app",
                concept="logical_syllogism",
                domain=ConceptDomain.LOGIC,
                keystone_type=KeystoneType.CLASSIFICATION,
                question_text="Is this syllogism valid? 'All birds can fly. Penguins are birds. Therefore, penguins can fly.'",
                correct_answer="Yes, the syllogism is logically valid. The conclusion follows from the premises even though the first premise is factually false.",
                explanation="Tests understanding of validity vs truth",
                human_failure_modes=["confusing validity with factual truth", "rejecting due to false premise", "misunderstanding logical structure"]
            )
        ]
        
        return keystones
    
    def _initialize_coherence_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize coherence testing procedures"""
        
        return {
            "self_consistency": [
                {
                    "name": "generate_then_classify",
                    "description": "Generate example, then classify own generation",
                    "procedure": "First ask model to generate example of concept, then ask if that example is correct"
                },
                {
                    "name": "explain_then_apply",
                    "description": "Explain concept, then apply it",
                    "procedure": "First ask for explanation, then test application of same concept"
                },
                {
                    "name": "temporal_consistency",
                    "description": "Test same question at different times",
                    "procedure": "Ask same question multiple times to check for consistent responses"
                }
            ],
            "cross_perspective": [
                {
                    "name": "system1_system2_agreement",
                    "description": "Check if System 1 and System 2 agree",
                    "procedure": "Compare fast recognition with slow reasoning outputs"
                },
                {
                    "name": "multi_agent_consensus",
                    "description": "Check agreement across agent team",
                    "procedure": "Test if agents with different temperatures reach same conclusions"
                }
            ]
        }
    
    async def evaluate_understanding(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> PotemkinTestResult:
        """
        Comprehensive evaluation for Potemkin understanding
        
        This is the main testing function that implements the paper's methodology
        """
        
        test_id = f"potemkin_test_{uuid4().hex[:8]}"
        
        logger.info("Starting Potemkin understanding evaluation",
                   test_id=test_id, concept=concept, domain=domain.value)
        
        # Step 1: Test keystone performance
        keystone_score = await self._test_keystone_performance(agent, concept, domain)
        
        # Step 2: Test application performance  
        application_score = await self._test_application_performance(agent, concept, domain)
        
        # Step 3: Test internal coherence
        coherence_score = await self._test_internal_coherence(agent, concept, domain)
        
        # Step 4: Determine if this represents Potemkin understanding
        is_potemkin, potemkin_type, evidence = self._analyze_potemkin_indicators(
            keystone_score, application_score, coherence_score, concept, domain
        )
        
        # Step 5: Collect reasoning trace
        reasoning_trace = await self._collect_reasoning_trace(agent, concept, domain)
        
        result = PotemkinTestResult(
            test_id=test_id,
            concept=concept,
            domain=domain,
            potemkin_type=potemkin_type,
            keystone_performance=keystone_score,
            application_performance=application_score,
            coherence_score=coherence_score,
            is_potemkin=is_potemkin,
            evidence=evidence,
            reasoning_trace=reasoning_trace,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.test_results.append(result)
        
        logger.info("Potemkin evaluation completed",
                   test_id=test_id,
                   is_potemkin=is_potemkin,
                   keystone_score=keystone_score,
                   application_score=application_score,
                   coherence_score=coherence_score)
        
        return result
    
    async def _test_keystone_performance(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test performance on keystone questions that humans can only answer with true understanding"""
        
        domain_key = domain.value
        if domain_key not in self.keystones:
            logger.warning(f"No keystones available for domain {domain_key}")
            return 0.0
        
        concept_keystones = [ks for ks in self.keystones[domain_key] if ks.concept == concept]
        if not concept_keystones:
            logger.warning(f"No keystones for concept {concept} in domain {domain_key}")
            return 0.0
        
        correct_answers = 0
        total_questions = len(concept_keystones)
        
        for keystone in concept_keystones:
            try:
                # Test with hybrid agent
                if isinstance(agent, HybridNWTNEngine):
                    result = await agent.process_query(keystone.question_text)
                    response = result.get("response", "")
                else:
                    # Test with manager
                    result = await agent.process_query_with_single_agent(keystone.question_text)
                    response = result.get("response", "")
                
                # Evaluate response quality
                is_correct = self._evaluate_keystone_response(response, keystone)
                if is_correct:
                    correct_answers += 1
                    
            except Exception as e:
                logger.error(f"Error testing keystone {keystone.question_id}: {e}")
        
        score = correct_answers / total_questions if total_questions > 0 else 0.0
        
        logger.debug("Keystone performance tested",
                    concept=concept, domain=domain.value,
                    correct=correct_answers, total=total_questions, score=score)
        
        return score
    
    async def _test_application_performance(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test performance on application tasks that require using the concept"""
        
        domain_key = domain.value
        if domain_key not in self.keystones:
            return 0.0
        
        # Get application keystones for this concept
        application_keystones = [
            ks for ks in self.keystones[domain_key] 
            if ks.concept == concept and ks.keystone_type in [
                KeystoneType.APPLICATION, KeystoneType.PREDICTION, 
                KeystoneType.ANALYSIS, KeystoneType.GENERATION
            ]
        ]
        
        if not application_keystones:
            return 0.0
        
        correct_applications = 0
        total_applications = len(application_keystones)
        
        for keystone in application_keystones:
            try:
                if isinstance(agent, HybridNWTNEngine):
                    result = await agent.process_query(keystone.question_text)
                    response = result.get("response", "")
                else:
                    result = await agent.process_query_with_single_agent(keystone.question_text)
                    response = result.get("response", "")
                
                is_correct = self._evaluate_application_response(response, keystone)
                if is_correct:
                    correct_applications += 1
                    
            except Exception as e:
                logger.error(f"Error testing application {keystone.question_id}: {e}")
        
        score = correct_applications / total_applications if total_applications > 0 else 0.0
        
        logger.debug("Application performance tested",
                    concept=concept, domain=domain.value,
                    correct=correct_applications, total=total_applications, score=score)
        
        return score
    
    async def _test_internal_coherence(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test internal coherence using the paper's methodology"""
        
        coherence_scores = []
        
        # Test 1: Generate-then-classify coherence
        generate_classify_score = await self._test_generate_classify_coherence(agent, concept, domain)
        coherence_scores.append(generate_classify_score)
        
        # Test 2: Explain-then-apply coherence
        explain_apply_score = await self._test_explain_apply_coherence(agent, concept, domain)
        coherence_scores.append(explain_apply_score)
        
        # Test 3: System 1 vs System 2 coherence (NWTN-specific)
        if isinstance(agent, HybridNWTNEngine):
            system_coherence = await self._test_system_coherence(agent, concept, domain)
            coherence_scores.append(system_coherence)
        
        # Test 4: Multi-agent coherence (if using manager)
        if isinstance(agent, HybridNWTNManager):
            multi_agent_coherence = await self._test_multi_agent_coherence(agent, concept, domain)
            coherence_scores.append(multi_agent_coherence)
        
        # Average coherence score
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        logger.debug("Internal coherence tested",
                    concept=concept, domain=domain.value,
                    individual_scores=coherence_scores,
                    overall_coherence=overall_coherence)
        
        return overall_coherence
    
    async def _test_generate_classify_coherence(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test coherence between generation and classification (from the paper)"""
        
        # Generate examples
        generate_prompt = f"Generate 3 examples of {concept} and 3 non-examples"
        
        try:
            if isinstance(agent, HybridNWTNEngine):
                gen_result = await agent.process_query(generate_prompt)
                generation = gen_result.get("response", "")
            else:
                gen_result = await agent.process_query_with_single_agent(generate_prompt)
                generation = gen_result.get("response", "")
            
            # Extract examples from generation (simplified parsing)
            examples = self._extract_examples_from_generation(generation)
            
            # Test classification of generated examples
            correct_classifications = 0
            total_examples = len(examples)
            
            for example in examples:
                classify_prompt = f"Is this an example of {concept}? {example['text']}"
                
                if isinstance(agent, HybridNWTNEngine):
                    class_result = await agent.process_query(classify_prompt)
                    classification = class_result.get("response", "")
                else:
                    class_result = await agent.process_query_with_single_agent(classify_prompt)
                    classification = class_result.get("response", "")
                
                # Check if classification matches intended type
                predicted_positive = "yes" in classification.lower() or "true" in classification.lower()
                if predicted_positive == example['is_positive']:
                    correct_classifications += 1
            
            coherence_score = correct_classifications / total_examples if total_examples > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in generate-classify coherence test: {e}")
            coherence_score = 0.0
        
        return coherence_score
    
    async def _test_explain_apply_coherence(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test coherence between explanation and application"""
        
        # Get explanation
        explain_prompt = f"Explain what {concept} means and how to identify it"
        
        try:
            if isinstance(agent, HybridNWTNEngine):
                explain_result = await agent.process_query(explain_prompt)
                explanation = explain_result.get("response", "")
            else:
                explain_result = await agent.process_query_with_single_agent(explain_prompt)
                explanation = explain_result.get("response", "")
            
            # Test application
            apply_prompt = f"Apply your understanding of {concept} to analyze this example: [concept-specific example]"
            
            if isinstance(agent, HybridNWTNEngine):
                apply_result = await agent.process_query(apply_prompt)
                application = apply_result.get("response", "")
            else:
                apply_result = await agent.process_query_with_single_agent(apply_prompt)
                application = apply_result.get("response", "")
            
            # Measure coherence between explanation and application
            coherence_score = self._measure_explanation_application_coherence(
                explanation, application, concept
            )
            
        except Exception as e:
            logger.error(f"Error in explain-apply coherence test: {e}")
            coherence_score = 0.0
        
        return coherence_score
    
    async def _test_system_coherence(
        self,
        agent: HybridNWTNEngine,
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test coherence between System 1 and System 2 (NWTN-specific)"""
        
        try:
            # Test query about the concept
            test_query = f"Analyze the concept of {concept} in {domain.value}"
            result = await agent.process_query(test_query)
            
            # Check if System 1 SOCs align with System 2 reasoning
            socs_used = result.get("socs_used", [])
            reasoning_trace = result.get("reasoning_trace", [])
            
            # Measure alignment between rapid recognition and detailed reasoning
            coherence_score = self._measure_system_alignment(socs_used, reasoning_trace, concept)
            
        except Exception as e:
            logger.error(f"Error in system coherence test: {e}")
            coherence_score = 0.0
        
        return coherence_score
    
    async def _test_multi_agent_coherence(
        self,
        manager: HybridNWTNManager,
        concept: str,
        domain: ConceptDomain
    ) -> float:
        """Test coherence across multiple agents (NWTN-specific)"""
        
        try:
            # Create small team for testing
            team_result = await manager.process_query_with_team(
                f"Explain and apply the concept of {concept}",
                domain=domain.value,
                team_size=3
            )
            
            # Measure agreement across team members
            individual_results = team_result.get("individual_results", [])
            agreement_score = team_result.get("team_agreement_score", 0.0)
            
            # Coherence is high when agents agree on core concepts
            return agreement_score
            
        except Exception as e:
            logger.error(f"Error in multi-agent coherence test: {e}")
            return 0.0
    
    def _analyze_potemkin_indicators(
        self,
        keystone_score: float,
        application_score: float,
        coherence_score: float,
        concept: str,
        domain: ConceptDomain
    ) -> Tuple[bool, PotemkinType, Dict[str, Any]]:
        """Analyze results to determine if Potemkin understanding is present"""
        
        evidence = {
            "keystone_score": keystone_score,
            "application_score": application_score,
            "coherence_score": coherence_score,
            "score_gap": keystone_score - application_score,
            "coherence_threshold": 0.7,
            "performance_threshold": 0.8
        }
        
        # Key indicator from paper: high keystone performance but low application
        explanation_application_gap = (keystone_score >= 0.8 and application_score < 0.6)
        
        # Internal incoherence indicator
        internal_incoherence = (coherence_score < 0.7)
        
        # Performance gap indicator
        significant_performance_gap = (keystone_score - application_score) > 0.3
        
        # Determine Potemkin type and presence
        if explanation_application_gap and significant_performance_gap:
            return True, PotemkinType.EXPLANATION_APPLICATION_GAP, evidence
        elif internal_incoherence:
            return True, PotemkinType.INTERNAL_INCOHERENCE, evidence
        elif significant_performance_gap:
            return True, PotemkinType.CLASSIFICATION_GENERATION_GAP, evidence
        else:
            return False, PotemkinType.EXPLANATION_APPLICATION_GAP, evidence  # Default type
    
    async def _collect_reasoning_trace(
        self,
        agent: Union[HybridNWTNEngine, HybridNWTNManager],
        concept: str,
        domain: ConceptDomain
    ) -> List[Dict[str, Any]]:
        """Collect detailed reasoning trace for analysis"""
        
        trace_query = f"Provide detailed reasoning about the concept {concept} in {domain.value}"
        
        try:
            if isinstance(agent, HybridNWTNEngine):
                result = await agent.process_query(trace_query)
            else:
                result = await agent.process_query_with_single_agent(trace_query)
            
            return result.get("reasoning_trace", [])
            
        except Exception as e:
            logger.error(f"Error collecting reasoning trace: {e}")
            return []
    
    def _evaluate_keystone_response(self, response: str, keystone: KeystoneQuestion) -> bool:
        """Evaluate if response correctly answers keystone question"""
        
        response_lower = response.lower()
        correct_answer_lower = str(keystone.correct_answer).lower()
        
        # Simple keyword matching (in production, this would be more sophisticated)
        key_concepts = correct_answer_lower.split()
        
        matched_concepts = 0
        for concept in key_concepts:
            if len(concept) > 3 and concept in response_lower:
                matched_concepts += 1
        
        # Consider correct if majority of key concepts are present
        return matched_concepts >= len(key_concepts) * 0.6
    
    def _evaluate_application_response(self, response: str, keystone: KeystoneQuestion) -> bool:
        """Evaluate if response correctly applies the concept"""
        
        # This would use more sophisticated evaluation in production
        # For now, use keyword matching and length heuristics
        
        response_lower = response.lower()
        
        # Check for application-specific keywords
        application_keywords = ["because", "therefore", "since", "due to", "results in", "leads to"]
        has_reasoning = any(keyword in response_lower for keyword in application_keywords)
        
        # Check response length (applications should be substantive)
        has_substance = len(response.split()) > 10
        
        # Check for concept-specific terms
        concept_terms = keystone.concept.lower().replace("_", " ").split()
        has_concept_terms = any(term in response_lower for term in concept_terms)
        
        return has_reasoning and has_substance and has_concept_terms
    
    def _extract_examples_from_generation(self, generation: str) -> List[Dict[str, Any]]:
        """Extract examples from generated text (simplified)"""
        
        # This is a simplified implementation
        # In production, this would use more sophisticated parsing
        
        lines = generation.split('\n')
        examples = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                # Heuristic: assume first half are positive examples
                is_positive = len(examples) < 3
                examples.append({
                    "text": line,
                    "is_positive": is_positive
                })
            
            if len(examples) >= 6:  # Stop at 6 examples
                break
        
        return examples
    
    def _measure_explanation_application_coherence(
        self,
        explanation: str,
        application: str,
        concept: str
    ) -> float:
        """Measure coherence between explanation and application"""
        
        # Simple coherence measurement
        # In production, this would use semantic similarity
        
        explanation_words = set(explanation.lower().split())
        application_words = set(application.lower().split())
        
        # Jaccard similarity
        intersection = len(explanation_words & application_words)
        union = len(explanation_words | application_words)
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Bonus for concept-specific terms
        concept_terms = concept.replace("_", " ").split()
        concept_in_both = all(
            term.lower() in explanation.lower() and term.lower() in application.lower()
            for term in concept_terms
        )
        
        if concept_in_both:
            similarity += 0.2
        
        return min(similarity, 1.0)
    
    def _measure_system_alignment(
        self,
        socs_used: List[Dict[str, Any]],
        reasoning_trace: List[Dict[str, Any]],
        concept: str
    ) -> float:
        """Measure alignment between System 1 SOCs and System 2 reasoning"""
        
        if not socs_used or not reasoning_trace:
            return 0.0
        
        # Extract SOC names
        soc_names = set(soc.get("name", "").lower() for soc in socs_used)
        
        # Extract reasoning concepts
        reasoning_text = " ".join(step.get("description", "") for step in reasoning_trace).lower()
        
        # Check if SOCs are reflected in reasoning
        soc_reflection_count = 0
        for soc_name in soc_names:
            if soc_name and soc_name in reasoning_text:
                soc_reflection_count += 1
        
        alignment_score = soc_reflection_count / len(soc_names) if soc_names else 0.0
        
        return alignment_score
    
    def is_genuine_understanding(self, result: PotemkinTestResult) -> bool:
        """Determine if result indicates genuine understanding"""
        return not result.is_potemkin
    
    def get_potemkin_rate(self, domain: Optional[ConceptDomain] = None) -> float:
        """Calculate overall Potemkin rate"""
        
        if domain:
            relevant_results = [r for r in self.test_results if r.domain == domain]
        else:
            relevant_results = self.test_results
        
        if not relevant_results:
            return 0.0
        
        potemkin_count = sum(1 for r in relevant_results if r.is_potemkin)
        return potemkin_count / len(relevant_results)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive Potemkin understanding report"""
        
        total_tests = len(self.test_results)
        potemkin_tests = [r for r in self.test_results if r.is_potemkin]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "potemkin_count": len(potemkin_tests),
                "potemkin_rate": len(potemkin_tests) / total_tests if total_tests > 0 else 0.0,
                "avg_keystone_performance": np.mean([r.keystone_performance for r in self.test_results]) if self.test_results else 0.0,
                "avg_application_performance": np.mean([r.application_performance for r in self.test_results]) if self.test_results else 0.0,
                "avg_coherence_score": np.mean([r.coherence_score for r in self.test_results]) if self.test_results else 0.0
            },
            "by_domain": {},
            "by_potemkin_type": {},
            "recommendations": self._generate_recommendations()
        }
        
        # Domain breakdown
        for domain in ConceptDomain:
            domain_results = [r for r in self.test_results if r.domain == domain]
            if domain_results:
                domain_potemkins = [r for r in domain_results if r.is_potemkin]
                report["by_domain"][domain.value] = {
                    "total_tests": len(domain_results),
                    "potemkin_count": len(domain_potemkins),
                    "potemkin_rate": len(domain_potemkins) / len(domain_results)
                }
        
        # Potemkin type breakdown
        for potemkin_type in PotemkinType:
            type_results = [r for r in potemkin_tests if r.potemkin_type == potemkin_type]
            if type_results:
                report["by_potemkin_type"][potemkin_type.value] = {
                    "count": len(type_results),
                    "examples": [r.concept for r in type_results[:3]]  # First 3 examples
                }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for addressing Potemkin understanding"""
        
        recommendations = []
        
        potemkin_rate = self.get_potemkin_rate()
        
        if potemkin_rate > 0.3:
            recommendations.append("High Potemkin rate detected. Consider strengthening world model consistency checking.")
        
        if potemkin_rate > 0.5:
            recommendations.append("Critical: Potemkin understanding is prevalent. Implement mandatory coherence validation.")
        
        # Domain-specific recommendations
        for domain in ConceptDomain:
            domain_rate = self.get_potemkin_rate(domain)
            if domain_rate > 0.4:
                recommendations.append(f"High Potemkin rate in {domain.value}. Enhance domain-specific world models.")
        
        # Type-specific recommendations
        potemkin_types = [r.potemkin_type for r in self.test_results if r.is_potemkin]
        if PotemkinType.EXPLANATION_APPLICATION_GAP in potemkin_types:
            recommendations.append("Strengthen connection between System 1 recognition and System 2 application.")
        
        if PotemkinType.INTERNAL_INCOHERENCE in potemkin_types:
            recommendations.append("Implement cross-validation between different reasoning paths.")
        
        if not recommendations:
            recommendations.append("Good news: Low Potemkin rates detected. Continue current approach.")
        
        return recommendations


# Integration with NWTN testing
async def test_nwtn_for_potemkins(
    nwtn_agent: Union[HybridNWTNEngine, HybridNWTNManager],
    concepts: List[str],
    domains: List[ConceptDomain]
) -> Dict[str, Any]:
    """Comprehensive Potemkin testing for NWTN"""
    
    detector = PotemkinDetector()
    results = []
    
    for domain in domains:
        for concept in concepts:
            try:
                result = await detector.evaluate_understanding(nwtn_agent, concept, domain)
                results.append(result)
            except Exception as e:
                logger.error(f"Error testing {concept} in {domain.value}: {e}")
    
    # Generate comprehensive report
    report = detector.generate_report()
    
    return {
        "individual_results": [r.to_dict() for r in results],
        "summary_report": report,
        "genuine_understanding_rate": 1.0 - detector.get_potemkin_rate(),
        "recommendations": report["recommendations"]
    }


# Example usage
if __name__ == "__main__":
    async def demo_potemkin_detection():
        """Demo of Potemkin detection for NWTN"""
        
        print("🔍 Potemkin Understanding Detection Demo")
        print("=" * 50)
        
        # Create detector
        detector = PotemkinDetector()
        
        # Test concepts
        concepts = ["lewis_structure", "chemical_equilibrium", "conservation_of_energy"]
        domains = [ConceptDomain.CHEMISTRY, ConceptDomain.PHYSICS]
        
        # In a real scenario, you would pass actual NWTN agents
        # For demo, we'll show the framework structure
        
        print(f"Keystone library contains {len(detector.keystones)} domains")
        print(f"Testing {len(concepts)} concepts across {len(domains)} domains")
        
        # Show sample keystones
        for domain_key, keystones in detector.keystones.items():
            print(f"\n{domain_key.upper()} Keystones:")
            for ks in keystones[:2]:  # Show first 2
                print(f"  - {ks.question_id}: {ks.question_text[:60]}...")
        
        print("\n" + "=" * 50)
        print("✅ Potemkin detection framework ready!")
        print("Key features:")
        print("  - Keystone question validation")
        print("  - Application performance testing")
        print("  - Internal coherence checking")
        print("  - System 1/System 2 alignment")
        print("  - Multi-agent consensus validation")
    
    asyncio.run(demo_potemkin_detection())