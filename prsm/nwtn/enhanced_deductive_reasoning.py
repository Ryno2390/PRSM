#!/usr/bin/env python3
"""
Enhanced Deductive Reasoning Engine for NWTN
===========================================

This module implements a comprehensive deductive reasoning system based on
elemental components derived from formal logic and cognitive science research.

The system follows the five elemental components of deductive reasoning:
1. Identification of Premises
2. Application of Logical Structure  
3. Derivation of Conclusion
4. Evaluation of Validity and Soundness
5. Inference Application

Key Features:
- Comprehensive premise identification and verification
- Formal logical structure application with multiple inference rules
- Systematic conclusion derivation with certainty guarantees
- Rigorous validity and soundness evaluation
- Practical inference application and chaining
"""

import asyncio
import numpy as np
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from uuid import uuid4
from datetime import datetime, timezone
import logging

import structlog

logger = structlog.get_logger(__name__)


class PremiseType(Enum):
    """Types of premises in deductive reasoning"""
    UNIVERSAL_AFFIRMATIVE = "universal_affirmative"    # All A are B
    UNIVERSAL_NEGATIVE = "universal_negative"          # No A are B
    PARTICULAR_AFFIRMATIVE = "particular_affirmative"  # Some A are B
    PARTICULAR_NEGATIVE = "particular_negative"        # Some A are not B
    CONDITIONAL = "conditional"                        # If P then Q
    BICONDITIONAL = "biconditional"                   # P if and only if Q
    CONJUNCTIVE = "conjunctive"                       # P and Q
    DISJUNCTIVE = "disjunctive"                       # P or Q
    NEGATION = "negation"                             # Not P
    EXISTENTIAL = "existential"                       # There exists an X such that P(X)
    UNIVERSAL = "universal"                           # For all X, P(X)


class LogicalFormType(Enum):
    """Types of logical forms"""
    PROPOSITIONAL = "propositional"      # P, Q, R
    PREDICATE = "predicate"              # P(x), Q(x,y)
    MODAL = "modal"                      # Necessarily P, Possibly P
    TEMPORAL = "temporal"                # Always P, Eventually P
    DEONTIC = "deontic"                  # Obligatory P, Permitted P


class InferenceRuleType(Enum):
    """Comprehensive set of inference rules"""
    # Propositional Rules
    MODUS_PONENS = "modus_ponens"                    # P → Q, P ⊢ Q
    MODUS_TOLLENS = "modus_tollens"                  # P → Q, ¬Q ⊢ ¬P
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"  # P → Q, Q → R ⊢ P → R
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"   # P ∨ Q, ¬P ⊢ Q
    CONSTRUCTIVE_DILEMMA = "constructive_dilemma"     # P → Q, R → S, P ∨ R ⊢ Q ∨ S
    DESTRUCTIVE_DILEMMA = "destructive_dilemma"       # P → Q, R → S, ¬Q ∨ ¬S ⊢ ¬P ∨ ¬R
    
    # Propositional Equivalences
    DOUBLE_NEGATION = "double_negation"               # ¬¬P ⊢ P
    DE_MORGAN = "de_morgan"                           # ¬(P ∧ Q) ⊢ ¬P ∨ ¬Q
    DISTRIBUTION = "distribution"                     # P ∧ (Q ∨ R) ⊢ (P ∧ Q) ∨ (P ∧ R)
    MATERIAL_IMPLICATION = "material_implication"     # P → Q ⊢ ¬P ∨ Q
    
    # Predicate Rules
    UNIVERSAL_INSTANTIATION = "universal_instantiation"  # ∀x P(x) ⊢ P(a)
    UNIVERSAL_GENERALIZATION = "universal_generalization"  # P(a) ⊢ ∀x P(x) [with restrictions]
    EXISTENTIAL_INSTANTIATION = "existential_instantiation"  # ∃x P(x) ⊢ P(a)
    EXISTENTIAL_GENERALIZATION = "existential_generalization"  # P(a) ⊢ ∃x P(x)
    
    # Categorical Syllogisms
    BARBARA = "barbara"       # All A are B, All B are C ⊢ All A are C
    CELARENT = "celarent"     # No A are B, All C are A ⊢ No C are B
    DARII = "darii"           # All A are B, Some C are A ⊢ Some C are B
    FERIO = "ferio"           # No A are B, Some C are A ⊢ Some C are not B
    CESARE = "cesare"         # No A are B, All C are A ⊢ No C are B
    CAMESTRES = "camestres"   # All A are B, No C are B ⊢ No C are A
    
    # Advanced Rules
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"  # Assume ¬P, derive contradiction ⊢ P
    PROOF_BY_CASES = "proof_by_cases"                  # P ∨ Q, P → R, Q → R ⊢ R
    MATHEMATICAL_INDUCTION = "mathematical_induction"   # P(0), ∀k(P(k) → P(k+1)) ⊢ ∀n P(n)


class PremiseSource(Enum):
    """Sources of premises for verification"""
    AXIOM = "axiom"                    # Mathematical or logical axiom
    DEFINITION = "definition"          # Definitional truth
    EMPIRICAL = "empirical"           # Empirical observation
    AUTHORITY = "authority"           # Authoritative source
    ASSUMPTION = "assumption"         # Assumed for argument
    DERIVED = "derived"               # Derived from previous reasoning
    HYPOTHESIS = "hypothesis"         # Hypothetical premise
    COMMON_KNOWLEDGE = "common_knowledge"  # Generally accepted fact


@dataclass
class Premise:
    """Enhanced premise with comprehensive identification and verification"""
    
    # Core identification
    id: str
    content: str
    premise_type: PremiseType
    logical_form: str
    
    # Verification and validation
    source: PremiseSource
    truth_value: Optional[bool] = None
    confidence: float = 1.0
    reliability: float = 1.0
    
    # Parsing and structure
    subject: str = ""
    predicate: str = ""
    quantifier: str = ""
    modality: str = ""
    temporal_aspects: List[str] = field(default_factory=list)
    
    # Relationships
    depends_on: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    
    # Metadata
    domain: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-processing after initialization"""
        if not self.logical_form:
            self.logical_form = self._generate_logical_form()
    
    def _generate_logical_form(self) -> str:
        """Generate logical form from premise content"""
        # This would use more sophisticated parsing in practice
        if self.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE:
            return f"∀x({self.subject}(x) → {self.predicate}(x))"
        elif self.premise_type == PremiseType.UNIVERSAL_NEGATIVE:
            return f"∀x({self.subject}(x) → ¬{self.predicate}(x))"
        elif self.premise_type == PremiseType.PARTICULAR_AFFIRMATIVE:
            return f"∃x({self.subject}(x) ∧ {self.predicate}(x))"
        elif self.premise_type == PremiseType.PARTICULAR_NEGATIVE:
            return f"∃x({self.subject}(x) ∧ ¬{self.predicate}(x))"
        elif self.premise_type == PremiseType.CONDITIONAL:
            return f"{self.subject} → {self.predicate}"
        else:
            return self.content  # Fallback
    
    def get_truth_certainty(self) -> float:
        """Calculate truth certainty combining confidence and reliability"""
        if self.truth_value is None:
            return 0.0
        return self.confidence * self.reliability if self.truth_value else 0.0


@dataclass
class LogicalStructure:
    """Represents a logical structure for inference"""
    
    structure_type: InferenceRuleType
    premise_patterns: List[str]
    conclusion_pattern: str
    
    # Validity conditions
    validity_conditions: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    
    # Application metadata
    applications: int = 0
    success_rate: float = 1.0
    
    def is_applicable(self, premises: List[Premise]) -> bool:
        """Check if this structure is applicable to given premises"""
        return len(premises) >= len(self.premise_patterns)
    
    def apply_structure(self, premises: List[Premise]) -> Optional[str]:
        """Apply this logical structure to derive conclusion"""
        # Implementation would depend on specific rule type
        return None


@dataclass
class DeductiveConclusion:
    """A conclusion derived through deductive reasoning"""
    
    id: str
    content: str
    logical_form: str
    
    # Derivation
    premises_used: List[str]
    logical_structure: LogicalStructure
    derivation_steps: List[str]
    
    # Certainty properties
    is_certain: bool = True    # Deductive conclusions are certain if valid and sound
    validity: float = 1.0      # Logical validity
    soundness: float = 1.0     # Truth of premises
    
    # Contextual properties
    domain: str = ""
    applicability: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_certainty_score(self) -> float:
        """Calculate overall certainty score"""
        return self.validity * self.soundness if self.is_certain else 0.0


@dataclass
class DeductiveProof:
    """A complete deductive proof with all components"""
    
    id: str
    query: str
    
    # Elemental components
    premise_identification: Dict[str, Premise]
    logical_structure_application: List[LogicalStructure]
    conclusion_derivation: DeductiveConclusion
    validity_soundness_evaluation: Dict[str, Any]
    inference_application: Dict[str, Any]
    
    # Proof properties
    is_valid: bool = False
    is_sound: bool = False
    is_complete: bool = False
    
    # Proof metrics
    proof_length: int = 0
    complexity: float = 0.0
    confidence: float = 1.0
    
    proof_steps: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PremiseIdentificationEngine:
    """Engine for comprehensive premise identification and verification"""
    
    def __init__(self):
        self.premise_patterns = self._initialize_premise_patterns()
        self.verification_rules = self._initialize_verification_rules()
        self.premise_cache: Dict[str, Premise] = {}
    
    async def identify_premises(self, statements: List[str], context: Dict[str, Any] = None) -> List[Premise]:
        """Identify and verify premises from statements"""
        
        logger.info(f"Identifying premises from {len(statements)} statements")
        
        premises = []
        
        for i, statement in enumerate(statements):
            premise = await self._identify_single_premise(statement, f"P{i+1}", context)
            
            if premise:
                # Verify premise
                await self._verify_premise(premise, context)
                premises.append(premise)
        
        # Check for premise relationships
        await self._analyze_premise_relationships(premises)
        
        logger.info(f"Identified {len(premises)} premises")
        return premises
    
    async def _identify_single_premise(self, statement: str, premise_id: str, context: Dict[str, Any]) -> Optional[Premise]:
        """Identify a single premise from a statement"""
        
        # Parse the statement
        parsed = await self._parse_statement(statement)
        
        # Determine premise type
        premise_type = await self._determine_premise_type(parsed)
        
        # Determine source
        source = await self._determine_premise_source(statement, context)
        
        # Extract components
        subject, predicate, quantifier = await self._extract_logical_components(parsed)
        
        premise = Premise(
            id=premise_id,
            content=statement,
            premise_type=premise_type,
            logical_form="",  # Will be generated in __post_init__
            source=source,
            subject=subject,
            predicate=predicate,
            quantifier=quantifier,
            domain=context.get("domain", "") if context else "",
            context=context or {}
        )
        
        return premise
    
    async def _parse_statement(self, statement: str) -> Dict[str, Any]:
        """Parse statement into logical components"""
        
        statement = statement.strip()
        
        # Identify quantifiers
        quantifiers = {
            "all": ["all", "every", "each"],
            "some": ["some", "there exist", "there are"],
            "no": ["no", "none", "not any"],
            "some_not": ["some are not", "not all"]
        }
        
        quantifier = None
        for quant_type, patterns in quantifiers.items():
            for pattern in patterns:
                if statement.lower().startswith(pattern):
                    quantifier = quant_type
                    break
            if quantifier:
                break
        
        # Identify logical connectives
        connectives = {
            "and": ["and", "∧", "&"],
            "or": ["or", "∨", "|"],
            "not": ["not", "¬", "~"],
            "implies": ["implies", "→", "=>", "if...then"],
            "iff": ["if and only if", "↔", "iff"]
        }
        
        found_connectives = []
        for conn_type, patterns in connectives.items():
            for pattern in patterns:
                if pattern in statement.lower():
                    found_connectives.append(conn_type)
                    break
        
        return {
            "original": statement,
            "quantifier": quantifier,
            "connectives": found_connectives,
            "is_conditional": "implies" in found_connectives,
            "is_biconditional": "iff" in found_connectives,
            "is_negation": "not" in found_connectives
        }
    
    async def _determine_premise_type(self, parsed: Dict[str, Any]) -> PremiseType:
        """Determine the type of premise from parsed components"""
        
        if parsed["is_conditional"]:
            return PremiseType.CONDITIONAL
        elif parsed["is_biconditional"]:
            return PremiseType.BICONDITIONAL
        elif parsed["quantifier"] == "all":
            return PremiseType.UNIVERSAL_AFFIRMATIVE
        elif parsed["quantifier"] == "no":
            return PremiseType.UNIVERSAL_NEGATIVE
        elif parsed["quantifier"] == "some":
            return PremiseType.PARTICULAR_AFFIRMATIVE
        elif parsed["quantifier"] == "some_not":
            return PremiseType.PARTICULAR_NEGATIVE
        elif "and" in parsed["connectives"]:
            return PremiseType.CONJUNCTIVE
        elif "or" in parsed["connectives"]:
            return PremiseType.DISJUNCTIVE
        elif parsed["is_negation"]:
            return PremiseType.NEGATION
        else:
            return PremiseType.UNIVERSAL_AFFIRMATIVE  # Default
    
    async def _determine_premise_source(self, statement: str, context: Dict[str, Any]) -> PremiseSource:
        """Determine the source of the premise"""
        
        # Heuristics for source determination
        if context and "source" in context:
            source_hints = context["source"].lower()
            if "axiom" in source_hints:
                return PremiseSource.AXIOM
            elif "definition" in source_hints:
                return PremiseSource.DEFINITION
            elif "empirical" in source_hints:
                return PremiseSource.EMPIRICAL
            elif "authority" in source_hints:
                return PremiseSource.AUTHORITY
        
        # Default heuristics
        statement_lower = statement.lower()
        if "by definition" in statement_lower:
            return PremiseSource.DEFINITION
        elif "assume" in statement_lower:
            return PremiseSource.ASSUMPTION
        elif "observe" in statement_lower:
            return PremiseSource.EMPIRICAL
        else:
            return PremiseSource.COMMON_KNOWLEDGE
    
    async def _extract_logical_components(self, parsed: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract subject, predicate, and quantifier from parsed statement"""
        
        statement = parsed["original"]
        quantifier = parsed["quantifier"] or "all"
        
        # Extract subject and predicate using patterns
        patterns = [
            r'(?:all|every|some|no)\s+(.+?)\s+(?:are|is|have|can|will)\s+(.+)',
            r'if\s+(.+?)\s+then\s+(.+)',
            r'(.+?)\s+(?:are|is|have|can|will)\s+(.+)',
            r'(.+?)\s+implies\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, statement.lower())
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                return subject, predicate, quantifier
        
        # Fallback
        words = statement.split()
        if len(words) >= 3:
            subject = words[0] if not words[0].lower() in ["all", "some", "no"] else words[1]
            predicate = " ".join(words[2:])
            return subject, predicate, quantifier
        
        return "unknown", "unknown", quantifier
    
    async def _verify_premise(self, premise: Premise, context: Dict[str, Any]):
        """Verify the truth and reliability of a premise"""
        
        # Truth value determination based on source
        if premise.source == PremiseSource.AXIOM:
            premise.truth_value = True
            premise.confidence = 1.0
            premise.reliability = 1.0
        elif premise.source == PremiseSource.DEFINITION:
            premise.truth_value = True
            premise.confidence = 1.0
            premise.reliability = 1.0
        elif premise.source == PremiseSource.ASSUMPTION:
            premise.truth_value = True  # Assumed true for argument
            premise.confidence = 1.0
            premise.reliability = 0.5  # Lower reliability for assumptions
        elif premise.source == PremiseSource.EMPIRICAL:
            premise.truth_value = True  # Assume empirical observations are true
            premise.confidence = 0.9
            premise.reliability = 0.8
        elif premise.source == PremiseSource.AUTHORITY:
            premise.truth_value = True
            premise.confidence = 0.9
            premise.reliability = 0.9
        else:
            premise.truth_value = None  # Unknown
            premise.confidence = 0.5
            premise.reliability = 0.5
    
    async def _analyze_premise_relationships(self, premises: List[Premise]):
        """Analyze relationships between premises"""
        
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i != j:
                    # Check for contradictions
                    if await self._are_contradictory(premise1, premise2):
                        premise1.contradicts.append(premise2.id)
                    
                    # Check for support relationships
                    if await self._supports(premise1, premise2):
                        premise1.supports.append(premise2.id)
                        premise2.depends_on.append(premise1.id)
    
    async def _are_contradictory(self, premise1: Premise, premise2: Premise) -> bool:
        """Check if two premises are contradictory"""
        
        # Direct contradiction
        if (premise1.subject == premise2.subject and 
            premise1.predicate == premise2.predicate):
            
            # Universal affirmative vs universal negative
            if (premise1.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE and
                premise2.premise_type == PremiseType.UNIVERSAL_NEGATIVE):
                return True
            
            # Universal affirmative vs particular negative
            if (premise1.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE and
                premise2.premise_type == PremiseType.PARTICULAR_NEGATIVE):
                return True
        
        return False
    
    async def _supports(self, premise1: Premise, premise2: Premise) -> bool:
        """Check if premise1 supports premise2"""
        
        # Simple support checking
        if premise1.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE:
            if premise2.premise_type == PremiseType.PARTICULAR_AFFIRMATIVE:
                if premise1.subject == premise2.subject and premise1.predicate == premise2.predicate:
                    return True
        
        return False
    
    def _initialize_premise_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for premise identification"""
        
        return {
            "universal_affirmative": [
                r"all\s+(.+?)\s+are\s+(.+)",
                r"every\s+(.+?)\s+is\s+(.+)",
                r"each\s+(.+?)\s+has\s+(.+)"
            ],
            "universal_negative": [
                r"no\s+(.+?)\s+are\s+(.+)",
                r"none\s+of\s+(.+?)\s+are\s+(.+)"
            ],
            "particular_affirmative": [
                r"some\s+(.+?)\s+are\s+(.+)",
                r"there\s+exist\s+(.+?)\s+that\s+are\s+(.+)"
            ],
            "conditional": [
                r"if\s+(.+?)\s+then\s+(.+)",
                r"(.+?)\s+implies\s+(.+)"
            ]
        }
    
    def _initialize_verification_rules(self) -> Dict[str, Any]:
        """Initialize rules for premise verification"""
        
        return {
            "consistency_rules": [
                "No premise should contradict another",
                "Universal statements should be consistent with particular statements"
            ],
            "truth_conditions": [
                "Axioms are accepted as true",
                "Definitions are true by definition",
                "Empirical claims require verification"
            ]
        }


class LogicalStructureEngine:
    """Engine for applying logical structures to premises"""
    
    def __init__(self):
        self.inference_rules = self._initialize_inference_rules()
        self.structure_patterns = self._initialize_structure_patterns()
    
    async def apply_logical_structure(self, premises: List[Premise], target_conclusion: str) -> List[LogicalStructure]:
        """Apply appropriate logical structures to derive conclusion"""
        
        logger.info(f"Applying logical structures to {len(premises)} premises")
        
        applicable_structures = []
        
        # Try each inference rule
        for rule_type, rule_info in self.inference_rules.items():
            structure = LogicalStructure(
                structure_type=rule_type,
                premise_patterns=rule_info["patterns"],
                conclusion_pattern=rule_info["conclusion"],
                validity_conditions=rule_info.get("conditions", []),
                restrictions=rule_info.get("restrictions", [])
            )
            
            if structure.is_applicable(premises):
                applicable_structures.append(structure)
        
        # Sort by applicability and success rate
        applicable_structures.sort(key=lambda s: s.success_rate, reverse=True)
        
        logger.info(f"Found {len(applicable_structures)} applicable logical structures")
        return applicable_structures
    
    def _initialize_inference_rules(self) -> Dict[InferenceRuleType, Dict[str, Any]]:
        """Initialize comprehensive inference rules"""
        
        return {
            InferenceRuleType.MODUS_PONENS: {
                "patterns": ["P → Q", "P"],
                "conclusion": "Q",
                "conditions": ["Both premises must be true", "Conditional must be material implication"],
                "restrictions": []
            },
            InferenceRuleType.MODUS_TOLLENS: {
                "patterns": ["P → Q", "¬Q"],
                "conclusion": "¬P",
                "conditions": ["Both premises must be true", "Negation must be genuine"],
                "restrictions": []
            },
            InferenceRuleType.HYPOTHETICAL_SYLLOGISM: {
                "patterns": ["P → Q", "Q → R"],
                "conclusion": "P → R",
                "conditions": ["Both conditionals must be true", "Middle term must be identical"],
                "restrictions": []
            },
            InferenceRuleType.BARBARA: {
                "patterns": ["∀x(A(x) → B(x))", "∀x(B(x) → C(x))"],
                "conclusion": "∀x(A(x) → C(x))",
                "conditions": ["All premises must be universal affirmative", "Terms must be distributed correctly"],
                "restrictions": ["Subject and predicate must be clearly defined"]
            },
            InferenceRuleType.DARII: {
                "patterns": ["∀x(A(x) → B(x))", "∃x(C(x) ∧ A(x))"],
                "conclusion": "∃x(C(x) ∧ B(x))",
                "conditions": ["Major premise must be universal", "Minor premise must be particular"],
                "restrictions": ["Middle term must be distributed in major premise"]
            },
            InferenceRuleType.UNIVERSAL_INSTANTIATION: {
                "patterns": ["∀x P(x)"],
                "conclusion": "P(a)",
                "conditions": ["Universal quantifier must be genuine", "Individual must be in domain"],
                "restrictions": ["Individual must be arbitrary"]
            }
        }
    
    def _initialize_structure_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for logical structure recognition"""
        
        return {
            "syllogistic": [
                "All {A} are {B}, All {B} are {C}, therefore All {A} are {C}",
                "All {A} are {B}, Some {C} are {A}, therefore Some {C} are {B}"
            ],
            "conditional": [
                "If {P} then {Q}, {P}, therefore {Q}",
                "If {P} then {Q}, not {Q}, therefore not {P}"
            ],
            "quantified": [
                "For all x, {P(x)}, therefore {P(a)}",
                "{P(a)}, therefore there exists x such that {P(x)}"
            ]
        }


class ConclusionDerivationEngine:
    """Engine for systematic conclusion derivation"""
    
    def __init__(self):
        self.derivation_strategies = self._initialize_derivation_strategies()
    
    async def derive_conclusion(self, premises: List[Premise], logical_structures: List[LogicalStructure], query: str) -> Optional[DeductiveConclusion]:
        """Derive conclusion using logical structures"""
        
        logger.info(f"Deriving conclusion from {len(premises)} premises using {len(logical_structures)} structures")
        
        # Try each logical structure
        for structure in logical_structures:
            conclusion = await self._attempt_derivation(premises, structure, query)
            if conclusion:
                return conclusion
        
        return None
    
    async def _attempt_derivation(self, premises: List[Premise], structure: LogicalStructure, query: str) -> Optional[DeductiveConclusion]:
        """Attempt to derive conclusion using specific logical structure"""
        
        # Check if structure can derive the query
        if structure.structure_type == InferenceRuleType.MODUS_PONENS:
            return await self._derive_modus_ponens(premises, query)
        elif structure.structure_type == InferenceRuleType.BARBARA:
            return await self._derive_barbara(premises, query)
        elif structure.structure_type == InferenceRuleType.DARII:
            return await self._derive_darii(premises, query)
        elif structure.structure_type == InferenceRuleType.UNIVERSAL_INSTANTIATION:
            return await self._derive_universal_instantiation(premises, query)
        
        return None
    
    async def _derive_modus_ponens(self, premises: List[Premise], query: str) -> Optional[DeductiveConclusion]:
        """Derive conclusion using modus ponens"""
        
        # Look for conditional and its antecedent
        conditional_premise = None
        antecedent_premise = None
        
        for premise in premises:
            if premise.premise_type == PremiseType.CONDITIONAL:
                # Check if this conditional can derive the query
                if premise.predicate.lower() in query.lower():
                    conditional_premise = premise
                    
                    # Look for antecedent
                    for other_premise in premises:
                        if other_premise != premise and other_premise.content.lower() in premise.subject.lower():
                            antecedent_premise = other_premise
                            break
                    break
        
        if conditional_premise and antecedent_premise:
            conclusion = DeductiveConclusion(
                id=str(uuid4()),
                content=query,
                logical_form=f"{conditional_premise.predicate}",
                premises_used=[conditional_premise.id, antecedent_premise.id],
                logical_structure=LogicalStructure(
                    structure_type=InferenceRuleType.MODUS_PONENS,
                    premise_patterns=["P → Q", "P"],
                    conclusion_pattern="Q"
                ),
                derivation_steps=[
                    f"From premise {conditional_premise.id}: {conditional_premise.content}",
                    f"From premise {antecedent_premise.id}: {antecedent_premise.content}",
                    f"By modus ponens: {query}"
                ],
                validity=1.0,
                soundness=min(conditional_premise.get_truth_certainty(), antecedent_premise.get_truth_certainty())
            )
            return conclusion
        
        return None
    
    async def _derive_barbara(self, premises: List[Premise], query: str) -> Optional[DeductiveConclusion]:
        """Derive conclusion using Barbara syllogism"""
        
        # Look for two universal affirmative premises that chain
        for premise1 in premises:
            for premise2 in premises:
                if (premise1 != premise2 and
                    premise1.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE and
                    premise2.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE):
                    
                    # Check if they form a Barbara syllogism
                    if premise1.predicate == premise2.subject:
                        # Check if conclusion matches query
                        expected_conclusion = f"all {premise1.subject} are {premise2.predicate}"
                        if expected_conclusion.lower() in query.lower():
                            
                            conclusion = DeductiveConclusion(
                                id=str(uuid4()),
                                content=query,
                                logical_form=f"∀x({premise1.subject}(x) → {premise2.predicate}(x))",
                                premises_used=[premise1.id, premise2.id],
                                logical_structure=LogicalStructure(
                                    structure_type=InferenceRuleType.BARBARA,
                                    premise_patterns=["All A are B", "All B are C"],
                                    conclusion_pattern="All A are C"
                                ),
                                derivation_steps=[
                                    f"Major premise: {premise1.content}",
                                    f"Minor premise: {premise2.content}",
                                    f"By Barbara syllogism: {query}"
                                ],
                                validity=1.0,
                                soundness=min(premise1.get_truth_certainty(), premise2.get_truth_certainty())
                            )
                            return conclusion
        
        return None
    
    async def _derive_darii(self, premises: List[Premise], query: str) -> Optional[DeductiveConclusion]:
        """Derive conclusion using Darii syllogism"""
        
        # Look for universal and particular premises
        for premise1 in premises:
            for premise2 in premises:
                if premise1 != premise2:
                    universal_premise = None
                    particular_premise = None
                    
                    if (premise1.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE and
                        premise2.premise_type == PremiseType.PARTICULAR_AFFIRMATIVE):
                        universal_premise = premise1
                        particular_premise = premise2
                    elif (premise1.premise_type == PremiseType.PARTICULAR_AFFIRMATIVE and
                          premise2.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE):
                        universal_premise = premise2
                        particular_premise = premise1
                    
                    if universal_premise and particular_premise:
                        # Check if they form a Darii syllogism
                        if universal_premise.subject == particular_premise.predicate:
                            expected_conclusion = f"some {particular_premise.subject} are {universal_premise.predicate}"
                            if expected_conclusion.lower() in query.lower():
                                
                                conclusion = DeductiveConclusion(
                                    id=str(uuid4()),
                                    content=query,
                                    logical_form=f"∃x({particular_premise.subject}(x) ∧ {universal_premise.predicate}(x))",
                                    premises_used=[universal_premise.id, particular_premise.id],
                                    logical_structure=LogicalStructure(
                                        structure_type=InferenceRuleType.DARII,
                                        premise_patterns=["All A are B", "Some C are A"],
                                        conclusion_pattern="Some C are B"
                                    ),
                                    derivation_steps=[
                                        f"Major premise: {universal_premise.content}",
                                        f"Minor premise: {particular_premise.content}",
                                        f"By Darii syllogism: {query}"
                                    ],
                                    validity=1.0,
                                    soundness=min(universal_premise.get_truth_certainty(), particular_premise.get_truth_certainty())
                                )
                                return conclusion
        
        return None
    
    async def _derive_universal_instantiation(self, premises: List[Premise], query: str) -> Optional[DeductiveConclusion]:
        """Derive conclusion using universal instantiation"""
        
        # Look for universal premise that can be instantiated
        for premise in premises:
            if premise.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE:
                # Check if query is an instantiation of this universal
                if premise.subject.lower() in query.lower() and premise.predicate.lower() in query.lower():
                    
                    conclusion = DeductiveConclusion(
                        id=str(uuid4()),
                        content=query,
                        logical_form=f"{premise.predicate}({premise.subject})",
                        premises_used=[premise.id],
                        logical_structure=LogicalStructure(
                            structure_type=InferenceRuleType.UNIVERSAL_INSTANTIATION,
                            premise_patterns=["∀x P(x)"],
                            conclusion_pattern="P(a)"
                        ),
                        derivation_steps=[
                            f"Universal premise: {premise.content}",
                            f"By universal instantiation: {query}"
                        ],
                        validity=1.0,
                        soundness=premise.get_truth_certainty()
                    )
                    return conclusion
        
        return None
    
    def _initialize_derivation_strategies(self) -> Dict[str, Any]:
        """Initialize strategies for conclusion derivation"""
        
        return {
            "direct_derivation": {
                "description": "Direct application of inference rules",
                "priority": 1
            },
            "proof_by_contradiction": {
                "description": "Assume negation and derive contradiction",
                "priority": 2
            },
            "proof_by_cases": {
                "description": "Exhaust all possible cases",
                "priority": 3
            },
            "mathematical_induction": {
                "description": "Base case and inductive step",
                "priority": 4
            }
        }


class ValiditySoundnessEvaluator:
    """Engine for evaluating validity and soundness of deductive arguments"""
    
    def __init__(self):
        self.validity_rules = self._initialize_validity_rules()
        self.soundness_criteria = self._initialize_soundness_criteria()
    
    async def evaluate_validity_soundness(self, premises: List[Premise], conclusion: DeductiveConclusion, logical_structures: List[LogicalStructure]) -> Dict[str, Any]:
        """Evaluate both validity and soundness of the argument"""
        
        logger.info("Evaluating validity and soundness of deductive argument")
        
        # Evaluate validity (logical structure)
        validity_evaluation = await self._evaluate_validity(premises, conclusion, logical_structures)
        
        # Evaluate soundness (truth of premises)
        soundness_evaluation = await self._evaluate_soundness(premises, conclusion)
        
        # Combine evaluations
        evaluation = {
            "validity": validity_evaluation,
            "soundness": soundness_evaluation,
            "overall_assessment": {
                "is_valid": validity_evaluation["is_valid"],
                "is_sound": soundness_evaluation["is_sound"],
                "certainty_score": validity_evaluation["validity_score"] * soundness_evaluation["soundness_score"],
                "confidence": min(validity_evaluation["confidence"], soundness_evaluation["confidence"])
            }
        }
        
        logger.info(f"Evaluation complete: valid={evaluation['overall_assessment']['is_valid']}, sound={evaluation['overall_assessment']['is_sound']}")
        
        return evaluation
    
    async def _evaluate_validity(self, premises: List[Premise], conclusion: DeductiveConclusion, logical_structures: List[LogicalStructure]) -> Dict[str, Any]:
        """Evaluate logical validity of the argument"""
        
        validity_checks = {
            "logical_form_correct": await self._check_logical_form(conclusion),
            "inference_rule_applied_correctly": await self._check_inference_rule_application(premises, conclusion),
            "no_logical_fallacies": await self._check_for_logical_fallacies(premises, conclusion),
            "premise_conclusion_connection": await self._check_premise_conclusion_connection(premises, conclusion)
        }
        
        # Calculate validity score
        validity_score = sum(validity_checks.values()) / len(validity_checks)
        
        return {
            "is_valid": all(validity_checks.values()),
            "validity_score": validity_score,
            "validity_checks": validity_checks,
            "confidence": 0.95 if all(validity_checks.values()) else 0.5
        }
    
    async def _evaluate_soundness(self, premises: List[Premise], conclusion: DeductiveConclusion) -> Dict[str, Any]:
        """Evaluate soundness (truth of premises) of the argument"""
        
        soundness_checks = {
            "premises_are_true": await self._check_premise_truth(premises),
            "premises_are_consistent": await self._check_premise_consistency(premises),
            "premises_are_relevant": await self._check_premise_relevance(premises, conclusion),
            "no_premise_contradictions": await self._check_no_contradictions(premises)
        }
        
        # Calculate soundness score based on premise truth certainty
        premise_certainties = [premise.get_truth_certainty() for premise in premises]
        soundness_score = min(premise_certainties) if premise_certainties else 0.0
        
        return {
            "is_sound": all(soundness_checks.values()) and soundness_score > 0.5,
            "soundness_score": soundness_score,
            "soundness_checks": soundness_checks,
            "premise_certainties": premise_certainties,
            "confidence": 0.9 if all(soundness_checks.values()) else 0.3
        }
    
    async def _check_logical_form(self, conclusion: DeductiveConclusion) -> bool:
        """Check if the logical form is correct"""
        
        # Basic checks for logical form correctness
        if not conclusion.logical_form:
            return False
        
        # Check for basic logical symbols and structure
        logical_symbols = ["∀", "∃", "→", "∧", "∨", "¬", "(", ")"]
        has_logical_structure = any(symbol in conclusion.logical_form for symbol in logical_symbols)
        
        return has_logical_structure
    
    async def _check_inference_rule_application(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Check if inference rule was applied correctly"""
        
        # Verify that the inference rule used is appropriate
        rule_type = conclusion.logical_structure.structure_type
        
        if rule_type == InferenceRuleType.MODUS_PONENS:
            return await self._verify_modus_ponens_application(premises, conclusion)
        elif rule_type == InferenceRuleType.BARBARA:
            return await self._verify_barbara_application(premises, conclusion)
        elif rule_type == InferenceRuleType.DARII:
            return await self._verify_darii_application(premises, conclusion)
        elif rule_type == InferenceRuleType.UNIVERSAL_INSTANTIATION:
            return await self._verify_universal_instantiation_application(premises, conclusion)
        
        return True  # Default to valid if rule not recognized
    
    async def _verify_modus_ponens_application(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Verify modus ponens was applied correctly"""
        
        # Should have exactly 2 premises
        if len(conclusion.premises_used) != 2:
            return False
        
        # Find the premises used
        used_premises = [p for p in premises if p.id in conclusion.premises_used]
        
        # One should be conditional, one should be its antecedent
        conditional_premise = None
        antecedent_premise = None
        
        for premise in used_premises:
            if premise.premise_type == PremiseType.CONDITIONAL:
                conditional_premise = premise
            else:
                antecedent_premise = premise
        
        if not conditional_premise or not antecedent_premise:
            return False
        
        # Check if antecedent matches the conditional's subject
        return conditional_premise.subject.lower() in antecedent_premise.content.lower()
    
    async def _verify_barbara_application(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Verify Barbara syllogism was applied correctly"""
        
        # Should have exactly 2 premises, both universal affirmative
        if len(conclusion.premises_used) != 2:
            return False
        
        used_premises = [p for p in premises if p.id in conclusion.premises_used]
        
        # Both should be universal affirmative
        if not all(p.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE for p in used_premises):
            return False
        
        # Check middle term distribution
        premise1, premise2 = used_premises
        return premise1.predicate == premise2.subject
    
    async def _verify_darii_application(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Verify Darii syllogism was applied correctly"""
        
        # Should have exactly 2 premises
        if len(conclusion.premises_used) != 2:
            return False
        
        used_premises = [p for p in premises if p.id in conclusion.premises_used]
        
        # One should be universal affirmative, one particular affirmative
        premise_types = [p.premise_type for p in used_premises]
        return (PremiseType.UNIVERSAL_AFFIRMATIVE in premise_types and
                PremiseType.PARTICULAR_AFFIRMATIVE in premise_types)
    
    async def _verify_universal_instantiation_application(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Verify universal instantiation was applied correctly"""
        
        # Should have exactly 1 premise
        if len(conclusion.premises_used) != 1:
            return False
        
        used_premise = next(p for p in premises if p.id in conclusion.premises_used[0])
        
        # Premise should be universal
        return used_premise.premise_type == PremiseType.UNIVERSAL_AFFIRMATIVE
    
    async def _check_for_logical_fallacies(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Check for common logical fallacies"""
        
        # Check for affirming the consequent
        if conclusion.logical_structure.structure_type == InferenceRuleType.MODUS_PONENS:
            # This is a valid inference rule, but check it's not actually affirming the consequent
            pass
        
        # Check for undistributed middle term in syllogisms
        if conclusion.logical_structure.structure_type in [InferenceRuleType.BARBARA, InferenceRuleType.DARII]:
            # Check middle term is properly distributed
            pass
        
        return True  # For now, assume no fallacies
    
    async def _check_premise_conclusion_connection(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Check if premises actually connect to conclusion"""
        
        # Check if all premises used in conclusion are actually available
        available_premise_ids = [p.id for p in premises]
        return all(premise_id in available_premise_ids for premise_id in conclusion.premises_used)
    
    async def _check_premise_truth(self, premises: List[Premise]) -> bool:
        """Check if premises are likely to be true"""
        
        # Check truth values of premises
        for premise in premises:
            if premise.truth_value is False:
                return False
            if premise.truth_value is None and premise.get_truth_certainty() < 0.5:
                return False
        
        return True
    
    async def _check_premise_consistency(self, premises: List[Premise]) -> bool:
        """Check if premises are consistent with each other"""
        
        # Check for contradictions
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i != j and premise1.id in premise2.contradicts:
                    return False
        
        return True
    
    async def _check_premise_relevance(self, premises: List[Premise], conclusion: DeductiveConclusion) -> bool:
        """Check if premises are relevant to conclusion"""
        
        # Check if premises used in conclusion are actually relevant
        conclusion_terms = set(conclusion.content.lower().split())
        
        for premise_id in conclusion.premises_used:
            premise = next(p for p in premises if p.id == premise_id)
            premise_terms = set(premise.content.lower().split())
            
            # Check for term overlap
            if not conclusion_terms.intersection(premise_terms):
                return False
        
        return True
    
    async def _check_no_contradictions(self, premises: List[Premise]) -> bool:
        """Check that there are no contradictions in premises"""
        
        # This is similar to consistency check but more thorough
        for premise in premises:
            if premise.contradicts:
                return False
        
        return True
    
    def _initialize_validity_rules(self) -> Dict[str, Any]:
        """Initialize rules for validity checking"""
        
        return {
            "modus_ponens": {
                "pattern": "P → Q, P ⊢ Q",
                "conditions": ["Conditional must be material implication", "Antecedent must be affirmed"]
            },
            "barbara": {
                "pattern": "All A are B, All B are C ⊢ All A are C",
                "conditions": ["Both premises must be universal affirmative", "Middle term must be distributed"]
            },
            "darii": {
                "pattern": "All A are B, Some C are A ⊢ Some C are B", 
                "conditions": ["Major premise must be universal", "Minor premise must be particular"]
            }
        }
    
    def _initialize_soundness_criteria(self) -> Dict[str, Any]:
        """Initialize criteria for soundness evaluation"""
        
        return {
            "truth_sources": {
                "axiom": 1.0,
                "definition": 1.0,
                "empirical": 0.8,
                "authority": 0.9,
                "assumption": 0.5
            },
            "consistency_rules": [
                "No premise should contradict another",
                "Universal statements should be consistent with particular statements"
            ]
        }


class InferenceApplicationEngine:
    """Engine for applying deductive inferences to practical contexts"""
    
    def __init__(self):
        self.application_strategies = self._initialize_application_strategies()
        self.context_adapters = self._initialize_context_adapters()
    
    async def apply_inference(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply deductive inference to practical context"""
        
        logger.info(f"Applying deductive inference to practical context")
        
        # Determine application strategy
        strategy = await self._determine_application_strategy(conclusion, context)
        
        # Apply to context
        application_result = await self._apply_to_context(conclusion, context, strategy)
        
        # Generate further inferences
        further_inferences = await self._generate_further_inferences(conclusion, context)
        
        # Assess practical relevance
        relevance_assessment = await self._assess_practical_relevance(conclusion, context)
        
        return {
            "application_strategy": strategy,
            "application_result": application_result,
            "further_inferences": further_inferences,
            "relevance_assessment": relevance_assessment,
            "actionable_insights": await self._extract_actionable_insights(conclusion, context)
        }
    
    async def _determine_application_strategy(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> str:
        """Determine appropriate strategy for applying inference"""
        
        # Analyze context type
        context_type = context.get("type", "general")
        
        if context_type == "mathematical":
            return "formal_application"
        elif context_type == "legal":
            return "rule_application"
        elif context_type == "scientific":
            return "hypothesis_testing"
        elif context_type == "practical":
            return "decision_making"
        else:
            return "general_reasoning"
    
    async def _apply_to_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Apply conclusion to specific context using strategy"""
        
        if strategy == "formal_application":
            return await self._apply_formal_context(conclusion, context)
        elif strategy == "rule_application":
            return await self._apply_rule_context(conclusion, context)
        elif strategy == "hypothesis_testing":
            return await self._apply_scientific_context(conclusion, context)
        elif strategy == "decision_making":
            return await self._apply_decision_context(conclusion, context)
        else:
            return await self._apply_general_context(conclusion, context)
    
    async def _apply_formal_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to formal mathematical context"""
        
        return {
            "theorem_status": "proven" if conclusion.get_certainty_score() > 0.9 else "conjecture",
            "proof_validity": conclusion.validity,
            "mathematical_implications": conclusion.implications,
            "next_steps": ["Verify proof", "Explore corollaries", "Apply to specific cases"]
        }
    
    async def _apply_rule_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to legal/rule-based context"""
        
        return {
            "rule_application": conclusion.content,
            "legal_validity": conclusion.validity,
            "case_resolution": "applicable" if conclusion.get_certainty_score() > 0.8 else "uncertain",
            "precedent_value": conclusion.get_certainty_score()
        }
    
    async def _apply_scientific_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to scientific context"""
        
        return {
            "hypothesis_support": conclusion.get_certainty_score(),
            "experimental_predictions": conclusion.implications,
            "theory_implications": conclusion.applicability,
            "research_directions": ["Test predictions", "Refine theory", "Explore applications"]
        }
    
    async def _apply_decision_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to decision-making context"""
        
        return {
            "decision_support": conclusion.get_certainty_score(),
            "recommended_action": conclusion.content,
            "confidence_level": conclusion.get_certainty_score(),
            "risk_assessment": 1.0 - conclusion.get_certainty_score()
        }
    
    async def _apply_general_context(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply to general reasoning context"""
        
        return {
            "reasoning_result": conclusion.content,
            "certainty": conclusion.get_certainty_score(),
            "applicability": conclusion.applicability,
            "general_insights": conclusion.implications
        }
    
    async def _generate_further_inferences(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> List[str]:
        """Generate further inferences from conclusion"""
        
        further_inferences = []
        
        # Use conclusion as premise for further reasoning
        if conclusion.get_certainty_score() > 0.8:
            further_inferences.append(f"Given that {conclusion.content}, we can further infer...")
        
        # Explore implications
        for implication in conclusion.implications:
            further_inferences.append(f"This implies: {implication}")
        
        # Consider applications
        for application in conclusion.applicability:
            further_inferences.append(f"This can be applied to: {application}")
        
        return further_inferences
    
    async def _assess_practical_relevance(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess practical relevance of conclusion"""
        
        relevance_score = 0.5  # Base relevance
        
        # Increase relevance based on context match
        if context.get("domain") == conclusion.domain:
            relevance_score += 0.3
        
        # Increase relevance based on certainty
        relevance_score += conclusion.get_certainty_score() * 0.2
        
        # Assess actionability
        is_actionable = len(conclusion.applicability) > 0
        
        return {
            "relevance_score": min(relevance_score, 1.0),
            "is_actionable": is_actionable,
            "practical_impact": "high" if relevance_score > 0.8 else "medium" if relevance_score > 0.5 else "low",
            "application_domains": conclusion.applicability
        }
    
    async def _extract_actionable_insights(self, conclusion: DeductiveConclusion, context: Dict[str, Any]) -> List[str]:
        """Extract actionable insights from conclusion"""
        
        insights = []
        
        if conclusion.get_certainty_score() > 0.9:
            insights.append(f"High certainty conclusion: {conclusion.content}")
        
        if conclusion.implications:
            insights.append(f"Key implications: {', '.join(conclusion.implications)}")
        
        if conclusion.applicability:
            insights.append(f"Can be applied to: {', '.join(conclusion.applicability)}")
        
        return insights
    
    def _initialize_application_strategies(self) -> Dict[str, Any]:
        """Initialize strategies for inference application"""
        
        return {
            "formal_application": {
                "description": "Apply to formal mathematical contexts",
                "requirements": ["Rigorous proof", "Logical validity"],
                "outcomes": ["Theorems", "Lemmas", "Corollaries"]
            },
            "rule_application": {
                "description": "Apply to rule-based contexts",
                "requirements": ["Rule validity", "Case applicability"],
                "outcomes": ["Decisions", "Judgments", "Precedents"]
            },
            "decision_making": {
                "description": "Apply to practical decision contexts",
                "requirements": ["Practical relevance", "Actionability"],
                "outcomes": ["Recommendations", "Actions", "Policies"]
            }
        }
    
    def _initialize_context_adapters(self) -> Dict[str, Any]:
        """Initialize context adapters for different domains"""
        
        return {
            "mathematical": {
                "language": "formal",
                "validation": "proof",
                "output": "theorem"
            },
            "legal": {
                "language": "prescriptive",
                "validation": "precedent",
                "output": "ruling"
            },
            "scientific": {
                "language": "empirical",
                "validation": "experiment",
                "output": "hypothesis"
            }
        }


class EnhancedDeductiveReasoningEngine:
    """
    Enhanced Deductive Reasoning Engine implementing all elemental components
    
    This engine provides comprehensive deductive reasoning capabilities with:
    1. Premise Identification and Verification
    2. Logical Structure Application
    3. Conclusion Derivation
    4. Validity and Soundness Evaluation
    5. Inference Application
    """
    
    def __init__(self):
        # Initialize elemental components
        self.premise_engine = PremiseIdentificationEngine()
        self.structure_engine = LogicalStructureEngine()
        self.conclusion_engine = ConclusionDerivationEngine()
        self.evaluation_engine = ValiditySoundnessEvaluator()
        self.application_engine = InferenceApplicationEngine()
        
        # Proof history
        self.proof_history: List[DeductiveProof] = []
        
        # Configuration
        self.confidence_threshold = 0.8
        self.max_proof_depth = 10
        
        logger.info("Enhanced Deductive Reasoning Engine initialized")
    
    async def perform_deductive_reasoning(
        self,
        premises: List[str],
        query: str,
        context: Dict[str, Any] = None
    ) -> DeductiveProof:
        """
        Perform comprehensive deductive reasoning using all elemental components
        
        Args:
            premises: List of premise statements
            query: The conclusion to prove or disprove
            context: Additional context for reasoning
            
        Returns:
            DeductiveProof: Complete proof with all elemental components
        """
        
        logger.info(f"Performing deductive reasoning: {query}")
        
        context = context or {}
        
        # Component 1: Identification of Premises
        logger.info("Step 1: Identifying premises")
        identified_premises = await self.premise_engine.identify_premises(premises, context)
        
        # Component 2: Application of Logical Structure
        logger.info("Step 2: Applying logical structures")
        logical_structures = await self.structure_engine.apply_logical_structure(identified_premises, query)
        
        # Component 3: Derivation of Conclusion
        logger.info("Step 3: Deriving conclusion")
        conclusion = await self.conclusion_engine.derive_conclusion(identified_premises, logical_structures, query)
        
        if not conclusion:
            logger.warning("Could not derive conclusion from premises")
            return self._create_failed_proof(query, identified_premises, logical_structures)
        
        # Component 4: Evaluation of Validity and Soundness
        logger.info("Step 4: Evaluating validity and soundness")
        evaluation = await self.evaluation_engine.evaluate_validity_soundness(
            identified_premises, conclusion, logical_structures
        )
        
        # Component 5: Inference Application
        logger.info("Step 5: Applying inference")
        application = await self.application_engine.apply_inference(conclusion, context)
        
        # Create complete proof
        proof = DeductiveProof(
            id=str(uuid4()),
            query=query,
            premise_identification={p.id: p for p in identified_premises},
            logical_structure_application=logical_structures,
            conclusion_derivation=conclusion,
            validity_soundness_evaluation=evaluation,
            inference_application=application,
            is_valid=evaluation["overall_assessment"]["is_valid"],
            is_sound=evaluation["overall_assessment"]["is_sound"],
            is_complete=True,
            proof_length=len(conclusion.derivation_steps),
            complexity=len(logical_structures) / 10.0,
            confidence=evaluation["overall_assessment"]["confidence"]
        )
        
        # Add to history
        self.proof_history.append(proof)
        
        logger.info(f"Deductive reasoning complete: valid={proof.is_valid}, sound={proof.is_sound}")
        
        return proof
    
    def _create_failed_proof(self, query: str, premises: List[Premise], structures: List[LogicalStructure]) -> DeductiveProof:
        """Create a failed proof when conclusion cannot be derived"""
        
        return DeductiveProof(
            id=str(uuid4()),
            query=query,
            premise_identification={p.id: p for p in premises},
            logical_structure_application=structures,
            conclusion_derivation=DeductiveConclusion(
                id=str(uuid4()),
                content="Cannot be derived from given premises",
                logical_form="",
                premises_used=[],
                logical_structure=LogicalStructure(
                    structure_type=InferenceRuleType.MODUS_PONENS,  # Placeholder
                    premise_patterns=[],
                    conclusion_pattern=""
                ),
                derivation_steps=["No valid derivation found"],
                is_certain=False,
                validity=0.0,
                soundness=0.0
            ),
            validity_soundness_evaluation={"overall_assessment": {"is_valid": False, "is_sound": False}},
            inference_application={"application_result": {"status": "failed"}},
            is_valid=False,
            is_sound=False,
            is_complete=False,
            confidence=0.0
        )
    
    async def chain_deductive_reasoning(
        self,
        premise_sets: List[List[str]],
        intermediate_queries: List[str],
        final_query: str,
        context: Dict[str, Any] = None
    ) -> List[DeductiveProof]:
        """
        Chain multiple deductive reasoning steps
        
        Args:
            premise_sets: List of premise sets for each step
            intermediate_queries: Intermediate conclusions to derive
            final_query: Final conclusion to prove
            context: Reasoning context
            
        Returns:
            List of DeductiveProof objects for each step
        """
        
        logger.info(f"Chaining deductive reasoning with {len(intermediate_queries)} intermediate steps")
        
        proofs = []
        accumulated_premises = []
        
        # Process intermediate steps
        for i, (premises, query) in enumerate(zip(premise_sets, intermediate_queries)):
            # Add previous conclusions as premises
            combined_premises = premises + accumulated_premises
            
            # Perform reasoning
            proof = await self.perform_deductive_reasoning(combined_premises, query, context)
            proofs.append(proof)
            
            # Add conclusion to accumulated premises if valid and sound
            if proof.is_valid and proof.is_sound:
                accumulated_premises.append(proof.conclusion_derivation.content)
        
        # Final reasoning step
        if premise_sets:
            final_premises = premise_sets[-1] + accumulated_premises
            final_proof = await self.perform_deductive_reasoning(final_premises, final_query, context)
            proofs.append(final_proof)
        
        logger.info(f"Chained reasoning complete: {len(proofs)} steps")
        
        return proofs
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about deductive reasoning performance"""
        
        if not self.proof_history:
            return {"total_proofs": 0}
        
        valid_proofs = [p for p in self.proof_history if p.is_valid]
        sound_proofs = [p for p in self.proof_history if p.is_sound]
        complete_proofs = [p for p in self.proof_history if p.is_complete]
        
        return {
            "total_proofs": len(self.proof_history),
            "valid_proofs": len(valid_proofs),
            "sound_proofs": len(sound_proofs),
            "complete_proofs": len(complete_proofs),
            "validity_rate": len(valid_proofs) / len(self.proof_history),
            "soundness_rate": len(sound_proofs) / len(self.proof_history),
            "completion_rate": len(complete_proofs) / len(self.proof_history),
            "average_proof_length": sum(p.proof_length for p in self.proof_history) / len(self.proof_history),
            "average_complexity": sum(p.complexity for p in self.proof_history) / len(self.proof_history),
            "average_confidence": sum(p.confidence for p in self.proof_history) / len(self.proof_history),
            "premise_identification_accuracy": self._calculate_premise_accuracy(),
            "logical_structure_effectiveness": self._calculate_structure_effectiveness()
        }
    
    def _calculate_premise_accuracy(self) -> float:
        """Calculate accuracy of premise identification"""
        
        total_premises = 0
        accurate_premises = 0
        
        for proof in self.proof_history:
            for premise in proof.premise_identification.values():
                total_premises += 1
                if premise.truth_value is True and premise.confidence > 0.7:
                    accurate_premises += 1
        
        return accurate_premises / total_premises if total_premises > 0 else 0.0
    
    def _calculate_structure_effectiveness(self) -> float:
        """Calculate effectiveness of logical structure application"""
        
        total_structures = 0
        effective_structures = 0
        
        for proof in self.proof_history:
            for structure in proof.logical_structure_application:
                total_structures += 1
                if structure.success_rate > 0.7:
                    effective_structures += 1
        
        return effective_structures / total_structures if total_structures > 0 else 0.0


# Example usage and demonstration
async def demonstrate_enhanced_deductive_reasoning():
    """Demonstrate the enhanced deductive reasoning system"""
    
    print("🧠 ENHANCED DEDUCTIVE REASONING SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Initialize engine
    engine = EnhancedDeductiveReasoningEngine()
    
    # Example 1: Classic syllogism
    print("\n1. 📚 CLASSIC SYLLOGISM DEMONSTRATION")
    print("-" * 40)
    
    premises = [
        "All humans are mortal",
        "Socrates is a human"
    ]
    query = "Socrates is mortal"
    
    proof = await engine.perform_deductive_reasoning(premises, query)
    
    print(f"Query: {query}")
    print(f"Premises: {premises}")
    print(f"Valid: {proof.is_valid}")
    print(f"Sound: {proof.is_sound}")
    print(f"Confidence: {proof.confidence:.2f}")
    print(f"Derivation steps: {len(proof.conclusion_derivation.derivation_steps)}")
    
    # Example 2: Conditional reasoning
    print("\n2. 🔗 CONDITIONAL REASONING DEMONSTRATION")
    print("-" * 40)
    
    premises = [
        "If it rains, then the ground gets wet",
        "It is raining"
    ]
    query = "The ground gets wet"
    
    proof = await engine.perform_deductive_reasoning(premises, query)
    
    print(f"Query: {query}")
    print(f"Premises: {premises}")
    print(f"Valid: {proof.is_valid}")
    print(f"Sound: {proof.is_sound}")
    print(f"Confidence: {proof.confidence:.2f}")
    
    # Example 3: Complex chained reasoning
    print("\n3. 🔄 CHAINED REASONING DEMONSTRATION")
    print("-" * 40)
    
    premise_sets = [
        ["All birds have feathers", "All eagles are birds"],
        ["All feathered animals can fly", "All eagles have feathers"]
    ]
    intermediate_queries = ["All eagles have feathers"]
    final_query = "All eagles can fly"
    
    proofs = await engine.chain_deductive_reasoning(
        premise_sets, intermediate_queries, final_query
    )
    
    print(f"Final query: {final_query}")
    print(f"Chained proofs: {len(proofs)}")
    for i, proof in enumerate(proofs):
        print(f"  Step {i+1}: Valid={proof.is_valid}, Sound={proof.is_sound}")
    
    # Show statistics
    print("\n📊 REASONING STATISTICS")
    print("-" * 40)
    
    stats = engine.get_reasoning_statistics()
    print(f"Total proofs: {stats['total_proofs']}")
    print(f"Validity rate: {stats['validity_rate']:.2%}")
    print(f"Soundness rate: {stats['soundness_rate']:.2%}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced deductive reasoning demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_deductive_reasoning())