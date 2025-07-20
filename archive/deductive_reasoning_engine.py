#!/usr/bin/env python3
"""
NWTN Deductive Reasoning Engine
Formal logic and rule-based inference for certain conclusions

This module implements NWTN's deductive reasoning capabilities, which allow the system to:
1. Apply formal logical rules (modus ponens, modus tollens, etc.)
2. Process syllogisms and logical structures
3. Derive certain conclusions from valid premises
4. Validate logical consistency and soundness

Deductive reasoning operates from general principles to specific conclusions, providing
logically certain results when premises are true and reasoning is valid.

Key Concepts:
- Syllogistic reasoning (All A are B, X is A, therefore X is B)
- Propositional logic (If P then Q, P, therefore Q)
- Predicate logic (For all x, P(x) implies Q(x))
- Logical validation and consistency checking
- Rule-based inference systems

Usage:
    from prsm.nwtn.deductive_reasoning_engine import DeductiveReasoningEngine
    
    engine = DeductiveReasoningEngine()
    result = await engine.deduce_conclusion(premises, query)
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timezone
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field

from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel
from prsm.nwtn.world_model_engine import WorldModelEngine
from prsm.agents.executors.model_executor import ModelExecutor

logger = structlog.get_logger(__name__)


class LogicalOperator(str, Enum):
    """Logical operators for propositional logic"""
    AND = "and"           # ∧
    OR = "or"             # ∨
    NOT = "not"           # ¬
    IMPLIES = "implies"   # →
    IFF = "iff"           # ↔
    EXISTS = "exists"     # ∃
    FORALL = "forall"     # ∀


class LogicalRuleType(str, Enum):
    """Types of logical inference rules"""
    MODUS_PONENS = "modus_ponens"           # P → Q, P ⊢ Q
    MODUS_TOLLENS = "modus_tollens"         # P → Q, ¬Q ⊢ ¬P
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"  # P → Q, Q → R ⊢ P → R
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"   # P ∨ Q, ¬P ⊢ Q
    ADDITION = "addition"                   # P ⊢ P ∨ Q
    SIMPLIFICATION = "simplification"       # P ∧ Q ⊢ P
    CONJUNCTION = "conjunction"             # P, Q ⊢ P ∧ Q
    RESOLUTION = "resolution"               # ¬P ∨ Q, P ∨ R ⊢ Q ∨ R
    UNIVERSAL_INSTANTIATION = "universal_instantiation"  # ∀x P(x) ⊢ P(a)
    EXISTENTIAL_GENERALIZATION = "existential_generalization"  # P(a) ⊢ ∃x P(x)


class SyllogismType(str, Enum):
    """Types of categorical syllogisms"""
    BARBARA = "barbara"     # All A are B, All B are C, therefore All A are C
    CELARENT = "celarent"   # No A are B, All C are A, therefore No C are B
    DARII = "darii"         # All A are B, Some C are A, therefore Some C are B
    FERIO = "ferio"         # No A are B, Some C are A, therefore Some C are not B


@dataclass
class LogicalProposition:
    """A logical proposition with subject, predicate, and quantifier"""
    
    id: str
    subject: str
    predicate: str
    quantifier: str  # "all", "some", "no", "some_not"
    negated: bool = False
    
    # Logical form
    propositional_form: str = ""  # P, Q, R, etc.
    predicate_form: str = ""      # P(x), Q(x), etc.
    
    # Validation
    is_valid: bool = True
    confidence: float = 1.0
    
    def __str__(self):
        neg = "not " if self.negated else ""
        return f"{neg}{self.quantifier} {self.subject} are {self.predicate}"


@dataclass
class LogicalRule:
    """A logical inference rule"""
    
    rule_type: LogicalRuleType
    premises: List[str]
    conclusion: str
    
    # Rule properties
    is_sound: bool = True
    is_valid: bool = True
    confidence: float = 1.0
    
    # Application tracking
    applications: int = 0
    success_rate: float = 1.0
    
    def apply(self, premises: List[LogicalProposition]) -> Optional[LogicalProposition]:
        """Apply this rule to given premises"""
        # Will be implemented in specific rule methods
        pass


@dataclass
class DeductiveProof:
    """A deductive proof with steps and justifications"""
    
    id: str
    query: str
    premises: List[LogicalProposition]
    
    # Proof steps
    proof_steps: List[Dict[str, Any]] = field(default_factory=list)
    conclusion: Optional[LogicalProposition] = None
    
    # Proof properties
    is_valid: bool = False
    is_sound: bool = False
    is_complete: bool = False
    
    # Metrics
    proof_length: int = 0
    complexity: float = 0.0
    confidence: float = 1.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DeductiveReasoningEngine:
    """
    Engine for deductive reasoning using formal logic and rule-based inference
    
    This system enables NWTN to perform rigorous logical deduction,
    ensuring certain conclusions when premises are valid.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="deductive_reasoning_engine")
        self.world_model = WorldModelEngine()
        
        # Logical rules database
        self.logical_rules = self._initialize_logical_rules()
        
        # Proof history
        self.proof_history: List[DeductiveProof] = []
        
        # Configuration
        self.max_proof_steps = 20
        self.confidence_threshold = 0.9
        
        logger.info("Initialized Deductive Reasoning Engine")
    
    async def deduce_conclusion(
        self, 
        premises: List[str], 
        query: str, 
        context: Dict[str, Any] = None
    ) -> DeductiveProof:
        """
        Perform deductive reasoning to derive conclusions from premises
        
        Args:
            premises: List of premise statements
            query: The conclusion to prove or disprove
            context: Additional context for reasoning
            
        Returns:
            DeductiveProof: Complete proof with steps and validation
        """
        
        logger.info(
            "Starting deductive reasoning",
            premises_count=len(premises),
            query=query
        )
        
        # Step 1: Parse premises into logical propositions
        logical_premises = await self._parse_premises(premises)
        
        # Step 2: Parse query into logical form
        query_proposition = await self._parse_query(query)
        
        # Step 3: Attempt to construct proof
        proof = await self._construct_proof(logical_premises, query_proposition, query)
        
        # Step 4: Validate proof
        validated_proof = await self._validate_proof(proof)
        
        # Step 5: Add to history
        self.proof_history.append(validated_proof)
        
        logger.info(
            "Deductive reasoning complete",
            proof_valid=validated_proof.is_valid,
            proof_sound=validated_proof.is_sound,
            steps=validated_proof.proof_length
        )
        
        return validated_proof
    
    async def _parse_premises(self, premises: List[str]) -> List[LogicalProposition]:
        """Parse natural language premises into logical propositions"""
        
        parsed_premises = []
        
        for i, premise in enumerate(premises):
            # Extract logical structure
            logical_prop = await self._parse_logical_proposition(premise, f"P{i+1}")
            parsed_premises.append(logical_prop)
        
        return parsed_premises
    
    async def _parse_logical_proposition(self, statement: str, prop_id: str) -> LogicalProposition:
        """Parse a single statement into a logical proposition"""
        
        statement = str(statement).strip().lower()
        
        # Identify quantifier
        quantifier = "all"  # Default
        if statement.startswith("all "):
            quantifier = "all"
            statement = statement[4:]
        elif statement.startswith("some "):
            quantifier = "some"
            statement = statement[5:]
        elif statement.startswith("no "):
            quantifier = "no"
            statement = statement[3:]
        elif "some" in statement and "not" in statement:
            quantifier = "some_not"
        
        # Check for negation
        negated = "not" in statement
        
        # Extract subject and predicate
        subject, predicate = await self._extract_subject_predicate(statement)
        
        # Create proposition
        proposition = LogicalProposition(
            id=prop_id,
            subject=subject,
            predicate=predicate,
            quantifier=quantifier,
            negated=negated,
            propositional_form=prop_id,
            predicate_form=f"{prop_id}({subject})"
        )
        
        return proposition
    
    async def _extract_subject_predicate(self, statement: str) -> Tuple[str, str]:
        """Extract subject and predicate from statement"""
        
        # Common patterns for subject-predicate extraction
        patterns = [
            r'(.+?)\s+are\s+(.+)',
            r'(.+?)\s+is\s+(.+)',
            r'(.+?)\s+have\s+(.+)',
            r'(.+?)\s+can\s+(.+)',
            r'(.+?)\s+will\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, statement)
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                return subject, predicate
        
        # Fallback: split on first verb
        words = statement.split()
        if len(words) >= 3:
            # Simple heuristic: subject is first word, predicate is rest
            subject = words[0]
            predicate = " ".join(words[2:])  # Skip the linking verb
            return subject, predicate
        
        return "unknown", "unknown"
    
    async def _parse_query(self, query: str) -> LogicalProposition:
        """Parse query into logical proposition"""
        
        return await self._parse_logical_proposition(query, "Q")
    
    async def _construct_proof(
        self, 
        premises: List[LogicalProposition], 
        query: LogicalProposition, 
        original_query: str
    ) -> DeductiveProof:
        """Construct a deductive proof from premises to conclusion"""
        
        proof = DeductiveProof(
            id=str(uuid4()),
            query=original_query,
            premises=premises
        )
        
        # Initialize proof with premises
        for premise in premises:
            proof.proof_steps.append({
                "step": len(proof.proof_steps) + 1,
                "statement": str(premise),
                "justification": "premise",
                "rule": None,
                "from_steps": []
            })
        
        # Try to prove the query
        success = await self._attempt_proof(proof, query)
        
        if success:
            proof.conclusion = query
            proof.is_complete = True
            proof.proof_length = len(proof.proof_steps)
        
        return proof
    
    async def _attempt_proof(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Attempt to prove target proposition using available rules"""
        
        # Check if target is already proven (in premises or derived)
        if await self._is_already_proven(proof, target):
            return True
        
        # Try each logical rule
        for rule in self.logical_rules:
            if await self._can_apply_rule(proof, rule, target):
                # Apply the rule
                success = await self._apply_rule(proof, rule, target)
                if success:
                    return True
        
        # Try syllogistic reasoning
        if await self._try_syllogistic_reasoning(proof, target):
            return True
        
        # Try direct logical equivalence
        if await self._try_logical_equivalence(proof, target):
            return True
        
        return False
    
    async def _is_already_proven(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Check if target proposition is already proven in the proof"""
        
        for step in proof.proof_steps:
            if self._propositions_equivalent(step["statement"], str(target)):
                return True
        
        return False
    
    def _propositions_equivalent(self, prop1: str, prop2: str) -> bool:
        """Check if two propositions are logically equivalent"""
        
        # Normalize propositions
        prop1_norm = str(prop1).lower().strip()
        prop2_norm = str(prop2).lower().strip()
        
        # Direct equality
        if prop1_norm == prop2_norm:
            return True
        
        # Check for logical equivalences
        equivalences = [
            ("all", "every"),
            ("some", "there exist"),
            ("no", "not any"),
        ]
        
        for equiv1, equiv2 in equivalences:
            if equiv1 in prop1_norm and equiv2 in prop2_norm:
                return True
            if equiv2 in prop1_norm and equiv1 in prop2_norm:
                return True
        
        return False
    
    async def _can_apply_rule(self, proof: DeductiveProof, rule: LogicalRule, target: LogicalProposition) -> bool:
        """Check if a logical rule can be applied to derive target"""
        
        # Simplified rule application checking
        if rule.rule_type == LogicalRuleType.MODUS_PONENS:
            return await self._can_apply_modus_ponens(proof, target)
        elif rule.rule_type == LogicalRuleType.MODUS_TOLLENS:
            return await self._can_apply_modus_tollens(proof, target)
        elif rule.rule_type == LogicalRuleType.HYPOTHETICAL_SYLLOGISM:
            return await self._can_apply_hypothetical_syllogism(proof, target)
        
        return False
    
    async def _can_apply_modus_ponens(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Check if modus ponens can be applied"""
        
        # Look for P → Q and P to derive Q
        for step1 in proof.proof_steps:
            for step2 in proof.proof_steps:
                if step1 != step2:
                    # Check if we have P → Q and P
                    if await self._is_conditional_statement(step1["statement"], target):
                        if await self._is_antecedent_of_conditional(step2["statement"], step1["statement"]):
                            return True
        
        return False
    
    async def _is_conditional_statement(self, statement: str, consequent: LogicalProposition) -> bool:
        """Check if statement is a conditional with given consequent"""
        
        # Look for "if...then", "implies", etc.
        conditional_patterns = [
            r'if\s+(.+?)\s+then\s+(.+)',
            r'(.+?)\s+implies\s+(.+)',
            r'(.+?)\s+→\s+(.+)',
        ]
        
        for pattern in conditional_patterns:
            match = re.search(pattern, str(statement).lower())
            if match:
                antecedent = match.group(1).strip()
                consequent_part = match.group(2).strip()
                
                # Check if consequent matches target
                if self._propositions_equivalent(consequent_part, str(consequent)):
                    return True
        
        return False
    
    async def _is_antecedent_of_conditional(self, statement: str, conditional: str) -> bool:
        """Check if statement is the antecedent of a conditional"""
        
        # Extract antecedent from conditional
        conditional_patterns = [
            r'if\s+(.+?)\s+then\s+(.+)',
            r'(.+?)\s+implies\s+(.+)',
            r'(.+?)\s+→\s+(.+)',
        ]
        
        for pattern in conditional_patterns:
            match = re.search(pattern, str(conditional).lower())
            if match:
                antecedent = match.group(1).strip()
                
                # Check if statement matches antecedent
                if self._propositions_equivalent(statement, antecedent):
                    return True
        
        return False
    
    async def _can_apply_modus_tollens(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Check if modus tollens can be applied"""
        
        # Look for P → Q and ¬Q to derive ¬P
        # Simplified implementation
        return False
    
    async def _can_apply_hypothetical_syllogism(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Check if hypothetical syllogism can be applied"""
        
        # Look for P → Q and Q → R to derive P → R
        # Simplified implementation
        return False
    
    async def _apply_rule(self, proof: DeductiveProof, rule: LogicalRule, target: LogicalProposition) -> bool:
        """Apply a logical rule to derive target"""
        
        if rule.rule_type == LogicalRuleType.MODUS_PONENS:
            return await self._apply_modus_ponens(proof, target)
        elif rule.rule_type == LogicalRuleType.MODUS_TOLLENS:
            return await self._apply_modus_tollens(proof, target)
        elif rule.rule_type == LogicalRuleType.HYPOTHETICAL_SYLLOGISM:
            return await self._apply_hypothetical_syllogism(proof, target)
        
        return False
    
    async def _apply_modus_ponens(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Apply modus ponens rule"""
        
        # Find the premises that allow modus ponens
        conditional_step = None
        antecedent_step = None
        
        for step1 in proof.proof_steps:
            for step2 in proof.proof_steps:
                if step1 != step2:
                    if await self._is_conditional_statement(step1["statement"], target):
                        if await self._is_antecedent_of_conditional(step2["statement"], step1["statement"]):
                            conditional_step = step1
                            antecedent_step = step2
                            break
            if conditional_step:
                break
        
        if conditional_step and antecedent_step:
            # Add the derived conclusion
            proof.proof_steps.append({
                "step": len(proof.proof_steps) + 1,
                "statement": str(target),
                "justification": "modus ponens",
                "rule": LogicalRuleType.MODUS_PONENS,
                "from_steps": [conditional_step["step"], antecedent_step["step"]]
            })
            return True
        
        return False
    
    async def _apply_modus_tollens(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Apply modus tollens rule"""
        # Simplified implementation
        return False
    
    async def _apply_hypothetical_syllogism(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Apply hypothetical syllogism rule"""
        # Simplified implementation
        return False
    
    async def _try_syllogistic_reasoning(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Try syllogistic reasoning patterns"""
        
        # Try Barbara syllogism: All A are B, All B are C, therefore All A are C
        if await self._try_barbara_syllogism(proof, target):
            return True
        
        # Try Darii syllogism: All A are B, Some C are A, therefore Some C are B
        if await self._try_darii_syllogism(proof, target):
            return True
        
        return False
    
    async def _try_barbara_syllogism(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Try Barbara syllogism: All A are B, All B are C, therefore All A are C"""
        
        # Look for two universal affirmative premises that chain together
        for step1 in proof.proof_steps:
            for step2 in proof.proof_steps:
                if step1 != step2:
                    # Check if we can form a Barbara syllogism
                    if await self._forms_barbara_syllogism(step1["statement"], step2["statement"], target):
                        # Add the derived conclusion
                        proof.proof_steps.append({
                            "step": len(proof.proof_steps) + 1,
                            "statement": str(target),
                            "justification": "Barbara syllogism",
                            "rule": "barbara",
                            "from_steps": [step1["step"], step2["step"]]
                        })
                        return True
        
        return False
    
    async def _forms_barbara_syllogism(self, premise1: str, premise2: str, conclusion: LogicalProposition) -> bool:
        """Check if two premises form a valid Barbara syllogism with given conclusion"""
        
        # Parse premises
        p1_parsed = await self._parse_logical_proposition(premise1, "P1")
        p2_parsed = await self._parse_logical_proposition(premise2, "P2")
        
        # Check for Barbara pattern: All A are B, All B are C, therefore All A are C
        if (p1_parsed.quantifier == "all" and p2_parsed.quantifier == "all" and 
            conclusion.quantifier == "all"):
            
            # Check if middle term connects the premises
            if p1_parsed.predicate == p2_parsed.subject:
                # Check if conclusion follows the pattern
                if (p1_parsed.subject == conclusion.subject and 
                    p2_parsed.predicate == conclusion.predicate):
                    return True
        
        return False
    
    async def _try_darii_syllogism(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Try Darii syllogism: All A are B, Some C are A, therefore Some C are B"""
        
        # Look for universal premise and particular premise
        for step1 in proof.proof_steps:
            for step2 in proof.proof_steps:
                if step1 != step2:
                    # Check if we can form a Darii syllogism
                    if await self._forms_darii_syllogism(step1["statement"], step2["statement"], target):
                        # Add the derived conclusion
                        proof.proof_steps.append({
                            "step": len(proof.proof_steps) + 1,
                            "statement": str(target),
                            "justification": "Darii syllogism",
                            "rule": "darii",
                            "from_steps": [step1["step"], step2["step"]]
                        })
                        return True
        
        return False
    
    async def _forms_darii_syllogism(self, premise1: str, premise2: str, conclusion: LogicalProposition) -> bool:
        """Check if two premises form a valid Darii syllogism with given conclusion"""
        
        # Parse premises
        p1_parsed = await self._parse_logical_proposition(premise1, "P1")
        p2_parsed = await self._parse_logical_proposition(premise2, "P2")
        
        # Check for Darii pattern: All A are B, Some C are A, therefore Some C are B
        if ((p1_parsed.quantifier == "all" and p2_parsed.quantifier == "some") or
            (p1_parsed.quantifier == "some" and p2_parsed.quantifier == "all")):
            
            if conclusion.quantifier == "some":
                # Determine which premise is universal and which is particular
                if p1_parsed.quantifier == "all":
                    universal, particular = p1_parsed, p2_parsed
                else:
                    universal, particular = p2_parsed, p1_parsed
                
                # Check if middle term connects the premises
                if universal.subject == particular.predicate:
                    # Check if conclusion follows the pattern
                    if (particular.subject == conclusion.subject and 
                        universal.predicate == conclusion.predicate):
                        return True
        
        return False
    
    async def _try_logical_equivalence(self, proof: DeductiveProof, target: LogicalProposition) -> bool:
        """Try direct logical equivalence"""
        
        # Check if target is logically equivalent to any proven statement
        for step in proof.proof_steps:
            if await self._are_logically_equivalent(step["statement"], str(target)):
                # Add equivalence step
                proof.proof_steps.append({
                    "step": len(proof.proof_steps) + 1,
                    "statement": str(target),
                    "justification": "logical equivalence",
                    "rule": "equivalence",
                    "from_steps": [step["step"]]
                })
                return True
        
        return False
    
    async def _are_logically_equivalent(self, statement1: str, statement2: str) -> bool:
        """Check if two statements are logically equivalent"""
        
        # Direct equivalence
        if self._propositions_equivalent(statement1, statement2):
            return True
        
        # Common logical equivalences
        equivalences = [
            ("all", "every"),
            ("some", "there exist"),
            ("no", "not any"),
            ("not all", "some are not")
        ]
        
        stmt1_lower = str(statement1).lower()
        stmt2_lower = str(statement2).lower()
        
        for equiv1, equiv2 in equivalences:
            if equiv1 in stmt1_lower and equiv2 in stmt2_lower:
                return True
            if equiv2 in stmt1_lower and equiv1 in stmt2_lower:
                return True
        
        return False
    
    async def _validate_proof(self, proof: DeductiveProof) -> DeductiveProof:
        """Validate the constructed proof"""
        
        # Check validity (structure)
        proof.is_valid = await self._check_proof_validity(proof)
        
        # Check soundness (truth of premises)
        proof.is_sound = await self._check_proof_soundness(proof)
        
        # Calculate confidence
        proof.confidence = await self._calculate_proof_confidence(proof)
        
        # Calculate complexity
        proof.complexity = len(proof.proof_steps) / self.max_proof_steps
        
        return proof
    
    async def _check_proof_validity(self, proof: DeductiveProof) -> bool:
        """Check if the proof structure is valid"""
        
        # Check if each step is justified
        for step in proof.proof_steps:
            if step["justification"] == "premise":
                continue
            
            # Check if the justification is valid
            if not await self._validate_inference_step(step, proof):
                return False
        
        return True
    
    async def _validate_inference_step(self, step: Dict[str, Any], proof: DeductiveProof) -> bool:
        """Validate a single inference step"""
        
        justification = step["justification"]
        
        if justification == "modus ponens":
            return await self._validate_modus_ponens_step(step, proof)
        elif justification == "Barbara syllogism":
            return await self._validate_barbara_step(step, proof)
        elif justification == "Darii syllogism":
            return await self._validate_darii_step(step, proof)
        elif justification == "logical equivalence":
            return await self._validate_equivalence_step(step, proof)
        
        return False
    
    async def _validate_modus_ponens_step(self, step: Dict[str, Any], proof: DeductiveProof) -> bool:
        """Validate a modus ponens inference step"""
        
        from_steps = step["from_steps"]
        if len(from_steps) != 2:
            return False
        
        # Get the referenced steps
        step1 = next((s for s in proof.proof_steps if s["step"] == from_steps[0]), None)
        step2 = next((s for s in proof.proof_steps if s["step"] == from_steps[1]), None)
        
        if not step1 or not step2:
            return False
        
        # Check if we have a valid modus ponens structure
        # One step should be a conditional, the other should be its antecedent
        return (await self._is_conditional_statement(step1["statement"], await self._parse_logical_proposition(step["statement"], "Q")) and
                await self._is_antecedent_of_conditional(step2["statement"], step1["statement"]))
    
    async def _validate_barbara_step(self, step: Dict[str, Any], proof: DeductiveProof) -> bool:
        """Validate a Barbara syllogism step"""
        
        from_steps = step["from_steps"]
        if len(from_steps) != 2:
            return False
        
        # Get the referenced steps
        step1 = next((s for s in proof.proof_steps if s["step"] == from_steps[0]), None)
        step2 = next((s for s in proof.proof_steps if s["step"] == from_steps[1]), None)
        
        if not step1 or not step2:
            return False
        
        # Check if we have a valid Barbara syllogism
        conclusion = await self._parse_logical_proposition(step["statement"], "Q")
        return await self._forms_barbara_syllogism(step1["statement"], step2["statement"], conclusion)
    
    async def _validate_darii_step(self, step: Dict[str, Any], proof: DeductiveProof) -> bool:
        """Validate a Darii syllogism step"""
        
        from_steps = step["from_steps"]
        if len(from_steps) != 2:
            return False
        
        # Get the referenced steps
        step1 = next((s for s in proof.proof_steps if s["step"] == from_steps[0]), None)
        step2 = next((s for s in proof.proof_steps if s["step"] == from_steps[1]), None)
        
        if not step1 or not step2:
            return False
        
        # Check if we have a valid Darii syllogism
        conclusion = await self._parse_logical_proposition(step["statement"], "Q")
        return await self._forms_darii_syllogism(step1["statement"], step2["statement"], conclusion)
    
    async def _validate_equivalence_step(self, step: Dict[str, Any], proof: DeductiveProof) -> bool:
        """Validate a logical equivalence step"""
        
        from_steps = step["from_steps"]
        if len(from_steps) != 1:
            return False
        
        # Get the referenced step
        ref_step = next((s for s in proof.proof_steps if s["step"] == from_steps[0]), None)
        
        if not ref_step:
            return False
        
        # Check if statements are logically equivalent
        return await self._are_logically_equivalent(ref_step["statement"], step["statement"])
    
    async def _check_proof_soundness(self, proof: DeductiveProof) -> bool:
        """Check if the proof is sound (premises are true)"""
        
        # In a complete implementation, this would check premises against world model
        # For now, assume premises are sound if they're consistent
        return await self._check_premise_consistency(proof.premises)
    
    async def _check_premise_consistency(self, premises: List[LogicalProposition]) -> bool:
        """Check if premises are consistent with each other"""
        
        # Simple consistency check: no direct contradictions
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i != j:
                    if await self._are_contradictory(premise1, premise2):
                        return False
        
        return True
    
    async def _are_contradictory(self, prop1: LogicalProposition, prop2: LogicalProposition) -> bool:
        """Check if two propositions are contradictory"""
        
        # Check for direct contradictions
        if (prop1.subject == prop2.subject and 
            prop1.predicate == prop2.predicate and
            prop1.negated != prop2.negated):
            return True
        
        # Check for quantifier contradictions
        if (prop1.subject == prop2.subject and 
            prop1.predicate == prop2.predicate and
            ((prop1.quantifier == "all" and prop2.quantifier == "no") or
             (prop1.quantifier == "no" and prop2.quantifier == "all"))):
            return True
        
        return False
    
    async def _calculate_proof_confidence(self, proof: DeductiveProof) -> float:
        """Calculate confidence in the proof"""
        
        if not proof.is_valid:
            return 0.0
        
        if not proof.is_sound:
            return 0.5
        
        # Deductive reasoning provides high confidence when valid and sound
        base_confidence = 0.95
        
        # Reduce confidence based on complexity
        complexity_penalty = min(0.1 * proof.complexity, 0.2)
        
        return base_confidence - complexity_penalty
    
    def _initialize_logical_rules(self) -> List[LogicalRule]:
        """Initialize database of logical inference rules"""
        
        rules = [
            LogicalRule(
                rule_type=LogicalRuleType.MODUS_PONENS,
                premises=["P → Q", "P"],
                conclusion="Q"
            ),
            LogicalRule(
                rule_type=LogicalRuleType.MODUS_TOLLENS,
                premises=["P → Q", "¬Q"],
                conclusion="¬P"
            ),
            LogicalRule(
                rule_type=LogicalRuleType.HYPOTHETICAL_SYLLOGISM,
                premises=["P → Q", "Q → R"],
                conclusion="P → R"
            ),
            LogicalRule(
                rule_type=LogicalRuleType.DISJUNCTIVE_SYLLOGISM,
                premises=["P ∨ Q", "¬P"],
                conclusion="Q"
            ),
            LogicalRule(
                rule_type=LogicalRuleType.CONJUNCTION,
                premises=["P", "Q"],
                conclusion="P ∧ Q"
            ),
            LogicalRule(
                rule_type=LogicalRuleType.SIMPLIFICATION,
                premises=["P ∧ Q"],
                conclusion="P"
            )
        ]
        
        return rules
    
    def get_deductive_stats(self) -> Dict[str, Any]:
        """Get statistics about deductive reasoning usage"""
        
        return {
            "total_proofs": len(self.proof_history),
            "valid_proofs": sum(1 for proof in self.proof_history if proof.is_valid),
            "sound_proofs": sum(1 for proof in self.proof_history if proof.is_sound),
            "complete_proofs": sum(1 for proof in self.proof_history if proof.is_complete),
            "average_proof_length": sum(proof.proof_length for proof in self.proof_history) / max(len(self.proof_history), 1),
            "average_confidence": sum(proof.confidence for proof in self.proof_history) / max(len(self.proof_history), 1),
            "available_rules": len(self.logical_rules),
            "rule_types": [rule.rule_type.value for rule in self.logical_rules]
        }