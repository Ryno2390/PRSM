"""
PRSM Neuro-Symbolic World Model Engine
=====================================

Implements the System 2 logic layer for PRSM. 
Combines neural intuition (SSM) with symbolic logic (Assertions) 
to ensure scientific breakthroughs follow physical and logical laws.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from prsm.core.cryptography.zk_proofs import ConstraintVerificationResult, AssertInfo

logger = logging.getLogger(__name__)

@dataclass
class ScientificConstraint:
    """A hard physical or logical law that must be obeyed"""
    name: str
    description: str
    validator: Callable[[Any, Dict[str, Any]], bool]
    severity: str = "critical" # critical, warning, advisory

@dataclass
class EthicalConstraint(ScientificConstraint):
    """A hard ethical rule that cannot be bypassed by autonomous agents"""
    category: str = "safety"

class NeuroSymbolicEngine:
    """
    The 'System 2' of the NWTN brain.
    Steers the neural engine by applying symbolic constraints to its outputs.
    """
    def __init__(self):
        self.constraints: Dict[str, ScientificConstraint] = {}
        self.ethical_kill_switches: Dict[str, EthicalConstraint] = {}
        self._initialize_builtin_constraints()

    def _initialize_builtin_constraints(self):
        """Register core laws of science and logic"""
        
        # 1. Mass/Energy Conservation (Simulated)
        self.register_constraint(ScientificConstraint(
            name="CONSERVATION_LAW",
            description="Ensure output entities do not exceed input mass/energy limits",
            validator=lambda output, context: True # Placeholder for actual logic
        ))

        # 2. Logical Non-Contradiction
        self.register_constraint(ScientificConstraint(
            name="NON_CONTRADICTION",
            description="Ensure reasoning step does not contradict previous verified steps",
            validator=self._check_non_contradiction
        ))

        # 3. ETHICAL KILL-SWITCHES
        self.register_ethical_constraint(EthicalConstraint(
            name="BIO_SAFETY_LEVEL_4",
            description="RESTRICTED: Research involving high-risk pathogens is forbidden without explicit DAO override.",
            validator=self._check_bio_safety,
            category="biological"
        ))

    def register_ethical_constraint(self, constraint: EthicalConstraint):
        self.ethical_kill_switches[constraint.name] = constraint
        logger.info(f"Registered ethical kill-switch: {constraint.name}")

    def register_constraint(self, constraint: ScientificConstraint):
        self.constraints[constraint.name] = constraint
        logger.info(f"Registered symbolic constraint: {constraint.name}")

    async def verify_constraints(self, proposal: Any, context: Dict[str, Any]) -> ConstraintVerificationResult:
        """
        Verify a neural proposal against all symbolic and ethical constraints.
        """
        result = ConstraintVerificationResult(success=True)
        
        # Check Ethical Kill-Switches FIRST
        for name, constraint in self.ethical_kill_switches.items():
            if not constraint.validator(proposal, context):
                result.success = False
                result.rejection_reason = f"🚫 ETHICAL KILL-SWITCH TRIGGERED [{name}]: {constraint.description}"
                logger.critical(f"🚨 {result.rejection_reason}")
                return result # Immediate halt

        for i, (name, constraint) in enumerate(self.constraints.items()):
            is_valid = constraint.validator(proposal, context)
            
            # Record the assertion for the ZK Proof layer
            result.add_assertion(
                pos=i, 
                content=f"Constraint {name}: {constraint.description}",
                result=is_valid
            )
            
            if not is_valid:
                result.rejection_reason = f"Violation of {name}: {constraint.description}"
                logger.warning(f"❌ Symbolic Rejection: {result.rejection_reason}")

        return result

    def _check_bio_safety(self, proposal: Any, context: Dict[str, Any]) -> bool:
        """Ethical check for restricted pathogens"""
        restricted_agents = ["ebola", "smallpox", "anthrax", "sars-cov-2-enhanced"]
        proposal_text = str(proposal).lower()
        return not any(agent in proposal_text for agent in restricted_agents)

    def _check_non_contradiction(self, proposal: Any, context: Dict[str, Any]) -> bool:
        """Simple symbolic check for contradictory statements"""
        proposal_text = str(proposal).lower()
        history = context.get("history", "").lower()
        
        # Example logic: if history says "A is True" and proposal says "A is False"
        contradictions = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("stable", "unstable")
        ]
        
        for word_a, word_b in contradictions:
            if word_a in history and word_b in proposal_text:
                return False
        return True

_world_model_instance: Optional[NeuroSymbolicEngine] = None

def get_world_model() -> NeuroSymbolicEngine:
    global _world_model_instance
    if _world_model_instance is None:
        _world_model_instance = NeuroSymbolicEngine()
    return _world_model_instance
