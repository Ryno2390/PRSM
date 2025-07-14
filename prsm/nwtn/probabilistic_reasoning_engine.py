#!/usr/bin/env python3
"""
NWTN Probabilistic Reasoning Engine
Bayesian inference and reasoning under uncertainty

This module implements NWTN's probabilistic reasoning capabilities, which allow the system to:
1. Perform Bayesian inference and belief updating
2. Handle uncertainty quantification and propagation
3. Make probabilistic predictions and assessments
4. Combine evidence from multiple sources
5. Reason about risk, likelihood, and probability distributions

Probabilistic reasoning is essential for handling uncertainty, making decisions under
incomplete information, and quantifying confidence in conclusions.

Key Concepts:
- Bayesian inference and belief updating
- Prior and posterior probability distributions
- Likelihood estimation and evidence combination
- Uncertainty quantification and propagation
- Risk assessment and decision making under uncertainty
- Probabilistic graphical models

Usage:
    from prsm.nwtn.probabilistic_reasoning_engine import ProbabilisticReasoningEngine
    
    engine = ProbabilisticReasoningEngine()
    result = await engine.probabilistic_inference(evidence, hypothesis, context)
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


class ProbabilityType(str, Enum):
    """Types of probability assessments"""
    SUBJECTIVE = "subjective"       # Subjective/personal probability
    OBJECTIVE = "objective"         # Objective/frequency-based
    LOGICAL = "logical"             # Logical probability
    EPISTEMIC = "epistemic"         # Degree of belief/knowledge
    ALEATORY = "aleatory"           # Natural randomness/variability


class EvidenceType(str, Enum):
    """Types of evidence for probabilistic reasoning"""
    DIRECT = "direct"               # Direct observation
    TESTIMONIAL = "testimonial"     # Testimony/report
    STATISTICAL = "statistical"     # Statistical data
    EXPERT_OPINION = "expert_opinion"  # Expert judgment
    THEORETICAL = "theoretical"     # Theoretical prediction
    ANALOGICAL = "analogical"       # Analogical inference
    CIRCUMSTANTIAL = "circumstantial"  # Circumstantial evidence


class InferenceMethod(str, Enum):
    """Methods for probabilistic inference"""
    BAYES_THEOREM = "bayes_theorem"
    LIKELIHOOD_RATIO = "likelihood_ratio"
    MONTE_CARLO = "monte_carlo"
    MARKOV_CHAIN = "markov_chain"
    BELIEF_PROPAGATION = "belief_propagation"
    VARIATIONAL = "variational"
    MAXIMUM_ENTROPY = "maximum_entropy"


class UncertaintyType(str, Enum):
    """Types of uncertainty"""
    ALEATORY = "aleatory"           # Irreducible randomness
    EPISTEMIC = "epistemic"         # Reducible knowledge uncertainty
    MODEL = "model"                 # Model structure uncertainty
    PARAMETER = "parameter"         # Parameter value uncertainty
    MEASUREMENT = "measurement"     # Measurement error
    LINGUISTIC = "linguistic"       # Linguistic/semantic uncertainty


@dataclass
class ProbabilisticEvidence:
    """Evidence for probabilistic reasoning"""
    
    id: str
    description: str
    evidence_type: EvidenceType
    
    # Probability assessments
    likelihood: float = 0.5         # P(evidence | hypothesis)
    base_rate: float = 0.5          # P(evidence)
    reliability: float = 1.0        # Reliability of evidence source
    
    # Uncertainty and confidence
    uncertainty: float = 0.1        # Uncertainty in likelihood
    confidence: float = 0.8         # Confidence in evidence
    
    # Context and metadata
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def adjusted_likelihood(self) -> float:
        """Calculate reliability-adjusted likelihood"""
        return self.likelihood * self.reliability + (1 - self.reliability) * 0.5


@dataclass
class ProbabilisticHypothesis:
    """A hypothesis for probabilistic reasoning"""
    
    id: str
    statement: str
    
    # Probability assessments
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    likelihood_given_evidence: float = 0.5
    
    # Uncertainty quantification
    probability_distribution: str = "point_estimate"  # "point_estimate", "beta", "normal"
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    credible_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Evidence tracking
    supporting_evidence: List[ProbabilisticEvidence] = field(default_factory=list)
    contradicting_evidence: List[ProbabilisticEvidence] = field(default_factory=list)
    
    # Model parameters
    alpha: float = 1.0              # Beta distribution alpha parameter
    beta: float = 1.0               # Beta distribution beta parameter
    
    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_posterior(self, evidence: ProbabilisticEvidence):
        """Update posterior probability using Bayes' theorem"""
        
        # Calculate likelihood ratio
        likelihood_ratio = evidence.adjusted_likelihood() / evidence.base_rate
        
        # Update using Bayes' theorem
        odds_prior = self.prior_probability / (1 - self.prior_probability)
        odds_posterior = odds_prior * likelihood_ratio
        
        self.posterior_probability = odds_posterior / (1 + odds_posterior)
        
        # Update beta distribution parameters
        if evidence in self.supporting_evidence:
            self.alpha += evidence.reliability
        elif evidence in self.contradicting_evidence:
            self.beta += evidence.reliability
        
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class ProbabilisticModel:
    """A probabilistic model for reasoning"""
    
    id: str
    name: str
    description: str
    
    # Model components
    hypotheses: List[ProbabilisticHypothesis] = field(default_factory=list)
    evidence: List[ProbabilisticEvidence] = field(default_factory=list)
    
    # Model structure
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # hypothesis_id -> [dependent_hypothesis_ids]
    conditionals: Dict[str, Dict[str, float]] = field(default_factory=dict)  # P(A|B) relationships
    
    # Inference parameters
    inference_method: InferenceMethod = InferenceMethod.BAYES_THEOREM
    convergence_threshold: float = 0.001
    max_iterations: int = 1000
    
    # Model evaluation
    log_likelihood: float = 0.0
    bayes_factor: float = 1.0
    model_evidence: float = 0.0
    
    # Uncertainty quantification
    epistemic_uncertainty: float = 0.0
    aleatory_uncertainty: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_hypothesis(self, hypothesis: ProbabilisticHypothesis):
        """Add a hypothesis to the model"""
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
    
    def add_evidence(self, evidence: ProbabilisticEvidence):
        """Add evidence to the model"""
        if evidence not in self.evidence:
            self.evidence.append(evidence)
    
    def get_joint_probability(self, hypothesis_ids: List[str]) -> float:
        """Calculate joint probability of multiple hypotheses"""
        if not hypothesis_ids:
            return 1.0
        
        # Simple independence assumption (can be enhanced)
        joint_prob = 1.0
        for hyp_id in hypothesis_ids:
            hypothesis = next((h for h in self.hypotheses if h.id == hyp_id), None)
            if hypothesis:
                joint_prob *= hypothesis.posterior_probability
        
        return joint_prob


@dataclass
class ProbabilisticAnalysis:
    """Result of probabilistic reasoning analysis"""
    
    id: str
    query: str
    evidence_items: List[ProbabilisticEvidence]
    
    # Probabilistic model
    model: ProbabilisticModel
    
    # Primary conclusions
    primary_hypothesis: ProbabilisticHypothesis
    alternative_hypotheses: List[ProbabilisticHypothesis] = field(default_factory=list)
    
    # Probability assessments
    final_probability: float = 0.5
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    # Uncertainty analysis
    uncertainty_sources: List[str] = field(default_factory=list)
    total_uncertainty: float = 0.0
    uncertainty_breakdown: Dict[UncertaintyType, float] = field(default_factory=dict)
    
    # Model diagnostics
    model_fit: float = 0.0
    convergence_achieved: bool = False
    iterations_required: int = 0
    
    # Sensitivity analysis
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    robustness_score: float = 0.0
    
    # Recommendations
    decision_recommendations: List[str] = field(default_factory=list)
    information_value: List[str] = field(default_factory=list)  # What additional info would be valuable
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProbabilisticReasoningEngine:
    """
    Engine for probabilistic reasoning using Bayesian inference
    
    This system enables NWTN to handle uncertainty, quantify confidence,
    and make rational decisions under incomplete information.
    """
    
    def __init__(self):
        self.model_executor = ModelExecutor(agent_id="probabilistic_reasoning_engine")
        self.world_model = WorldModelEngine()
        
        # Storage for models and analyses
        self.probabilistic_models: List[ProbabilisticModel] = []
        self.probabilistic_analyses: List[ProbabilisticAnalysis] = []
        
        # Configuration
        self.default_prior = 0.5
        self.min_evidence_reliability = 0.1
        self.convergence_threshold = 0.001
        self.max_iterations = 1000
        
        # Prior distributions for common domains
        self.domain_priors = {
            "medical": {"disease_prevalence": 0.05, "test_accuracy": 0.85, "symptom_sensitivity": 0.7},
            "legal": {"guilt_probability": 0.1, "witness_reliability": 0.8, "evidence_strength": 0.6},
            "scientific": {"hypothesis_truth": 0.2, "experimental_success": 0.3, "theory_validity": 0.4},
            "financial": {"market_movement": 0.5, "investment_success": 0.4, "risk_realization": 0.3},
            "technical": {"system_failure": 0.1, "bug_presence": 0.05, "performance_degradation": 0.2}
        }
        
        logger.info("Initialized Probabilistic Reasoning Engine")
    
    async def probabilistic_inference(
        self, 
        evidence_items: List[str], 
        hypothesis: str, 
        context: Dict[str, Any] = None
    ) -> ProbabilisticAnalysis:
        """
        Perform probabilistic inference given evidence and hypothesis
        
        Args:
            evidence_items: List of evidence statements
            hypothesis: The hypothesis to evaluate
            context: Additional context for inference
            
        Returns:
            ProbabilisticAnalysis: Complete probabilistic analysis with uncertainty quantification
        """
        
        logger.info(
            "Starting probabilistic inference",
            evidence_count=len(evidence_items),
            hypothesis=hypothesis
        )
        
        # Step 1: Parse evidence and hypothesis
        parsed_evidence = await self._parse_evidence(evidence_items, context)
        parsed_hypothesis = await self._parse_hypothesis(hypothesis, context)
        
        # Step 2: Build probabilistic model
        model = await self._build_probabilistic_model(parsed_evidence, parsed_hypothesis, context)
        
        # Step 3: Perform Bayesian inference
        updated_model = await self._perform_bayesian_inference(model, parsed_evidence)
        
        # Step 4: Quantify uncertainty
        uncertainty_analysis = await self._quantify_uncertainty(updated_model)
        
        # Step 5: Generate probabilistic analysis
        analysis = await self._generate_probabilistic_analysis(
            updated_model, parsed_evidence, hypothesis, uncertainty_analysis, context
        )
        
        # Step 6: Perform sensitivity analysis
        enhanced_analysis = await self._enhance_with_sensitivity_analysis(analysis)
        
        # Step 7: Store results
        self.probabilistic_models.append(updated_model)
        self.probabilistic_analyses.append(enhanced_analysis)
        
        logger.info(
            "Probabilistic inference complete",
            final_probability=enhanced_analysis.final_probability,
            total_uncertainty=enhanced_analysis.total_uncertainty,
            model_fit=enhanced_analysis.model_fit
        )
        
        return enhanced_analysis
    
    async def _parse_evidence(self, evidence_items: List[str], context: Dict[str, Any] = None) -> List[ProbabilisticEvidence]:
        """Parse evidence items into probabilistic evidence objects"""
        
        parsed_evidence = []
        
        for i, evidence_text in enumerate(evidence_items):
            # Classify evidence type
            evidence_type = await self._classify_evidence_type(evidence_text)
            
            # Estimate likelihood
            likelihood = await self._estimate_likelihood(evidence_text, context)
            
            # Estimate base rate
            base_rate = await self._estimate_base_rate(evidence_text, context)
            
            # Assess reliability
            reliability = await self._assess_evidence_reliability(evidence_text, evidence_type)
            
            # Create evidence object
            evidence = ProbabilisticEvidence(
                id=f"evidence_{i+1}",
                description=evidence_text,
                evidence_type=evidence_type,
                likelihood=likelihood,
                base_rate=base_rate,
                reliability=reliability,
                source=context.get("source", "unknown") if context else "unknown"
            )
            
            parsed_evidence.append(evidence)
        
        return parsed_evidence
    
    async def _classify_evidence_type(self, evidence_text: str) -> EvidenceType:
        """Classify the type of evidence"""
        
        text_lower = evidence_text.lower()
        
        # Statistical evidence
        if any(indicator in text_lower for indicator in ["statistics", "data", "study", "research", "percentage", "rate"]):
            return EvidenceType.STATISTICAL
        
        # Expert opinion
        if any(indicator in text_lower for indicator in ["expert", "professor", "doctor", "specialist", "authority"]):
            return EvidenceType.EXPERT_OPINION
        
        # Testimonial evidence
        if any(indicator in text_lower for indicator in ["reported", "said", "claimed", "testified", "stated"]):
            return EvidenceType.TESTIMONIAL
        
        # Theoretical evidence
        if any(indicator in text_lower for indicator in ["theory", "model", "predicts", "expected", "theoretical"]):
            return EvidenceType.THEORETICAL
        
        # Analogical evidence
        if any(indicator in text_lower for indicator in ["similar", "like", "analogous", "comparable"]):
            return EvidenceType.ANALOGICAL
        
        # Circumstantial evidence
        if any(indicator in text_lower for indicator in ["circumstantial", "indirect", "suggests", "implies"]):
            return EvidenceType.CIRCUMSTANTIAL
        
        # Default to direct observation
        return EvidenceType.DIRECT
    
    async def _estimate_likelihood(self, evidence_text: str, context: Dict[str, Any] = None) -> float:
        """Estimate likelihood P(evidence | hypothesis)"""
        
        # Domain-specific likelihood estimation
        domain = context.get("domain", "general") if context else "general"
        
        # Use AI assistance for likelihood estimation
        likelihood_prompt = f"""
        Estimate the likelihood of observing this evidence if the hypothesis is true:
        
        Evidence: {evidence_text}
        Domain: {domain}
        
        Consider:
        1. How likely is this evidence to occur if the hypothesis is true?
        2. What is the strength of the connection between evidence and hypothesis?
        3. Are there alternative explanations for this evidence?
        
        Provide a probability estimate between 0 and 1, where:
        - 0.9-1.0: Very likely evidence given hypothesis
        - 0.7-0.9: Likely evidence given hypothesis
        - 0.5-0.7: Moderately likely evidence given hypothesis
        - 0.3-0.5: Unlikely evidence given hypothesis
        - 0.0-0.3: Very unlikely evidence given hypothesis
        
        Return just the numerical probability.
        """
        
        try:
            response = await self.model_executor.execute_request(
                prompt=likelihood_prompt,
                model_name="gpt-4",
                temperature=0.2
            )
            
            # Extract probability from response
            probability = await self._extract_probability_from_response(response)
            return probability
            
        except Exception as e:
            logger.error("Error estimating likelihood", error=str(e))
            return 0.5  # Default neutral likelihood
    
    async def _estimate_base_rate(self, evidence_text: str, context: Dict[str, Any] = None) -> float:
        """Estimate base rate P(evidence)"""
        
        # Simple base rate estimation
        text_lower = evidence_text.lower()
        
        # Common evidence gets higher base rate
        if any(word in text_lower for word in ["common", "frequent", "often", "usual", "typical"]):
            return 0.7
        
        # Rare evidence gets lower base rate
        if any(word in text_lower for word in ["rare", "uncommon", "unusual", "exceptional", "unique"]):
            return 0.1
        
        # Statistical indicators
        if any(word in text_lower for word in ["statistics", "data", "research", "study"]):
            return 0.3  # Statistical evidence is less common
        
        # Default moderate base rate
        return 0.5
    
    async def _assess_evidence_reliability(self, evidence_text: str, evidence_type: EvidenceType) -> float:
        """Assess reliability of evidence source"""
        
        # Base reliability by evidence type
        type_reliability = {
            EvidenceType.STATISTICAL: 0.9,
            EvidenceType.EXPERT_OPINION: 0.8,
            EvidenceType.DIRECT: 0.8,
            EvidenceType.THEORETICAL: 0.7,
            EvidenceType.TESTIMONIAL: 0.6,
            EvidenceType.ANALOGICAL: 0.5,
            EvidenceType.CIRCUMSTANTIAL: 0.4
        }
        
        base_reliability = type_reliability.get(evidence_type, 0.5)
        
        # Adjust for reliability indicators
        text_lower = evidence_text.lower()
        
        # Positive reliability indicators
        if any(indicator in text_lower for indicator in ["verified", "confirmed", "established", "proven"]):
            base_reliability += 0.1
        
        # Negative reliability indicators
        if any(indicator in text_lower for indicator in ["unverified", "alleged", "rumored", "claimed"]):
            base_reliability -= 0.2
        
        # Uncertainty indicators
        if any(indicator in text_lower for indicator in ["possibly", "maybe", "might", "could"]):
            base_reliability -= 0.1
        
        return max(0.1, min(1.0, base_reliability))
    
    async def _extract_probability_from_response(self, response: str) -> float:
        """Extract probability value from AI response"""
        
        # Look for probability patterns
        probability_patterns = [
            r'(\d+\.?\d*)',  # Decimal number
            r'(\d+)%',       # Percentage
            r'(\d+)/(\d+)',  # Fraction
        ]
        
        for pattern in probability_patterns:
            matches = re.findall(pattern, response)
            if matches:
                match = matches[0]
                
                if isinstance(match, tuple) and len(match) == 2:
                    # Fraction
                    numerator, denominator = match
                    return float(numerator) / float(denominator)
                else:
                    # Decimal or percentage
                    value = float(match)
                    if value > 1.0:
                        return value / 100.0  # Convert percentage
                    return value
        
        return 0.5  # Default if no probability found
    
    async def _parse_hypothesis(self, hypothesis: str, context: Dict[str, Any] = None) -> ProbabilisticHypothesis:
        """Parse hypothesis into probabilistic hypothesis object"""
        
        # Estimate prior probability
        prior_prob = await self._estimate_prior_probability(hypothesis, context)
        
        # Create hypothesis object
        parsed_hypothesis = ProbabilisticHypothesis(
            id="primary_hypothesis",
            statement=hypothesis,
            prior_probability=prior_prob,
            posterior_probability=prior_prob  # Initialize with prior
        )
        
        return parsed_hypothesis
    
    async def _estimate_prior_probability(self, hypothesis: str, context: Dict[str, Any] = None) -> float:
        """Estimate prior probability of hypothesis"""
        
        # Domain-specific priors
        domain = context.get("domain", "general") if context else "general"
        
        if domain in self.domain_priors:
            domain_priors = self.domain_priors[domain]
            
            # Simple keyword matching for domain-specific priors
            hyp_lower = hypothesis.lower()
            
            for prior_type, prior_value in domain_priors.items():
                if any(keyword in hyp_lower for keyword in prior_type.split("_")):
                    return prior_value
        
        # Use AI assistance for prior estimation
        prior_prompt = f"""
        Estimate the prior probability of this hypothesis before considering specific evidence:
        
        Hypothesis: {hypothesis}
        Domain: {domain}
        
        Consider:
        1. How common or rare is this type of hypothesis?
        2. What is the base rate in this domain?
        3. What would be a reasonable prior belief?
        
        Provide a probability estimate between 0 and 1.
        Return just the numerical probability.
        """
        
        try:
            response = await self.model_executor.execute_request(
                prompt=prior_prompt,
                model_name="gpt-4",
                temperature=0.2
            )
            
            probability = await self._extract_probability_from_response(response)
            return probability
            
        except Exception as e:
            logger.error("Error estimating prior probability", error=str(e))
            return self.default_prior
    
    async def _build_probabilistic_model(
        self, 
        evidence_items: List[ProbabilisticEvidence], 
        hypothesis: ProbabilisticHypothesis, 
        context: Dict[str, Any] = None
    ) -> ProbabilisticModel:
        """Build probabilistic model from evidence and hypothesis"""
        
        model = ProbabilisticModel(
            id=str(uuid4()),
            name="Probabilistic Inference Model",
            description="Model for Bayesian inference",
            evidence=evidence_items,
            inference_method=InferenceMethod.BAYES_THEOREM
        )
        
        # Add primary hypothesis
        model.add_hypothesis(hypothesis)
        
        # Generate alternative hypotheses
        alternative_hypotheses = await self._generate_alternative_hypotheses(hypothesis, context)
        for alt_hyp in alternative_hypotheses:
            model.add_hypothesis(alt_hyp)
        
        # Classify evidence as supporting or contradicting
        for evidence in evidence_items:
            if evidence.likelihood > 0.5:
                hypothesis.supporting_evidence.append(evidence)
            else:
                hypothesis.contradicting_evidence.append(evidence)
        
        return model
    
    async def _generate_alternative_hypotheses(
        self, 
        primary_hypothesis: ProbabilisticHypothesis, 
        context: Dict[str, Any] = None
    ) -> List[ProbabilisticHypothesis]:
        """Generate alternative hypotheses for comparison"""
        
        alternatives = []
        
        # Negation of primary hypothesis
        negation = ProbabilisticHypothesis(
            id="negation_hypothesis",
            statement=f"NOT ({primary_hypothesis.statement})",
            prior_probability=1.0 - primary_hypothesis.prior_probability,
            posterior_probability=1.0 - primary_hypothesis.posterior_probability
        )
        alternatives.append(negation)
        
        # Generate additional alternatives using AI
        alt_prompt = f"""
        Generate 2-3 alternative hypotheses to this primary hypothesis:
        
        Primary: {primary_hypothesis.statement}
        
        Alternative hypotheses should be:
        1. Plausible explanations for the same observations
        2. Mutually exclusive with the primary hypothesis
        3. Testable and specific
        
        Return just the alternative hypothesis statements, one per line.
        """
        
        try:
            response = await self.model_executor.execute_request(
                prompt=alt_prompt,
                model_name="gpt-4",
                temperature=0.6
            )
            
            alt_statements = [line.strip() for line in response.split('\n') if line.strip()]
            
            for i, statement in enumerate(alt_statements[:3]):
                alt_hyp = ProbabilisticHypothesis(
                    id=f"alternative_{i+1}",
                    statement=statement,
                    prior_probability=0.2,  # Lower prior for alternatives
                    posterior_probability=0.2
                )
                alternatives.append(alt_hyp)
                
        except Exception as e:
            logger.error("Error generating alternative hypotheses", error=str(e))
        
        return alternatives
    
    async def _perform_bayesian_inference(
        self, 
        model: ProbabilisticModel, 
        evidence_items: List[ProbabilisticEvidence]
    ) -> ProbabilisticModel:
        """Perform Bayesian inference to update posterior probabilities"""
        
        # Update each hypothesis with evidence
        for hypothesis in model.hypotheses:
            for evidence in evidence_items:
                hypothesis.update_posterior(evidence)
        
        # Normalize probabilities if needed
        await self._normalize_probabilities(model)
        
        # Calculate model evidence
        model.model_evidence = await self._calculate_model_evidence(model)
        
        # Calculate log likelihood
        model.log_likelihood = await self._calculate_log_likelihood(model)
        
        return model
    
    async def _normalize_probabilities(self, model: ProbabilisticModel):
        """Normalize probabilities to ensure they sum to 1"""
        
        total_prob = sum(hyp.posterior_probability for hyp in model.hypotheses)
        
        if total_prob > 0:
            for hypothesis in model.hypotheses:
                hypothesis.posterior_probability /= total_prob
    
    async def _calculate_model_evidence(self, model: ProbabilisticModel) -> float:
        """Calculate model evidence P(evidence | model)"""
        
        # Simplified model evidence calculation
        evidence_prob = 1.0
        
        for evidence in model.evidence:
            # Weighted likelihood across all hypotheses
            weighted_likelihood = 0.0
            for hypothesis in model.hypotheses:
                weighted_likelihood += hypothesis.prior_probability * evidence.likelihood
            
            evidence_prob *= weighted_likelihood
        
        return evidence_prob
    
    async def _calculate_log_likelihood(self, model: ProbabilisticModel) -> float:
        """Calculate log likelihood of the model"""
        
        log_likelihood = 0.0
        
        for evidence in model.evidence:
            if evidence.likelihood > 0:
                log_likelihood += math.log(evidence.likelihood)
        
        return log_likelihood
    
    async def _quantify_uncertainty(self, model: ProbabilisticModel) -> Dict[UncertaintyType, float]:
        """Quantify different types of uncertainty"""
        
        uncertainty_breakdown = {}
        
        # Epistemic uncertainty (uncertainty about parameter values)
        epistemic = 0.0
        for hypothesis in model.hypotheses:
            # Use confidence interval width as uncertainty measure
            ci_width = hypothesis.confidence_interval[1] - hypothesis.confidence_interval[0]
            epistemic += ci_width / len(model.hypotheses)
        
        uncertainty_breakdown[UncertaintyType.EPISTEMIC] = epistemic
        
        # Aleatory uncertainty (inherent randomness)
        aleatory = 0.0
        for evidence in model.evidence:
            aleatory += evidence.uncertainty / len(model.evidence)
        
        uncertainty_breakdown[UncertaintyType.ALEATORY] = aleatory
        
        # Model uncertainty (uncertainty about model structure)
        model_uncertainty = 1.0 - model.model_evidence
        uncertainty_breakdown[UncertaintyType.MODEL] = model_uncertainty
        
        # Parameter uncertainty (uncertainty in parameter estimates)
        param_uncertainty = 0.0
        for hypothesis in model.hypotheses:
            if hypothesis.alpha + hypothesis.beta > 0:
                # Beta distribution variance
                variance = (hypothesis.alpha * hypothesis.beta) / ((hypothesis.alpha + hypothesis.beta)**2 * (hypothesis.alpha + hypothesis.beta + 1))
                param_uncertainty += variance
        
        uncertainty_breakdown[UncertaintyType.PARAMETER] = param_uncertainty / len(model.hypotheses)
        
        # Measurement uncertainty
        measurement_uncertainty = 0.0
        for evidence in model.evidence:
            measurement_uncertainty += (1.0 - evidence.reliability) / len(model.evidence)
        
        uncertainty_breakdown[UncertaintyType.MEASUREMENT] = measurement_uncertainty
        
        return uncertainty_breakdown
    
    async def _generate_probabilistic_analysis(
        self, 
        model: ProbabilisticModel, 
        evidence_items: List[ProbabilisticEvidence], 
        query: str,
        uncertainty_analysis: Dict[UncertaintyType, float],
        context: Dict[str, Any] = None
    ) -> ProbabilisticAnalysis:
        """Generate comprehensive probabilistic analysis"""
        
        # Find primary hypothesis (highest posterior probability)
        primary_hypothesis = max(model.hypotheses, key=lambda h: h.posterior_probability)
        
        # Alternative hypotheses
        alternative_hypotheses = [h for h in model.hypotheses if h != primary_hypothesis]
        
        # Calculate confidence interval
        confidence_interval = await self._calculate_confidence_interval(primary_hypothesis)
        
        # Calculate total uncertainty
        total_uncertainty = sum(uncertainty_analysis.values())
        
        # Generate uncertainty sources
        uncertainty_sources = await self._identify_uncertainty_sources(model, uncertainty_analysis)
        
        # Generate decision recommendations
        decision_recommendations = await self._generate_decision_recommendations(model, primary_hypothesis)
        
        # Generate information value assessment
        information_value = await self._assess_information_value(model)
        
        # Calculate model fit
        model_fit = await self._calculate_model_fit(model)
        
        analysis = ProbabilisticAnalysis(
            id=str(uuid4()),
            query=query,
            evidence_items=evidence_items,
            model=model,
            primary_hypothesis=primary_hypothesis,
            alternative_hypotheses=alternative_hypotheses,
            final_probability=primary_hypothesis.posterior_probability,
            confidence_interval=confidence_interval,
            uncertainty_sources=uncertainty_sources,
            total_uncertainty=total_uncertainty,
            uncertainty_breakdown=uncertainty_analysis,
            model_fit=model_fit,
            convergence_achieved=True,  # Simplified
            decision_recommendations=decision_recommendations,
            information_value=information_value
        )
        
        return analysis
    
    async def _calculate_confidence_interval(self, hypothesis: ProbabilisticHypothesis) -> Tuple[float, float]:
        """Calculate confidence interval for hypothesis probability"""
        
        # Use beta distribution for confidence interval
        if hypothesis.alpha > 0 and hypothesis.beta > 0:
            # Beta distribution parameters
            alpha = hypothesis.alpha
            beta = hypothesis.beta
            
            # Calculate mean and variance
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            
            # Simple confidence interval (mean ± 1.96 * std)
            std = math.sqrt(variance)
            lower = max(0.0, mean - 1.96 * std)
            upper = min(1.0, mean + 1.96 * std)
            
            return (lower, upper)
        
        # Fallback to simple interval around point estimate
        prob = hypothesis.posterior_probability
        margin = 0.1  # 10% margin
        
        return (max(0.0, prob - margin), min(1.0, prob + margin))
    
    async def _identify_uncertainty_sources(
        self, 
        model: ProbabilisticModel, 
        uncertainty_analysis: Dict[UncertaintyType, float]
    ) -> List[str]:
        """Identify major sources of uncertainty"""
        
        sources = []
        
        # Check each uncertainty type
        for uncertainty_type, value in uncertainty_analysis.items():
            if value > 0.2:  # Significant uncertainty
                type_name = uncertainty_type.value.replace("_", " ")
                sources.append(f"High {type_name} uncertainty ({value:.2f})")
        
        # Evidence-specific uncertainty
        unreliable_evidence = [e for e in model.evidence if e.reliability < 0.7]
        if unreliable_evidence:
            sources.append(f"{len(unreliable_evidence)} pieces of evidence have low reliability")
        
        # Model-specific uncertainty
        if model.model_evidence < 0.5:
            sources.append("Model evidence is low, suggesting model uncertainty")
        
        # Hypothesis competition
        sorted_hypotheses = sorted(model.hypotheses, key=lambda h: h.posterior_probability, reverse=True)
        if len(sorted_hypotheses) > 1:
            top_prob = sorted_hypotheses[0].posterior_probability
            second_prob = sorted_hypotheses[1].posterior_probability
            
            if top_prob - second_prob < 0.3:
                sources.append("Competing hypotheses have similar probabilities")
        
        return sources
    
    async def _generate_decision_recommendations(
        self, 
        model: ProbabilisticModel, 
        primary_hypothesis: ProbabilisticHypothesis
    ) -> List[str]:
        """Generate decision recommendations based on probabilistic analysis"""
        
        recommendations = []
        
        prob = primary_hypothesis.posterior_probability
        
        # Probability-based recommendations
        if prob > 0.8:
            recommendations.append("High probability supports acting on this hypothesis")
        elif prob > 0.6:
            recommendations.append("Moderate probability suggests cautious action")
        elif prob > 0.4:
            recommendations.append("Low probability suggests gathering more evidence")
        else:
            recommendations.append("Very low probability suggests rejecting this hypothesis")
        
        # Uncertainty-based recommendations
        if model.model_evidence < 0.5:
            recommendations.append("High model uncertainty suggests considering alternative models")
        
        # Evidence-based recommendations
        weak_evidence = [e for e in model.evidence if e.reliability < 0.5]
        if weak_evidence:
            recommendations.append(f"Verify {len(weak_evidence)} pieces of weak evidence")
        
        # Risk-based recommendations
        if prob > 0.5 and any(e.evidence_type == EvidenceType.CIRCUMSTANTIAL for e in model.evidence):
            recommendations.append("Consider direct evidence to confirm circumstantial findings")
        
        return recommendations
    
    async def _assess_information_value(self, model: ProbabilisticModel) -> List[str]:
        """Assess what additional information would be most valuable"""
        
        value_assessments = []
        
        # High-value evidence types
        missing_types = set(EvidenceType) - set(e.evidence_type for e in model.evidence)
        
        for missing_type in missing_types:
            if missing_type == EvidenceType.STATISTICAL:
                value_assessments.append("Statistical data would significantly improve confidence")
            elif missing_type == EvidenceType.EXPERT_OPINION:
                value_assessments.append("Expert opinion would provide valuable validation")
            elif missing_type == EvidenceType.DIRECT:
                value_assessments.append("Direct observation would strengthen the evidence base")
        
        # Evidence quality improvements
        low_reliability_evidence = [e for e in model.evidence if e.reliability < 0.7]
        if low_reliability_evidence:
            value_assessments.append("Improving reliability of existing evidence would be valuable")
        
        # Hypothesis testing
        untested_hypotheses = [h for h in model.hypotheses if len(h.supporting_evidence) == 0]
        if untested_hypotheses:
            value_assessments.append("Evidence specifically testing alternative hypotheses needed")
        
        return value_assessments
    
    async def _calculate_model_fit(self, model: ProbabilisticModel) -> float:
        """Calculate goodness of fit for the probabilistic model"""
        
        # Simple fit measure based on evidence consistency
        fit_score = 0.0
        
        for evidence in model.evidence:
            # Check if evidence likelihood is consistent with reliability
            consistency = 1.0 - abs(evidence.likelihood - evidence.reliability)
            fit_score += consistency
        
        if model.evidence:
            fit_score /= len(model.evidence)
        
        return fit_score
    
    async def _enhance_with_sensitivity_analysis(self, analysis: ProbabilisticAnalysis) -> ProbabilisticAnalysis:
        """Enhance analysis with sensitivity analysis"""
        
        # Sensitivity to evidence reliability
        sensitivity_analysis = {}
        
        for evidence in analysis.evidence_items:
            # Calculate sensitivity to this evidence
            original_prob = analysis.final_probability
            
            # Simulate removing this evidence
            modified_prob = await self._calculate_probability_without_evidence(analysis.model, evidence)
            
            sensitivity = abs(original_prob - modified_prob)
            sensitivity_analysis[evidence.id] = sensitivity
        
        analysis.sensitivity_analysis = sensitivity_analysis
        
        # Calculate robustness score
        max_sensitivity = max(sensitivity_analysis.values()) if sensitivity_analysis else 0.0
        analysis.robustness_score = 1.0 - max_sensitivity
        
        return analysis
    
    async def _calculate_probability_without_evidence(
        self, 
        model: ProbabilisticModel, 
        excluded_evidence: ProbabilisticEvidence
    ) -> float:
        """Calculate probability without specific evidence"""
        
        # Simplified calculation - exclude evidence and recalculate
        remaining_evidence = [e for e in model.evidence if e != excluded_evidence]
        
        if not remaining_evidence:
            return model.hypotheses[0].prior_probability
        
        # Recalculate with remaining evidence
        modified_prob = model.hypotheses[0].prior_probability
        
        for evidence in remaining_evidence:
            likelihood_ratio = evidence.adjusted_likelihood() / evidence.base_rate
            odds = modified_prob / (1 - modified_prob)
            odds *= likelihood_ratio
            modified_prob = odds / (1 + odds)
        
        return modified_prob
    
    def get_probabilistic_stats(self) -> Dict[str, Any]:
        """Get statistics about probabilistic reasoning usage"""
        
        total_evidence = sum(len(model.evidence) for model in self.probabilistic_models)
        total_hypotheses = sum(len(model.hypotheses) for model in self.probabilistic_models)
        
        return {
            "total_models": len(self.probabilistic_models),
            "total_analyses": len(self.probabilistic_analyses),
            "total_evidence": total_evidence,
            "total_hypotheses": total_hypotheses,
            "evidence_types": {
                et.value: sum(
                    sum(1 for e in model.evidence if e.evidence_type == et)
                    for model in self.probabilistic_models
                ) for et in EvidenceType
            },
            "inference_methods": {
                im.value: sum(1 for model in self.probabilistic_models if model.inference_method == im)
                for im in InferenceMethod
            },
            "average_final_probability": sum(
                analysis.final_probability for analysis in self.probabilistic_analyses
            ) / max(len(self.probabilistic_analyses), 1),
            "average_total_uncertainty": sum(
                analysis.total_uncertainty for analysis in self.probabilistic_analyses
            ) / max(len(self.probabilistic_analyses), 1),
            "average_model_fit": sum(
                analysis.model_fit for analysis in self.probabilistic_analyses
            ) / max(len(self.probabilistic_analyses), 1),
            "uncertainty_breakdown": {
                ut.value: sum(
                    analysis.uncertainty_breakdown.get(ut, 0) for analysis in self.probabilistic_analyses
                ) / max(len(self.probabilistic_analyses), 1) for ut in UncertaintyType
            }
        }