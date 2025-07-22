#!/usr/bin/env python3
"""
Contrarian Paper Identification System for NWTN
===============================================

This module implements the Contrarian Paper Identification system from the NWTN Novel Idea Generation Roadmap Phase 6.
It systematically identifies research papers that contradict mainstream thinking and leverages contrarian insights
for breakthrough reasoning and innovation discovery.

Key Innovations:
1. **Mainstream Consensus Detection**: Identifies dominant paradigms and consensus positions in research domains
2. **Contradiction Pattern Analysis**: Analyzes how and why papers contradict established thinking
3. **Contrarian Evidence Validation**: Assesses the quality and credibility of contrarian claims
4. **Breakthrough Potential Assessment**: Evaluates contrarian papers for paradigm-shift potential
5. **Contrarian Knowledge Integration**: Integrates contrarian insights into reasoning processes

Architecture:
- MainstreamConsensusDetector: Identifies dominant paradigms and consensus positions
- ContradictionAnalyzer: Analyzes patterns of contradiction and challenge
- ContrarianEvidenceValidator: Validates contrarian claims and evidence quality
- BreakthroughPotentialEvaluator: Assesses paradigm-shift potential of contrarian work
- ContrarianKnowledgeIntegrator: Integrates contrarian insights into reasoning

Based on NWTN Roadmap Phase 6 - Contrarian Paper Identification (P3 Priority, Medium Effort)
Expected Impact: Enhanced breakthrough reasoning through systematic contrarian perspective integration
"""

import asyncio
import time
import math
import re
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from collections import defaultdict, Counter
import structlog

logger = structlog.get_logger(__name__)

class ConsensusType(Enum):
    """Types of consensus that can be identified in research"""
    THEORETICAL = "theoretical"           # Consensus on theories or models
    METHODOLOGICAL = "methodological"     # Consensus on methods or approaches
    EMPIRICAL = "empirical"               # Consensus on empirical findings
    CAUSAL = "causal"                     # Consensus on causal relationships
    INTERPRETIVE = "interpretive"         # Consensus on interpretation of data
    NORMATIVE = "normative"               # Consensus on values or should-statements
    CATEGORICAL = "categorical"           # Consensus on classifications or definitions
    QUANTITATIVE = "quantitative"         # Consensus on measurements or quantities

class ContradictionType(Enum):
    """Types of contradiction patterns in contrarian papers"""
    DIRECT_REFUTATION = "direct_refutation"           # Direct denial of mainstream claims
    ALTERNATIVE_EXPLANATION = "alternative_explanation" # Different explanation for same phenomena
    SCOPE_LIMITATION = "scope_limitation"              # Limits scope of mainstream claims
    ASSUMPTION_CHALLENGE = "assumption_challenge"      # Challenges underlying assumptions
    METHOD_CRITIQUE = "method_critique"                # Critiques methodological approaches
    EVIDENCE_REINTERPRETATION = "evidence_reinterpretation" # Reinterprets existing evidence
    PARADIGM_INVERSION = "paradigm_inversion"          # Inverts fundamental paradigms
    BOUNDARY_REDEFINITION = "boundary_redefinition"    # Redefines conceptual boundaries

class ContrarianCredibility(Enum):
    """Credibility levels for contrarian papers"""
    HIGH = "high"                 # Strong methodology, peer review, replication
    MEDIUM = "medium"             # Good methodology, some validation
    LOW = "low"                   # Weak methodology, limited validation
    SPECULATIVE = "speculative"   # Highly speculative, little validation
    DEBUNKED = "debunked"         # Contradicted by subsequent evidence

class BreakthroughPotential(Enum):
    """Potential for contrarian work to create breakthroughs"""
    REVOLUTIONARY = "revolutionary"   # Could overturn entire fields
    TRANSFORMATIVE = "transformative" # Could significantly change understanding
    SIGNIFICANT = "significant"       # Could modify current theories
    INCREMENTAL = "incremental"       # Could refine current understanding
    MINIMAL = "minimal"               # Limited potential for change

@dataclass
class MainstreamConsensus:
    """Represents identified mainstream consensus in a research domain"""
    id: str = field(default_factory=lambda: str(uuid4()))
    domain: str = ""
    consensus_type: ConsensusType = ConsensusType.THEORETICAL
    consensus_statement: str = ""
    supporting_papers: List[str] = field(default_factory=list)
    consensus_strength: float = 0.0  # 0.0 to 1.0
    key_proponents: List[str] = field(default_factory=list)
    historical_development: List[str] = field(default_factory=list)
    underlying_assumptions: List[str] = field(default_factory=list)
    scope_and_limitations: List[str] = field(default_factory=list)
    vulnerable_points: List[str] = field(default_factory=list)  # Potential points of contradiction
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ContrarianPaper:
    """Represents a paper that contradicts mainstream consensus"""
    id: str = field(default_factory=lambda: str(uuid4()))
    paper_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    publication_year: int = 0
    journal: str = ""
    domain: str = ""
    contradicted_consensus: List[str] = field(default_factory=list)  # IDs of MainstreamConsensus
    contradiction_type: ContradictionType = ContradictionType.DIRECT_REFUTATION
    contrarian_claims: List[str] = field(default_factory=list)
    evidence_presented: List[str] = field(default_factory=list)
    methodological_approach: str = ""
    credibility_level: ContrarianCredibility = ContrarianCredibility.MEDIUM
    breakthrough_potential: BreakthroughPotential = BreakthroughPotential.INCREMENTAL
    paradigm_shift_score: float = 0.0  # 0.0 to 1.0
    validation_status: str = ""
    follow_up_work: List[str] = field(default_factory=list)
    mainstream_response: str = ""
    contrarian_insights: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ContradictionAnalysis:
    """Analysis of how a contrarian paper contradicts mainstream thinking"""
    id: str = field(default_factory=lambda: str(uuid4()))
    contrarian_paper_id: str = ""
    consensus_id: str = ""
    contradiction_mechanism: str = ""
    logical_structure: str = ""
    evidence_comparison: Dict[str, Any] = field(default_factory=dict)
    assumption_differences: List[str] = field(default_factory=list)
    methodological_differences: List[str] = field(default_factory=list)
    scope_differences: List[str] = field(default_factory=list)
    interpretive_differences: List[str] = field(default_factory=list)
    strength_of_contradiction: float = 0.0  # 0.0 to 1.0
    resolution_possibilities: List[str] = field(default_factory=list)
    synthesis_opportunities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ContrarianInsight:
    """Insight extracted from contrarian papers for breakthrough reasoning"""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_paper_id: str = ""
    insight_type: str = ""
    insight_description: str = ""
    applicable_domains: List[str] = field(default_factory=list)
    breakthrough_applications: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    limitations_and_caveats: List[str] = field(default_factory=list)
    integration_opportunities: List[str] = field(default_factory=list)
    novelty_score: float = 0.0
    applicability_score: float = 0.0
    confidence_level: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ContrarianIdentificationResult:
    """Complete result from contrarian paper identification analysis"""
    query: str = ""
    domain: str = ""
    mainstream_consensus: List[MainstreamConsensus] = field(default_factory=list)
    contrarian_papers: List[ContrarianPaper] = field(default_factory=list)
    contradiction_analyses: List[ContradictionAnalysis] = field(default_factory=list)
    contrarian_insights: List[ContrarianInsight] = field(default_factory=list)
    consensus_challenge_score: float = 0.0
    paradigm_disruption_potential: float = 0.0
    breakthrough_opportunity_score: float = 0.0
    contrarian_evidence_quality: float = 0.0
    processing_time: float = 0.0
    contrarian_summary: str = ""
    mainstream_vs_contrarian_comparison: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class MainstreamConsensusDetector:
    """Detects mainstream consensus positions in research domains"""
    
    def __init__(self):
        self.consensus_indicators = {
            ConsensusType.THEORETICAL: [
                "widely accepted", "established theory", "dominant paradigm", "consensus view",
                "orthodox position", "standard model", "received wisdom", "conventional theory"
            ],
            ConsensusType.METHODOLOGICAL: [
                "standard method", "established protocol", "best practice", "gold standard",
                "standard procedure", "accepted methodology", "conventional approach"
            ],
            ConsensusType.EMPIRICAL: [
                "well-established", "documented fact", "established finding", "confirmed result",
                "replicated study", "consistent evidence", "robust finding"
            ],
            ConsensusType.CAUSAL: [
                "established relationship", "proven cause", "demonstrated effect", "causal link",
                "established mechanism", "proven pathway", "accepted causation"
            ],
            ConsensusType.INTERPRETIVE: [
                "standard interpretation", "accepted meaning", "conventional understanding",
                "established significance", "orthodox reading", "received interpretation"
            ]
        }
        
        self.vulnerability_indicators = [
            "however", "but", "nevertheless", "despite", "although", "yet",
            "limitation", "exception", "anomaly", "inconsistent", "paradox",
            "unexplained", "unclear", "debate", "controversy", "disagreement"
        ]
    
    async def detect_consensus(self, papers: List[Dict[str, Any]], domain: str) -> List[MainstreamConsensus]:
        """Detect mainstream consensus positions from paper corpus"""
        
        consensus_positions = []
        
        # Group papers by topic/theme
        topic_groups = await self._group_papers_by_topic(papers, domain)
        
        for topic, topic_papers in topic_groups.items():
            # Analyze consensus for this topic
            consensus = await self._analyze_topic_consensus(topic, topic_papers, domain)
            if consensus and consensus.consensus_strength > 0.5:  # Only strong consensus
                consensus_positions.append(consensus)
        
        # Sort by consensus strength
        consensus_positions.sort(key=lambda c: c.consensus_strength, reverse=True)
        return consensus_positions[:10]  # Top 10 consensus positions
    
    async def _group_papers_by_topic(self, papers: List[Dict[str, Any]], domain: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group papers by topic or theme"""
        
        # Simple topic grouping based on keywords (in practice, would use more sophisticated clustering)
        topic_groups = defaultdict(list)
        
        for paper in papers:
            # Extract key topics from title and abstract
            topics = await self._extract_topics(paper, domain)
            
            for topic in topics:
                topic_groups[topic].append(paper)
        
        # Only keep topics with sufficient papers for consensus analysis
        filtered_groups = {topic: papers for topic, papers in topic_groups.items() if len(papers) >= 5}
        
        return filtered_groups
    
    async def _extract_topics(self, paper: Dict[str, Any], domain: str) -> List[str]:
        """Extract key topics from a paper"""
        
        # Combine title and abstract for topic extraction
        text = ""
        if paper.get('title'):
            text += paper['title'] + " "
        if paper.get('abstract'):
            text += paper['abstract']
        
        text = text.lower()
        
        # Domain-specific topic keywords (simplified - in practice would use NLP)
        domain_keywords = {
            'machine_learning': ['neural networks', 'deep learning', 'supervised learning', 'unsupervised learning', 'reinforcement learning'],
            'physics': ['quantum mechanics', 'relativity', 'particle physics', 'thermodynamics', 'electromagnetism'],
            'biology': ['evolution', 'genetics', 'molecular biology', 'ecology', 'neuroscience'],
            'psychology': ['cognition', 'behavior', 'learning', 'memory', 'perception'],
            'economics': ['market efficiency', 'behavioral economics', 'game theory', 'macroeconomics', 'microeconomics']
        }
        
        keywords = domain_keywords.get(domain, [])
        
        # Find topics present in the text
        found_topics = [keyword for keyword in keywords if keyword in text]
        
        # If no domain-specific topics, extract general topics
        if not found_topics:
            # Simple topic extraction based on common research terms
            general_topics = ['methodology', 'theory', 'empirical', 'analysis', 'model', 'framework']
            found_topics = [topic for topic in general_topics if topic in text]
        
        return found_topics or ['general']
    
    async def _analyze_topic_consensus(self, topic: str, papers: List[Dict[str, Any]], domain: str) -> MainstreamConsensus:
        """Analyze consensus for a specific topic"""
        
        # Identify consensus statements
        consensus_statements = await self._extract_consensus_statements(papers, topic)
        
        if not consensus_statements:
            return None
        
        # Find most common consensus statement
        statement_counts = Counter(consensus_statements)
        dominant_statement = statement_counts.most_common(1)[0][0]
        consensus_strength = statement_counts[dominant_statement] / len(papers)
        
        # Extract supporting information
        supporting_papers = [p.get('id', '') for p in papers if self._supports_statement(p, dominant_statement)]
        key_proponents = await self._extract_key_proponents(papers, dominant_statement)
        underlying_assumptions = await self._extract_assumptions(papers, dominant_statement)
        vulnerable_points = await self._identify_vulnerable_points(papers, dominant_statement)
        
        consensus = MainstreamConsensus(
            domain=domain,
            consensus_type=self._classify_consensus_type(dominant_statement),
            consensus_statement=dominant_statement,
            supporting_papers=supporting_papers,
            consensus_strength=consensus_strength,
            key_proponents=key_proponents,
            underlying_assumptions=underlying_assumptions,
            vulnerable_points=vulnerable_points
        )
        
        return consensus
    
    async def _extract_consensus_statements(self, papers: List[Dict[str, Any]], topic: str) -> List[str]:
        """Extract consensus statements from papers about a topic"""
        
        statements = []
        
        for paper in papers:
            text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            
            # Look for consensus indicators
            for consensus_type, indicators in self.consensus_indicators.items():
                for indicator in indicators:
                    if indicator in text and topic in text:
                        # Extract statement around the indicator
                        statement = self._extract_statement_around_indicator(text, indicator, topic)
                        if statement:
                            statements.append(statement)
        
        return statements
    
    def _extract_statement_around_indicator(self, text: str, indicator: str, topic: str) -> str:
        """Extract a statement around a consensus indicator"""
        
        # Find the position of the indicator
        indicator_pos = text.find(indicator)
        if indicator_pos == -1:
            return ""
        
        # Extract a window around the indicator (simplified)
        start = max(0, indicator_pos - 100)
        end = min(len(text), indicator_pos + 100)
        
        window = text[start:end]
        
        # Simple statement extraction (in practice, would use sentence segmentation)
        sentences = window.split('.')
        for sentence in sentences:
            if indicator in sentence and topic in sentence:
                return sentence.strip()
        
        return ""
    
    def _supports_statement(self, paper: Dict[str, Any], statement: str) -> bool:
        """Check if a paper supports a consensus statement"""
        
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        
        # Simple keyword matching (in practice, would use semantic similarity)
        statement_keywords = set(statement.split())
        text_keywords = set(text.split())
        
        overlap = len(statement_keywords.intersection(text_keywords))
        return overlap / len(statement_keywords) > 0.3
    
    async def _extract_key_proponents(self, papers: List[Dict[str, Any]], statement: str) -> List[str]:
        """Extract key proponents of a consensus statement"""
        
        proponents = []
        
        for paper in papers:
            if self._supports_statement(paper, statement):
                authors = paper.get('authors', [])
                if isinstance(authors, list):
                    proponents.extend(authors)
                elif isinstance(authors, str):
                    proponents.append(authors)
        
        # Count author mentions and return most frequent
        author_counts = Counter(proponents)
        return [author for author, count in author_counts.most_common(5)]
    
    async def _extract_assumptions(self, papers: List[Dict[str, Any]], statement: str) -> List[str]:
        """Extract underlying assumptions of a consensus statement"""
        
        assumption_indicators = [
            "assume", "assumes", "assuming", "assumption", "presuppose", "given that",
            "take for granted", "premise", "foundation", "basis", "underlying"
        ]
        
        assumptions = []
        
        for paper in papers:
            if self._supports_statement(paper, statement):
                text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
                
                for indicator in assumption_indicators:
                    if indicator in text:
                        assumption = self._extract_statement_around_indicator(text, indicator, "")
                        if assumption and len(assumption) > 10:
                            assumptions.append(assumption)
        
        return list(set(assumptions))[:5]  # Remove duplicates, keep top 5
    
    async def _identify_vulnerable_points(self, papers: List[Dict[str, Any]], statement: str) -> List[str]:
        """Identify points where consensus might be vulnerable to contradiction"""
        
        vulnerabilities = []
        
        for paper in papers:
            if self._supports_statement(paper, statement):
                text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
                
                for indicator in self.vulnerability_indicators:
                    if indicator in text:
                        vulnerability = self._extract_statement_around_indicator(text, indicator, "")
                        if vulnerability and len(vulnerability) > 10:
                            vulnerabilities.append(vulnerability)
        
        return list(set(vulnerabilities))[:5]  # Remove duplicates, keep top 5
    
    def _classify_consensus_type(self, statement: str) -> ConsensusType:
        """Classify the type of consensus based on statement content"""
        
        statement_lower = statement.lower()
        
        # Check for different types of consensus indicators
        for consensus_type, indicators in self.consensus_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    return consensus_type
        
        # Default classification based on content
        if any(word in statement_lower for word in ['theory', 'model', 'framework']):
            return ConsensusType.THEORETICAL
        elif any(word in statement_lower for word in ['method', 'approach', 'procedure']):
            return ConsensusType.METHODOLOGICAL
        elif any(word in statement_lower for word in ['finding', 'result', 'evidence']):
            return ConsensusType.EMPIRICAL
        elif any(word in statement_lower for word in ['cause', 'effect', 'relationship']):
            return ConsensusType.CAUSAL
        else:
            return ConsensusType.THEORETICAL

class ContradictionAnalyzer:
    """Analyzes how contrarian papers contradict mainstream consensus"""
    
    def __init__(self):
        self.contradiction_patterns = {
            ContradictionType.DIRECT_REFUTATION: [
                "not", "false", "incorrect", "wrong", "refute", "disprove", "contradiction"
            ],
            ContradictionType.ALTERNATIVE_EXPLANATION: [
                "alternatively", "different explanation", "instead", "rather", "better explanation"
            ],
            ContradictionType.SCOPE_LIMITATION: [
                "limited to", "only applies", "except", "excluding", "boundary", "constraint"
            ],
            ContradictionType.ASSUMPTION_CHALLENGE: [
                "assumes", "assumption", "presuppose", "taken for granted", "question"
            ],
            ContradictionType.METHOD_CRITIQUE: [
                "methodology", "approach", "flawed method", "inappropriate", "inadequate"
            ],
            ContradictionType.EVIDENCE_REINTERPRETATION: [
                "reinterpret", "different interpretation", "alternative view", "suggests instead"
            ]
        }
    
    async def analyze_contradictions(self, 
                                   contrarian_papers: List[ContrarianPaper], 
                                   consensus_positions: List[MainstreamConsensus]) -> List[ContradictionAnalysis]:
        """Analyze how contrarian papers contradict consensus positions"""
        
        analyses = []
        
        for contrarian in contrarian_papers:
            for consensus_id in contrarian.contradicted_consensus:
                # Find the corresponding consensus
                consensus = next((c for c in consensus_positions if c.id == consensus_id), None)
                if not consensus:
                    continue
                
                # Analyze the contradiction
                analysis = await self._analyze_single_contradiction(contrarian, consensus)
                if analysis:
                    analyses.append(analysis)
        
        return analyses
    
    async def _analyze_single_contradiction(self, 
                                          contrarian: ContrarianPaper, 
                                          consensus: MainstreamConsensus) -> ContradictionAnalysis:
        """Analyze a single contrarian paper's contradiction of consensus"""
        
        # Analyze contradiction mechanism
        mechanism = await self._identify_contradiction_mechanism(contrarian, consensus)
        
        # Compare evidence
        evidence_comparison = await self._compare_evidence(contrarian, consensus)
        
        # Identify differences
        assumption_diffs = await self._identify_assumption_differences(contrarian, consensus)
        method_diffs = await self._identify_methodological_differences(contrarian, consensus)
        scope_diffs = await self._identify_scope_differences(contrarian, consensus)
        
        # Calculate strength of contradiction
        contradiction_strength = await self._calculate_contradiction_strength(contrarian, consensus)
        
        # Identify resolution possibilities
        resolutions = await self._identify_resolution_possibilities(contrarian, consensus)
        
        # Find synthesis opportunities
        synthesis_ops = await self._identify_synthesis_opportunities(contrarian, consensus)
        
        analysis = ContradictionAnalysis(
            contrarian_paper_id=contrarian.id,
            consensus_id=consensus.id,
            contradiction_mechanism=mechanism,
            logical_structure=f"{contrarian.contradiction_type.value} contradiction pattern",
            evidence_comparison=evidence_comparison,
            assumption_differences=assumption_diffs,
            methodological_differences=method_diffs,
            scope_differences=scope_diffs,
            strength_of_contradiction=contradiction_strength,
            resolution_possibilities=resolutions,
            synthesis_opportunities=synthesis_ops
        )
        
        return analysis
    
    async def _identify_contradiction_mechanism(self, 
                                              contrarian: ContrarianPaper, 
                                              consensus: MainstreamConsensus) -> str:
        """Identify how the contrarian paper contradicts consensus"""
        
        # Analyze contrarian claims vs consensus statement
        consensus_text = consensus.consensus_statement.lower()
        
        # Check for different contradiction patterns
        for contradiction_type, indicators in self.contradiction_patterns.items():
            if contrarian.contradiction_type == contradiction_type:
                # Find specific mechanism
                for claim in contrarian.contrarian_claims:
                    claim_lower = claim.lower()
                    for indicator in indicators:
                        if indicator in claim_lower:
                            return f"{contradiction_type.value}: {claim} challenges '{consensus.consensus_statement}' through {indicator}"
        
        return f"General {contrarian.contradiction_type.value} of consensus position"
    
    async def _compare_evidence(self, 
                               contrarian: ContrarianPaper, 
                               consensus: MainstreamConsensus) -> Dict[str, Any]:
        """Compare evidence presented by contrarian vs consensus"""
        
        return {
            'contrarian_evidence_count': len(contrarian.evidence_presented),
            'consensus_support_count': len(consensus.supporting_papers),
            'evidence_overlap': self._calculate_evidence_overlap(contrarian, consensus),
            'evidence_quality_comparison': {
                'contrarian': contrarian.credibility_level.value,
                'consensus': 'established' if consensus.consensus_strength > 0.8 else 'moderate'
            },
            'methodological_approach': contrarian.methodological_approach,
            'evidence_types': {
                'contrarian': self._classify_evidence_types(contrarian.evidence_presented),
                'consensus': 'multiple_studies'
            }
        }
    
    def _calculate_evidence_overlap(self, contrarian: ContrarianPaper, consensus: MainstreamConsensus) -> float:
        """Calculate overlap between contrarian and consensus evidence"""
        
        # Simplified overlap calculation
        contrarian_papers = set([contrarian.paper_id])  # Contrarian cites itself
        consensus_papers = set(consensus.supporting_papers)
        
        if not consensus_papers:
            return 0.0
        
        overlap = len(contrarian_papers.intersection(consensus_papers))
        return overlap / len(consensus_papers.union(contrarian_papers))
    
    def _classify_evidence_types(self, evidence_list: List[str]) -> List[str]:
        """Classify types of evidence presented"""
        
        evidence_types = []
        
        for evidence in evidence_list:
            evidence_lower = evidence.lower()
            
            if any(term in evidence_lower for term in ['experiment', 'trial', 'test']):
                evidence_types.append('experimental')
            elif any(term in evidence_lower for term in ['survey', 'questionnaire', 'interview']):
                evidence_types.append('survey')
            elif any(term in evidence_lower for term in ['observation', 'case study', 'field study']):
                evidence_types.append('observational')
            elif any(term in evidence_lower for term in ['simulation', 'model', 'computation']):
                evidence_types.append('computational')
            elif any(term in evidence_lower for term in ['meta-analysis', 'systematic review', 'review']):
                evidence_types.append('meta-analysis')
            else:
                evidence_types.append('other')
        
        return list(set(evidence_types))  # Remove duplicates
    
    async def _identify_assumption_differences(self, 
                                             contrarian: ContrarianPaper, 
                                             consensus: MainstreamConsensus) -> List[str]:
        """Identify differences in underlying assumptions"""
        
        differences = []
        
        # Compare contrarian claims with consensus assumptions
        for claim in contrarian.contrarian_claims:
            for assumption in consensus.underlying_assumptions:
                if self._assumptions_conflict(claim, assumption):
                    differences.append(f"Contrarian: '{claim}' vs Consensus: '{assumption}'")
        
        return differences[:5]  # Top 5 differences
    
    def _assumptions_conflict(self, claim: str, assumption: str) -> bool:
        """Check if contrarian claim conflicts with consensus assumption"""
        
        # Simple keyword-based conflict detection
        claim_words = set(claim.lower().split())
        assumption_words = set(assumption.lower().split())
        
        # Look for negation or contradiction keywords
        contradiction_words = {'not', 'no', 'never', 'false', 'incorrect', 'wrong'}
        
        if any(word in claim_words for word in contradiction_words):
            # Check if claim and assumption share content words
            content_overlap = len(claim_words.intersection(assumption_words))
            return content_overlap > 2  # Arbitrary threshold
        
        return False
    
    async def _identify_methodological_differences(self, 
                                                 contrarian: ContrarianPaper, 
                                                 consensus: MainstreamConsensus) -> List[str]:
        """Identify methodological differences"""
        
        differences = []
        
        # Compare methodological approaches
        if contrarian.methodological_approach:
            # Extract method keywords
            method_keywords = contrarian.methodological_approach.lower().split()
            
            # Common methodological differences
            if any(word in method_keywords for word in ['qualitative', 'ethnographic', 'interpretive']):
                differences.append("Uses qualitative methods vs quantitative consensus")
            
            if any(word in method_keywords for word in ['longitudinal', 'long-term', 'extended']):
                differences.append("Uses longitudinal approach vs cross-sectional consensus")
            
            if any(word in method_keywords for word in ['experimental', 'controlled', 'randomized']):
                differences.append("Uses experimental methods vs observational consensus")
        
        return differences
    
    async def _identify_scope_differences(self, 
                                        contrarian: ContrarianPaper, 
                                        consensus: MainstreamConsensus) -> List[str]:
        """Identify differences in scope or boundaries"""
        
        differences = []
        
        # Check contrarian claims for scope limitations
        for claim in contrarian.contrarian_claims:
            claim_lower = claim.lower()
            
            if any(word in claim_lower for word in ['limited', 'only', 'specific', 'particular']):
                differences.append(f"Scope limitation: {claim}")
            
            if any(word in claim_lower for word in ['broader', 'wider', 'general', 'universal']):
                differences.append(f"Scope expansion: {claim}")
            
            if any(word in claim_lower for word in ['context', 'situational', 'conditional']):
                differences.append(f"Context dependency: {claim}")
        
        return differences[:3]  # Top 3 scope differences
    
    async def _calculate_contradiction_strength(self, 
                                              contrarian: ContrarianPaper, 
                                              consensus: MainstreamConsensus) -> float:
        """Calculate the strength of contradiction"""
        
        # Factors affecting contradiction strength
        factors = {
            'directness': 0.0,
            'evidence_quality': 0.0,
            'scope_coverage': 0.0,
            'credibility': 0.0
        }
        
        # Directness of contradiction
        if contrarian.contradiction_type == ContradictionType.DIRECT_REFUTATION:
            factors['directness'] = 1.0
        elif contrarian.contradiction_type == ContradictionType.PARADIGM_INVERSION:
            factors['directness'] = 0.9
        elif contrarian.contradiction_type in [ContradictionType.ALTERNATIVE_EXPLANATION, ContradictionType.ASSUMPTION_CHALLENGE]:
            factors['directness'] = 0.7
        else:
            factors['directness'] = 0.5
        
        # Evidence quality
        credibility_scores = {
            ContrarianCredibility.HIGH: 1.0,
            ContrarianCredibility.MEDIUM: 0.7,
            ContrarianCredibility.LOW: 0.4,
            ContrarianCredibility.SPECULATIVE: 0.2,
            ContrarianCredibility.DEBUNKED: 0.0
        }
        factors['evidence_quality'] = credibility_scores.get(contrarian.credibility_level, 0.5)
        
        # Scope coverage (how much of consensus is challenged)
        factors['scope_coverage'] = len(contrarian.contrarian_claims) / max(len(consensus.underlying_assumptions), 1)
        factors['scope_coverage'] = min(factors['scope_coverage'], 1.0)
        
        # Contrarian credibility
        factors['credibility'] = contrarian.paradigm_shift_score
        
        # Weighted average
        weights = {'directness': 0.3, 'evidence_quality': 0.3, 'scope_coverage': 0.2, 'credibility': 0.2}
        
        strength = sum(factors[factor] * weights[factor] for factor in factors)
        return min(max(strength, 0.0), 1.0)
    
    async def _identify_resolution_possibilities(self, 
                                               contrarian: ContrarianPaper, 
                                               consensus: MainstreamConsensus) -> List[str]:
        """Identify possible ways to resolve the contradiction"""
        
        resolutions = []
        
        # Based on contradiction type, suggest resolutions
        if contrarian.contradiction_type == ContradictionType.SCOPE_LIMITATION:
            resolutions.append("Define scope boundaries - both may be correct in different contexts")
        
        elif contrarian.contradiction_type == ContradictionType.ALTERNATIVE_EXPLANATION:
            resolutions.append("Test competing explanations empirically")
            resolutions.append("Look for higher-level framework that encompasses both")
        
        elif contrarian.contradiction_type == ContradictionType.METHOD_CRITIQUE:
            resolutions.append("Methodological improvement or standardization")
            resolutions.append("Cross-validation with multiple methods")
        
        elif contrarian.contradiction_type == ContradictionType.ASSUMPTION_CHALLENGE:
            resolutions.append("Make assumptions explicit and test them")
            resolutions.append("Develop assumption-free approaches")
        
        elif contrarian.contradiction_type == ContradictionType.EVIDENCE_REINTERPRETATION:
            resolutions.append("Gather additional evidence to distinguish interpretations")
            resolutions.append("Develop better theoretical frameworks")
        
        else:
            resolutions.append("Further research and replication")
            resolutions.append("Theoretical integration attempts")
        
        return resolutions
    
    async def _identify_synthesis_opportunities(self, 
                                              contrarian: ContrarianPaper, 
                                              consensus: MainstreamConsensus) -> List[str]:
        """Identify opportunities for synthesizing contrarian and mainstream views"""
        
        synthesis_ops = []
        
        # Look for complementary aspects
        if len(contrarian.contrarian_claims) > 0 and len(consensus.underlying_assumptions) > 0:
            synthesis_ops.append("Dialectical synthesis: Integrate thesis and antithesis into higher synthesis")
        
        # Context-dependent synthesis
        if any('context' in claim.lower() for claim in contrarian.contrarian_claims):
            synthesis_ops.append("Context-dependent integration: Both valid in different contexts")
        
        # Level-based synthesis  
        synthesis_ops.append("Multi-level integration: Different levels of analysis may both be valid")
        
        # Temporal synthesis
        synthesis_ops.append("Temporal integration: Different stages of development or evolution")
        
        # Complementarity synthesis
        synthesis_ops.append("Complementarity principle: Both perspectives needed for complete understanding")
        
        return synthesis_ops

class ContrarianEvidenceValidator:
    """Validates the quality and credibility of contrarian evidence"""
    
    def __init__(self):
        self.quality_indicators = {
            'methodology': {
                'high': ['randomized', 'controlled', 'double-blind', 'systematic', 'meta-analysis'],
                'medium': ['cohort', 'case-control', 'cross-sectional', 'survey'],
                'low': ['case study', 'anecdotal', 'opinion', 'commentary']
            },
            'publication': {
                'high': ['peer-reviewed', 'impact factor', 'prestigious journal', 'nature', 'science'],
                'medium': ['peer-reviewed', 'academic journal', 'conference'],
                'low': ['preprint', 'blog', 'website', 'self-published']
            },
            'replication': {
                'high': ['replicated', 'reproduced', 'validated', 'confirmed'],
                'medium': ['partial replication', 'similar findings', 'supported'],
                'low': ['unreplicated', 'contradicted', 'failed replication']
            }
        }
    
    async def validate_contrarian_evidence(self, contrarian_papers: List[ContrarianPaper]) -> List[ContrarianPaper]:
        """Validate evidence quality for contrarian papers"""
        
        validated_papers = []
        
        for paper in contrarian_papers:
            # Assess credibility
            credibility = await self._assess_credibility(paper)
            paper.credibility_level = credibility
            
            # Assess breakthrough potential
            breakthrough_potential = await self._assess_breakthrough_potential(paper)
            paper.breakthrough_potential = breakthrough_potential
            
            # Calculate paradigm shift score
            paradigm_score = await self._calculate_paradigm_shift_score(paper)
            paper.paradigm_shift_score = paradigm_score
            
            validated_papers.append(paper)
        
        return validated_papers
    
    async def _assess_credibility(self, paper: ContrarianPaper) -> ContrarianCredibility:
        """Assess the credibility level of a contrarian paper"""
        
        credibility_score = 0.0
        
        # Methodology assessment
        method_score = self._assess_methodology_quality(paper.methodological_approach)
        credibility_score += method_score * 0.4
        
        # Publication quality
        pub_score = self._assess_publication_quality(paper.journal)
        credibility_score += pub_score * 0.3
        
        # Evidence quality
        evidence_score = self._assess_evidence_quality(paper.evidence_presented)
        credibility_score += evidence_score * 0.3
        
        # Convert to credibility level
        if credibility_score >= 0.8:
            return ContrarianCredibility.HIGH
        elif credibility_score >= 0.6:
            return ContrarianCredibility.MEDIUM
        elif credibility_score >= 0.4:
            return ContrarianCredibility.LOW
        elif credibility_score >= 0.2:
            return ContrarianCredibility.SPECULATIVE
        else:
            return ContrarianCredibility.DEBUNKED
    
    def _assess_methodology_quality(self, methodology: str) -> float:
        """Assess methodology quality"""
        
        if not methodology:
            return 0.3  # Default/unknown
        
        method_lower = methodology.lower()
        
        for quality_level in ['high', 'medium', 'low']:
            indicators = self.quality_indicators['methodology'][quality_level]
            for indicator in indicators:
                if indicator in method_lower:
                    if quality_level == 'high':
                        return 1.0
                    elif quality_level == 'medium':
                        return 0.6
                    else:  # low
                        return 0.2
        
        return 0.3  # Default if no indicators found
    
    def _assess_publication_quality(self, journal: str) -> float:
        """Assess publication quality"""
        
        if not journal:
            return 0.3  # Default/unknown
        
        journal_lower = journal.lower()
        
        for quality_level in ['high', 'medium', 'low']:
            indicators = self.quality_indicators['publication'][quality_level]
            for indicator in indicators:
                if indicator in journal_lower:
                    if quality_level == 'high':
                        return 1.0
                    elif quality_level == 'medium':
                        return 0.6
                    else:  # low
                        return 0.2
        
        return 0.3  # Default if no indicators found
    
    def _assess_evidence_quality(self, evidence_list: List[str]) -> float:
        """Assess quality of evidence presented"""
        
        if not evidence_list:
            return 0.2  # No evidence
        
        total_score = 0.0
        
        for evidence in evidence_list:
            evidence_lower = evidence.lower()
            
            # Check for quality indicators
            for quality_level in ['high', 'medium', 'low']:
                indicators = self.quality_indicators['replication'][quality_level]
                for indicator in indicators:
                    if indicator in evidence_lower:
                        if quality_level == 'high':
                            total_score += 1.0
                        elif quality_level == 'medium':
                            total_score += 0.6
                        else:  # low
                            total_score += 0.2
                        break
                else:
                    continue
                break
            else:
                total_score += 0.3  # Default for unclassified evidence
        
        return min(total_score / len(evidence_list), 1.0)
    
    async def _assess_breakthrough_potential(self, paper: ContrarianPaper) -> BreakthroughPotential:
        """Assess breakthrough potential of contrarian paper"""
        
        potential_score = 0.0
        
        # Novelty of contrarian claims
        novelty_score = len(paper.contrarian_claims) / 10.0  # Assume more claims = more novel
        potential_score += min(novelty_score, 1.0) * 0.3
        
        # Scope of contradiction
        if paper.contradiction_type in [ContradictionType.PARADIGM_INVERSION, ContradictionType.ASSUMPTION_CHALLENGE]:
            potential_score += 0.4
        elif paper.contradiction_type in [ContradictionType.ALTERNATIVE_EXPLANATION, ContradictionType.DIRECT_REFUTATION]:
            potential_score += 0.3
        else:
            potential_score += 0.1
        
        # Evidence strength
        credibility_bonus = {
            ContrarianCredibility.HIGH: 0.3,
            ContrarianCredibility.MEDIUM: 0.2,
            ContrarianCredibility.LOW: 0.1,
            ContrarianCredibility.SPECULATIVE: 0.05,
            ContrarianCredibility.DEBUNKED: 0.0
        }
        potential_score += credibility_bonus.get(paper.credibility_level, 0.1)
        
        # Convert to breakthrough potential level
        if potential_score >= 0.9:
            return BreakthroughPotential.REVOLUTIONARY
        elif potential_score >= 0.7:
            return BreakthroughPotential.TRANSFORMATIVE
        elif potential_score >= 0.5:
            return BreakthroughPotential.SIGNIFICANT
        elif potential_score >= 0.3:
            return BreakthroughPotential.INCREMENTAL
        else:
            return BreakthroughPotential.MINIMAL
    
    async def _calculate_paradigm_shift_score(self, paper: ContrarianPaper) -> float:
        """Calculate numerical paradigm shift score"""
        
        score = 0.0
        
        # Base score from contradiction type
        type_scores = {
            ContradictionType.PARADIGM_INVERSION: 1.0,
            ContradictionType.ASSUMPTION_CHALLENGE: 0.8,
            ContradictionType.ALTERNATIVE_EXPLANATION: 0.6,
            ContradictionType.DIRECT_REFUTATION: 0.7,
            ContradictionType.METHOD_CRITIQUE: 0.4,
            ContradictionType.EVIDENCE_REINTERPRETATION: 0.5,
            ContradictionType.SCOPE_LIMITATION: 0.3,
            ContradictionType.BOUNDARY_REDEFINITION: 0.6
        }
        score += type_scores.get(paper.contradiction_type, 0.5) * 0.4
        
        # Credibility factor
        credibility_scores = {
            ContrarianCredibility.HIGH: 1.0,
            ContrarianCredibility.MEDIUM: 0.7,
            ContrarianCredibility.LOW: 0.4,
            ContrarianCredibility.SPECULATIVE: 0.2,
            ContrarianCredibility.DEBUNKED: 0.0
        }
        score += credibility_scores.get(paper.credibility_level, 0.5) * 0.3
        
        # Evidence quantity and quality
        evidence_factor = min(len(paper.evidence_presented) / 5.0, 1.0)  # Normalize to max 5 pieces
        score += evidence_factor * 0.2
        
        # Claims impact
        claims_factor = min(len(paper.contrarian_claims) / 3.0, 1.0)  # Normalize to max 3 claims
        score += claims_factor * 0.1
        
        return min(max(score, 0.0), 1.0)

class ContrarianKnowledgeIntegrator:
    """Integrates contrarian insights into reasoning processes"""
    
    def __init__(self):
        self.integration_strategies = {
            'direct_application': self._direct_application_integration,
            'dialectical_synthesis': self._dialectical_synthesis_integration,
            'context_dependent': self._context_dependent_integration,
            'complementarity': self._complementarity_integration,
            'evolutionary': self._evolutionary_integration
        }
    
    async def extract_contrarian_insights(self, 
                                        contrarian_papers: List[ContrarianPaper],
                                        contradiction_analyses: List[ContradictionAnalysis]) -> List[ContrarianInsight]:
        """Extract actionable insights from contrarian papers"""
        
        insights = []
        
        for paper in contrarian_papers:
            # Only process high-quality contrarian papers
            if paper.credibility_level in [ContrarianCredibility.HIGH, ContrarianCredibility.MEDIUM]:
                paper_insights = await self._extract_paper_insights(paper, contradiction_analyses)
                insights.extend(paper_insights)
        
        # Sort by breakthrough potential
        insights.sort(key=lambda i: i.novelty_score * i.applicability_score, reverse=True)
        return insights[:20]  # Top 20 insights
    
    async def _extract_paper_insights(self, 
                                    paper: ContrarianPaper,
                                    analyses: List[ContradictionAnalysis]) -> List[ContrarianInsight]:
        """Extract insights from a single contrarian paper"""
        
        insights = []
        
        # Find analyses for this paper
        paper_analyses = [a for a in analyses if a.contrarian_paper_id == paper.id]
        
        # Extract insights from contrarian claims
        for i, claim in enumerate(paper.contrarian_claims):
            insight = await self._create_insight_from_claim(claim, paper, paper_analyses)
            if insight:
                insights.append(insight)
        
        # Extract synthesis insights
        if paper_analyses:
            synthesis_insights = await self._create_synthesis_insights(paper, paper_analyses)
            insights.extend(synthesis_insights)
        
        return insights
    
    async def _create_insight_from_claim(self, 
                                       claim: str,
                                       paper: ContrarianPaper,
                                       analyses: List[ContradictionAnalysis]) -> ContrarianInsight:
        """Create insight from a contrarian claim"""
        
        # Determine insight type
        insight_type = self._classify_insight_type(claim, paper.contradiction_type)
        
        # Generate breakthrough applications
        applications = await self._generate_breakthrough_applications(claim, paper.domain)
        
        # Assess applicability
        applicable_domains = await self._identify_applicable_domains(claim, paper.domain)
        
        # Extract supporting evidence
        supporting_evidence = paper.evidence_presented[:3]  # Top 3 pieces of evidence
        
        # Identify limitations
        limitations = await self._identify_limitations(claim, analyses)
        
        # Calculate scores
        novelty_score = self._calculate_novelty_score(claim, paper)
        applicability_score = len(applicable_domains) / 5.0  # Normalize to max 5 domains
        confidence = paper.paradigm_shift_score * 0.7  # Discount for uncertainty
        
        insight = ContrarianInsight(
            source_paper_id=paper.id,
            insight_type=insight_type,
            insight_description=claim,
            applicable_domains=applicable_domains,
            breakthrough_applications=applications,
            supporting_evidence=supporting_evidence,
            limitations_and_caveats=limitations,
            novelty_score=novelty_score,
            applicability_score=min(applicability_score, 1.0),
            confidence_level=confidence
        )
        
        return insight
    
    def _classify_insight_type(self, claim: str, contradiction_type: ContradictionType) -> str:
        """Classify the type of insight"""
        
        claim_lower = claim.lower()
        
        if contradiction_type == ContradictionType.ASSUMPTION_CHALLENGE:
            return "assumption_challenge"
        elif contradiction_type == ContradictionType.ALTERNATIVE_EXPLANATION:
            return "alternative_explanation"
        elif contradiction_type == ContradictionType.PARADIGM_INVERSION:
            return "paradigm_shift"
        elif any(word in claim_lower for word in ['method', 'approach', 'technique']):
            return "methodological_insight"
        elif any(word in claim_lower for word in ['cause', 'effect', 'relationship']):
            return "causal_insight"
        elif any(word in claim_lower for word in ['boundary', 'limit', 'scope']):
            return "boundary_insight"
        else:
            return "general_insight"
    
    async def _generate_breakthrough_applications(self, claim: str, domain: str) -> List[str]:
        """Generate potential breakthrough applications of the insight"""
        
        applications = []
        claim_lower = claim.lower()
        
        # Domain-specific applications
        if domain == 'machine_learning':
            if 'learning' in claim_lower:
                applications.append("Novel machine learning algorithms based on contrarian learning principles")
            if 'data' in claim_lower:
                applications.append("Alternative data processing approaches challenging conventional wisdom")
        elif domain == 'physics':
            if 'energy' in claim_lower:
                applications.append("Alternative energy systems based on contrarian physical principles")
            if 'matter' in claim_lower:
                applications.append("New materials science approaches challenging conventional matter theory")
        elif domain == 'biology':
            if 'evolution' in claim_lower:
                applications.append("Alternative evolutionary mechanisms for artificial life systems")
            if 'gene' in claim_lower:
                applications.append("Novel genetic engineering approaches based on contrarian genetic theory")
        
        # General applications
        applications.extend([
            f"Contrarian approach to {domain} challenging mainstream assumptions",
            f"Alternative {domain} methodologies based on contrarian insights",
            f"Paradigm shift opportunities in {domain} through contrarian thinking"
        ])
        
        return applications[:5]  # Top 5 applications
    
    async def _identify_applicable_domains(self, claim: str, source_domain: str) -> List[str]:
        """Identify domains where the insight might be applicable"""
        
        domains = [source_domain]  # Always applicable to source domain
        claim_lower = claim.lower()
        
        # Cross-domain applicability based on keywords
        domain_keywords = {
            'machine_learning': ['learning', 'pattern', 'algorithm', 'data', 'prediction'],
            'physics': ['energy', 'matter', 'force', 'field', 'particle'],
            'biology': ['evolution', 'organism', 'gene', 'cell', 'adaptation'],
            'psychology': ['behavior', 'cognition', 'learning', 'memory', 'perception'],
            'economics': ['market', 'decision', 'value', 'trade', 'optimization'],
            'social_science': ['society', 'culture', 'group', 'interaction', 'behavior'],
            'philosophy': ['knowledge', 'reality', 'truth', 'logic', 'existence']
        }
        
        for domain, keywords in domain_keywords.items():
            if domain != source_domain:  # Don't double-add source domain
                if any(keyword in claim_lower for keyword in keywords):
                    domains.append(domain)
        
        return domains[:5]  # Max 5 domains
    
    async def _identify_limitations(self, 
                                  claim: str, 
                                  analyses: List[ContradictionAnalysis]) -> List[str]:
        """Identify limitations and caveats of the contrarian insight"""
        
        limitations = []
        
        # From contradiction analyses
        for analysis in analyses:
            if analysis.strength_of_contradiction < 0.6:
                limitations.append("Moderate contradiction strength - may not fully invalidate consensus")
            
            if analysis.evidence_comparison.get('evidence_overlap', 0) < 0.3:
                limitations.append("Limited evidence overlap with consensus - may address different aspects")
            
            # Add scope limitations
            limitations.extend(analysis.scope_differences[:2])
        
        # General limitations
        claim_lower = claim.lower()
        if any(word in claim_lower for word in ['preliminary', 'initial', 'exploratory']):
            limitations.append("Preliminary findings - requires further validation")
        
        if any(word in claim_lower for word in ['limited', 'specific', 'particular']):
            limitations.append("Limited scope - may not generalize broadly")
        
        return limitations[:3]  # Top 3 limitations
    
    def _calculate_novelty_score(self, claim: str, paper: ContrarianPaper) -> float:
        """Calculate novelty score for the insight"""
        
        score = 0.0
        
        # Base score from paradigm shift score
        score += paper.paradigm_shift_score * 0.4
        
        # Contradiction type factor
        type_novelty = {
            ContradictionType.PARADIGM_INVERSION: 1.0,
            ContradictionType.ASSUMPTION_CHALLENGE: 0.8,
            ContradictionType.ALTERNATIVE_EXPLANATION: 0.7,
            ContradictionType.DIRECT_REFUTATION: 0.6,
            ContradictionType.BOUNDARY_REDEFINITION: 0.6,
            ContradictionType.EVIDENCE_REINTERPRETATION: 0.5,
            ContradictionType.METHOD_CRITIQUE: 0.4,
            ContradictionType.SCOPE_LIMITATION: 0.3
        }
        score += type_novelty.get(paper.contradiction_type, 0.5) * 0.3
        
        # Breakthrough potential factor
        potential_novelty = {
            BreakthroughPotential.REVOLUTIONARY: 1.0,
            BreakthroughPotential.TRANSFORMATIVE: 0.8,
            BreakthroughPotential.SIGNIFICANT: 0.6,
            BreakthroughPotential.INCREMENTAL: 0.4,
            BreakthroughPotential.MINIMAL: 0.2
        }
        score += potential_novelty.get(paper.breakthrough_potential, 0.5) * 0.3
        
        return min(max(score, 0.0), 1.0)
    
    async def _create_synthesis_insights(self, 
                                       paper: ContrarianPaper,
                                       analyses: List[ContradictionAnalysis]) -> List[ContrarianInsight]:
        """Create insights from synthesis opportunities"""
        
        insights = []
        
        for analysis in analyses:
            for synthesis_op in analysis.synthesis_opportunities[:2]:  # Top 2 synthesis opportunities
                insight = ContrarianInsight(
                    source_paper_id=paper.id,
                    insight_type="synthesis_opportunity",
                    insight_description=synthesis_op,
                    applicable_domains=[paper.domain],
                    breakthrough_applications=[f"Synthesis approach: {synthesis_op}"],
                    supporting_evidence=paper.evidence_presented[:2],
                    novelty_score=0.6,  # Moderate novelty for synthesis
                    applicability_score=0.8,  # High applicability
                    confidence_level=0.7
                )
                insights.append(insight)
        
        return insights

class ContrarianPaperIdentificationEngine:
    """Main engine for identifying and analyzing contrarian papers"""
    
    def __init__(self):
        self.consensus_detector = MainstreamConsensusDetector()
        self.contradiction_analyzer = ContradictionAnalyzer()
        self.evidence_validator = ContrarianEvidenceValidator()
        self.knowledge_integrator = ContrarianKnowledgeIntegrator()
    
    async def identify_contrarian_papers(self, 
                                       papers: List[Dict[str, Any]], 
                                       query: str,
                                       domain: str) -> ContrarianIdentificationResult:
        """Complete contrarian paper identification and analysis"""
        
        start_time = time.time()
        
        try:
            # 1. Detect mainstream consensus positions
            consensus_positions = await self.consensus_detector.detect_consensus(papers, domain)
            
            # 2. Identify papers that contradict consensus
            contrarian_candidates = await self._identify_contrarian_candidates(papers, consensus_positions, domain)
            
            # 3. Validate contrarian evidence
            validated_contrarians = await self.evidence_validator.validate_contrarian_evidence(contrarian_candidates)
            
            # 4. Analyze contradictions
            contradiction_analyses = await self.contradiction_analyzer.analyze_contradictions(
                validated_contrarians, consensus_positions
            )
            
            # 5. Extract contrarian insights
            contrarian_insights = await self.knowledge_integrator.extract_contrarian_insights(
                validated_contrarians, contradiction_analyses
            )
            
            # 6. Calculate summary metrics
            consensus_challenge_score = self._calculate_consensus_challenge_score(
                validated_contrarians, consensus_positions
            )
            
            paradigm_disruption_potential = self._calculate_paradigm_disruption_potential(
                validated_contrarians
            )
            
            breakthrough_opportunity_score = self._calculate_breakthrough_opportunity_score(
                contrarian_insights
            )
            
            evidence_quality = self._calculate_evidence_quality_score(validated_contrarians)
            
            # 7. Generate summary and comparison
            contrarian_summary = await self._generate_contrarian_summary(
                validated_contrarians, consensus_positions, contrarian_insights
            )
            
            comparison = await self._generate_mainstream_vs_contrarian_comparison(
                consensus_positions, validated_contrarians
            )
            
            processing_time = time.time() - start_time
            
            return ContrarianIdentificationResult(
                query=query,
                domain=domain,
                mainstream_consensus=consensus_positions,
                contrarian_papers=validated_contrarians,
                contradiction_analyses=contradiction_analyses,
                contrarian_insights=contrarian_insights,
                consensus_challenge_score=consensus_challenge_score,
                paradigm_disruption_potential=paradigm_disruption_potential,
                breakthrough_opportunity_score=breakthrough_opportunity_score,
                contrarian_evidence_quality=evidence_quality,
                processing_time=processing_time,
                contrarian_summary=contrarian_summary,
                mainstream_vs_contrarian_comparison=comparison
            )
            
        except Exception as e:
            logger.error(f"Contrarian paper identification failed: {e}")
            
            return ContrarianIdentificationResult(
                query=query,
                domain=domain,
                processing_time=time.time() - start_time,
                contrarian_summary=f"Error in contrarian analysis: {e}"
            )
    
    async def _identify_contrarian_candidates(self, 
                                            papers: List[Dict[str, Any]], 
                                            consensus_positions: List[MainstreamConsensus],
                                            domain: str) -> List[ContrarianPaper]:
        """Identify papers that potentially contradict consensus"""
        
        contrarian_candidates = []
        
        for paper in papers:
            # Check if paper contradicts any consensus position
            contradictions = await self._check_paper_contradictions(paper, consensus_positions)
            
            if contradictions:
                contrarian = await self._create_contrarian_paper(paper, contradictions, domain)
                contrarian_candidates.append(contrarian)
        
        return contrarian_candidates
    
    async def _check_paper_contradictions(self, 
                                        paper: Dict[str, Any], 
                                        consensus_positions: List[MainstreamConsensus]) -> List[str]:
        """Check if paper contradicts consensus positions"""
        
        contradictions = []
        paper_text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        
        # Look for contradiction indicators
        contradiction_words = [
            'not', 'no', 'never', 'false', 'incorrect', 'wrong', 'refute', 'disprove',
            'challenge', 'question', 'doubt', 'dispute', 'contrary', 'opposite',
            'however', 'but', 'nevertheless', 'although', 'despite'
        ]
        
        has_contradiction_indicators = any(word in paper_text for word in contradiction_words)
        
        if has_contradiction_indicators:
            for consensus in consensus_positions:
                if self._paper_contradicts_consensus(paper_text, consensus):
                    contradictions.append(consensus.id)
        
        return contradictions
    
    def _paper_contradicts_consensus(self, paper_text: str, consensus: MainstreamConsensus) -> bool:
        """Check if paper text contradicts a specific consensus"""
        
        consensus_keywords = set(consensus.consensus_statement.lower().split())
        paper_keywords = set(paper_text.split())
        
        # Check for keyword overlap (paper discusses same topic)
        overlap = len(consensus_keywords.intersection(paper_keywords))
        discusses_topic = overlap > 2  # Arbitrary threshold
        
        if not discusses_topic:
            return False
        
        # Check for contradiction patterns
        contradiction_patterns = [
            f"not {consensus.consensus_statement.lower()}",
            f"challenge {consensus.consensus_statement.lower()}",
            f"contrary to {consensus.consensus_statement.lower()}",
            "alternative explanation",
            "different interpretation"
        ]
        
        return any(pattern in paper_text for pattern in contradiction_patterns)
    
    async def _create_contrarian_paper(self, 
                                     paper: Dict[str, Any], 
                                     contradicted_consensus: List[str],
                                     domain: str) -> ContrarianPaper:
        """Create ContrarianPaper object from paper data"""
        
        # Extract contrarian claims
        contrarian_claims = await self._extract_contrarian_claims(paper)
        
        # Extract evidence
        evidence = await self._extract_evidence(paper)
        
        # Classify contradiction type
        contradiction_type = await self._classify_contradiction_type(paper, contrarian_claims)
        
        # Extract methodological approach
        methodology = self._extract_methodology(paper)
        
        contrarian = ContrarianPaper(
            paper_id=paper.get('id', ''),
            title=paper.get('title', ''),
            authors=paper.get('authors', []),
            publication_year=paper.get('year', 0),
            journal=paper.get('journal', ''),
            domain=domain,
            contradicted_consensus=contradicted_consensus,
            contradiction_type=contradiction_type,
            contrarian_claims=contrarian_claims,
            evidence_presented=evidence,
            methodological_approach=methodology
        )
        
        return contrarian
    
    async def _extract_contrarian_claims(self, paper: Dict[str, Any]) -> List[str]:
        """Extract contrarian claims from paper"""
        
        claims = []
        
        # Look in title for contrarian claims
        title = paper.get('title', '')
        if any(word in title.lower() for word in ['not', 'challenge', 'alternative', 'contrary']):
            claims.append(title)
        
        # Look in abstract for contrarian statements
        abstract = paper.get('abstract', '')
        if abstract:
            # Simple sentence splitting
            sentences = abstract.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['not', 'however', 'challenge', 'alternative', 'contrary']):
                    claims.append(sentence.strip())
        
        return claims[:5]  # Top 5 claims
    
    async def _extract_evidence(self, paper: Dict[str, Any]) -> List[str]:
        """Extract evidence presented in paper"""
        
        evidence = []
        
        # Look for evidence keywords in abstract
        abstract = paper.get('abstract', '')
        evidence_keywords = [
            'experiment', 'study', 'analysis', 'data', 'result', 'finding',
            'observation', 'measurement', 'test', 'trial', 'survey'
        ]
        
        if abstract:
            sentences = abstract.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in evidence_keywords):
                    evidence.append(sentence.strip())
        
        return evidence[:3]  # Top 3 pieces of evidence
    
    async def _classify_contradiction_type(self, 
                                         paper: Dict[str, Any], 
                                         claims: List[str]) -> ContradictionType:
        """Classify the type of contradiction"""
        
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        claims_text = ' '.join(claims).lower()
        
        # Check for different contradiction patterns
        if any(word in text for word in ['not', 'false', 'incorrect', 'refute', 'disprove']):
            return ContradictionType.DIRECT_REFUTATION
        
        elif any(phrase in text for phrase in ['alternative explanation', 'different interpretation']):
            return ContradictionType.ALTERNATIVE_EXPLANATION
        
        elif any(word in text for word in ['assume', 'assumption', 'presuppose']):
            return ContradictionType.ASSUMPTION_CHALLENGE
        
        elif any(word in text for word in ['method', 'approach', 'methodology']):
            return ContradictionType.METHOD_CRITIQUE
        
        elif any(word in text for word in ['limited', 'scope', 'boundary']):
            return ContradictionType.SCOPE_LIMITATION
        
        elif any(phrase in text for phrase in ['reinterpret', 'different view', 'alternative view']):
            return ContradictionType.EVIDENCE_REINTERPRETATION
        
        elif any(word in claims_text for word in ['paradigm', 'fundamental', 'basic']):
            return ContradictionType.PARADIGM_INVERSION
        
        else:
            return ContradictionType.DIRECT_REFUTATION  # Default
    
    def _extract_methodology(self, paper: Dict[str, Any]) -> str:
        """Extract methodological approach from paper"""
        
        abstract = paper.get('abstract', '')
        
        # Look for methodology keywords
        method_keywords = [
            'experiment', 'survey', 'interview', 'observation', 'analysis',
            'simulation', 'model', 'statistical', 'qualitative', 'quantitative',
            'longitudinal', 'cross-sectional', 'randomized', 'controlled'
        ]
        
        methods_found = []
        for keyword in method_keywords:
            if keyword in abstract.lower():
                methods_found.append(keyword)
        
        return ', '.join(methods_found) if methods_found else 'unspecified'
    
    def _calculate_consensus_challenge_score(self, 
                                           contrarian_papers: List[ContrarianPaper],
                                           consensus_positions: List[MainstreamConsensus]) -> float:
        """Calculate how much consensus is challenged"""
        
        if not consensus_positions:
            return 0.0
        
        challenged_consensus = set()
        for paper in contrarian_papers:
            challenged_consensus.update(paper.contradicted_consensus)
        
        return len(challenged_consensus) / len(consensus_positions)
    
    def _calculate_paradigm_disruption_potential(self, contrarian_papers: List[ContrarianPaper]) -> float:
        """Calculate potential for paradigm disruption"""
        
        if not contrarian_papers:
            return 0.0
        
        total_potential = sum(paper.paradigm_shift_score for paper in contrarian_papers)
        return total_potential / len(contrarian_papers)
    
    def _calculate_breakthrough_opportunity_score(self, insights: List[ContrarianInsight]) -> float:
        """Calculate breakthrough opportunity score from insights"""
        
        if not insights:
            return 0.0
        
        total_opportunity = sum(insight.novelty_score * insight.applicability_score for insight in insights)
        return total_opportunity / len(insights)
    
    def _calculate_evidence_quality_score(self, contrarian_papers: List[ContrarianPaper]) -> float:
        """Calculate overall evidence quality score"""
        
        if not contrarian_papers:
            return 0.0
        
        quality_scores = {
            ContrarianCredibility.HIGH: 1.0,
            ContrarianCredibility.MEDIUM: 0.7,
            ContrarianCredibility.LOW: 0.4,
            ContrarianCredibility.SPECULATIVE: 0.2,
            ContrarianCredibility.DEBUNKED: 0.0
        }
        
        total_quality = sum(quality_scores.get(paper.credibility_level, 0.5) for paper in contrarian_papers)
        return total_quality / len(contrarian_papers)
    
    async def _generate_contrarian_summary(self, 
                                         contrarian_papers: List[ContrarianPaper],
                                         consensus_positions: List[MainstreamConsensus],
                                         insights: List[ContrarianInsight]) -> str:
        """Generate summary of contrarian analysis"""
        
        if not contrarian_papers:
            return "No significant contrarian papers identified challenging mainstream consensus."
        
        summary_parts = []
        
        # High-level summary
        high_credibility = sum(1 for p in contrarian_papers if p.credibility_level == ContrarianCredibility.HIGH)
        medium_credibility = sum(1 for p in contrarian_papers if p.credibility_level == ContrarianCredibility.MEDIUM)
        
        summary_parts.append(f"Identified {len(contrarian_papers)} contrarian papers: {high_credibility} high credibility, {medium_credibility} medium credibility.")
        
        # Consensus challenge summary
        challenged_count = len(set(consensus_id for paper in contrarian_papers for consensus_id in paper.contradicted_consensus))
        summary_parts.append(f"These papers challenge {challenged_count} of {len(consensus_positions)} identified consensus positions.")
        
        # Breakthrough potential
        revolutionary_count = sum(1 for p in contrarian_papers if p.breakthrough_potential == BreakthroughPotential.REVOLUTIONARY)
        transformative_count = sum(1 for p in contrarian_papers if p.breakthrough_potential == BreakthroughPotential.TRANSFORMATIVE)
        
        if revolutionary_count > 0:
            summary_parts.append(f"{revolutionary_count} papers have revolutionary breakthrough potential.")
        if transformative_count > 0:
            summary_parts.append(f"{transformative_count} papers have transformative breakthrough potential.")
        
        # Top insights
        if insights:
            top_insight = max(insights, key=lambda i: i.novelty_score * i.applicability_score)
            summary_parts.append(f"Top contrarian insight: {top_insight.insight_description}")
        
        return " ".join(summary_parts)
    
    async def _generate_mainstream_vs_contrarian_comparison(self, 
                                                          consensus_positions: List[MainstreamConsensus],
                                                          contrarian_papers: List[ContrarianPaper]) -> Dict[str, Any]:
        """Generate comparison between mainstream and contrarian perspectives"""
        
        return {
            'consensus_count': len(consensus_positions),
            'contrarian_count': len(contrarian_papers),
            'consensus_strength_avg': statistics.mean([c.consensus_strength for c in consensus_positions]) if consensus_positions else 0,
            'contrarian_paradigm_shift_avg': statistics.mean([p.paradigm_shift_score for p in contrarian_papers]) if contrarian_papers else 0,
            'contradiction_types': Counter([p.contradiction_type.value for p in contrarian_papers]),
            'credibility_distribution': Counter([p.credibility_level.value for p in contrarian_papers]),
            'breakthrough_potential_distribution': Counter([p.breakthrough_potential.value for p in contrarian_papers]),
            'domains_challenged': list(set([p.domain for p in contrarian_papers])),
            'methodology_comparison': {
                'consensus_methods': 'established methodologies',
                'contrarian_methods': list(set([p.methodological_approach for p in contrarian_papers if p.methodological_approach != 'unspecified']))
            }
        }

# Integration with meta-reasoning engine
async def contrarian_paper_identification_integration(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Integration function for meta-reasoning engine"""
    
    engine = ContrarianPaperIdentificationEngine()
    
    # Extract papers and domain from context
    papers = context.get('papers', context.get('external_papers', []))
    domain = context.get('domain', 'general')
    
    if not papers:
        return {
            'conclusion': 'No papers provided for contrarian analysis',
            'confidence': 0.1,
            'evidence': [],
            'reasoning_chain': ['No paper corpus available for contrarian identification'],
            'processing_time': 0.0,
            'quality_score': 0.0,
            'contrarian_papers_found': 0,
            'consensus_positions_identified': 0
        }
    
    result = await engine.identify_contrarian_papers(papers, query, domain)
    
    return {
        'conclusion': result.contrarian_summary,
        'confidence': result.contrarian_evidence_quality,
        'evidence': [f"Consensus challenge score: {result.consensus_challenge_score:.2f}",
                    f"Paradigm disruption potential: {result.paradigm_disruption_potential:.2f}",
                    f"Breakthrough opportunity score: {result.breakthrough_opportunity_score:.2f}"],
        'reasoning_chain': [
            f"Identified {len(result.mainstream_consensus)} mainstream consensus positions",
            f"Found {len(result.contrarian_papers)} contrarian papers challenging consensus",
            f"Generated {len(result.contrarian_insights)} actionable contrarian insights",
            result.contrarian_summary
        ],
        'processing_time': result.processing_time,
        'quality_score': result.breakthrough_opportunity_score,
        'contrarian_papers_found': len(result.contrarian_papers),
        'consensus_positions_identified': len(result.mainstream_consensus),
        'contrarian_insights_count': len(result.contrarian_insights),
        'metadata': {
            'consensus_challenge_score': result.consensus_challenge_score,
            'paradigm_disruption_potential': result.paradigm_disruption_potential,
            'breakthrough_opportunity_score': result.breakthrough_opportunity_score,
            'mainstream_vs_contrarian_comparison': result.mainstream_vs_contrarian_comparison
        }
    }

if __name__ == "__main__":
    # Example usage and testing
    async def test_contrarian_paper_identification():
        """Test the contrarian paper identification engine"""
        
        engine = ContrarianPaperIdentificationEngine()
        
        # Mock paper data for testing
        mock_papers = [
            {
                'id': 'paper_1',
                'title': 'Traditional Machine Learning Approaches Work Best',
                'abstract': 'Our study shows that traditional machine learning methods consistently outperform newer approaches. We found strong evidence supporting established methodologies.',
                'authors': ['Smith, J.', 'Johnson, M.'],
                'year': 2023,
                'journal': 'Journal of AI Research',
                'domain': 'machine_learning'
            },
            {
                'id': 'paper_2', 
                'title': 'Deep Learning is Not Always the Answer: A Contrarian View',
                'abstract': 'Contrary to popular belief, our experiments demonstrate that deep learning approaches often fail where simpler methods succeed. We challenge the assumption that complexity equals effectiveness.',
                'authors': ['Wilson, K.', 'Brown, L.'],
                'year': 2023,
                'journal': 'Alternative AI Perspectives',
                'domain': 'machine_learning'
            },
            {
                'id': 'paper_3',
                'title': 'Established Consensus in Machine Learning',
                'abstract': 'The widely accepted view in machine learning is that more data and deeper networks lead to better performance. This established finding has been replicated across studies.',
                'authors': ['Davis, R.', 'Miller, S.'],
                'year': 2022,
                'journal': 'ML Consensus',
                'domain': 'machine_learning'
            }
        ]
        
        query = "What are the best approaches in machine learning?"
        domain = "machine_learning"
        
        print(f"Testing contrarian paper identification for query: '{query}'")
        print(f"Domain: {domain}")
        print(f"Number of papers: {len(mock_papers)}")
        
        result = await engine.identify_contrarian_papers(mock_papers, query, domain)
        
        print(f"\n=== RESULTS ===")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"\nMainstream consensus positions identified: {len(result.mainstream_consensus)}")
        for i, consensus in enumerate(result.mainstream_consensus):
            print(f"  {i+1}. {consensus.consensus_statement} (strength: {consensus.consensus_strength:.2f})")
        
        print(f"\nContrarian papers identified: {len(result.contrarian_papers)}")
        for i, paper in enumerate(result.contrarian_papers):
            print(f"  {i+1}. {paper.title}")
            print(f"     Contradiction type: {paper.contradiction_type.value}")
            print(f"     Credibility: {paper.credibility_level.value}")
            print(f"     Breakthrough potential: {paper.breakthrough_potential.value}")
            print(f"     Paradigm shift score: {paper.paradigm_shift_score:.2f}")
        
        print(f"\nContrarian insights extracted: {len(result.contrarian_insights)}")
        for i, insight in enumerate(result.contrarian_insights[:3]):  # Top 3
            print(f"  {i+1}. {insight.insight_description}")
            print(f"     Type: {insight.insight_type}")
            print(f"     Novelty: {insight.novelty_score:.2f}, Applicability: {insight.applicability_score:.2f}")
        
        print(f"\nSummary Metrics:")
        print(f"  Consensus challenge score: {result.consensus_challenge_score:.2f}")
        print(f"  Paradigm disruption potential: {result.paradigm_disruption_potential:.2f}")
        print(f"  Breakthrough opportunity score: {result.breakthrough_opportunity_score:.2f}")
        print(f"  Evidence quality score: {result.contrarian_evidence_quality:.2f}")
        
        print(f"\nContrarian Summary:")
        print(f"  {result.contrarian_summary}")
    
    # Uncomment to test
    # asyncio.run(test_contrarian_paper_identification())