"""
NWTN Contrarian Candidate Generation Engine
===========================================

Implements Phase 2.1 of the Novel Idea Generation Roadmap:
Generates reasoning candidates that explicitly contradict consensus thinking.

Based on NWTN Novel Idea Generation Roadmap Phase 2.1.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import random

logger = logging.getLogger(__name__)

class ContrarianType(Enum):
    """Types of contrarian reasoning approaches"""
    DIRECT_OPPOSITION = "direct_opposition"          # Direct contradiction of consensus
    ALTERNATIVE_INTERPRETATION = "alternative_interpretation"  # Different way to interpret evidence  
    ASSUMPTION_CHALLENGE = "assumption_challenge"    # Question underlying assumptions
    REVERSE_CAUSATION = "reverse_causation"         # Flip cause-effect relationships
    TEMPORAL_INVERSION = "temporal_inversion"       # Challenge timing assumptions
    SCOPE_REFRAMING = "scope_reframing"            # Reframe the problem scope

@dataclass
class ContrarianCandidate:
    """A contrarian reasoning candidate that opposes consensus"""
    candidate_id: str
    contrarian_type: ContrarianType
    
    # Core contrarian content
    consensus_position: str                 # What the consensus believes
    contrarian_position: str               # Our opposing position
    contrarian_reasoning: str              # Why the opposition might be valid
    
    # Supporting evidence
    supporting_evidence: List[str]         # Evidence that might support contrarian view
    potential_weaknesses: List[str]        # Where contrarian view might be weak
    
    # Metadata
    confidence_score: float               # How confident we are in this contrarian view
    novelty_score: float                 # How novel/surprising this opposition is
    testability_score: float             # How testable this contrarian position is
    
    # Context
    domain: str                          # Domain this applies to
    keywords: List[str]                  # Key terms and concepts
    generated_from_papers: List[str]     # Papers that inspired this contrarian view

class ContrarianCandidateEngine:
    """
    Generate reasoning candidates that systematically oppose consensus thinking.
    
    This engine looks for opportunities to challenge established views and
    generate novel perspectives by taking opposing positions.
    """
    
    def __init__(self):
        self.consensus_patterns = self._initialize_consensus_patterns()
        self.contrarian_templates = self._initialize_contrarian_templates()
        self.opposition_strategies = self._initialize_opposition_strategies()
        
    def _initialize_consensus_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns that indicate consensus thinking"""
        return {
            # Consensus language indicators
            "universal_claims": [
                r"all .+ are",
                r"every .+ must",
                r"always .+ when",
                r"never .+ unless",
                r"universally .+ that"
            ],
            
            # Established beliefs
            "established_facts": [
                r"it is well[- ]known that",
                r"established .+ shows?",
                r"proven .+ demonstrates?",
                r"consensus .+ indicates?",
                r"widely accepted .+ suggests?"
            ],
            
            # Causal certainty
            "causal_certainty": [
                r".+ causes? .+",
                r".+ leads? to .+",
                r".+ results? in .+",
                r"due to .+",
                r"because of .+"
            ],
            
            # Temporal assumptions
            "temporal_assumptions": [
                r"first .+ then .+",
                r"before .+ after .+",
                r"precedes? .+",
                r"follows? .+"
            ]
        }
    
    def _initialize_contrarian_templates(self) -> Dict[ContrarianType, List[str]]:
        """Initialize templates for generating contrarian positions"""
        return {
            ContrarianType.DIRECT_OPPOSITION: [
                "What if the opposite is true: instead of {consensus}, consider that {opposite}",
                "Contrary to conventional wisdom that {consensus}, evidence suggests {opposite}",
                "The established view {consensus} may be wrong - {opposite} could be the reality"
            ],
            
            ContrarianType.ALTERNATIVE_INTERPRETATION: [
                "While most interpret {evidence} as supporting {consensus}, it could actually indicate {alternative}",
                "The same data showing {consensus} might be better explained by {alternative}",
                "Rather than proving {consensus}, the evidence for {alternative} may be stronger"
            ],
            
            ContrarianType.ASSUMPTION_CHALLENGE: [
                "The fundamental assumption behind {consensus} - that {assumption} - may be flawed",
                "What if we question the basic premise that {assumption} underlying {consensus}",
                "Challenging the core assumption {assumption} reveals problems with {consensus}"
            ],
            
            ContrarianType.REVERSE_CAUSATION: [
                "Instead of {cause} causing {effect}, what if {effect} actually causes {cause}",
                "The causal relationship may be reversed: {effect} leads to {cause}, not vice versa",
                "Contrary to believing {cause} → {effect}, consider {effect} → {cause}"
            ],
            
            ContrarianType.TEMPORAL_INVERSION: [
                "What if {later_event} actually happens before {earlier_event}, not after",
                "The temporal sequence may be wrong: {effect} precedes {cause}",
                "Challenging the timeline: {conclusion} may come before {premise}"
            ],
            
            ContrarianType.SCOPE_REFRAMING: [
                "Instead of viewing this as a {narrow_scope} problem, it's actually about {broader_scope}",
                "The real issue isn't {surface_problem} but the deeper {underlying_problem}",
                "Reframing from {current_frame} to {alternative_frame} reveals new possibilities"
            ]
        }
    
    def _initialize_opposition_strategies(self) -> List[Dict[str, Any]]:
        """Initialize strategies for systematic opposition generation"""
        return [
            {
                "name": "Polarity Reversal",
                "description": "Flip positive/negative, good/bad, beneficial/harmful",
                "keywords": ["beneficial", "harmful", "positive", "negative", "good", "bad"]
            },
            {
                "name": "Causal Inversion", 
                "description": "Reverse cause and effect relationships",
                "keywords": ["cause", "effect", "leads to", "results in", "due to"]
            },
            {
                "name": "Scope Expansion",
                "description": "Challenge narrow framing with broader perspective",
                "keywords": ["specific", "particular", "individual", "local", "limited"]
            },
            {
                "name": "Temporal Challenge",
                "description": "Question timing and sequence assumptions",
                "keywords": ["first", "then", "before", "after", "precedes", "follows"]
            },
            {
                "name": "Quantity Inversion",
                "description": "Challenge more/less, bigger/smaller assumptions",
                "keywords": ["more", "less", "larger", "smaller", "increase", "decrease"]
            },
            {
                "name": "Category Disruption",
                "description": "Challenge classification and grouping assumptions",
                "keywords": ["type", "category", "class", "group", "kind", "species"]
            }
        ]
    
    async def generate_contrarian_candidates(
        self, 
        query: str, 
        context: Dict[str, Any],
        papers: List[Dict[str, Any]] = None,
        max_candidates: int = 5
    ) -> List[ContrarianCandidate]:
        """
        Generate contrarian candidates that oppose consensus thinking
        
        Args:
            query: Research question or topic
            context: Breakthrough mode and other context
            papers: Retrieved papers to analyze for consensus
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            List of contrarian candidates
        """
        try:
            logger.info(f"Generating contrarian candidates for query: {query[:50]}...")
            
            # Extract consensus positions from papers and query
            consensus_positions = await self._extract_consensus_positions(query, papers or [])
            
            # Generate contrarian candidates
            candidates = []
            
            for i, consensus in enumerate(consensus_positions[:max_candidates]):
                # Determine contrarian type based on consensus content
                contrarian_type = self._determine_contrarian_type(consensus)
                
                # Generate contrarian position
                contrarian_candidate = await self._generate_single_contrarian_candidate(
                    consensus, contrarian_type, query, context, papers or []
                )
                
                if contrarian_candidate:
                    candidates.append(contrarian_candidate)
            
            # If we don't have enough from papers, generate synthetic ones
            while len(candidates) < max_candidates:
                synthetic_candidate = await self._generate_synthetic_contrarian_candidate(
                    query, context, len(candidates)
                )
                if synthetic_candidate:
                    candidates.append(synthetic_candidate)
                else:
                    break
            
            # Sort by novelty and confidence
            candidates.sort(key=lambda c: (c.novelty_score + c.confidence_score) / 2, reverse=True)
            
            logger.info(f"Generated {len(candidates)} contrarian candidates")
            return candidates[:max_candidates]
            
        except Exception as e:
            logger.error(f"Failed to generate contrarian candidates: {e}")
            return []
    
    async def _extract_consensus_positions(self, query: str, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract consensus positions from papers and query"""
        consensus_positions = []
        
        # Extract from query
        query_consensus = self._extract_consensus_from_text(query)
        consensus_positions.extend(query_consensus)
        
        # Extract from papers
        for paper in papers[:10]:  # Limit to avoid overwhelming
            paper_text = paper.get('content', '') or paper.get('abstract', '')
            if paper_text:
                paper_consensus = self._extract_consensus_from_text(paper_text)
                consensus_positions.extend(paper_consensus)
        
        # Remove duplicates and return top positions
        unique_positions = list(set(consensus_positions))
        return unique_positions[:10]
    
    def _extract_consensus_from_text(self, text: str) -> List[str]:
        """Extract statements that represent consensus thinking"""
        consensus_statements = []
        
        for pattern_type, patterns in self.consensus_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract a sentence around the match
                    start = max(0, text.rfind('.', 0, match.start()) + 1)
                    end = text.find('.', match.end())
                    if end == -1:
                        end = len(text)
                    
                    statement = text[start:end].strip()
                    if len(statement) > 20 and len(statement) < 200:
                        consensus_statements.append(statement)
        
        return consensus_statements[:5]  # Return top 5 per text
    
    def _determine_contrarian_type(self, consensus: str) -> ContrarianType:
        """Determine the best contrarian type for opposing this consensus"""
        consensus_lower = consensus.lower()
        
        # Causal language → Reverse causation
        if any(word in consensus_lower for word in ['cause', 'effect', 'leads to', 'results in']):
            return ContrarianType.REVERSE_CAUSATION
        
        # Temporal language → Temporal inversion
        if any(word in consensus_lower for word in ['first', 'then', 'before', 'after']):
            return ContrarianType.TEMPORAL_INVERSION
        
        # Universal claims → Direct opposition
        if any(word in consensus_lower for word in ['all', 'every', 'always', 'never']):
            return ContrarianType.DIRECT_OPPOSITION
        
        # Evidence language → Alternative interpretation
        if any(word in consensus_lower for word in ['evidence', 'shows', 'demonstrates', 'proves']):
            return ContrarianType.ALTERNATIVE_INTERPRETATION
        
        # Default to assumption challenge
        return ContrarianType.ASSUMPTION_CHALLENGE
    
    async def _generate_single_contrarian_candidate(
        self,
        consensus: str,
        contrarian_type: ContrarianType,
        query: str,
        context: Dict[str, Any],
        papers: List[Dict[str, Any]]
    ) -> Optional[ContrarianCandidate]:
        """Generate a single contrarian candidate"""
        try:
            # Select template for this contrarian type
            templates = self.contrarian_templates[contrarian_type]
            template = random.choice(templates)
            
            # Generate contrarian position based on type
            contrarian_position = await self._generate_contrarian_position(
                consensus, contrarian_type, template, query
            )
            
            # Generate reasoning for why this contrarian view might be valid
            contrarian_reasoning = await self._generate_contrarian_reasoning(
                consensus, contrarian_position, contrarian_type
            )
            
            # Find supporting evidence and weaknesses
            supporting_evidence = await self._find_supporting_evidence(
                contrarian_position, papers
            )
            potential_weaknesses = await self._identify_potential_weaknesses(
                contrarian_position, consensus
            )
            
            # Calculate scores
            confidence_score = self._calculate_confidence_score(
                contrarian_position, supporting_evidence, potential_weaknesses
            )
            novelty_score = self._calculate_novelty_score(contrarian_position, consensus)
            testability_score = self._calculate_testability_score(contrarian_position)
            
            return ContrarianCandidate(
                candidate_id=f"contrarian_{hash(contrarian_position) % 10000}",
                contrarian_type=contrarian_type,
                consensus_position=consensus,
                contrarian_position=contrarian_position,
                contrarian_reasoning=contrarian_reasoning,
                supporting_evidence=supporting_evidence,
                potential_weaknesses=potential_weaknesses,
                confidence_score=confidence_score,
                novelty_score=novelty_score,
                testability_score=testability_score,
                domain=self._extract_domain(query),
                keywords=self._extract_keywords(contrarian_position),
                generated_from_papers=[p.get('title', 'Unknown') for p in papers[:3]]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate contrarian candidate: {e}")
            return None
    
    async def _generate_contrarian_position(
        self, consensus: str, contrarian_type: ContrarianType, template: str, query: str
    ) -> str:
        """Generate the actual contrarian position"""
        
        if contrarian_type == ContrarianType.DIRECT_OPPOSITION:
            # Simple opposition
            if "beneficial" in consensus.lower():
                opposite = consensus.replace("beneficial", "harmful")
            elif "positive" in consensus.lower():
                opposite = consensus.replace("positive", "negative")
            elif "increases" in consensus.lower():
                opposite = consensus.replace("increases", "decreases")
            else:
                opposite = f"the opposite of what is commonly believed about {consensus}"
            
            return template.format(consensus=consensus, opposite=opposite)
        
        elif contrarian_type == ContrarianType.REVERSE_CAUSATION:
            # Extract cause and effect if possible
            if " causes " in consensus:
                parts = consensus.split(" causes ")
                cause, effect = parts[0].strip(), parts[1].strip()
                return template.format(cause=cause, effect=effect)
            else:
                return f"The causal relationship in '{consensus}' may be reversed"
        
        elif contrarian_type == ContrarianType.ASSUMPTION_CHALLENGE:
            # Identify and challenge assumption
            if "because" in consensus.lower():
                assumption = consensus.split("because")[1].strip()
            elif "due to" in consensus.lower():
                assumption = consensus.split("due to")[1].strip()
            else:
                assumption = "the underlying premise"
            
            return template.format(consensus=consensus, assumption=assumption)
        
        else:
            # Generic contrarian position
            return f"Contrary to the belief that {consensus}, the opposite may be true"
    
    async def _generate_contrarian_reasoning(
        self, consensus: str, contrarian_position: str, contrarian_type: ContrarianType
    ) -> str:
        """Generate reasoning for why the contrarian position might be valid"""
        
        reasoning_templates = {
            ContrarianType.DIRECT_OPPOSITION: [
                "Historical examples show that consensus views are often wrong",
                "The majority opinion may suffer from groupthink and confirmation bias",
                "Contrarian positions have led to major breakthroughs in the past"
            ],
            ContrarianType.REVERSE_CAUSATION: [
                "Correlation doesn't imply causation in the assumed direction",
                "Complex systems often have bidirectional or circular causality",
                "What appears to be cause may actually be effect"
            ],
            ContrarianType.ASSUMPTION_CHALLENGE: [
                "Fundamental assumptions are rarely questioned but often wrong",
                "Challenging basic premises can reveal new possibilities",
                "The foundation may be weaker than it appears"
            ]
        }
        
        templates = reasoning_templates.get(contrarian_type, reasoning_templates[ContrarianType.DIRECT_OPPOSITION])
        base_reasoning = random.choice(templates)
        
        return f"{base_reasoning}. In this case, {contrarian_position.lower()} because conventional thinking may have overlooked alternative explanations or been influenced by cognitive biases."
    
    async def _find_supporting_evidence(self, contrarian_position: str, papers: List[Dict[str, Any]]) -> List[str]:
        """Find evidence that might support the contrarian position"""
        evidence = []
        
        # Look for contradictory evidence in papers
        keywords = self._extract_keywords(contrarian_position)
        
        for paper in papers[:5]:
            paper_text = paper.get('content', '') or paper.get('abstract', '')
            
            # Look for qualifying language that suggests uncertainty
            uncertainty_indicators = [
                'however', 'but', 'although', 'despite', 'nevertheless', 
                'contrary to', 'unexpectedly', 'surprisingly', 'paradoxically'
            ]
            
            for indicator in uncertainty_indicators:
                if indicator in paper_text.lower():
                    # Extract sentence containing the uncertainty
                    sentences = paper_text.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            evidence.append(f"Paper '{paper.get('title', 'Unknown')}' notes: {sentence.strip()}")
                            break
        
        # Add generic supporting arguments if no specific evidence found
        if not evidence:
            evidence = [
                "Historical precedent shows established views can be overturned",
                "Alternative interpretations of existing data may be possible",
                "Cognitive biases may influence consensus formation"
            ]
        
        return evidence[:3]
    
    async def _identify_potential_weaknesses(self, contrarian_position: str, consensus: str) -> List[str]:
        """Identify potential weaknesses in the contrarian position"""
        return [
            "Limited empirical evidence for this contrarian view",
            "May conflict with well-established theoretical frameworks",
            "Could be difficult to test or validate experimentally",
            "Might represent minority view for valid scientific reasons"
        ][:2]  # Keep it concise
    
    def _calculate_confidence_score(self, position: str, evidence: List[str], weaknesses: List[str]) -> float:
        """Calculate confidence score for contrarian position"""
        base_score = 0.3  # Start lower for contrarian positions
        
        # Boost for evidence
        evidence_boost = len(evidence) * 0.1
        
        # Penalty for weaknesses  
        weakness_penalty = len(weaknesses) * 0.05
        
        return max(0.1, min(0.8, base_score + evidence_boost - weakness_penalty))
    
    def _calculate_novelty_score(self, contrarian_position: str, consensus: str) -> float:
        """Calculate how novel/surprising the contrarian position is"""
        # More different from consensus = higher novelty
        if "opposite" in contrarian_position.lower():
            return 0.9
        elif "contrary" in contrarian_position.lower():
            return 0.8
        elif "reverse" in contrarian_position.lower():
            return 0.7
        else:
            return 0.6
    
    def _calculate_testability_score(self, position: str) -> float:
        """Calculate how testable the contrarian position is"""
        testable_indicators = ['measure', 'test', 'experiment', 'data', 'evidence', 'study']
        
        score = 0.4  # Base testability
        for indicator in testable_indicators:
            if indicator in position.lower():
                score += 0.1
        
        return min(0.9, score)
    
    def _extract_domain(self, query: str) -> str:
        """Extract the domain/field from the query"""
        domains = {
            'medical': ['medical', 'health', 'disease', 'treatment', 'clinical'],
            'technology': ['technology', 'software', 'AI', 'computer', 'digital'],
            'business': ['business', 'market', 'strategy', 'economic', 'financial'],
            'science': ['science', 'research', 'experiment', 'theory', 'hypothesis']
        }
        
        query_lower = query.lower()
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords[:10]
    
    async def _generate_synthetic_contrarian_candidate(
        self, query: str, context: Dict[str, Any], candidate_index: int
    ) -> Optional[ContrarianCandidate]:
        """Generate synthetic contrarian candidate when no papers available"""
        
        synthetic_templates = [
            {
                "consensus": f"The conventional approach to {query} is the best method",
                "contrarian": f"The conventional approach to {query} may be fundamentally flawed",
                "type": ContrarianType.ASSUMPTION_CHALLENGE
            },
            {
                "consensus": f"More research in {query} will lead to better outcomes", 
                "contrarian": f"More research in {query} might actually hinder progress",
                "type": ContrarianType.DIRECT_OPPOSITION
            },
            {
                "consensus": f"The primary factors in {query} cause the observed effects",
                "contrarian": f"The observed effects in {query} may actually cause what we think are the primary factors",
                "type": ContrarianType.REVERSE_CAUSATION
            }
        ]
        
        if candidate_index >= len(synthetic_templates):
            return None
        
        template = synthetic_templates[candidate_index]
        
        return ContrarianCandidate(
            candidate_id=f"synthetic_contrarian_{candidate_index}",
            contrarian_type=template["type"],
            consensus_position=template["consensus"],
            contrarian_position=template["contrarian"],
            contrarian_reasoning="This contrarian position challenges conventional wisdom by questioning fundamental assumptions",
            supporting_evidence=["Alternative frameworks may exist", "Historical precedent for paradigm shifts"],
            potential_weaknesses=["Limited direct evidence", "May conflict with established practice"],
            confidence_score=0.4,
            novelty_score=0.7,
            testability_score=0.5,
            domain=self._extract_domain(query),
            keywords=self._extract_keywords(query),
            generated_from_papers=["Synthetic generation"]
        )

# Global instance
contrarian_candidate_engine = ContrarianCandidateEngine()

async def generate_contrarian_candidates(
    query: str, 
    context: Dict[str, Any], 
    papers: List[Dict[str, Any]] = None,
    max_candidates: int = 5
) -> List[ContrarianCandidate]:
    """Convenience function to generate contrarian candidates"""
    return await contrarian_candidate_engine.generate_contrarian_candidates(
        query, context, papers, max_candidates
    )