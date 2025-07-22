"""
NWTN Cross-Domain Transplant Generation Engine
==============================================

Implements Phase 2.2 of the Novel Idea Generation Roadmap:
Generates breakthrough candidates by transplanting solutions from maximally distant domains.

This engine identifies solutions from completely different fields and adapts them
to the current problem domain, enabling unprecedented cross-pollination of ideas.

Based on NWTN Novel Idea Generation Roadmap Phase 2.2.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import random
import math

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Types of knowledge domains for cross-pollination"""
    BIOLOGICAL = "biological"              # Nature, evolution, living systems
    PHYSICAL = "physical"                  # Physics, chemistry, materials
    MATHEMATICAL = "mathematical"          # Mathematics, algorithms, logic
    TECHNOLOGICAL = "technological"        # Engineering, computing, automation
    SOCIAL = "social"                      # Psychology, sociology, economics
    ARTISTIC = "artistic"                  # Art, music, design, creativity
    HISTORICAL = "historical"             # Past events, ancient solutions
    MECHANICAL = "mechanical"             # Mechanical systems, robotics
    CHEMICAL = "chemical"                  # Chemical processes, reactions
    QUANTUM = "quantum"                   # Quantum mechanics, physics
    LINGUISTIC = "linguistic"             # Language, communication, semantics
    ECOLOGICAL = "ecological"             # Ecosystems, environmental systems

class TransplantType(Enum):
    """Types of cross-domain solution transplantation"""
    STRUCTURAL_ANALOGY = "structural_analogy"        # Copy structural patterns
    FUNCTIONAL_MIMICRY = "functional_mimicry"        # Mimic functional behavior
    PROCESS_ADAPTATION = "process_adaptation"        # Adapt processes/workflows
    PRINCIPLE_EXTRACTION = "principle_extraction"   # Extract underlying principles
    PATTERN_MAPPING = "pattern_mapping"             # Map behavioral patterns
    SYSTEM_HYBRIDIZATION = "system_hybridization"   # Combine system characteristics

@dataclass
class DomainDistance:
    """Measure of conceptual distance between domains"""
    source_domain: DomainType
    target_domain: DomainType
    distance_score: float  # 0.0 = identical, 1.0 = maximally distant
    conceptual_bridges: List[str]  # Potential connection points
    transplant_viability: float   # How likely transplant will work

@dataclass
class SolutionPattern:
    """A solution pattern identified in a source domain"""
    pattern_id: str
    source_domain: DomainType
    solution_name: str
    problem_addressed: str
    mechanism_description: str
    key_principles: List[str]
    structural_elements: List[str]
    functional_elements: List[str]
    success_factors: List[str]
    constraints_assumptions: List[str]

@dataclass
class CrossDomainTransplant:
    """A cross-domain transplant candidate"""
    transplant_id: str
    transplant_type: TransplantType
    
    # Source information
    source_domain: DomainType
    source_solution: SolutionPattern
    
    # Target adaptation
    target_query: str
    target_domain: DomainType
    transplanted_solution: str
    adaptation_reasoning: str
    
    # Viability assessment
    transplant_feasibility: float     # How feasible is the transplant
    novelty_score: float              # How novel/surprising is this connection
    potential_impact: float           # Potential impact if successful
    implementation_complexity: float  # How complex to implement
    
    # Evidence and validation
    analogous_elements: List[str]     # What elements are analogous
    key_differences: List[str]        # Key differences to address
    validation_criteria: List[str]    # How to validate this transplant
    
    # Metadata
    domain_distance: float            # How distant are the domains
    confidence_score: float           # Overall confidence in transplant
    generated_from_papers: List[str]  # Papers that inspired this transplant

class CrossDomainTransplantEngine:
    """
    Generate breakthrough candidates by transplanting solutions across maximally distant domains.
    
    This engine identifies successful solutions from completely different fields and adapts
    them to the current problem domain, enabling unprecedented cross-pollination of ideas.
    """
    
    def __init__(self):
        self.domain_mappings = self._initialize_domain_mappings()
        self.solution_patterns = self._initialize_solution_patterns()
        self.transplant_strategies = self._initialize_transplant_strategies()
        self.domain_distances = self._calculate_domain_distances()
        
    def _initialize_domain_mappings(self) -> Dict[DomainType, Dict[str, Any]]:
        """Initialize domain characteristic mappings"""
        return {
            DomainType.BIOLOGICAL: {
                "keywords": ["evolution", "adaptation", "organism", "ecosystem", "natural", "biological", "genetic", "cellular", "neural"],
                "problem_types": ["optimization", "adaptation", "self-organization", "resilience", "efficiency"],
                "solution_patterns": ["evolutionary_algorithms", "swarm_intelligence", "neural_networks", "symbiosis", "adaptation"]
            },
            DomainType.PHYSICAL: {
                "keywords": ["physics", "mechanics", "thermodynamics", "energy", "force", "momentum", "equilibrium"],
                "problem_types": ["stability", "efficiency", "energy_transfer", "structural_integrity", "optimization"],
                "solution_patterns": ["conservation_laws", "equilibrium_systems", "energy_minimization", "phase_transitions"]
            },
            DomainType.MATHEMATICAL: {
                "keywords": ["algorithm", "theorem", "proof", "optimization", "graph", "topology", "statistics", "probability"],
                "problem_types": ["optimization", "pattern_recognition", "decision_making", "resource_allocation"],
                "solution_patterns": ["graph_algorithms", "optimization_theory", "probability_distributions", "mathematical_proofs"]
            },
            DomainType.TECHNOLOGICAL: {
                "keywords": ["engineering", "system", "design", "automation", "software", "hardware", "network", "protocol"],
                "problem_types": ["scalability", "reliability", "performance", "automation", "integration"],
                "solution_patterns": ["distributed_systems", "fault_tolerance", "load_balancing", "caching", "protocols"]
            },
            DomainType.SOCIAL: {
                "keywords": ["social", "behavior", "psychology", "economics", "organization", "collaboration", "incentive"],
                "problem_types": ["coordination", "motivation", "trust", "cooperation", "decision_making"],
                "solution_patterns": ["incentive_mechanisms", "social_networks", "reputation_systems", "collective_intelligence"]
            },
            DomainType.ARTISTIC: {
                "keywords": ["art", "design", "creativity", "aesthetic", "composition", "harmony", "rhythm", "pattern"],
                "problem_types": ["creativity", "expression", "harmony", "balance", "innovation"],
                "solution_patterns": ["compositional_rules", "aesthetic_principles", "creative_processes", "pattern_recognition"]
            },
            DomainType.MECHANICAL: {
                "keywords": ["mechanical", "gear", "lever", "pulley", "engine", "machine", "robotics", "automation"],
                "problem_types": ["motion", "force_multiplication", "precision", "automation", "efficiency"],
                "solution_patterns": ["mechanical_advantage", "gear_ratios", "feedback_control", "precision_mechanisms"]
            },
            DomainType.CHEMICAL: {
                "keywords": ["chemical", "reaction", "catalyst", "synthesis", "molecular", "compound", "process"],
                "problem_types": ["synthesis", "purification", "reaction_optimization", "molecular_design"],
                "solution_patterns": ["catalysis", "chemical_equilibrium", "reaction_mechanisms", "molecular_assembly"]
            },
            DomainType.QUANTUM: {
                "keywords": ["quantum", "superposition", "entanglement", "coherence", "uncertainty", "probability"],
                "problem_types": ["parallelism", "uncertainty_handling", "information_processing", "optimization"],
                "solution_patterns": ["quantum_parallelism", "superposition_principles", "entanglement_effects", "measurement_theory"]
            },
            DomainType.LINGUISTIC: {
                "keywords": ["language", "communication", "semantics", "grammar", "meaning", "translation", "parsing"],
                "problem_types": ["communication", "understanding", "translation", "interpretation", "meaning"],
                "solution_patterns": ["grammar_rules", "semantic_networks", "translation_mechanisms", "parsing_algorithms"]
            },
            DomainType.ECOLOGICAL: {
                "keywords": ["ecosystem", "environment", "sustainability", "balance", "cycle", "resource", "habitat"],
                "problem_types": ["sustainability", "resource_management", "balance", "adaptation", "resilience"],
                "solution_patterns": ["resource_cycles", "ecosystem_balance", "sustainable_systems", "adaptive_capacity"]
            }
        }
    
    def _initialize_solution_patterns(self) -> Dict[DomainType, List[SolutionPattern]]:
        """Initialize known solution patterns for each domain"""
        patterns = {}
        
        # Biological domain patterns
        patterns[DomainType.BIOLOGICAL] = [
            SolutionPattern(
                pattern_id="ant_colony_optimization",
                source_domain=DomainType.BIOLOGICAL,
                solution_name="Ant Colony Foraging",
                problem_addressed="Finding optimal paths in complex networks",
                mechanism_description="Ants use pheromone trails to collectively find shortest paths to food sources",
                key_principles=["collective_intelligence", "positive_feedback", "indirect_coordination"],
                structural_elements=["agents", "pheromone_trails", "environment", "local_decisions"],
                functional_elements=["exploration", "exploitation", "path_optimization", "collective_decision"],
                success_factors=["simple_local_rules", "emergent_global_behavior", "adaptive_feedback"],
                constraints_assumptions=["agents_can_leave_trails", "environment_persistence", "local_information"]
            ),
            SolutionPattern(
                pattern_id="immune_system_defense",
                source_domain=DomainType.BIOLOGICAL,
                solution_name="Immune System Recognition",
                problem_addressed="Detecting and neutralizing unknown threats",
                mechanism_description="Multi-layered defense with adaptive recognition and memory",
                key_principles=["pattern_recognition", "adaptive_learning", "memory_formation", "layered_defense"],
                structural_elements=["antibodies", "memory_cells", "recognition_sites", "communication_molecules"],
                functional_elements=["detection", "classification", "response", "memory_storage"],
                success_factors=["diversity_generation", "clonal_selection", "immune_memory"],
                constraints_assumptions=["self_vs_nonself_recognition", "molecular_binding", "cellular_communication"]
            )
        ]
        
        # Physical domain patterns
        patterns[DomainType.PHYSICAL] = [
            SolutionPattern(
                pattern_id="resonance_amplification",
                source_domain=DomainType.PHYSICAL,
                solution_name="Resonance Phenomena",
                problem_addressed="Amplifying weak signals or effects",
                mechanism_description="Small periodic forces can create large amplitude oscillations at resonant frequency",
                key_principles=["frequency_matching", "energy_accumulation", "constructive_interference"],
                structural_elements=["oscillator", "driving_force", "resonant_frequency", "feedback_mechanism"],
                functional_elements=["signal_detection", "amplitude_amplification", "frequency_tuning", "energy_transfer"],
                success_factors=["precise_frequency_matching", "low_damping", "sustained_input"],
                constraints_assumptions=["periodic_driving_force", "oscillatory_system", "energy_conservation"]
            )
        ]
        
        # Mathematical domain patterns  
        patterns[DomainType.MATHEMATICAL] = [
            SolutionPattern(
                pattern_id="graph_coloring",
                source_domain=DomainType.MATHEMATICAL,
                solution_name="Graph Coloring Algorithms",
                problem_addressed="Assigning resources without conflicts",
                mechanism_description="Color graph nodes so no adjacent nodes have same color",
                key_principles=["constraint_satisfaction", "conflict_minimization", "resource_optimization"],
                structural_elements=["nodes", "edges", "colors", "constraints"],
                functional_elements=["conflict_detection", "resource_assignment", "constraint_checking", "optimization"],
                success_factors=["minimal_colors", "conflict_free_assignment", "efficient_algorithm"],
                constraints_assumptions=["finite_colors", "adjacency_constraints", "complete_assignment"]
            )
        ]
        
        # Add more patterns for other domains...
        for domain in DomainType:
            if domain not in patterns:
                patterns[domain] = []  # Initialize empty for now
        
        return patterns
    
    def _initialize_transplant_strategies(self) -> Dict[TransplantType, Dict[str, Any]]:
        """Initialize transplant strategy templates"""
        return {
            TransplantType.STRUCTURAL_ANALOGY: {
                "description": "Map structural components from source to target domain",
                "adaptation_template": "Just as {source_domain} uses {source_structure} to achieve {source_function}, {target_domain} could use {target_structure} to achieve {target_function}",
                "focus_areas": ["structural_elements", "component_relationships", "hierarchical_organization"]
            },
            TransplantType.FUNCTIONAL_MIMICRY: {
                "description": "Mimic functional behavior across domains",
                "adaptation_template": "The functional behavior of {source_mechanism} in {source_domain} could be replicated in {target_domain} through {target_mechanism}",
                "focus_areas": ["functional_elements", "behavioral_patterns", "input_output_relationships"]
            },
            TransplantType.PROCESS_ADAPTATION: {
                "description": "Adapt processes and workflows across domains",
                "adaptation_template": "The process of {source_process} used in {source_domain} could be adapted for {target_domain} by {adaptation_strategy}",
                "focus_areas": ["process_steps", "workflow_patterns", "optimization_strategies"]
            },
            TransplantType.PRINCIPLE_EXTRACTION: {
                "description": "Extract and apply underlying principles",
                "adaptation_template": "The principle of {source_principle} from {source_domain} suggests that {target_domain} could benefit from {principle_application}",
                "focus_areas": ["key_principles", "underlying_mechanisms", "fundamental_laws"]
            },
            TransplantType.PATTERN_MAPPING: {
                "description": "Map behavioral and interaction patterns",
                "adaptation_template": "The pattern of {source_pattern} observed in {source_domain} could inform {target_pattern} in {target_domain}",
                "focus_areas": ["behavioral_patterns", "interaction_dynamics", "emergent_properties"]
            },
            TransplantType.SYSTEM_HYBRIDIZATION: {
                "description": "Combine characteristics from multiple systems",
                "adaptation_template": "Combining {source_characteristics} from {source_domain} with {target_characteristics} could create a hybrid approach for {target_problem}",
                "focus_areas": ["system_characteristics", "integration_points", "synergistic_effects"]
            }
        }
    
    def _calculate_domain_distances(self) -> Dict[Tuple[DomainType, DomainType], DomainDistance]:
        """Calculate conceptual distances between all domain pairs"""
        distances = {}
        
        for source_domain in DomainType:
            for target_domain in DomainType:
                if source_domain != target_domain:
                    distance_score = self._calculate_single_domain_distance(source_domain, target_domain)
                    conceptual_bridges = self._find_conceptual_bridges(source_domain, target_domain)
                    transplant_viability = self._assess_transplant_viability(source_domain, target_domain, distance_score)
                    
                    distances[(source_domain, target_domain)] = DomainDistance(
                        source_domain=source_domain,
                        target_domain=target_domain,
                        distance_score=distance_score,
                        conceptual_bridges=conceptual_bridges,
                        transplant_viability=transplant_viability
                    )
        
        return distances
    
    def _calculate_single_domain_distance(self, source: DomainType, target: DomainType) -> float:
        """Calculate distance score between two specific domains"""
        source_info = self.domain_mappings.get(source, {})
        target_info = self.domain_mappings.get(target, {})
        
        source_keywords = set(source_info.get("keywords", []))
        target_keywords = set(target_info.get("keywords", []))
        
        # Calculate keyword overlap (lower overlap = higher distance)
        if len(source_keywords) == 0 or len(target_keywords) == 0:
            return 0.9  # High distance for missing data
        
        overlap = len(source_keywords & target_keywords)
        total = len(source_keywords | target_keywords)
        
        # Distance is inverse of overlap ratio
        overlap_ratio = overlap / total if total > 0 else 0
        distance = 1.0 - overlap_ratio
        
        # Add domain-specific distance adjustments
        distance = self._adjust_domain_specific_distance(source, target, distance)
        
        return min(1.0, max(0.0, distance))
    
    def _adjust_domain_specific_distance(self, source: DomainType, target: DomainType, base_distance: float) -> float:
        """Apply domain-specific distance adjustments"""
        
        # Define domain clusters for distance adjustment
        abstract_domains = {DomainType.MATHEMATICAL, DomainType.QUANTUM, DomainType.LINGUISTIC}
        concrete_domains = {DomainType.MECHANICAL, DomainType.CHEMICAL, DomainType.BIOLOGICAL}
        systems_domains = {DomainType.TECHNOLOGICAL, DomainType.SOCIAL, DomainType.ECOLOGICAL}
        creative_domains = {DomainType.ARTISTIC, DomainType.HISTORICAL}
        
        # Increase distance between abstract and concrete domains
        if (source in abstract_domains and target in concrete_domains) or \
           (source in concrete_domains and target in abstract_domains):
            base_distance += 0.2
        
        # Increase distance between creative and technical domains
        if (source in creative_domains and target in {DomainType.TECHNOLOGICAL, DomainType.MATHEMATICAL}) or \
           (source in {DomainType.TECHNOLOGICAL, DomainType.MATHEMATICAL} and target in creative_domains):
            base_distance += 0.3
        
        # Maximum distance for most contrasting pairs
        extreme_pairs = [
            (DomainType.QUANTUM, DomainType.ARTISTIC),
            (DomainType.MATHEMATICAL, DomainType.BIOLOGICAL),
            (DomainType.MECHANICAL, DomainType.LINGUISTIC)
        ]
        
        for pair in extreme_pairs:
            if (source, target) == pair or (target, source) == pair:
                base_distance = max(base_distance, 0.9)
        
        return base_distance
    
    def _find_conceptual_bridges(self, source: DomainType, target: DomainType) -> List[str]:
        """Find potential conceptual bridges between domains"""
        source_info = self.domain_mappings.get(source, {})
        target_info = self.domain_mappings.get(target, {})
        
        bridges = []
        
        # Common problem types as bridges
        source_problems = set(source_info.get("problem_types", []))
        target_problems = set(target_info.get("problem_types", []))
        common_problems = source_problems & target_problems
        
        for problem in common_problems:
            bridges.append(f"shared_problem_type_{problem}")
        
        # Common keywords as bridges
        source_keywords = set(source_info.get("keywords", []))
        target_keywords = set(target_info.get("keywords", []))
        common_keywords = source_keywords & target_keywords
        
        for keyword in common_keywords:
            bridges.append(f"shared_concept_{keyword}")
        
        # Add metaphorical bridges
        metaphorical_bridges = self._generate_metaphorical_bridges(source, target)
        bridges.extend(metaphorical_bridges)
        
        return bridges[:5]  # Limit to top 5 bridges
    
    def _generate_metaphorical_bridges(self, source: DomainType, target: DomainType) -> List[str]:
        """Generate metaphorical connections between domains"""
        
        metaphors = {
            (DomainType.BIOLOGICAL, DomainType.TECHNOLOGICAL): ["evolution_as_optimization", "cells_as_components", "organisms_as_systems"],
            (DomainType.PHYSICAL, DomainType.SOCIAL): ["forces_as_influences", "equilibrium_as_stability", "resonance_as_alignment"],
            (DomainType.MATHEMATICAL, DomainType.ARTISTIC): ["patterns_as_beauty", "symmetry_as_harmony", "algorithms_as_composition"],
            (DomainType.MECHANICAL, DomainType.LINGUISTIC): ["gears_as_grammar", "mechanisms_as_meaning", "precision_as_clarity"]
        }
        
        bridges = metaphors.get((source, target), [])
        if not bridges:
            bridges = metaphors.get((target, source), [])
        
        return bridges
    
    def _assess_transplant_viability(self, source: DomainType, target: DomainType, distance: float) -> float:
        """Assess how viable transplantation between domains might be"""
        
        # Higher distance can mean higher novelty but lower viability
        # Sweet spot is medium-high distance with good bridges
        
        bridges = self._find_conceptual_bridges(source, target)
        bridge_count = len(bridges)
        
        # Viability calculation
        # - Too low distance (< 0.3): low novelty
        # - Too high distance (> 0.9): impractical transplant
        # - Medium-high distance (0.5-0.8): optimal range
        
        if distance < 0.3:
            base_viability = 0.3  # Low novelty
        elif distance > 0.9:
            base_viability = 0.2  # Too distant
        else:
            # Optimal range - higher distance gives higher base viability
            base_viability = min(0.8, distance)
        
        # Adjust for conceptual bridges (more bridges = higher viability)
        bridge_bonus = min(0.3, bridge_count * 0.1)
        
        final_viability = base_viability + bridge_bonus
        return min(1.0, max(0.1, final_viability))
    
    async def generate_cross_domain_transplants(
        self,
        query: str,
        context: Dict[str, Any],
        papers: List[Dict[str, Any]] = None,
        max_transplants: int = 5
    ) -> List[CrossDomainTransplant]:
        """
        Generate cross-domain transplant candidates for a query
        
        Args:
            query: Research question or problem
            context: Context including breakthrough mode
            papers: Retrieved papers to analyze for domains
            max_transplants: Maximum transplants to generate
        
        Returns:
            List of cross-domain transplant candidates
        """
        try:
            logger.info(f"Generating cross-domain transplants for query: {query[:50]}...")
            
            # Identify target domain from query
            target_domain = self._identify_query_domain(query)
            
            # Find relevant source domains and solutions
            source_candidates = await self._find_source_domain_candidates(query, papers or [], target_domain)
            
            # Generate transplants from promising source domains
            transplants = []
            
            for source_domain, source_solutions in source_candidates:
                if len(transplants) >= max_transplants:
                    break
                
                # Generate transplants for this source domain
                domain_transplants = await self._generate_domain_specific_transplants(
                    query, target_domain, source_domain, source_solutions, context
                )
                
                transplants.extend(domain_transplants)
            
            # Sort by overall promise (novelty × feasibility × impact)
            transplants.sort(key=lambda t: t.novelty_score * t.transplant_feasibility * t.potential_impact, reverse=True)
            
            # Return top transplants
            selected_transplants = transplants[:max_transplants]
            
            logger.info(f"Generated {len(selected_transplants)} cross-domain transplant candidates")
            return selected_transplants
            
        except Exception as e:
            logger.error(f"Failed to generate cross-domain transplants: {e}")
            return []
    
    def _identify_query_domain(self, query: str) -> DomainType:
        """Identify the primary domain of the query"""
        query_lower = query.lower()
        
        domain_scores = {}
        
        for domain, domain_info in self.domain_mappings.items():
            score = 0
            keywords = domain_info.get("keywords", [])
            
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            domain_scores[domain] = score
        
        # Return domain with highest score, or TECHNOLOGICAL as default
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return DomainType.TECHNOLOGICAL  # Default domain
    
    async def _find_source_domain_candidates(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        target_domain: DomainType
    ) -> List[Tuple[DomainType, List[SolutionPattern]]]:
        """Find promising source domains for transplantation"""
        
        candidates = []
        
        # Get domains ordered by distance (prefer medium-high distance)
        domain_candidates = []
        for source_domain in DomainType:
            if source_domain != target_domain:
                distance_info = self.domain_distances.get((source_domain, target_domain))
                if distance_info and 0.4 <= distance_info.distance_score <= 0.9:  # Optimal range
                    domain_candidates.append((source_domain, distance_info))
        
        # Sort by transplant viability
        domain_candidates.sort(key=lambda x: x[1].transplant_viability, reverse=True)
        
        # Generate source solutions for top domains
        for source_domain, distance_info in domain_candidates[:8]:  # Top 8 source domains
            source_solutions = await self._generate_source_solutions(source_domain, query, papers)
            if source_solutions:
                candidates.append((source_domain, source_solutions))
        
        return candidates
    
    async def _generate_source_solutions(
        self,
        source_domain: DomainType,
        query: str,
        papers: List[Dict[str, Any]]
    ) -> List[SolutionPattern]:
        """Generate or retrieve solution patterns for a source domain"""
        
        solutions = []
        
        # Use pre-defined patterns if available
        predefined_patterns = self.solution_patterns.get(source_domain, [])
        solutions.extend(predefined_patterns)
        
        # Generate solutions from papers
        paper_solutions = await self._extract_solutions_from_papers(source_domain, papers, query)
        solutions.extend(paper_solutions)
        
        # Generate synthetic solutions if needed
        if len(solutions) < 2:
            synthetic_solutions = self._generate_synthetic_solutions(source_domain, query)
            solutions.extend(synthetic_solutions)
        
        return solutions[:3]  # Limit to top 3 solutions per domain
    
    async def _extract_solutions_from_papers(
        self,
        source_domain: DomainType,
        papers: List[Dict[str, Any]],
        query: str
    ) -> List[SolutionPattern]:
        """Extract solution patterns from papers in the source domain"""
        
        solutions = []
        domain_info = self.domain_mappings.get(source_domain, {})
        domain_keywords = domain_info.get("keywords", [])
        
        for paper in papers[:5]:  # Limit paper processing
            paper_text = paper.get('content', '') or paper.get('abstract', '')
            
            # Check if paper is relevant to source domain
            relevance_score = 0
            for keyword in domain_keywords:
                if keyword in paper_text.lower():
                    relevance_score += 1
            
            if relevance_score >= 2:  # Paper is relevant to source domain
                solution = self._extract_solution_pattern_from_paper(paper, source_domain, query)
                if solution:
                    solutions.append(solution)
        
        return solutions
    
    def _extract_solution_pattern_from_paper(
        self,
        paper: Dict[str, Any],
        source_domain: DomainType,
        query: str
    ) -> Optional[SolutionPattern]:
        """Extract a solution pattern from a single paper"""
        
        title = paper.get('title', 'Unknown Paper')
        content = paper.get('content', '') or paper.get('abstract', '')
        
        if len(content) < 100:  # Skip papers with insufficient content
            return None
        
        # Extract key information using simple heuristics
        problem_addressed = self._extract_problem_from_content(content)
        mechanism = self._extract_mechanism_from_content(content)
        principles = self._extract_principles_from_content(content, source_domain)
        
        return SolutionPattern(
            pattern_id=f"paper_{hash(title) % 10000}",
            source_domain=source_domain,
            solution_name=title[:50],
            problem_addressed=problem_addressed,
            mechanism_description=mechanism,
            key_principles=principles,
            structural_elements=["component_1", "component_2", "component_3"],  # Simplified
            functional_elements=["function_1", "function_2"],  # Simplified
            success_factors=["factor_1", "factor_2"],  # Simplified
            constraints_assumptions=["assumption_1"]  # Simplified
        )
    
    def _extract_problem_from_content(self, content: str) -> str:
        """Extract problem statement from paper content"""
        # Look for problem indicators
        problem_indicators = ["problem", "challenge", "issue", "difficulty", "limitation"]
        
        sentences = content.split('.')
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in problem_indicators):
                return sentence.strip()[:200]
        
        return "Optimization and efficiency challenges"  # Default
    
    def _extract_mechanism_from_content(self, content: str) -> str:
        """Extract mechanism description from paper content"""
        # Look for mechanism indicators
        mechanism_indicators = ["method", "approach", "technique", "algorithm", "mechanism", "process"]
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in mechanism_indicators):
                if len(sentence.strip()) > 50:  # Prefer longer descriptions
                    return sentence.strip()[:300]
        
        return "Novel approach using systematic methodology"  # Default
    
    def _extract_principles_from_content(self, content: str, domain: DomainType) -> List[str]:
        """Extract key principles from paper content"""
        domain_info = self.domain_mappings.get(domain, {})
        domain_keywords = domain_info.get("keywords", [])
        
        principles = []
        content_lower = content.lower()
        
        # Find domain-specific principles
        for keyword in domain_keywords:
            if keyword in content_lower:
                principles.append(f"{keyword}_based_principle")
        
        # Add generic principles if needed
        if len(principles) < 2:
            principles.extend(["optimization_principle", "efficiency_principle", "scalability_principle"])
        
        return principles[:3]
    
    def _generate_synthetic_solutions(self, source_domain: DomainType, query: str) -> List[SolutionPattern]:
        """Generate synthetic solution patterns for domains lacking specific examples"""
        
        domain_info = self.domain_mappings.get(source_domain, {})
        solution_patterns = domain_info.get("solution_patterns", [])
        
        if not solution_patterns:
            solution_patterns = ["generic_optimization", "systematic_approach", "adaptive_mechanism"]
        
        solutions = []
        
        for i, pattern_name in enumerate(solution_patterns[:2]):
            solution = SolutionPattern(
                pattern_id=f"synthetic_{source_domain.value}_{i}",
                source_domain=source_domain,
                solution_name=f"Synthetic {source_domain.value.title()} Solution {i+1}",
                problem_addressed=f"{source_domain.value.title()} optimization challenges",
                mechanism_description=f"Uses {pattern_name} to achieve optimal results in {source_domain.value} domain",
                key_principles=[f"{source_domain.value}_principle", "optimization_principle", "adaptation_principle"],
                structural_elements=["core_component", "interface_layer", "control_system"],
                functional_elements=["sensing", "processing", "actuating"],
                success_factors=["proper_calibration", "adaptive_feedback", "efficient_implementation"],
                constraints_assumptions=["domain_specific_constraints", "resource_availability"]
            )
            solutions.append(solution)
        
        return solutions
    
    async def _generate_domain_specific_transplants(
        self,
        query: str,
        target_domain: DomainType,
        source_domain: DomainType,
        source_solutions: List[SolutionPattern],
        context: Dict[str, Any]
    ) -> List[CrossDomainTransplant]:
        """Generate transplants from specific source domain to target domain"""
        
        transplants = []
        distance_info = self.domain_distances.get((source_domain, target_domain))
        
        if not distance_info:
            return transplants
        
        for solution in source_solutions[:2]:  # Max 2 solutions per domain
            for transplant_type in TransplantType:
                transplant = await self._create_single_transplant(
                    query, target_domain, source_domain, solution, transplant_type, distance_info, context
                )
                
                if transplant:
                    transplants.append(transplant)
                    
                    if len(transplants) >= 3:  # Max 3 transplants per domain
                        break
            
            if len(transplants) >= 3:
                break
        
        return transplants
    
    async def _create_single_transplant(
        self,
        query: str,
        target_domain: DomainType,
        source_domain: DomainType,
        source_solution: SolutionPattern,
        transplant_type: TransplantType,
        distance_info: DomainDistance,
        context: Dict[str, Any]
    ) -> Optional[CrossDomainTransplant]:
        """Create a single cross-domain transplant candidate"""
        
        try:
            # Generate transplanted solution
            transplanted_solution = await self._generate_transplanted_solution(
                query, target_domain, source_solution, transplant_type
            )
            
            # Generate adaptation reasoning
            adaptation_reasoning = self._generate_adaptation_reasoning(
                source_domain, target_domain, source_solution, transplant_type
            )
            
            # Calculate scores
            feasibility_score = self._calculate_transplant_feasibility(
                source_domain, target_domain, transplant_type, distance_info
            )
            novelty_score = self._calculate_transplant_novelty(distance_info, transplant_type)
            impact_score = self._calculate_potential_impact(query, transplanted_solution)
            complexity_score = self._calculate_implementation_complexity(transplant_type, source_solution)
            
            # Generate supporting elements
            analogous_elements = self._identify_analogous_elements(source_solution, target_domain)
            key_differences = self._identify_key_differences(source_domain, target_domain)
            validation_criteria = self._generate_validation_criteria(transplanted_solution, target_domain)
            
            # Calculate overall confidence
            confidence_score = (feasibility_score + novelty_score + impact_score) / 3
            
            return CrossDomainTransplant(
                transplant_id=f"transplant_{hash(transplanted_solution) % 10000}",
                transplant_type=transplant_type,
                source_domain=source_domain,
                source_solution=source_solution,
                target_query=query,
                target_domain=target_domain,
                transplanted_solution=transplanted_solution,
                adaptation_reasoning=adaptation_reasoning,
                transplant_feasibility=feasibility_score,
                novelty_score=novelty_score,
                potential_impact=impact_score,
                implementation_complexity=complexity_score,
                analogous_elements=analogous_elements,
                key_differences=key_differences,
                validation_criteria=validation_criteria,
                domain_distance=distance_info.distance_score,
                confidence_score=confidence_score,
                generated_from_papers=[source_solution.solution_name]
            )
            
        except Exception as e:
            logger.error(f"Failed to create transplant: {e}")
            return None
    
    async def _generate_transplanted_solution(
        self,
        query: str,
        target_domain: DomainType,
        source_solution: SolutionPattern,
        transplant_type: TransplantType
    ) -> str:
        """Generate the actual transplanted solution description"""
        
        strategy_info = self.transplant_strategies[transplant_type]
        template = strategy_info["adaptation_template"]
        
        # Extract key elements from source solution
        source_mechanism = source_solution.mechanism_description
        source_principles = ", ".join(source_solution.key_principles[:2])
        
        # Generate target-specific adaptations
        target_mechanism = self._adapt_mechanism_to_target(source_mechanism, target_domain)
        target_context = self._generate_target_context(query, target_domain)
        
        # Create transplanted solution using template
        transplanted_solution = template.format(
            source_domain=source_solution.source_domain.value,
            target_domain=target_domain.value,
            source_mechanism=source_mechanism[:100],
            target_mechanism=target_mechanism,
            source_principle=source_principles,
            target_function=target_context,
            source_structure="core components",
            target_structure="adapted components",
            source_process="original process",
            adaptation_strategy="domain-specific adaptation",
            principle_application="systematic application"
        )
        
        return transplanted_solution
    
    def _adapt_mechanism_to_target(self, source_mechanism: str, target_domain: DomainType) -> str:
        """Adapt source mechanism description to target domain"""
        
        target_info = self.domain_mappings.get(target_domain, {})
        target_keywords = target_info.get("keywords", [])
        
        if target_keywords:
            primary_keyword = target_keywords[0]
            return f"{primary_keyword}-based adaptation of the mechanism"
        else:
            return f"domain-adapted mechanism for {target_domain.value}"
    
    def _generate_target_context(self, query: str, target_domain: DomainType) -> str:
        """Generate target domain context for the query"""
        
        target_info = self.domain_mappings.get(target_domain, {})
        problem_types = target_info.get("problem_types", ["optimization"])
        
        if problem_types:
            return f"{problem_types[0]} in {target_domain.value} context"
        else:
            return f"problem-solving in {target_domain.value}"
    
    def _generate_adaptation_reasoning(
        self,
        source_domain: DomainType,
        target_domain: DomainType,
        source_solution: SolutionPattern,
        transplant_type: TransplantType
    ) -> str:
        """Generate reasoning for why this adaptation makes sense"""
        
        strategy_info = self.transplant_strategies[transplant_type]
        
        reasoning_elements = [
            f"The {transplant_type.value} approach leverages the successful {source_solution.solution_name} from {source_domain.value}",
            f"Key principle of {source_solution.key_principles[0] if source_solution.key_principles else 'optimization'} is applicable across domains",
            f"Adaptation to {target_domain.value} maintains core benefits while addressing domain-specific constraints",
            strategy_info["description"]
        ]
        
        return ". ".join(reasoning_elements)
    
    def _calculate_transplant_feasibility(
        self,
        source_domain: DomainType,
        target_domain: DomainType,
        transplant_type: TransplantType,
        distance_info: DomainDistance
    ) -> float:
        """Calculate how feasible this transplant is likely to be"""
        
        # Base feasibility from domain distance
        base_feasibility = distance_info.transplant_viability
        
        # Adjust for transplant type complexity
        type_complexity = {
            TransplantType.PRINCIPLE_EXTRACTION: 0.9,      # Easiest
            TransplantType.FUNCTIONAL_MIMICRY: 0.8,        
            TransplantType.PATTERN_MAPPING: 0.7,           
            TransplantType.PROCESS_ADAPTATION: 0.6,        
            TransplantType.STRUCTURAL_ANALOGY: 0.5,        
            TransplantType.SYSTEM_HYBRIDIZATION: 0.3       # Hardest
        }
        
        type_factor = type_complexity.get(transplant_type, 0.5)
        
        # Combine factors
        feasibility = (base_feasibility * 0.7) + (type_factor * 0.3)
        
        return min(1.0, max(0.1, feasibility))
    
    def _calculate_transplant_novelty(self, distance_info: DomainDistance, transplant_type: TransplantType) -> float:
        """Calculate novelty score for the transplant"""
        
        # Higher domain distance generally means higher novelty
        distance_novelty = distance_info.distance_score
        
        # Some transplant types are inherently more novel
        type_novelty = {
            TransplantType.SYSTEM_HYBRIDIZATION: 1.0,      # Most novel
            TransplantType.STRUCTURAL_ANALOGY: 0.9,        
            TransplantType.PATTERN_MAPPING: 0.8,           
            TransplantType.PROCESS_ADAPTATION: 0.7,        
            TransplantType.FUNCTIONAL_MIMICRY: 0.6,        
            TransplantType.PRINCIPLE_EXTRACTION: 0.5       # Least novel
        }
        
        type_factor = type_novelty.get(transplant_type, 0.5)
        
        # Combine distance and type novelty
        novelty = (distance_novelty * 0.6) + (type_factor * 0.4)
        
        return min(1.0, max(0.1, novelty))
    
    def _calculate_potential_impact(self, query: str, transplanted_solution: str) -> float:
        """Calculate potential impact of the transplanted solution"""
        
        # Look for impact indicators in query and solution
        high_impact_terms = ["optimization", "efficiency", "breakthrough", "innovation", "revolutionary", "novel"]
        medium_impact_terms = ["improvement", "enhancement", "better", "advanced", "effective"]
        
        query_lower = query.lower()
        solution_lower = transplanted_solution.lower()
        
        high_impact_score = sum(1 for term in high_impact_terms if term in query_lower or term in solution_lower)
        medium_impact_score = sum(1 for term in medium_impact_terms if term in query_lower or term in solution_lower)
        
        # Calculate impact score
        impact_score = (high_impact_score * 0.3) + (medium_impact_score * 0.15) + 0.4  # Base impact
        
        return min(1.0, max(0.2, impact_score))
    
    def _calculate_implementation_complexity(self, transplant_type: TransplantType, source_solution: SolutionPattern) -> float:
        """Calculate implementation complexity"""
        
        # Base complexity by transplant type
        type_complexity = {
            TransplantType.PRINCIPLE_EXTRACTION: 0.3,      # Lowest complexity
            TransplantType.FUNCTIONAL_MIMICRY: 0.4,        
            TransplantType.PATTERN_MAPPING: 0.5,           
            TransplantType.PROCESS_ADAPTATION: 0.6,        
            TransplantType.STRUCTURAL_ANALOGY: 0.7,        
            TransplantType.SYSTEM_HYBRIDIZATION: 0.9       # Highest complexity
        }
        
        base_complexity = type_complexity.get(transplant_type, 0.5)
        
        # Adjust for solution complexity (based on number of elements)
        solution_complexity_factor = min(0.3, len(source_solution.structural_elements) * 0.1)
        
        total_complexity = base_complexity + solution_complexity_factor
        
        return min(1.0, max(0.1, total_complexity))
    
    def _identify_analogous_elements(self, source_solution: SolutionPattern, target_domain: DomainType) -> List[str]:
        """Identify elements that are analogous between source and target"""
        
        analogous = []
        
        # Map structural elements
        for element in source_solution.structural_elements:
            adapted_element = f"{element} → {target_domain.value}_equivalent"
            analogous.append(adapted_element)
        
        # Map functional elements
        for element in source_solution.functional_elements:
            adapted_element = f"{element} → {target_domain.value}_function"
            analogous.append(adapted_element)
        
        return analogous[:5]  # Limit to top 5
    
    def _identify_key_differences(self, source_domain: DomainType, target_domain: DomainType) -> List[str]:
        """Identify key differences that need to be addressed in transplantation"""
        
        differences = [
            f"{source_domain.value} operates in {source_domain.value} context while {target_domain.value} has different constraints",
            f"Scale and resource requirements may differ between {source_domain.value} and {target_domain.value}",
            f"Success metrics and validation approaches differ across domains",
            f"Implementation technologies and tools are domain-specific"
        ]
        
        # Add domain-specific differences
        source_info = self.domain_mappings.get(source_domain, {})
        target_info = self.domain_mappings.get(target_domain, {})
        
        source_problems = set(source_info.get("problem_types", []))
        target_problems = set(target_info.get("problem_types", []))
        
        unique_source = source_problems - target_problems
        unique_target = target_problems - source_problems
        
        if unique_source:
            differences.append(f"{source_domain.value} focuses on {list(unique_source)[0]} which is not primary in {target_domain.value}")
        
        if unique_target:
            differences.append(f"{target_domain.value} requires attention to {list(unique_target)[0]} which is not emphasized in {source_domain.value}")
        
        return differences[:4]  # Limit to top 4
    
    def _generate_validation_criteria(self, transplanted_solution: str, target_domain: DomainType) -> List[str]:
        """Generate criteria for validating the transplanted solution"""
        
        criteria = [
            "Functional equivalence testing between source and target implementations",
            "Performance benchmarking against existing approaches",
            "Scalability assessment for target domain constraints",
            "Resource efficiency evaluation"
        ]
        
        # Add domain-specific validation criteria
        target_info = self.domain_mappings.get(target_domain, {})
        problem_types = target_info.get("problem_types", [])
        
        if "optimization" in problem_types:
            criteria.append("Optimization effectiveness measurement")
        
        if "reliability" in problem_types:
            criteria.append("Reliability and fault tolerance testing")
        
        if "scalability" in problem_types:
            criteria.append("Scalability limits and breaking points analysis")
        
        return criteria[:5]  # Limit to top 5 criteria

# Global instance
cross_domain_transplant_engine = CrossDomainTransplantEngine()

async def generate_cross_domain_transplants(
    query: str,
    context: Dict[str, Any],
    papers: List[Dict[str, Any]] = None,
    max_transplants: int = 5
) -> List[CrossDomainTransplant]:
    """Convenience function to generate cross-domain transplants"""
    return await cross_domain_transplant_engine.generate_cross_domain_transplants(
        query, context, papers, max_transplants
    )