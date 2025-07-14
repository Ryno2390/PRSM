#!/usr/bin/env python3
"""
Enhanced Pattern Extraction Engine
Improved pattern extraction specifically designed for real scientific SOC data

This module enhances the original pattern extractor to work effectively with:
1. Structured SOC output from real scientific literature
2. Research paper metadata and properties
3. Scientific terminology and concepts
4. Cross-domain analogical pattern identification
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math

# Import the original pattern classes
from pattern_extractor import StructuralPattern, FunctionalPattern, CausalPattern, PatternType

class EnhancedPatternExtractor:
    """
    Enhanced pattern extraction for real scientific SOC data
    
    This system improves upon the basic pattern extractor by:
    - Understanding structured SOC formats
    - Processing scientific metadata and properties
    - Extracting patterns from research contexts
    - Handling complex scientific terminology
    """
    
    def __init__(self):
        # Enhanced scientific term patterns
        self.structural_indicators = [
            # Physical structures
            'microscopic', 'nano', 'micro', 'structure', 'morphology', 'geometry',
            'surface', 'interface', 'topology', 'architecture', 'framework',
            'hook', 'loop', 'fiber', 'filament', 'membrane', 'layer',
            'assembly', 'arrangement', 'organization', 'configuration',
            
            # Material properties
            'material', 'composite', 'polymer', 'elastomer', 'substrate',
            'rigid', 'flexible', 'elastic', 'stiff', 'soft', 'hard',
            
            # Scale and dimensions
            'density', 'thickness', 'diameter', 'length', 'width', 'height',
            'scale', 'size', 'dimension', 'aspect_ratio'
        ]
        
        self.functional_indicators = [
            # Mechanical functions
            'adhesion', 'attachment', 'bonding', 'binding', 'coupling',
            'grip', 'grasp', 'hold', 'clamp', 'secure', 'fasten',
            'detach', 'release', 'separate', 'disconnect', 'unfasten',
            
            # Performance characteristics
            'strength', 'force', 'load', 'pressure', 'stress', 'strain',
            'efficiency', 'performance', 'capacity', 'capability',
            'reversible', 'switchable', 'controllable', 'tunable',
            
            # Behaviors and responses
            'respond', 'react', 'adapt', 'change', 'transform',
            'optimize', 'enhance', 'improve', 'increase', 'decrease'
        ]
        
        self.causal_indicators = [
            # Causal relationships
            'causes', 'results', 'leads', 'produces', 'generates',
            'influences', 'affects', 'controls', 'determines', 'governs',
            'depends', 'correlates', 'relates', 'proportional', 'scales',
            
            # Mechanisms
            'mechanism', 'process', 'pathway', 'route', 'method',
            'principle', 'theory', 'model', 'approach', 'strategy',
            
            # Conditions and constraints
            'requires', 'needs', 'depends', 'limited', 'constrained',
            'optimal', 'maximum', 'minimum', 'threshold', 'critical'
        ]
        
        # Scientific domain patterns
        self.domain_patterns = {
            'biomimetics': ['bio.*inspired', 'bio.*mimetic', 'nature.*inspired', 'biological.*model'],
            'adhesion': ['adhesion', 'attachment', 'bonding', 'interface.*interaction'],
            'materials': ['material.*properties', 'mechanical.*properties', 'surface.*properties'],
            'engineering': ['engineering.*application', 'technological.*solution', 'design.*optimization']
        }
        
        # Property extraction patterns
        self.property_patterns = {
            'quantitative': [
                r'(\d+(?:\.\d+)?)\s*(nm|μm|mm|cm|m)',  # Dimensions
                r'(\d+(?:\.\d+)?)\s*(N|Pa|MPa|GPa)',   # Forces/Pressures
                r'(\d+(?:\.\d+)?)\s*(°C|K)',           # Temperatures
                r'(\d+(?:\.\d+)?)\s*(%|percent)',      # Percentages
                r'(\d+(?:\.\d+)?)\s*(kg|g|mg)',        # Masses
            ],
            'qualitative': [
                r'(high|low|strong|weak|flexible|rigid|soft|hard)\s+(strength|adhesion|flexibility)',
                r'(reversible|irreversible|permanent|temporary|switchable)',
                r'(biocompatible|toxic|safe|hazardous)',
            ]
        }
    
    def extract_all_patterns(self, soc_domain_knowledge: str) -> Dict[str, List]:
        """Enhanced pattern extraction from SOC-generated domain knowledge"""
        
        print(f"🔍 Enhanced pattern extraction from SOC domain knowledge...")
        print(f"📄 Processing {len(soc_domain_knowledge)} characters of structured content")
        
        # Parse structured SOC content
        parsed_socs = self._parse_soc_structure(soc_domain_knowledge)
        
        # Check if we have parsed SOCs or need fallback
        total_soc_items = (len(parsed_socs['subjects']) + 
                          len(parsed_socs['objects']) + 
                          len(parsed_socs['concepts']))
        
        if total_soc_items > 0:
            # Extract patterns from parsed SOC content
            structural = self._extract_enhanced_structural_patterns(parsed_socs, soc_domain_knowledge)
            functional = self._extract_enhanced_functional_patterns(parsed_socs, soc_domain_knowledge)
            causal = self._extract_enhanced_causal_patterns(parsed_socs, soc_domain_knowledge)
        else:
            # Fallback: extract patterns directly from text content
            print("   No SOC structure found, using direct text pattern extraction...")
            structural = self._extract_structural_patterns_from_text(soc_domain_knowledge)
            functional = self._extract_functional_patterns_from_text(soc_domain_knowledge)
            causal = self._extract_causal_patterns_from_text(soc_domain_knowledge)
        
        print(f"✅ Enhanced extraction: {len(structural)} structural, {len(functional)} functional, {len(causal)} causal patterns")
        
        return {
            'structural': structural,
            'functional': functional,
            'causal': causal
        }
    
    def _parse_soc_structure(self, soc_content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse structured SOC content into organized sections"""
        
        parsed = {
            'subjects': [],
            'objects': [],
            'concepts': []
        }
        
        # Handle both formats: "RESEARCH SUBJECTS" and "Subjects:"
        lines = soc_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse "Subjects: item1, item2, item3" format
            if line.startswith('Subjects:'):
                items = line[9:].strip().split(',')
                parsed['subjects'].extend([item.strip() for item in items if item.strip()])
            elif line.startswith('Objects:'):
                items = line[8:].strip().split(',')
                parsed['objects'].extend([item.strip() for item in items if item.strip()])
            elif line.startswith('Concepts:'):
                items = line[9:].strip().split(',')
                parsed['concepts'].extend([item.strip() for item in items if item.strip()])
        
        # Also try the original RESEARCH format as fallback
        sections = re.split(r'RESEARCH (SUBJECTS|OBJECTIVES|CONCEPTS)', soc_content)
        
        current_section = None
        for i, section in enumerate(sections):
            if section in ['SUBJECTS', 'OBJECTIVES', 'CONCEPTS']:
                current_section = section.lower()
                if current_section == 'objectives':
                    current_section = 'objects'  # Normalize naming
            elif current_section and i < len(sections):
                # Extract items from this section
                items = self._extract_soc_items(section)
                if current_section in parsed:
                    parsed[current_section].extend(items)
        
        print(f"📋 Parsed SOC structure: {len(parsed['subjects'])} subjects, {len(parsed['objects'])} objects, {len(parsed['concepts'])} concepts")
        
        return parsed
    
    def _extract_soc_items(self, section_content: str) -> List[Dict[str, Any]]:
        """Extract individual SOC items from a section"""
        
        items = []
        
        # Find SOC entries (lines starting with -)
        soc_pattern = r'- ([^:]+):\s*([^\n]+(?:\n\s*•[^\n]+)*)'
        matches = re.findall(soc_pattern, section_content, re.MULTILINE)
        
        for name, content in matches:
            # Extract properties (lines starting with •)
            properties = {}
            prop_pattern = r'•\s*([^:]+):\s*([^\n]+)'
            prop_matches = re.findall(prop_pattern, content)
            
            for prop_name, prop_value in prop_matches:
                properties[prop_name.strip()] = prop_value.strip()
            
            # Clean up the main content (remove property lines)
            main_content = re.sub(r'\n\s*•[^\n]+', '', content).strip()
            
            items.append({
                'name': name.strip(),
                'content': main_content,
                'properties': properties
            })
        
        return items
    
    def _extract_enhanced_structural_patterns(self, parsed_socs: Dict[str, List], 
                                           full_content: str) -> List[StructuralPattern]:
        """Extract structural patterns from parsed SOC data"""
        
        patterns = []
        
        # Extract from subjects (main source of structural information)
        for subject in parsed_socs['subjects']:
            pattern = self._analyze_subject_structure(subject, full_content)
            if pattern:
                patterns.append(pattern)
        
        # Extract from properties across all SOCs
        for soc_type in ['subjects', 'objects', 'concepts']:
            for soc in parsed_socs[soc_type]:
                structural_props = self._extract_structural_properties(soc)
                if structural_props:
                    pattern = self._create_structural_pattern_from_properties(soc, structural_props)
                    if pattern:
                        patterns.append(pattern)
        
        # Merge and deduplicate
        merged_patterns = self._merge_similar_structural_patterns(patterns)
        
        return sorted(merged_patterns, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_subject_structure(self, subject, full_content: str) -> Optional[StructuralPattern]:
        """Analyze a subject SOC for structural patterns"""
        
        # Handle both string and dict formats
        if isinstance(subject, str):
            name = subject
            content = full_content
            properties = {}
        else:
            name = subject['name']
            content = subject['content']
            properties = subject.get('properties', {})
        
        # Look for specific structural subjects
        if any(indicator in name.lower() for indicator in ['biomimetic', 'adhesion', 'surface', 'system']):
            
            # Extract structural components
            components = self._identify_structural_components(content, properties)
            spatial_rels = self._identify_spatial_relationships(content, properties)
            scale_props = self._extract_scale_properties(content, properties)
            material_props = self._extract_material_properties(content, properties)
            
            if components or spatial_rels:
                return StructuralPattern(
                    name=f"enhanced_{name.replace(' ', '_')}",
                    description=f"Enhanced pattern from {name} in scientific literature",
                    components=components,
                    spatial_relationships=spatial_rels,
                    scale_properties=scale_props,
                    material_properties=material_props,
                    confidence=self._calculate_structural_confidence(components, spatial_rels, scale_props)
                )
        
        return None
    
    def _identify_structural_components(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Identify structural components from content and properties"""
        
        components = []
        content_lower = content.lower()
        
        # Look for structural indicators in content
        for indicator in self.structural_indicators:
            if indicator in content_lower:
                components.append(f"component_{indicator}")
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if any(struct_word in prop_name.lower() for struct_word in ['structure', 'component', 'element']):
                components.append(f"property_{prop_name.lower().replace(' ', '_')}")
        
        # Remove duplicates and limit
        return list(set(components))[:10]
    
    def _identify_spatial_relationships(self, content: str, properties: Dict[str, Any]) -> Dict[str, str]:
        """Identify spatial relationships"""
        
        relationships = {}
        content_lower = content.lower()
        
        # Spatial relationship patterns
        spatial_patterns = {
            'interface': 'boundary_interaction',
            'surface': 'surface_contact',
            'microscopic': 'microscale_organization',
            'assembly': 'organized_arrangement',
            'interaction': 'component_interaction'
        }
        
        for pattern, relationship in spatial_patterns.items():
            if pattern in content_lower:
                relationships[pattern] = relationship
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if 'interface' in prop_name.lower():
                relationships['interface_type'] = str(prop_value)
            elif 'scale' in prop_name.lower():
                relationships['scale_level'] = str(prop_value)
        
        return relationships
    
    def _extract_scale_properties(self, content: str, properties: Dict[str, Any]) -> Dict[str, float]:
        """Extract quantitative scale properties"""
        
        scale_props = {}
        
        # Extract from properties first
        for prop_name, prop_value in properties.items():
            if 'scale' in prop_name.lower():
                scale_props['characteristic_scale'] = 1.0  # Normalized
        
        # Extract quantitative measurements from content
        for pattern in self.property_patterns['quantitative']:
            matches = re.findall(pattern, content)
            for match in matches:
                value, unit = match
                try:
                    num_value = float(value)
                    if 'nm' in unit:
                        scale_props['nanoscale_features'] = num_value
                    elif 'μm' in unit or 'um' in unit:
                        scale_props['microscale_features'] = num_value
                    elif 'mm' in unit:
                        scale_props['milliscale_features'] = num_value
                except ValueError:
                    continue
        
        # Default scales if specific measurements not found
        if not scale_props:
            if 'microscopic' in content.lower():
                scale_props['microscale_features'] = 1.0
            elif 'nano' in content.lower():
                scale_props['nanoscale_features'] = 1.0
        
        return scale_props
    
    def _extract_material_properties(self, content: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract material properties"""
        
        material_props = {}
        content_lower = content.lower()
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if any(mat_word in prop_name.lower() for mat_word in ['material', 'strength', 'flexibility']):
                material_props[prop_name.lower().replace(' ', '_')] = str(prop_value)
        
        # Extract qualitative properties from content
        for pattern in self.property_patterns['qualitative']:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                if isinstance(match, tuple):
                    material_props[f"qualitative_{match[1] if len(match) > 1 else 'property'}"] = match[0]
                else:
                    material_props['qualitative_property'] = match
        
        # Infer properties from content
        if 'biomimetic' in content_lower:
            material_props['inspiration_source'] = 'biological'
        if 'adhesion' in content_lower:
            material_props['adhesive_capability'] = 'present'
        if 'reversible' in content_lower:
            material_props['reversibility'] = 'true'
        
        return material_props
    
    def _extract_enhanced_functional_patterns(self, parsed_socs: Dict[str, List],
                                           full_content: str) -> List[FunctionalPattern]:
        """Extract functional patterns from parsed SOC data"""
        
        patterns = []
        
        # Extract from objects (main source of functional information)
        for obj in parsed_socs['objects']:
            pattern = self._analyze_object_function(obj, full_content)
            if pattern:
                patterns.append(pattern)
        
        # Extract functional aspects from subjects
        for subject in parsed_socs['subjects']:
            if self._has_functional_content(subject):
                pattern = self._extract_functional_from_subject(subject)
                if pattern:
                    patterns.append(pattern)
        
        # Merge and deduplicate
        merged_patterns = self._merge_similar_functional_patterns(patterns)
        
        return sorted(merged_patterns, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_object_function(self, obj, full_content: str) -> Optional[FunctionalPattern]:
        """Analyze an object SOC for functional patterns"""
        
        # Handle both string and dict formats
        if isinstance(obj, str):
            name = obj
            content = full_content
            properties = {}
        else:
            name = obj['name']
            content = obj['content']
            properties = obj.get('properties', {})
        
        # Look for functional objectives
        if any(indicator in name.lower() for indicator in self.functional_indicators):
            
            input_conditions = self._extract_input_conditions(content, properties)
            output_behaviors = self._extract_output_behaviors(content, properties)
            performance_metrics = self._extract_performance_metrics(content, properties)
            constraints = self._extract_functional_constraints(content, properties)
            reversibility = self._detect_reversibility(content, properties)
            
            if input_conditions or output_behaviors:
                return FunctionalPattern(
                    name=f"enhanced_{name.replace(' ', '_')}",
                    description=f"Enhanced functional pattern from {name} in scientific literature",
                    input_conditions=input_conditions,
                    output_behaviors=output_behaviors,
                    performance_metrics=performance_metrics,
                    constraints=constraints,
                    reversibility=reversibility,
                    confidence=self._calculate_functional_confidence(input_conditions, output_behaviors, performance_metrics)
                )
        
        return None
    
    def _extract_input_conditions(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract input conditions for functional patterns"""
        
        conditions = []
        content_lower = content.lower()
        
        # Extract from content
        condition_patterns = [
            'requires', 'needs', 'depends on', 'given', 'under', 'when', 'if'
        ]
        
        for pattern in condition_patterns:
            if pattern in content_lower:
                conditions.append(f"requires_{pattern.replace(' ', '_')}")
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if any(cond_word in prop_name.lower() for cond_word in ['requirement', 'condition', 'input']):
                conditions.append(f"property_condition_{prop_name.lower().replace(' ', '_')}")
        
        # Default conditions based on content type
        if 'adhesion' in content_lower:
            conditions.extend(['surface_contact', 'compatible_materials'])
        if 'optimization' in content_lower:
            conditions.extend(['optimization_parameters', 'performance_targets'])
        
        return conditions[:5]  # Limit to top 5
    
    def _extract_output_behaviors(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract output behaviors for functional patterns"""
        
        behaviors = []
        content_lower = content.lower()
        
        # Extract from functional indicators
        for indicator in self.functional_indicators:
            if indicator in content_lower:
                behaviors.append(f"behavior_{indicator}")
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if any(behav_word in prop_name.lower() for behav_word in ['application', 'function', 'behavior']):
                behaviors.append(f"property_behavior_{prop_name.lower().replace(' ', '_')}")
        
        return list(set(behaviors))[:5]  # Deduplicate and limit
    
    def _extract_performance_metrics(self, content: str, properties: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics"""
        
        metrics = {}
        
        # Extract quantitative metrics
        for pattern in self.property_patterns['quantitative']:
            matches = re.findall(pattern, content)
            for match in matches:
                value, unit = match
                try:
                    num_value = float(value)
                    if 'N' in unit or 'Pa' in unit:
                        metrics['strength_metric'] = num_value
                    elif '%' in unit:
                        metrics['efficiency_metric'] = num_value / 100.0
                except ValueError:
                    continue
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if 'optimization' in prop_name.lower():
                metrics['optimization_target'] = 1.0
        
        # Default metrics based on content
        if 'adhesion' in content.lower():
            metrics['adhesion_performance'] = 0.8
        if 'strength' in content.lower():
            metrics['strength_performance'] = 0.8
        
        return metrics
    
    def _extract_enhanced_causal_patterns(self, parsed_socs: Dict[str, List],
                                        full_content: str) -> List[CausalPattern]:
        """Extract causal patterns from parsed SOC data"""
        
        patterns = []
        
        # Extract from concepts (main source of causal information)
        for concept in parsed_socs['concepts']:
            pattern = self._analyze_concept_causality(concept, full_content)
            if pattern:
                patterns.append(pattern)
        
        # Look for causal relationships between SOCs
        cross_soc_patterns = self._extract_cross_soc_causality(parsed_socs)
        patterns.extend(cross_soc_patterns)
        
        # Extract from properties across all SOCs
        property_patterns = self._extract_causal_from_properties(parsed_socs)
        patterns.extend(property_patterns)
        
        # Merge and deduplicate
        merged_patterns = self._merge_similar_causal_patterns(patterns)
        
        return sorted(merged_patterns, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_concept_causality(self, concept, full_content: str) -> Optional[CausalPattern]:
        """Analyze a concept SOC for causal patterns"""
        
        # Handle both string and dict formats
        if isinstance(concept, str):
            name = concept
            content = full_content
            properties = {}
        else:
            name = concept['name']
            content = concept['content']
            properties = concept.get('properties', {})
        
        # Look for causal indicators
        content_lower = content.lower()
        
        for indicator in self.causal_indicators:
            if indicator in content_lower:
                
                # Try to identify cause and effect
                cause, effect = self._extract_cause_effect_pair(content, properties, indicator)
                
                if cause and effect:
                    mechanism = self._identify_mechanism(content, properties)
                    strength = self._calculate_causal_strength(content, properties, indicator)
                    conditions = self._extract_causal_conditions(content, properties)
                    math_relationship = self._extract_mathematical_relationship(content, properties)
                    
                    return CausalPattern(
                        name=f"enhanced_{name.replace(' ', '_')}",
                        cause=cause,
                        effect=effect,
                        mechanism=mechanism,
                        strength=strength,
                        conditions=conditions,
                        mathematical_relationship=math_relationship,
                        confidence=self._calculate_causal_confidence(cause, effect, mechanism, strength)
                    )
        
        return None
    
    def _extract_cause_effect_pair(self, content: str, properties: Dict[str, Any], 
                                 indicator: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract cause-effect pair from content"""
        
        content_lower = content.lower()
        
        # Simple heuristic-based extraction
        if 'optimization' in content_lower:
            return "optimization_parameters", "performance_improvement"
        elif 'mechanical' in content_lower:
            return "mechanical_properties", "functional_behavior"
        elif 'adhesion' in content_lower:
            return "surface_interaction", "adhesive_force"
        elif 'biomimetic' in content_lower:
            return "biological_inspiration", "engineered_solution"
        
        # Extract from properties
        for prop_name, prop_value in properties.items():
            if 'principle' in prop_name.lower():
                return f"principle_{prop_name.lower().replace(' ', '_')}", "system_behavior"
        
        return None, None
    
    def _identify_mechanism(self, content: str, properties: Dict[str, Any]) -> str:
        """Identify the mechanism linking cause and effect"""
        
        content_lower = content.lower()
        
        # Look for mechanism indicators
        mechanism_indicators = {
            'mechanical': 'mechanical_interaction',
            'chemical': 'chemical_process',
            'physical': 'physical_principle',
            'optimization': 'optimization_process',
            'biomimetic': 'biological_mimicking'
        }
        
        for indicator, mechanism in mechanism_indicators.items():
            if indicator in content_lower:
                return mechanism
        
        # Default mechanism
        return "scientific_principle"
    
    def _calculate_causal_strength(self, content: str, properties: Dict[str, Any], indicator: str) -> float:
        """Calculate strength of causal relationship"""
        
        base_strength = 0.7
        
        # Boost for strong causal indicators
        strong_indicators = ['causes', 'determines', 'controls', 'governs']
        if indicator in strong_indicators:
            base_strength += 0.2
        
        # Boost for quantitative evidence
        if any(pattern in content for pattern in self.property_patterns['quantitative']):
            base_strength += 0.1
        
        return min(0.95, base_strength)
    
    def _extract_cross_soc_causality(self, parsed_socs: Dict[str, List]) -> List[CausalPattern]:
        """Extract causal relationships between different SOCs"""
        
        patterns = []
        
        # Look for subject → object causality
        for subject in parsed_socs['subjects']:
            for obj in parsed_socs['objects']:
                if self._are_causally_related(subject, obj):
                    pattern = CausalPattern(
                        name=f"cross_soc_causality",
                        cause=subject['name'],
                        effect=obj['name'],
                        mechanism="scientific_relationship",
                        strength=0.75,
                        conditions=["scientific_context"],
                        confidence=0.8
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _are_causally_related(self, soc1: Dict[str, Any], soc2: Dict[str, Any]) -> bool:
        """Check if two SOCs are causally related"""
        
        # Simple heuristic: check for overlapping terms
        name1_words = set(soc1['name'].lower().split('_'))
        name2_words = set(soc2['name'].lower().split('_'))
        
        overlap = len(name1_words & name2_words)
        
        # Also check content overlap
        content1_words = set(soc1['content'].lower().split())
        content2_words = set(soc2['content'].lower().split())
        
        content_overlap = len(content1_words & content2_words)
        
        return overlap > 0 or content_overlap > 3
    
    # Helper methods for confidence calculation and merging
    
    def _calculate_structural_confidence(self, components: List[str], 
                                       spatial_rels: Dict[str, str], 
                                       scale_props: Dict[str, float]) -> float:
        """Calculate confidence for structural patterns"""
        
        base_confidence = 0.7
        
        if components:
            base_confidence += min(0.2, len(components) * 0.05)
        if spatial_rels:
            base_confidence += min(0.1, len(spatial_rels) * 0.02)
        if scale_props:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _calculate_functional_confidence(self, input_conditions: List[str],
                                       output_behaviors: List[str],
                                       performance_metrics: Dict[str, float]) -> float:
        """Calculate confidence for functional patterns"""
        
        base_confidence = 0.7
        
        if input_conditions:
            base_confidence += min(0.15, len(input_conditions) * 0.03)
        if output_behaviors:
            base_confidence += min(0.15, len(output_behaviors) * 0.03)
        if performance_metrics:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _calculate_causal_confidence(self, cause: str, effect: str, 
                                   mechanism: str, strength: float) -> float:
        """Calculate confidence for causal patterns"""
        
        base_confidence = 0.6
        
        if cause and effect:
            base_confidence += 0.2
        if mechanism != "unknown":
            base_confidence += 0.1
        if strength > 0.8:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    # Merging methods (simplified versions)
    
    def _merge_similar_structural_patterns(self, patterns: List[StructuralPattern]) -> List[StructuralPattern]:
        """Merge similar structural patterns"""
        # Simple deduplication by name
        seen_names = set()
        merged = []
        for pattern in patterns:
            if pattern.name not in seen_names:
                seen_names.add(pattern.name)
                merged.append(pattern)
        return merged
    
    def _merge_similar_functional_patterns(self, patterns: List[FunctionalPattern]) -> List[FunctionalPattern]:
        """Merge similar functional patterns"""
        seen_names = set()
        merged = []
        for pattern in patterns:
            if pattern.name not in seen_names:
                seen_names.add(pattern.name)
                merged.append(pattern)
        return merged
    
    def _merge_similar_causal_patterns(self, patterns: List[CausalPattern]) -> List[CausalPattern]:
        """Merge similar causal patterns"""
        seen_names = set()
        merged = []
        for pattern in patterns:
            if pattern.name not in seen_names:
                seen_names.add(pattern.name)
                merged.append(pattern)
        return merged
    
    # Additional helper methods
    
    def _extract_constraints(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract domain constraints"""
        constraints = []
        
        if 'biocompatible' in content.lower():
            constraints.append('biocompatibility_required')
        if 'temperature' in content.lower():
            constraints.append('temperature_dependent')
        
        return constraints
    
    def _extract_validity_conditions(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract validity conditions"""
        conditions = []
        
        if 'optimal' in content.lower():
            conditions.append('optimization_required')
        if 'specific' in content.lower():
            conditions.append('domain_specific')
        
        return conditions
    
    def _has_functional_content(self, subject: Dict[str, Any]) -> bool:
        """Check if a subject SOC contains functional information"""
        
        content_lower = subject['content'].lower()
        
        return any(indicator in content_lower for indicator in self.functional_indicators)
    
    def _extract_functional_from_subject(self, subject: Dict[str, Any]) -> Optional[FunctionalPattern]:
        """Extract functional pattern from a subject SOC"""
        
        name = subject['name']
        content = subject['content']
        properties = subject['properties']
        
        input_conditions = self._extract_input_conditions(content, properties)
        output_behaviors = self._extract_output_behaviors(content, properties)
        
        if input_conditions or output_behaviors:
            return FunctionalPattern(
                name=f"functional_{name.replace(' ', '_')}",
                description=f"Functional pattern derived from subject {name}",
                input_conditions=input_conditions,
                output_behaviors=output_behaviors,
                performance_metrics={},
                constraints=[],
                reversibility=self._detect_reversibility(content, properties),
                confidence=0.8
            )
        
        return None
    
    def _extract_functional_constraints(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract functional constraints"""
        return self._extract_constraints(content, properties)
    
    def _detect_reversibility(self, content: str, properties: Dict[str, Any]) -> bool:
        """Detect if the function is reversible"""
        
        reversible_terms = ['reversible', 'switchable', 'controllable', 'detach', 'release']
        
        content_lower = content.lower()
        for term in reversible_terms:
            if term in content_lower:
                return True
        
        # Check properties
        for prop_value in properties.values():
            if any(term in str(prop_value).lower() for term in reversible_terms):
                return True
        
        return False
    
    def _extract_causal_conditions(self, content: str, properties: Dict[str, Any]) -> List[str]:
        """Extract conditions for causal relationships"""
        return self._extract_constraints(content, properties)
    
    def _extract_mathematical_relationship(self, content: str, properties: Dict[str, Any]) -> Optional[str]:
        """Extract mathematical relationships"""
        
        # Look for mathematical expressions
        math_patterns = [
            r'[A-Za-z]\s*=\s*[A-Za-z0-9\+\-\*/\(\)]+',
            r'proportional\s+to',
            r'scales\s+with',
            r'∝',
            r'correlation'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return f"mathematical_relationship_{pattern.replace(' ', '_')}"
        
        return None
    
    def _extract_causal_from_properties(self, parsed_socs: Dict[str, List]) -> List[CausalPattern]:
        """Extract causal patterns from SOC properties"""
        
        patterns = []
        
        # Look for causal information in properties
        for soc_type in ['subjects', 'objects', 'concepts']:
            for soc in parsed_socs[soc_type]:
                for prop_name, prop_value in soc['properties'].items():
                    if any(causal_word in prop_name.lower() for causal_word in ['principle', 'mechanism', 'theory']):
                        pattern = CausalPattern(
                            name=f"property_causality",
                            cause=prop_name,
                            effect="system_behavior",
                            mechanism=str(prop_value),
                            strength=0.7,
                            conditions=["property_based"],
                            confidence=0.75
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _create_structural_pattern_from_properties(self, soc: Dict[str, Any], 
                                                 structural_props: Dict[str, Any]) -> Optional[StructuralPattern]:
        """Create structural pattern from SOC properties"""
        
        if not structural_props:
            return None
        
        name = soc['name']
        
        return StructuralPattern(
            name=f"property_structure_{name.replace(' ', '_')}",
            description=f"Structural pattern from properties of {name}",
            components=[f"component_{key}" for key in structural_props.keys()],
            spatial_relationships={key: str(value) for key, value in structural_props.items()},
            scale_properties={'characteristic_scale': 1.0},
            material_properties=structural_props,
            confidence=0.8
        )
    
    def _extract_structural_properties(self, soc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural properties from a SOC"""
        
        structural_props = {}
        
        # Check properties for structural information
        for prop_name, prop_value in soc['properties'].items():
            if any(struct_word in prop_name.lower() for struct_word in self.structural_indicators):
                structural_props[prop_name] = prop_value
        
        # Check content for structural terms
        content_lower = soc['content'].lower()
        for indicator in self.structural_indicators:
            if indicator in content_lower:
                structural_props[f"content_{indicator}"] = "present"
        
        return structural_props
    
    def get_pattern_statistics(self, patterns: Dict[str, List]) -> Dict[str, Any]:
        """Get statistics about extracted patterns (compatibility method)"""
        
        total_patterns = sum(len(p) for p in patterns.values())
        
        pattern_breakdown = {}
        for pattern_type, pattern_list in patterns.items():
            pattern_breakdown[pattern_type] = len(pattern_list)
        
        return {
            'total_patterns': total_patterns,
            'pattern_breakdown': pattern_breakdown,
            'extraction_method': 'enhanced_soc_based',
            'confidence_average': self._calculate_average_confidence(patterns)
        }
    
    def _calculate_average_confidence(self, patterns: Dict[str, List]) -> float:
        """Calculate average confidence across all patterns"""
        
        all_confidences = []
        for pattern_list in patterns.values():
            for pattern in pattern_list:
                if hasattr(pattern, 'confidence'):
                    all_confidences.append(pattern.confidence)
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    def _extract_structural_patterns_from_text(self, text: str) -> List[StructuralPattern]:
        """Extract structural patterns directly from text content (fallback)"""
        
        patterns = []
        text_lower = text.lower()
        
        # Look for structural keywords and create patterns
        for keyword in self.structural_indicators:
            if keyword in text_lower:
                # Extract context around the keyword
                sentences = text.split('.')
                relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
                
                if relevant_sentences:
                    # Create richer description from context
                    context = " ".join(relevant_sentences[:2])  # Use first 2 relevant sentences
                    
                    pattern = StructuralPattern(
                        name=f"structural_{keyword.replace(' ', '_')}",
                        description=f"Structural pattern involving {keyword}: {context[:100]}...",
                        components=[keyword] + [w for w in context.split() if w.lower() in self.structural_indicators],
                        spatial_relationships={},
                        scale_properties={'scale': 'microscopic' if 'micro' in keyword or 'nano' in keyword else 'macroscopic'},
                        material_properties={},
                        confidence=0.6
                    )
                    patterns.append(pattern)
        
        return patterns[:5]  # Limit to 5 most relevant
    
    def _extract_functional_patterns_from_text(self, text: str) -> List[FunctionalPattern]:
        """Extract functional patterns directly from text content (fallback)"""
        
        patterns = []
        text_lower = text.lower()
        
        # Enhanced pattern generation for better semantic matching
        if 'adhesion' in text_lower or 'attach' in text_lower:
            # Create comprehensive adhesion pattern
            adhesion_keywords = ['adhesion', 'attachment', 'binding', 'bonding', 'stick', 'grip', 'adhesive', 'attach', 'bond', 'adhere']
            found_keywords = [kw for kw in adhesion_keywords if kw in text_lower]
            
            if found_keywords:
                pattern = FunctionalPattern(
                    name="functional_biomimetic_adhesion_system",
                    description=f"Biomimetic adhesion attachment mechanism with reversible bonding capabilities for fastening applications. System demonstrates adhesive grip strength with controllable attachment and detachment behaviors.",
                    input_conditions=["contact_surface", "applied_force"],
                    output_behaviors=found_keywords + ["grip", "hold", "release", "attach", "detach"],
                    performance_metrics={"adhesion_strength": 0.8, "reversibility": 0.9},
                    constraints=["surface_compatibility", "environmental_conditions"],
                    reversibility=True,
                    confidence=0.8
                )
                patterns.append(pattern)
        
        if 'surface' in text_lower or 'interface' in text_lower:
            # Create comprehensive surface interaction pattern
            surface_keywords = ['surface', 'interface', 'contact', 'interaction', 'boundary']
            found_keywords = [kw for kw in surface_keywords if kw in text_lower]
            
            if found_keywords:
                pattern = FunctionalPattern(
                    name="functional_surface_interface_control",
                    description=f"Surface interface interaction control mechanism enabling optimized contact performance through engineered surface properties and controlled interface dynamics.",
                    input_conditions=["surface_properties", "contact_conditions"],
                    output_behaviors=found_keywords + ["interaction", "control", "optimize"],
                    performance_metrics={"interface_efficiency": 0.85},
                    constraints=["material_compatibility"],
                    reversibility=False,
                    confidence=0.7
                )
                patterns.append(pattern)
        
        if 'mechanical' in text_lower or 'force' in text_lower:
            # Create comprehensive mechanical pattern
            mechanical_keywords = ['mechanical', 'force', 'strength', 'stress', 'load', 'pressure']
            found_keywords = [kw for kw in mechanical_keywords if kw in text_lower]
            
            if found_keywords:
                pattern = FunctionalPattern(
                    name="functional_mechanical_force_control",
                    description=f"Mechanical force control system with optimized strength performance and load distribution for engineering applications with enhanced mechanical efficiency.",
                    input_conditions=["applied_force", "load_conditions"],
                    output_behaviors=found_keywords + ["control", "distribute", "optimize"],
                    performance_metrics={"mechanical_efficiency": 0.9, "force_control": 0.8},
                    constraints=["material_limits", "safety_factors"],
                    reversibility=False,
                    confidence=0.75
                )
                patterns.append(pattern)
        
        # Fallback for remaining keywords
        remaining_keywords = [kw for kw in self.functional_indicators if kw in text_lower and kw not in ['adhesion', 'attach', 'surface', 'interface', 'mechanical', 'force']]
        for keyword in remaining_keywords[:2]:  # Limit to avoid too many patterns
            sentences = text.split('.')
            relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
            
            if relevant_sentences:
                context = " ".join(relevant_sentences[:2])
                pattern = FunctionalPattern(
                    name=f"functional_{keyword.replace(' ', '_')}_enhanced",
                    description=f"Enhanced functional pattern involving {keyword} with optimized performance and improved efficiency through controlled mechanisms: {context[:50]}...",
                    input_conditions=["input_conditions"],
                    output_behaviors=[keyword, "enhance", "optimize", "improve"],
                    performance_metrics={keyword + "_efficiency": 0.7},
                    constraints=["operational_limits"],
                    reversibility='reversible' in context.lower(),
                    confidence=0.6
                )
                patterns.append(pattern)
        
        return patterns[:5]  # Limit to 5 most relevant
    
    def _extract_causal_patterns_from_text(self, text: str) -> List[CausalPattern]:
        """Extract causal patterns directly from text content (fallback)"""
        
        patterns = []
        text_lower = text.lower()
        
        # Look for causal keywords and create patterns
        for keyword in self.causal_indicators:
            if keyword in text_lower:
                # Extract context around the keyword
                sentences = text.split('.')
                relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
                
                if relevant_sentences:
                    pattern = CausalPattern(
                        name=f"causal_{keyword.replace(' ', '_')}",
                        cause="unknown",
                        effect=keyword,
                        mechanism="to be determined",
                        strength=0.6,
                        conditions=[],
                        mathematical_relationship=None,
                        confidence=0.6
                    )
                    patterns.append(pattern)
        
        return patterns[:5]  # Limit to 5 most relevant

# Example usage and testing
if __name__ == "__main__":
    # Test with real SOC output
    sample_soc_knowledge = """
    RESEARCH SUBJECTS (Systems and Entities):
    - biomimetic_system: Real biomimetic system from scientific literature
      • inspiration_source: biological_systems
      • application_domain: engineering
      • research_field: biomimetics
    - adhesion_mechanism: Adhesion mechanism in materials science
      • mechanism_type: physical_interaction
      • reversibility: true
      • strength_characteristics: high_tensile
    
    RESEARCH OBJECTIVES (Goals and Applications):
    - adhesion_strength_optimization: Optimization of adhesion strength
      • optimization_target: maximize_adhesion_force
      • measurement_units: force_per_area
      • application_domains: fastening, biomedical, manufacturing
    
    RESEARCH CONCEPTS (Principles and Methods):
    - mechanical_interaction_principle: Mechanical interaction principle
      • principle_type: physical_law
      • domain: mechanics
      • mathematical_description: force_relationships
    """
    
    extractor = EnhancedPatternExtractor()
    patterns = extractor.extract_all_patterns(sample_soc_knowledge)
    
    print("🔍 Enhanced Pattern Extraction Test Results:")
    print("=" * 50)
    
    for pattern_type, pattern_list in patterns.items():
        print(f"\n{pattern_type.upper()} PATTERNS ({len(pattern_list)}):")
        for i, pattern in enumerate(pattern_list, 1):
            print(f"  {i}. {pattern.name} (confidence: {pattern.confidence:.2f})")
            
            if hasattr(pattern, 'structural_components') and pattern.structural_components:
                print(f"     Components: {pattern.structural_components[:3]}")
            elif hasattr(pattern, 'input_conditions') and pattern.input_conditions:
                print(f"     Inputs: {pattern.input_conditions[:3]}")
            elif hasattr(pattern, 'cause') and hasattr(pattern, 'effect'):
                print(f"     Causality: {pattern.cause} → {pattern.effect}")
    
    total_patterns = sum(len(p) for p in patterns.values())
    print(f"\n📊 Total patterns extracted: {total_patterns}")