#!/usr/bin/env python3
"""
Domain Knowledge Integration
Integrates enhanced domain knowledge generator into the existing pipeline

This module:
1. Replaces mock SOCs with real enhanced domain knowledge
2. Provides compatibility layer for existing pattern extraction
3. Enhances the overall breakthrough discovery quality
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import json

# Import enhanced domain knowledge generator
from enhanced_domain_knowledge_generator import (
    EnhancedDomainKnowledgeGenerator, 
    replace_mock_socs_with_enhanced_knowledge
)

class DomainKnowledgeIntegration:
    """Integrates enhanced domain knowledge into existing pipeline"""
    
    def __init__(self):
        self.generator = EnhancedDomainKnowledgeGenerator()
        
    def enhance_pipeline_soc_extraction(self, paper_content: str, paper_id: str) -> List[Dict]:
        """
        Replace mock SOC extraction with enhanced domain knowledge
        
        This function can be used to replace the mock SOC generation in:
        - pipeline_tester.py (lines 125-140)
        - Any other components using hardcoded SOCs
        """
        
        print(f"🔄 Replacing mock SOCs with enhanced domain knowledge for {paper_id}")
        
        # Generate enhanced SOCs
        enhanced_socs = replace_mock_socs_with_enhanced_knowledge(paper_content, paper_id)
        
        # Convert to legacy format for compatibility with existing pipeline
        legacy_socs = self._convert_to_legacy_format(enhanced_socs)
        
        return legacy_socs
    
    def _convert_to_legacy_format(self, enhanced_socs: List[Dict]) -> List[Dict]:
        """Convert enhanced SOCs to legacy format for pipeline compatibility"""
        
        legacy_socs = []
        
        for soc in enhanced_socs:
            # Extract key information for legacy format
            subjects = []
            objects = []
            concepts = []
            
            # Extract subjects from material composition and mechanisms
            if 'material_composition' in soc:
                subjects.extend([name for name in soc.get('material_composition', []) if name])
            
            # Extract subjects from qualitative properties
            subjects.extend(soc.get('qualitative_properties', {}).keys())
            
            # Extract mechanisms as concepts
            concepts.extend(soc.get('mechanisms', []))
            
            # Extract domain as concept
            if soc.get('domain'):
                concepts.append(soc['domain'])
            
            # Add type as concept
            if soc.get('type'):
                concepts.append(soc['type'])
            
            # Default values if empty
            if not subjects:
                subjects = ['material', 'structure', 'system']
            if not objects:
                objects = ['property', 'performance', 'behavior']
            if not concepts:
                concepts = ['mechanism', 'interaction', 'function']
            
            # Create legacy SOC format
            legacy_soc = {
                'subjects': subjects[:5],  # Limit to 5 items
                'objects': objects[:5],
                'concepts': concepts[:5],
                'confidence': soc.get('confidence', 0.7),
                'source_paper': soc.get('source_paper', ''),
                # Preserve quantitative properties and causal relationships for quality metrics
                'quantitative_properties': soc.get('quantitative_properties', []),
                'causal_relationships': soc.get('causal_relationships', []),
                'domain': soc.get('domain', 'unknown'),
                # Add enhanced data as metadata
                'enhanced_data': {
                    'quantitative_properties': len(soc.get('quantitative_properties', [])),
                    'causal_relationships': len(soc.get('causal_relationships', [])),
                    'domain_category': soc.get('domain', 'unknown'),
                    'soc_type': soc.get('type', 'unknown')
                }
            }
            
            legacy_socs.append(legacy_soc)
        
        # Ensure we have at least one SOC for compatibility
        if not legacy_socs:
            legacy_socs = [{
                'subjects': ['material', 'surface', 'structure'],
                'objects': ['property', 'performance', 'behavior'],
                'concepts': ['mechanism', 'interaction', 'function'],
                'confidence': 0.5,
                'source_paper': '',
                'quantitative_properties': [],
                'causal_relationships': [],
                'domain': 'unknown',
                'enhanced_data': {
                    'quantitative_properties': 0,
                    'causal_relationships': 0,
                    'domain_category': 'unknown',
                    'soc_type': 'fallback'
                }
            }]
        
        return legacy_socs
    
    def get_quality_metrics(self, enhanced_socs: List[Dict]) -> Dict:
        """Calculate quality metrics for enhanced SOCs"""
        
        if not enhanced_socs:
            return {
                'total_socs': 0,
                'avg_confidence': 0.0,
                'quantitative_properties_count': 0,
                'causal_relationships_count': 0,
                'domain_diversity': 0,
                'quality_score': 0.0
            }
        
        total_confidence = sum(soc.get('confidence', 0) for soc in enhanced_socs)
        avg_confidence = total_confidence / len(enhanced_socs)
        
        quantitative_count = sum(len(soc.get('quantitative_properties', [])) for soc in enhanced_socs)
        causal_count = sum(len(soc.get('causal_relationships', [])) for soc in enhanced_socs)
        
        # Calculate domain diversity
        domains = set(soc.get('domain', 'unknown') for soc in enhanced_socs)
        domain_diversity = len(domains)
        
        # Calculate overall quality score
        quality_score = (
            avg_confidence * 0.3 +
            min(quantitative_count / len(enhanced_socs), 2.0) * 0.25 +  # Normalize to max 2 per SOC
            min(causal_count / len(enhanced_socs), 1.0) * 0.25 +        # Normalize to max 1 per SOC  
            min(domain_diversity / 5.0, 1.0) * 0.2                     # Normalize to max 5 domains
        )
        
        return {
            'total_socs': len(enhanced_socs),
            'avg_confidence': avg_confidence,
            'quantitative_properties_count': quantitative_count,
            'causal_relationships_count': causal_count,
            'domain_diversity': domain_diversity,
            'quality_score': quality_score
        }

def create_enhanced_pipeline_tester_patch():
    """
    Create a patch for pipeline_tester.py to use enhanced domain knowledge
    """
    
    patch_code = '''
# Enhanced SOC extraction using domain knowledge generator
# Replace the mock SOC generation around lines 125-140 in pipeline_tester.py

from domain_knowledge_integration import DomainKnowledgeIntegration

class EnhancedPipelineTester(PipelineTester):
    """Enhanced pipeline tester with real domain knowledge extraction"""
    
    def __init__(self):
        super().__init__()
        self.domain_integration = DomainKnowledgeIntegration()
    
    def test_soc_extraction(self, paper: Dict) -> Tuple[bool, List[Dict], float]:
        """Enhanced SOC extraction with real domain knowledge"""
        
        try:
            # Get paper content
            content = paper.get('content', paper.get('abstract', ''))
            paper_id = paper.get('arxiv_id', 'unknown')
            
            # Use enhanced domain knowledge extraction
            enhanced_socs = self.domain_integration.enhance_pipeline_soc_extraction(content, paper_id)
            
            # Calculate quality metrics
            quality_metrics = self.domain_integration.get_quality_metrics(enhanced_socs)
            
            print(f"   📊 Enhanced SOCs: {quality_metrics['total_socs']}")
            print(f"   📈 Quality Score: {quality_metrics['quality_score']:.3f}")
            print(f"   🔢 Quantitative Properties: {quality_metrics['quantitative_properties_count']}")
            print(f"   🔗 Causal Relationships: {quality_metrics['causal_relationships_count']}")
            
            processing_time = 0.1  # Placeholder
            
            return True, enhanced_socs, processing_time
            
        except Exception as e:
            print(f"   ❌ Enhanced SOC extraction failed: {e}")
            # Fallback to original mock SOCs
            return super().test_soc_extraction(paper)
'''
    
    return patch_code

def demonstrate_integration():
    """Demonstrate the enhanced domain knowledge integration"""
    
    print("🔧 DOMAIN KNOWLEDGE INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Sample paper content
    sample_paper = {
        'arxiv_id': 'demo_integration',
        'title': 'Gecko-Inspired Adhesion for Advanced Fastening',
        'content': '''
        We developed a bioinspired adhesive system using polydimethylsiloxane (PDMS) with hierarchical 
        micro and nano structures. The adhesive demonstrates reversible adhesion with a normal pull-off 
        force of 45 N/cm². The mechanism relies on van der Waals forces between micro-fibrils and the 
        target surface. Surface roughness measurements showed Ra = 0.3 μm, which is optimal for 
        maximizing contact area. When normal force is applied, the micro-structures deform to increase 
        real contact area, resulting in enhanced adhesion strength. The adhesion can be easily released 
        by applying shear force, demonstrating the reversible nature of the gecko-inspired mechanism.
        Temperature testing showed stable performance from -20°C to 80°C.
        '''
    }
    
    # Initialize integration
    integration = DomainKnowledgeIntegration()
    
    # Test enhanced SOC extraction
    print(f"\n🧠 Testing enhanced SOC extraction...")
    enhanced_socs = integration.enhance_pipeline_soc_extraction(
        sample_paper['content'], 
        sample_paper['arxiv_id']
    )
    
    # Calculate quality metrics
    quality_metrics = integration.get_quality_metrics(enhanced_socs)
    
    # Display results
    print(f"\n📊 ENHANCED SOC EXTRACTION RESULTS:")
    print(f"   Total SOCs: {quality_metrics['total_socs']}")
    print(f"   Average Confidence: {quality_metrics['avg_confidence']:.3f}")
    print(f"   Quality Score: {quality_metrics['quality_score']:.3f}")
    print(f"   Quantitative Properties: {quality_metrics['quantitative_properties_count']}")
    print(f"   Causal Relationships: {quality_metrics['causal_relationships_count']}")
    print(f"   Domain Diversity: {quality_metrics['domain_diversity']}")
    
    print(f"\n🔍 SAMPLE ENHANCED SOC:")
    if enhanced_socs:
        soc = enhanced_socs[0]
        print(f"   Subjects: {soc['subjects'][:3]}")
        print(f"   Objects: {soc['objects'][:3]}")
        print(f"   Concepts: {soc['concepts'][:3]}")
        print(f"   Confidence: {soc['confidence']:.3f}")
        
        enhanced_data = soc.get('enhanced_data', {})
        print(f"   Enhanced Features:")
        print(f"      Domain: {enhanced_data.get('domain_category', 'unknown')}")
        print(f"      SOC Type: {enhanced_data.get('soc_type', 'unknown')}")
        print(f"      Quantitative Data: {enhanced_data.get('quantitative_properties', 0)} properties")
        print(f"      Causal Links: {enhanced_data.get('causal_relationships', 0)} relationships")
    
    print(f"\n✅ Integration demonstration complete!")
    
    return enhanced_socs, quality_metrics

if __name__ == "__main__":
    demonstrate_integration()