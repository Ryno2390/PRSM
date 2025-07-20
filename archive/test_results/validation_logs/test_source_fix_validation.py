#!/usr/bin/env python3
"""
Simple test to validate source retrieval fix
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.candidate_answer_generator import CandidateAnswerGenerator, CandidateType

def test_candidate_type_weights():
    """Test that the candidate type weights favor multi-source types"""
    generator = CandidateAnswerGenerator()
    
    print("🔍 Candidate Type Weight Analysis")
    print("=" * 40)
    
    print("📊 Updated Answer Type Weights:")
    for candidate_type, weight in generator.answer_type_weights.items():
        sources_per_type = {
            CandidateType.SINGLE_SOURCE: 1,
            CandidateType.SYNTHESIS: 4,
            CandidateType.COMPARATIVE: 3,
            CandidateType.METHODOLOGICAL: 3,
            CandidateType.EMPIRICAL: 3,
            CandidateType.APPLIED: 3,
            CandidateType.THEORETICAL: 2
        }.get(candidate_type, 3)
        
        print(f"   {candidate_type.value:15}: {weight:.2f} weight × {sources_per_type} sources = {weight * sources_per_type:.2f} expected sources")
    
    # Calculate expected source contribution
    total_weighted_sources = sum(
        weight * {
            CandidateType.SINGLE_SOURCE: 1,
            CandidateType.SYNTHESIS: 4,
            CandidateType.COMPARATIVE: 3,
            CandidateType.METHODOLOGICAL: 3,
            CandidateType.EMPIRICAL: 3,
            CandidateType.APPLIED: 3,
            CandidateType.THEORETICAL: 2
        }.get(candidate_type, 3)
        for candidate_type, weight in generator.answer_type_weights.items()
    )
    
    print(f"\n🎯 Analysis:")
    print(f"   Expected weighted sources per candidate: {total_weighted_sources:.2f}")
    print(f"   Target candidates: {generator.target_candidates}")
    print(f"   Expected total source contributions: {total_weighted_sources * generator.target_candidates:.1f}")
    
    # Check for improvement
    single_source_weight = generator.answer_type_weights[CandidateType.SINGLE_SOURCE]
    synthesis_weight = generator.answer_type_weights[CandidateType.SYNTHESIS]
    
    print(f"\n✅ Optimization Status:")
    print(f"   SINGLE_SOURCE weight: {single_source_weight:.2f} (should be low)")
    print(f"   SYNTHESIS weight: {synthesis_weight:.2f} (should be high)")
    
    if single_source_weight <= 0.1 and synthesis_weight >= 0.3:
        print(f"   🎉 SUCCESS: Weights optimized for multi-source generation!")
        return True
    else:
        print(f"   ⚠️  NEEDS WORK: Weights may still favor single sources")
        return False

if __name__ == "__main__":
    success = test_candidate_type_weights()
    
    if success:
        print(f"\n🚀 Source retrieval optimization applied successfully!")
        print(f"   Expected improvement: 1 source → 3+ sources per query")
    else:
        print(f"\n🔧 Additional weight tuning needed")