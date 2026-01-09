#!/usr/bin/env python3
"""
Test Chunk Classification System
================================

Tests the semantic chunk classification system with sample research paper content.
Validates chunk type identification, confidence scoring, and metadata extraction.
"""

import asyncio
import time
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from engines.chunk_classification_system import (
    ChunkType, SemanticChunk, ChunkTypeClassifier, 
    classify_paper_chunks, get_chunk_type_descriptions
)

# Sample research paper content for testing
SAMPLE_PAPER_CONTENT = """
Abstract

This paper presents a novel approach to context rot mitigation in large language models through dynamic attention reallocation. We introduce the Adaptive Context Preservation (ACP) framework, which maintains context integrity during extended conversations. Our experimental evaluation demonstrates a 34% improvement in context retention over existing methods, with statistical significance (p < 0.01). The approach shows particular promise for deployment in production environments where context drift has been a persistent challenge.

1. Introduction

The problem of context rot in artificial intelligence systems has become increasingly critical as models are deployed in dynamic, real-world environments. Context rot refers to the degradation of performance when the operational context differs significantly from the training context. This phenomenon manifests through several mechanisms: distribution shift between training and deployment data, temporal drift in data patterns over time, domain adaptation challenges, and catastrophic forgetting during model updates.

Existing approaches to context preservation suffer from computational overhead and limited adaptability. However, there is a need for more efficient and robust solutions that can maintain performance across diverse operational contexts.

2. Methodology

We develop the Adaptive Context Preservation (ACP) framework using three core components. Our approach employs dynamic attention mechanisms that selectively focus on relevant context elements. The method involves hierarchical context encoding, where information is organized across multiple temporal scales.

Step 1: Context Analysis - We analyze incoming information for relevance and importance.
Step 2: Attention Reallocation - The system dynamically adjusts attention weights based on context analysis.
Step 3: Memory Consolidation - Important context elements are preserved in long-term memory structures.

3. Experimental Setup

We evaluate our approach on three benchmark datasets: ConText-1000, DynamicQA, and RealWorld-Chat. The experimental setup includes baseline comparisons with five state-of-the-art context preservation methods. Our evaluation metrics include context retention accuracy, response coherence, and computational efficiency.

For each dataset, we conduct 10-fold cross-validation with stratified sampling. The control group uses standard attention mechanisms, while our experimental condition implements the ACP framework.

4. Results

Table 1 shows the performance comparison across all datasets. Our ACP framework achieves 87.3% context retention accuracy compared to 65.2% for the best baseline method. This represents a statistically significant improvement (p < 0.001) with a 95% confidence interval.

The results indicate that our method outperforms existing approaches across all evaluation metrics. We observe particularly strong performance in long-context scenarios, where traditional methods show rapid degradation. Runtime analysis shows our approach adds only 12ms overhead per query, making it practical for production deployment.

5. Discussion

We find that the dynamic attention mechanism is the key contributor to improved performance. The results show that hierarchical encoding enables better long-term context preservation. Our findings suggest that adaptive approaches significantly outperform static context management strategies.

This indicates that context rot can be effectively mitigated through intelligent attention reallocation rather than simply increasing model capacity.

6. Limitations

Our approach has several limitations that should be considered. The method cannot handle extremely long contexts beyond 10,000 tokens due to memory constraints. Additionally, the framework requires fine-tuning for domain-specific applications, which may limit its generalizability. Performance degradation is observed in highly specialized domains with limited training data.

7. Future Work

Future research could explore integration with retrieval-augmented generation systems. We recommend investigating the scalability of our approach to even larger context windows. Potential improvements include adaptive learning rates and multi-modal context integration.

8. Mathematical Formulation

Let C(t) be the context state at time t, and A(t) be the attention weights. The adaptive context preservation mechanism is defined as:

C(t+1) = αC(t) + (1-α)f(A(t), I(t))

where α is the retention factor, f is the attention function, and I(t) is the input at time t. The optimization objective minimizes context drift while maximizing information retention:

minimize: ||C(t+1) - C_target||²
subject to: Σᵢ A_i(t) = 1, A_i(t) ≥ 0

9. Algorithm Description

Algorithm 1: Adaptive Context Preservation

Input: Context sequence C = [c₁, c₂, ..., cₙ]
Output: Preserved context C'

Begin:
1. Initialize attention weights A ← uniform(n)
2. For each time step t:
   a. Compute relevance scores R(t) ← relevance(C, cₜ)
   b. Update attention weights A(t) ← softmax(R(t))
   c. Apply context preservation C'(t) ← A(t) ⊙ C(t)
3. Return C'
End

Time complexity: O(n²), Space complexity: O(n)

10. Conclusion

In conclusion, our Adaptive Context Preservation framework provides a significant advancement in mitigating context rot in large language models. The experimental results demonstrate both statistical significance and practical applicability. This work has important implications for the deployment of AI systems in production environments, particularly where context continuity is critical for user experience and system reliability.
"""

async def test_chunk_classification():
    """Test the chunk classification system with sample content"""
    print("🧪 Testing Chunk Classification System")
    print("=" * 60)
    
    # Test paper metadata
    paper_metadata = {
        'paper_id': 'test_001',
        'title': 'Adaptive Context Preservation in Large Language Models',
        'authors': ['Test Author'],
        'year': 2024
    }
    
    print(f"Paper: {paper_metadata['title']}")
    print(f"Content length: {len(SAMPLE_PAPER_CONTENT):,} characters")
    print()
    
    # Run classification
    start_time = time.time()
    
    result = await classify_paper_chunks(
        paper_content=SAMPLE_PAPER_CONTENT,
        paper_metadata=paper_metadata,
        chunk_size=200,  # Smaller chunks for testing
        overlap=25
    )
    
    classification_time = time.time() - start_time
    
    # Display results
    print("📊 CLASSIFICATION RESULTS:")
    print(f"   Total chunks created: {result.total_chunks_created}")
    print(f"   Valid chunks: {result.valid_chunks}")
    print(f"   Classification time: {result.classification_time:.3f}s")
    print(f"   Overall confidence: {result.classification_confidence:.2f}")
    print()
    
    # Analyze chunk types
    chunk_type_counts = {}
    for chunk in result.chunks:
        chunk_type = chunk.chunk_type
        if chunk_type not in chunk_type_counts:
            chunk_type_counts[chunk_type] = 0
        chunk_type_counts[chunk_type] += 1
    
    print("📋 CHUNK TYPE DISTRIBUTION:")
    type_descriptions = get_chunk_type_descriptions()
    for chunk_type, count in sorted(chunk_type_counts.items(), key=lambda x: x[1], reverse=True):
        description = type_descriptions.get(chunk_type, "Unknown")
        print(f"   {chunk_type.value:20} | {count:2} chunks | {description}")
    print()
    
    # Show sample chunks
    print("📄 SAMPLE CLASSIFIED CHUNKS:")
    print("-" * 60)
    
    sample_chunks = result.chunks[:5]  # Show first 5 chunks
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n{i}. CHUNK TYPE: {chunk.chunk_type.value.upper()}")
        print(f"   Section: {chunk.section_name}")
        print(f"   Confidence: {chunk.confidence_score:.2f} | Evidence: {chunk.evidence_strength:.2f} | Novelty: {chunk.novelty_score:.2f}")
        print(f"   Word count: {chunk.word_count} | Concepts: {len(chunk.key_concepts)}")
        
        if chunk.key_concepts:
            print(f"   Key concepts: {', '.join(chunk.key_concepts[:5])}")
        
        if chunk.quantitative_data:
            print(f"   Quantitative data: {len(chunk.quantitative_data)} items")
            for data in chunk.quantitative_data[:2]:
                print(f"     - {data.get('type', 'unknown')}: {data.get('context', 'N/A')}")
        
        # Show content preview
        content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(f"   Content: {content_preview}")
        print()
    
    # Test specific chunk types
    print("🎯 SPECIFIC CHUNK TYPE ANALYSIS:")
    print("-" * 60)
    
    target_types = [ChunkType.ABSTRACT, ChunkType.METHODOLOGY, ChunkType.RESULTS_QUANTITATIVE, ChunkType.MATHEMATICAL_FORMULATION]
    
    for target_type in target_types:
        matching_chunks = [c for c in result.chunks if c.chunk_type == target_type]
        if matching_chunks:
            best_chunk = max(matching_chunks, key=lambda x: x.confidence_score)
            print(f"\nBEST {target_type.value.upper()} CHUNK:")
            print(f"   Confidence: {best_chunk.confidence_score:.2f}")
            print(f"   Evidence strength: {best_chunk.evidence_strength:.2f}")
            print(f"   Quantitative items: {len(best_chunk.quantitative_data)}")
            print(f"   Content preview: {best_chunk.content[:150]}...")
        else:
            print(f"\n❌ No {target_type.value} chunks found")
    
    # Quality assessment
    print(f"\n📈 QUALITY ASSESSMENT:")
    avg_confidence = sum(c.confidence_score for c in result.chunks) / len(result.chunks)
    avg_evidence = sum(c.evidence_strength for c in result.chunks) / len(result.chunks)
    avg_novelty = sum(c.novelty_score for c in result.chunks) / len(result.chunks)
    
    chunks_with_quantitative = sum(1 for c in result.chunks if c.quantitative_data)
    chunks_with_concepts = sum(1 for c in result.chunks if c.key_concepts)
    
    print(f"   Average confidence: {avg_confidence:.2f}")
    print(f"   Average evidence strength: {avg_evidence:.2f}")
    print(f"   Average novelty score: {avg_novelty:.2f}")
    print(f"   Chunks with quantitative data: {chunks_with_quantitative}/{len(result.chunks)} ({chunks_with_quantitative/len(result.chunks):.1%})")
    print(f"   Chunks with key concepts: {chunks_with_concepts}/{len(result.chunks)} ({chunks_with_concepts/len(result.chunks):.1%})")
    
    # Validation checks
    print(f"\n✅ VALIDATION CHECKS:")
    validations = [
        ("All chunks are valid", all(c.is_valid() for c in result.chunks)),
        ("Confidence scores in range", all(0 <= c.confidence_score <= 1 for c in result.chunks)),
        ("Evidence scores in range", all(0 <= c.evidence_strength <= 1 for c in result.chunks)),
        ("Found abstract chunks", any(c.chunk_type == ChunkType.ABSTRACT for c in result.chunks)),
        ("Found methodology chunks", any(c.chunk_type == ChunkType.METHODOLOGY for c in result.chunks)),
        ("Found results chunks", any(c.chunk_type == ChunkType.RESULTS_QUANTITATIVE for c in result.chunks)),
        ("Extracted quantitative data", sum(len(c.quantitative_data) for c in result.chunks) > 0),
        ("Extracted key concepts", sum(len(c.key_concepts) for c in result.chunks) > 0),
        ("Classification completed quickly", classification_time < 10.0)
    ]
    
    passed_checks = 0
    for check_name, passed in validations:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if passed:
            passed_checks += 1
    
    success_rate = passed_checks / len(validations)
    print(f"\n🎯 VALIDATION SUMMARY: {passed_checks}/{len(validations)} checks passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("\n🎉 CHUNK CLASSIFICATION SYSTEM: SUCCESS!")
        print("The system successfully identifies semantic chunk types with high accuracy.")
        print("Ready for integration with meta-paper generation pipeline.")
    else:
        print(f"\n⚠️  CHUNK CLASSIFICATION SYSTEM: PARTIAL SUCCESS")
        print(f"Success rate: {success_rate:.1%} (need ≥80%)")
        print("Some components may need refinement.")
    
    return success_rate >= 0.8

async def main():
    """Run the chunk classification test"""
    print("🚀 Starting Chunk Classification System Test...")
    print()
    
    success = await test_chunk_classification()
    
    if success:
        print("\n🎊 Chunk classification system test completed successfully!")
        print("System is ready for meta-paper generation implementation.")
    else:
        print("\n⚠️  Chunk classification system test had issues.")
        print("Review the output above for specific problems.")
    
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)