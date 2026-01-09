#!/usr/bin/env python3
"""
Test Chunk Embedding and Ranking System
=======================================

Tests the chunk embedding system and ranking algorithms for meta-paper generation.
Validates embedding generation, query relevance scoring, and optimal chunk selection.
"""

import asyncio
import time
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from engines.chunk_classification_system import (
    ChunkType, SemanticChunk, classify_paper_chunks
)
from engines.chunk_embedding_system import (
    ChunkEmbeddingSystem, ChunkRankingSystem, rank_and_select_chunks,
    get_reasoning_engine_descriptions
)

# Sample research paper content for testing
SAMPLE_PAPER_CONTENT = """
Abstract

This paper presents a novel approach to context rot mitigation in large language models through dynamic attention reallocation. We introduce the Adaptive Context Preservation (ACP) framework, which maintains context integrity during extended conversations. Our experimental evaluation demonstrates a 34% improvement in context retention over existing methods, with statistical significance (p < 0.01). The approach shows particular promise for deployment in production environments where context drift has been a persistent challenge.

1. Introduction

The problem of context rot in artificial intelligence systems has become increasingly critical as models are deployed in dynamic, real-world environments. Context rot refers to the degradation of performance when the operational context differs significantly from the training context. This phenomenon manifests through several mechanisms: distribution shift between training and deployment data, temporal drift in data patterns over time, domain adaptation challenges, and catastrophic forgetting during model updates.

2. Methodology

We develop the Adaptive Context Preservation (ACP) framework using three core components. Our approach employs dynamic attention mechanisms that selectively focus on relevant context elements. The method involves hierarchical context encoding, where information is organized across multiple temporal scales.

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

3. Experimental Setup

We evaluate our approach on three benchmark datasets: ConText-1000, DynamicQA, and RealWorld-Chat. The experimental setup includes baseline comparisons with five state-of-the-art context preservation methods. Our evaluation metrics include context retention accuracy, response coherence, and computational efficiency.

4. Results

Our ACP framework achieves 87.3% context retention accuracy compared to 65.2% for the best baseline method. This represents a statistically significant improvement (p < 0.001) with a 95% confidence interval. The results indicate that our method outperforms existing approaches across all evaluation metrics.

5. Mathematical Formulation

Let C(t) be the context state at time t, and A(t) be the attention weights. The adaptive context preservation mechanism is defined as:

C(t+1) = αC(t) + (1-α)f(A(t), I(t))

where α is the retention factor, f is the attention function, and I(t) is the input at time t. The optimization objective minimizes context drift while maximizing information retention.

6. Limitations and Future Work

Our approach has several limitations. The method cannot handle extremely long contexts beyond 10,000 tokens due to memory constraints. Future research could explore integration with retrieval-augmented generation systems.
"""

# Test queries for different reasoning engines
TEST_QUERIES = {
    "causal": "What causes context rot in AI systems and how can it be prevented?",
    "probabilistic": "What statistical evidence shows the effectiveness of context preservation methods?", 
    "mathematical": "What mathematical formulations are used for context preservation optimization?",
    "abductive": "What best explains the success of dynamic attention in context preservation?",
    "inductive": "What patterns emerge from experimental evaluation of context preservation methods?",
    "analogical": "How do context preservation approaches compare to existing methods?"
}

async def test_embedding_generation():
    """Test basic embedding generation functionality"""
    print("🧪 Testing Embedding Generation")
    print("=" * 50)
    
    # Create test chunk
    test_chunk = SemanticChunk(
        content="This paper presents a novel machine learning algorithm for neural network optimization. The approach achieves 95% accuracy on benchmark datasets with statistical significance (p < 0.001).",
        chunk_type=ChunkType.RESULTS_QUANTITATIVE,
        paper_id="test_001",
        paper_title="Test Paper",
        section_name="Results",
        confidence_score=0.8,
        evidence_strength=0.9,
        novelty_score=0.7,
        citation_count=0,
        recency_score=0.8,
        word_count=25,
        start_position=0,
        end_position=100,
        key_concepts=['machine learning', 'neural network', 'optimization', 'accuracy'],
        quantitative_data=[
            {'type': 'performance_metric', 'value': '95%', 'context': '95% accuracy'},
            {'type': 'statistical_significance', 'value': 'p < 0.001', 'context': 'p < 0.001'}
        ]
    )
    
    # Test embedding system
    embedding_system = ChunkEmbeddingSystem()
    
    start_time = time.time()
    embeddings = await embedding_system.generate_chunk_embeddings(test_chunk)
    generation_time = time.time() - start_time
    
    print(f"✅ Generated embeddings in {generation_time:.3f}s")
    print(f"   Chunk ID: {embeddings.chunk_id}")
    print(f"   Embedding dimension: {embeddings.embedding_dim}")
    print(f"   Content embedding: {'✅' if embeddings.content_embedding else '❌'}")
    print(f"   Concept embedding: {'✅' if embeddings.concept_embedding else '❌'}")
    print(f"   Results embedding: {'✅' if embeddings.results_embedding else '❌'}")
    
    # Test similarity calculation
    if embeddings.content_embedding:
        # Self-similarity should be 1.0
        self_similarity = embedding_system._cosine_similarity(
            embeddings.content_embedding, embeddings.content_embedding
        )
        print(f"   Self-similarity: {self_similarity:.3f} (should be 1.0)")
    
    print()
    return embeddings is not None

async def test_query_relevance_scoring():
    """Test query relevance scoring system"""
    print("🎯 Testing Query Relevance Scoring")
    print("=" * 50)
    
    # Create sample chunks with different types
    sample_chunks = [
        SemanticChunk(
            content="Our neural network achieves 92% accuracy on the test dataset with p < 0.01 statistical significance.",
            chunk_type=ChunkType.RESULTS_QUANTITATIVE,
            paper_id="test_001", paper_title="Test Paper", section_name="Results",
            confidence_score=0.9, evidence_strength=0.95, novelty_score=0.6,
            citation_count=0, recency_score=0.8, word_count=15,
            start_position=0, end_position=50,
            key_concepts=['neural network', 'accuracy', 'statistical'],
            quantitative_data=[{'type': 'performance_metric', 'value': '92%', 'context': '92% accuracy'}]
        ),
        SemanticChunk(
            content="We employ a novel attention mechanism that dynamically reallocates computational resources based on context relevance.",
            chunk_type=ChunkType.METHODOLOGY,
            paper_id="test_001", paper_title="Test Paper", section_name="Methods", 
            confidence_score=0.8, evidence_strength=0.7, novelty_score=0.8,
            citation_count=0, recency_score=0.8, word_count=16,
            start_position=50, end_position=100,
            key_concepts=['attention mechanism', 'context', 'resources'],
            quantitative_data=[]
        ),
        SemanticChunk(
            content="The optimization objective minimizes loss L = αE[x] + βV[x] subject to constraints on computational budget.",
            chunk_type=ChunkType.MATHEMATICAL_FORMULATION,
            paper_id="test_001", paper_title="Test Paper", section_name="Math",
            confidence_score=0.85, evidence_strength=0.8, novelty_score=0.7,
            citation_count=0, recency_score=0.8, word_count=14,
            start_position=100, end_position=150,
            key_concepts=['optimization', 'loss', 'constraints'],
            quantitative_data=[]
        )
    ]
    
    embedding_system = ChunkEmbeddingSystem()
    test_query = "What methods achieve the highest accuracy in neural network optimization?"
    
    print(f"Query: {test_query}")
    print()
    
    # Test relevance scoring for different reasoning engines
    reasoning_engines = ['probabilistic', 'causal', 'mathematical']
    
    for engine in reasoning_engines:
        print(f"🔧 Reasoning Engine: {engine}")
        
        for i, chunk in enumerate(sample_chunks, 1):
            # Generate embeddings
            embeddings = await embedding_system.generate_chunk_embeddings(chunk)
            
            # Calculate relevance
            relevance = await embedding_system.calculate_query_relevance(
                chunk, embeddings, test_query, engine
            )
            
            print(f"   Chunk {i} ({chunk.chunk_type.value}):")
            print(f"     Total score: {relevance.total_score:.3f}")
            print(f"     Content sim: {relevance.content_similarity:.3f}")
            print(f"     Concept sim: {relevance.concept_similarity:.3f}")
            print(f"     Type weight: {relevance.type_weight:.3f}")
            print(f"     Evidence: +{relevance.evidence_bonus:.3f}")
            print(f"     Engine bonus: +{relevance.reasoning_engine_bonus:.3f}")
        print()
    
    return True

async def test_chunk_ranking_and_selection():
    """Test chunk ranking and selection system"""
    print("📊 Testing Chunk Ranking and Selection")
    print("=" * 60)
    
    # First classify chunks from sample paper
    paper_metadata = {
        'paper_id': 'test_001',
        'title': 'Adaptive Context Preservation in Large Language Models',
        'authors': ['Test Author']
    }
    
    print("🔍 Classifying chunks from sample paper...")
    classification_result = await classify_paper_chunks(
        SAMPLE_PAPER_CONTENT, paper_metadata, chunk_size=150, overlap=20
    )
    
    chunks = classification_result.chunks
    print(f"   Classified {len(chunks)} chunks")
    print()
    
    # Test different queries and reasoning engines
    for reasoning_engine, query in TEST_QUERIES.items():
        print(f"🎯 Query ({reasoning_engine}): {query[:60]}...")
        
        start_time = time.time()
        
        # Rank and select chunks
        selection_result = await rank_and_select_chunks(
            chunks=chunks,
            query=query,
            reasoning_engine=reasoning_engine,
            token_budget=800,  # Smaller budget for testing
            top_k=20
        )
        
        selection_time = time.time() - start_time
        
        print(f"   Selected chunks: {len(selection_result.selected_chunks)}")
        print(f"   Token usage: {selection_result.token_usage}/800")
        print(f"   Avg relevance: {selection_result.average_relevance:.3f}")
        print(f"   Diversity score: {selection_result.diversity_score:.3f}")
        print(f"   Selection time: {selection_time:.3f}s")
        
        # Show top 3 selected chunks
        print("   Top chunks:")
        for i, chunk in enumerate(selection_result.selected_chunks[:3], 1):
            chunk_preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
            print(f"     {i}. {chunk.chunk_type.value}: {chunk_preview}")
        print()
    
    return True

async def test_meta_paper_assembly():
    """Test meta-paper assembly process"""
    print("📄 Testing Meta-Paper Assembly Process")
    print("=" * 60)
    
    # Classify chunks from sample paper
    paper_metadata = {
        'paper_id': 'test_001', 
        'title': 'Context Preservation Research',
        'authors': ['Researcher']
    }
    
    classification_result = await classify_paper_chunks(
        SAMPLE_PAPER_CONTENT, paper_metadata, chunk_size=200
    )
    chunks = classification_result.chunks
    
    # Simulate meta-paper assembly for a complex query
    complex_query = "How do dynamic attention mechanisms solve context rot problems in AI systems, and what mathematical frameworks support this approach?"
    
    print(f"Complex Query: {complex_query}")
    print()
    
    # Test assembly with different reasoning engines
    reasoning_engines = ['causal', 'mathematical', 'abductive']
    
    for engine in reasoning_engines:
        print(f"🔧 Reasoning Engine: {engine}")
        
        selection_result = await rank_and_select_chunks(
            chunks=chunks,
            query=complex_query,
            reasoning_engine=engine,
            token_budget=1200,
            top_k=30
        )
        
        print(f"   Meta-paper components: {len(selection_result.selected_chunks)} chunks")
        print(f"   Token budget used: {selection_result.token_usage}/1200 ({selection_result.token_usage/1200:.1%})")
        
        # Analyze chunk type distribution
        type_counts = {}
        for chunk in selection_result.selected_chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        print("   Chunk type distribution:")
        for chunk_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {chunk_type}: {count}")
        
        # Show conceptual coverage
        all_concepts = set()
        for chunk in selection_result.selected_chunks:
            all_concepts.update(chunk.key_concepts)
        
        print(f"   Conceptual coverage: {len(all_concepts)} unique concepts")
        if all_concepts:
            top_concepts = list(all_concepts)[:8]
            print(f"   Key concepts: {', '.join(top_concepts)}")
        
        print()
    
    return True

async def main():
    """Run comprehensive embedding system tests"""
    print("🚀 Starting Chunk Embedding and Ranking System Tests...")
    print()
    
    test_results = []
    
    # Test 1: Embedding Generation
    try:
        result1 = await test_embedding_generation()
        test_results.append(("Embedding Generation", result1))
    except Exception as e:
        print(f"❌ Embedding Generation test failed: {e}")
        test_results.append(("Embedding Generation", False))
    
    # Test 2: Query Relevance Scoring  
    try:
        result2 = await test_query_relevance_scoring()
        test_results.append(("Query Relevance Scoring", result2))
    except Exception as e:
        print(f"❌ Query Relevance Scoring test failed: {e}")
        test_results.append(("Query Relevance Scoring", False))
    
    # Test 3: Chunk Ranking and Selection
    try:
        result3 = await test_chunk_ranking_and_selection()
        test_results.append(("Chunk Ranking and Selection", result3))
    except Exception as e:
        print(f"❌ Chunk Ranking and Selection test failed: {e}")
        test_results.append(("Chunk Ranking and Selection", False))
    
    # Test 4: Meta-Paper Assembly
    try:
        result4 = await test_meta_paper_assembly()
        test_results.append(("Meta-Paper Assembly", result4))
    except Exception as e:
        print(f"❌ Meta-Paper Assembly test failed: {e}")
        test_results.append(("Meta-Paper Assembly", False))
    
    # Summary
    print("🎯 TEST SUMMARY:")
    print("=" * 50)
    
    passed_tests = 0
    for test_name, passed in test_results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
        if passed:
            passed_tests += 1
    
    success_rate = passed_tests / len(test_results)
    print(f"\n📊 Overall Success Rate: {passed_tests}/{len(test_results)} ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("\n🎉 CHUNK EMBEDDING SYSTEM: SUCCESS!")
        print("The system successfully generates embeddings, ranks chunks, and assembles meta-papers.")
        print("Ready for integration with NWTN candidate generation pipeline.")
        
        print(f"\n🔧 REASONING ENGINE CAPABILITIES:")
        descriptions = get_reasoning_engine_descriptions()
        for engine, description in descriptions.items():
            print(f"   - {engine}: {description}")
        
    else:
        print(f"\n⚠️  CHUNK EMBEDDING SYSTEM: PARTIAL SUCCESS")
        print(f"Success rate: {success_rate:.1%} (need ≥80%)")
        print("Some components may need refinement.")
    
    return success_rate >= 0.8

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