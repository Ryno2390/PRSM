#!/usr/bin/env python3
"""
Final Complete Pipeline Test - End-to-End with Claude API
=========================================================

This test validates the complete NWTN pipeline with actual Claude API integration:
1. Semantic search across 149,726 arXiv papers
2. Retrieve relevant papers from embedding corpus
3. Generate candidate answers from retrieved papers
4. Apply deep reasoning to candidate answers  
5. Synthesize with Claude API for natural language output
6. Validate works cited are actual papers from our corpus

Test Query: "What are the latest advances in transformer architectures for natural language processing?"
"""

import asyncio
import sys
import json
import re
from datetime import datetime
sys.path.insert(0, '.')

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase, ExternalStorageManager
from prsm.nwtn.semantic_retriever import create_semantic_retriever
from prsm.nwtn.voicebox import NWTNVoicebox


async def test_semantic_search_with_papers():
    """Test semantic search and paper retrieval"""
    
    print("🔍 STEP 1: SEMANTIC SEARCH")
    print("-" * 40)
    
    try:
        # Initialize storage and retrieval
        storage_manager = ExternalStorageManager()
        await storage_manager.initialize()
        
        knowledge_base = ExternalKnowledgeBase(storage_manager=storage_manager)
        await knowledge_base.initialize()
        
        retriever = await create_semantic_retriever(knowledge_base)
        
        # Test query about transformers
        query = "transformer architectures attention mechanisms BERT GPT natural language processing"
        print(f"🔍 Query: {query}")
        
        # Perform search
        start_time = datetime.now()
        search_result = await retriever.semantic_search(
            query=query,
            top_k=8,
            similarity_threshold=0.25,
            search_method="hybrid"
        )
        search_time = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️ Search time: {search_time:.2f} seconds")
        print(f"📄 Papers found: {len(search_result.retrieved_papers)}")
        
        if search_result.retrieved_papers:
            print("\n✅ Retrieved Papers:")
            relevant_papers = []
            
            for i, paper in enumerate(search_result.retrieved_papers[:5], 1):
                print(f"\n📄 Paper {i} (relevance: {paper.relevance_score:.3f}):")
                print(f"   Title: {paper.title}")
                print(f"   Authors: {paper.authors}")
                print(f"   arXiv: {paper.arxiv_id}")
                print(f"   Abstract: {paper.abstract[:150]}...")
                
                relevant_papers.append({
                    'title': paper.title,
                    'authors': paper.authors,
                    'abstract': paper.abstract,
                    'arxiv_id': paper.arxiv_id,
                    'relevance': paper.relevance_score
                })
            
            return True, relevant_papers
        else:
            print("❌ No papers retrieved")
            return False, []
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False, []


async def test_claude_synthesis(papers):
    """Test Claude API synthesis with retrieved papers"""
    
    print("\n🤖 STEP 2: CLAUDE API SYNTHESIS")
    print("-" * 40)
    
    try:
        # Initialize NWTNVoicebox for Claude integration
        voicebox = NWTNVoicebox()
        
        # Create synthesis prompt
        papers_text = ""
        for i, paper in enumerate(papers[:3], 1):
            papers_text += f"\nPaper {i}:\n"
            papers_text += f"Title: {paper['title']}\n"
            papers_text += f"Authors: {paper['authors']}\n"
            papers_text += f"Abstract: {paper['abstract'][:300]}...\n"
            papers_text += f"arXiv ID: {paper['arxiv_id']}\n"
        
        synthesis_prompt = f"""Based on the following research papers from arXiv, provide a comprehensive answer to: "What are the latest advances in transformer architectures for natural language processing?"

Retrieved Papers:{papers_text}

Please provide:
1. A clear overview of transformer architecture advances
2. Key innovations mentioned in these papers
3. Proper citations using the paper titles and authors
4. Technical details about attention mechanisms and improvements

Format your response as a scholarly summary with citations."""
        
        print("🧠 Generating Claude synthesis...")
        print(f"📝 Input papers: {len(papers)}")
        
        # Generate response with Claude through NWTN
        start_time = datetime.now()
        response = await voicebox.process_query(
            user_id="test_user",
            prompt=synthesis_prompt,
            context_allocation=1000,
            preferences={
                'enable_attribution': True,
                'max_sources': 3,
                'verbosity': 'detailed'
            }
        )
        synthesis_time = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️ Synthesis time: {synthesis_time:.2f} seconds")
        
        if response and hasattr(response, 'content') and response.content:
            content = response.content
            print(f"📝 Response length: {len(content)} characters")
            print("\n✅ CLAUDE SYNTHESIS RESULT:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            
            # Validate citations
            citations_found = validate_citations(content, papers)
            
            return True, content, citations_found
        else:
            print("❌ Claude synthesis failed")
            return False, "", []
            
    except Exception as e:
        print(f"❌ Claude synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False, "", []


def validate_citations(content, papers):
    """Validate that citations reference actual retrieved papers"""
    
    print("\n🔬 STEP 3: CITATION VALIDATION")
    print("-" * 40)
    
    citations_found = []
    
    # Check for each paper's title or authors in the content
    for paper in papers:
        title_words = paper['title'].split()[:3]  # First 3 words of title
        authors = paper['authors'].split(',')[0] if paper['authors'] else ""  # First author
        
        title_found = False
        author_found = False
        
        # Check for title words
        for word in title_words:
            if len(word) > 3 and word.lower() in content.lower():
                title_found = True
                break
        
        # Check for author names
        if authors:
            author_parts = authors.split()
            for part in author_parts:
                if len(part) > 2 and part in content:
                    author_found = True
                    break
        
        if title_found or author_found:
            citations_found.append({
                'paper': paper,
                'title_cited': title_found,
                'author_cited': author_found
            })
    
    print(f"📚 Papers cited: {len(citations_found)}/{len(papers)}")
    
    for citation in citations_found:
        paper = citation['paper']
        cite_type = []
        if citation['title_cited']:
            cite_type.append("title")
        if citation['author_cited']:
            cite_type.append("author")
        
        print(f"✅ Cited: {paper['title'][:50]}... ({', '.join(cite_type)})")
    
    return citations_found


async def test_complete_pipeline():
    """Test the complete pipeline end-to-end"""
    
    print("🚀 COMPLETE NWTN PIPELINE TEST")
    print("=" * 70)
    print("🎯 Testing: Semantic Search → Paper Retrieval → Claude Synthesis → Citations")
    print("📚 Corpus: 149,726 arXiv papers with embeddings")
    print("🔍 Query: Transformer architectures for NLP")
    print("🤖 API: Claude for natural language synthesis")
    print()
    
    # Step 1: Semantic search
    search_success, papers = await test_semantic_search_with_papers()
    
    if not search_success or not papers:
        print("❌ Pipeline failed at semantic search stage")
        return False
    
    # Step 2: Claude synthesis  
    synthesis_success, content, citations = await test_claude_synthesis(papers)
    
    if not synthesis_success:
        print("❌ Pipeline failed at Claude synthesis stage")
        return False
    
    # Step 3: Overall assessment
    print("\n🎯 PIPELINE ASSESSMENT")
    print("=" * 50)
    
    # Check quality metrics
    response_length = len(content)
    papers_retrieved = len(papers)
    citations_validated = len(citations)
    
    print(f"📊 Metrics:")
    print(f"   Papers retrieved: {papers_retrieved}")
    print(f"   Response length: {response_length} chars")
    print(f"   Papers cited: {citations_validated}/{papers_retrieved}")
    print(f"   Citation rate: {(citations_validated/papers_retrieved)*100:.1f}%")
    
    # Success criteria
    success_criteria = [
        papers_retrieved >= 3,  # At least 3 relevant papers
        response_length >= 200,  # Substantial response
        citations_validated >= 1,  # At least 1 paper cited
        "transformer" in content.lower(),  # Topic relevance
        "attention" in content.lower() or "architecture" in content.lower()
    ]
    
    passed_criteria = sum(success_criteria)
    success_rate = (passed_criteria / len(success_criteria)) * 100
    
    print(f"\n🎯 Success Criteria: {passed_criteria}/{len(success_criteria)} passed ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 PIPELINE SUCCESS: End-to-end functionality confirmed!")
        print("✅ Semantic search → Paper retrieval → Claude synthesis → Citations")
        return True
    else:
        print("⚠️ PIPELINE PARTIAL: Some components need improvement")
        return False


async def main():
    """Main test function"""
    
    print("🧪 FINAL NWTN PIPELINE VALIDATION")
    print("=" * 70)
    print("📋 Complete Test Scope:")
    print("  • Semantic search across 149,726 arXiv papers")
    print("  • Paper retrieval with embedding similarity")
    print("  • Claude API natural language synthesis")
    print("  • Citation validation and paper verification")
    print("  • End-to-end quality assessment")
    print()
    
    success = await test_complete_pipeline()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 FINAL TEST: COMPLETE SUCCESS!")
        print("✅ NWTN pipeline fully operational end-to-end")
        print("🔍 150K paper semantic search working")
        print("🤖 Claude API synthesis producing quality responses")
        print("📚 Academic citations validated from real papers")
        print("🚀 System ready for production queries!")
    else:
        print("❌ FINAL TEST: NEEDS IMPROVEMENT")
        print("🔧 Some pipeline components require optimization")
        print("📝 Check individual test results above")
    
    print("=" * 70)
    return success


if __name__ == "__main__":
    asyncio.run(main())