#!/usr/bin/env python3
"""
PRSM Integration Demo
End-to-end demonstration of the complete PRSM pipeline

This demo showcases:
1. User query processing through NWTN Orchestrator
2. Vector similarity search with provenance tracking
3. FTNS token economics and creator royalties
4. Complete reasoning trace for transparency
5. Performance monitoring and optimization

Perfect for investor demonstrations and technical validation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from prsm.core.models import UserInput
    from prsm.vector_store.base import VectorStoreConfig, VectorStoreType, ContentType
    
    # Import mock components for demo
    from test_vector_store_mock import MockVectorStore, SimplePerformanceTracker
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the PRSM root directory")
    exit(1)


class MockEmbeddingService:
    """
    Mock embedding service that simulates OpenAI/Anthropic API calls
    
    In production, this would connect to real AI services.
    For the demo, we generate realistic embeddings and simulate API costs.
    """
    
    def __init__(self):
        self.api_calls = 0
        self.total_cost = 0.0
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (simulated)"""
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Generate deterministic embedding based on text content
        # This ensures similar queries get similar embeddings
        hash_value = hash(text.lower()) % (2**32)
        np.random.seed(hash_value)
        embedding = np.random.random(384).astype(np.float32)
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Track API usage
        self.api_calls += 1
        self.total_cost += 0.0001  # Simulate $0.0001 per embedding
        
        return embedding
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "api_calls": self.api_calls,
            "total_cost_usd": self.total_cost,
            "average_cost_per_call": self.total_cost / max(1, self.api_calls)
        }


class MockFTNSService:
    """
    Mock FTNS token service for economic demonstration
    
    Simulates the complete token economy including:
    - User balance management
    - Context allocation and charging
    - Creator royalty distribution
    - Economic transaction tracking
    """
    
    def __init__(self):
        # User balances (simulated)
        self.user_balances = {
            "demo_user": 1000.0,
            "researcher_alice": 50.0,
            "data_scientist_bob": 200.0
        }
        
        # Creator earnings
        self.creator_earnings = {}
        
        # Transaction history
        self.transactions = []
        
    async def get_user_balance(self, user_id: str) -> float:
        """Get user's current FTNS balance"""
        return self.user_balances.get(user_id, 0.0)
    
    async def charge_context_usage(self, user_id: str, context_cost: float) -> bool:
        """Charge user for context usage"""
        current_balance = await self.get_user_balance(user_id)
        
        if current_balance >= context_cost:
            self.user_balances[user_id] = current_balance - context_cost
            
            transaction = {
                "timestamp": datetime.utcnow(),
                "user_id": user_id,
                "type": "context_charge",
                "amount": -context_cost,
                "description": "NWTN context processing"
            }
            self.transactions.append(transaction)
            
            logger.info(f"💰 Charged {user_id}: {context_cost:.4f} FTNS (balance: {self.user_balances[user_id]:.4f})")
            return True
        else:
            logger.warning(f"❌ Insufficient FTNS balance for {user_id}: {current_balance:.4f} < {context_cost:.4f}")
            return False
    
    async def distribute_royalties(self, content_matches: List[Dict[str, Any]], total_value: float):
        """Distribute royalties to content creators"""
        if not content_matches:
            return
        
        total_royalty_value = 0.0
        
        for match in content_matches:
            creator_id = match.get("creator_id")
            royalty_rate = match.get("royalty_rate", 0.08)
            relevance_weight = match.get("similarity_score", 1.0)
            
            if creator_id:
                # Calculate weighted royalty based on relevance
                creator_royalty = total_value * royalty_rate * relevance_weight
                
                if creator_id not in self.creator_earnings:
                    self.creator_earnings[creator_id] = 0.0
                
                self.creator_earnings[creator_id] += creator_royalty
                total_royalty_value += creator_royalty
                
                transaction = {
                    "timestamp": datetime.utcnow(),
                    "creator_id": creator_id,
                    "type": "royalty_payment",
                    "amount": creator_royalty,
                    "description": f"Content usage royalty ({royalty_rate*100:.1f}%)"
                }
                self.transactions.append(transaction)
                
                logger.info(f"👨‍🎨 Royalty to {creator_id}: {creator_royalty:.6f} FTNS ({royalty_rate*100:.1f}%)")
        
        logger.info(f"💎 Total royalties distributed: {total_royalty_value:.6f} FTNS")
    
    def get_economics_summary(self) -> Dict[str, Any]:
        """Get complete economics summary"""
        return {
            "user_balances": self.user_balances.copy(),
            "creator_earnings": self.creator_earnings.copy(),
            "total_transactions": len(self.transactions),
            "total_volume": sum(abs(t.get("amount", 0)) for t in self.transactions)
        }


class PRSMIntegrationDemo:
    """
    Complete PRSM integration demonstration
    
    Orchestrates the entire pipeline from user query to final response,
    showcasing all major PRSM components working together.
    """
    
    def __init__(self):
        self.vector_store = None
        self.embedding_service = MockEmbeddingService()
        self.ftns_service = MockFTNSService()
        self.demo_content_loaded = False
        
        # Demo performance tracking
        self.demo_stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0
        }
    
    async def initialize(self):
        """Initialize all PRSM components"""
        print("🚀 INITIALIZING PRSM INTEGRATION DEMO")
        print("=" * 60)
        
        # Initialize vector store
        print("📦 Initializing vector store...")
        config = VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            host="demo_host",
            port=5432,
            database="prsm_demo",
            collection_name="research_content",
            vector_dimension=384
        )
        
        self.vector_store = MockVectorStore(config)
        await self.vector_store.connect()
        print("✅ Vector store ready")
        
        # Load demo content
        await self._load_demo_content()
        print("✅ Demo content loaded")
        
        # Initialize FTNS service
        print("💰 FTNS token service ready")
        
        print("\n🎯 PRSM INTEGRATION DEMO READY!")
        print("=" * 60)
    
    async def _load_demo_content(self):
        """Load sample academic content for demonstration"""
        print("📚 Loading sample research content...")
        
        demo_papers = [
            {
                "title": "Decentralized AI Governance: A Framework for Democratic Control",
                "authors": ["Dr. Sarah Chen", "Prof. Michael Rodriguez"],
                "abstract": "This paper presents a novel framework for democratic governance of artificial intelligence systems through decentralized protocols and token-based voting mechanisms.",
                "content_type": ContentType.RESEARCH_PAPER,
                "creator_id": "sarah_chen_university",
                "royalty_rate": 0.08,
                "quality_score": 0.94,
                "citation_count": 47,
                "keywords": ["AI governance", "decentralization", "democracy", "blockchain"]
            },
            {
                "title": "Provenance Tracking in Large-Scale AI Training: Legal and Technical Considerations",
                "authors": ["Dr. Alex Kim", "Dr. Jennifer Walsh"],
                "abstract": "An analysis of content provenance tracking systems for AI training data, addressing both technical implementation and legal compliance requirements.",
                "content_type": ContentType.RESEARCH_PAPER,
                "creator_id": "alex_kim_legal_ai",
                "royalty_rate": 0.08,
                "quality_score": 0.91,
                "citation_count": 73,
                "keywords": ["provenance", "AI training", "copyright", "compliance"]
            },
            {
                "title": "Economic Incentives in Distributed AI Networks: A Game-Theoretic Analysis",
                "authors": ["Prof. David Thompson", "Dr. Lisa Patel"],
                "abstract": "This study examines economic mechanisms for sustainable distributed AI networks, focusing on token incentives and creator compensation models.",
                "content_type": ContentType.RESEARCH_PAPER,
                "creator_id": "david_thompson_economics",
                "royalty_rate": 0.08,
                "quality_score": 0.89,
                "citation_count": 62,
                "keywords": ["economics", "game theory", "AI networks", "incentives"]
            },
            {
                "title": "Climate Impact Dataset: Global Temperature Anomalies 2000-2024",
                "authors": ["Climate Research Consortium"],
                "abstract": "Comprehensive dataset of global temperature measurements with high spatial and temporal resolution, processed for machine learning applications.",
                "content_type": ContentType.DATASET,
                "creator_id": "climate_research_consortium",
                "royalty_rate": 0.06,
                "quality_score": 0.96,
                "citation_count": 234,
                "keywords": ["climate", "temperature", "dataset", "global warming"]
            },
            {
                "title": "Efficient Vector Database Implementation for Similarity Search",
                "authors": ["Dr. Bob Johnson"],
                "abstract": "Open-source implementation of high-performance vector database optimized for similarity search in large-scale AI applications.",
                "content_type": ContentType.CODE,
                "creator_id": "bob_johnson_dev",
                "royalty_rate": 0.05,
                "quality_score": 0.87,
                "citation_count": 156,
                "keywords": ["vector database", "similarity search", "performance", "optimization"]
            }
        ]
        
        # Generate embeddings and store content
        for i, paper in enumerate(demo_papers):
            # Create content text for embedding
            content_text = f"{paper['title']} {paper['abstract']} {' '.join(paper['keywords'])}"
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content_text)
            
            # Store in vector database
            content_cid = f"QmDemo{i+1}_{paper['title'].replace(' ', '_')[:30]}"
            
            metadata = {
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "content_type": paper["content_type"].value,
                "creator_id": paper["creator_id"],
                "royalty_rate": paper["royalty_rate"],
                "quality_score": paper["quality_score"],
                "citation_count": paper["citation_count"],
                "keywords": paper["keywords"]
            }
            
            await self.vector_store.store_content_with_embeddings(
                content_cid, embedding, metadata
            )
            
            print(f"  📄 Loaded: {paper['title'][:50]}...")
        
        self.demo_content_loaded = True
        
        # Display content statistics
        stats = await self.vector_store.get_collection_stats()
        print(f"\n📊 Content Library Statistics:")
        print(f"   Total papers: {stats.get('total_vectors', 0)}")
        print(f"   Unique creators: {stats.get('unique_creators', 0)}")
        print(f"   Average citations: {stats.get('average_citations', 0):.1f}")
        print(f"   Average quality: {stats.get('average_quality', 0):.2f}")
    
    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a complete user query through the PRSM pipeline"""
        start_time = time.time()
        
        print(f"\n🔍 PROCESSING QUERY FROM {user_id}")
        print("=" * 60)
        print(f"Query: \"{query}\"")
        print()
        
        # Step 1: Check user balance
        print("💰 Step 1: Checking FTNS balance...")
        user_balance = await self.ftns_service.get_user_balance(user_id)
        print(f"   User balance: {user_balance:.4f} FTNS")
        
        if user_balance < 0.1:  # Minimum required
            print("❌ Insufficient FTNS balance for query processing")
            return {"error": "Insufficient FTNS balance"}
        
        # Step 2: Intent analysis and context estimation
        print("\n🧠 Step 2: NWTN intent analysis...")
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Simple intent classification
        query_lower = query.lower()
        if "governance" in query_lower or "democratic" in query_lower:
            intent_category = "ai_governance"
            complexity_estimate = 0.8
        elif "provenance" in query_lower or "copyright" in query_lower:
            intent_category = "legal_compliance"
            complexity_estimate = 0.7
        elif "economic" in query_lower or "incentive" in query_lower:
            intent_category = "economics"
            complexity_estimate = 0.6
        else:
            intent_category = "general_research"
            complexity_estimate = 0.5
        
        # Calculate context cost
        base_cost = 0.05
        complexity_multiplier = 1 + complexity_estimate
        context_cost = base_cost * complexity_multiplier
        
        print(f"   Intent category: {intent_category}")
        print(f"   Complexity: {complexity_estimate:.1f}")
        print(f"   Context cost: {context_cost:.4f} FTNS")
        
        # Step 3: Generate query embedding
        print("\n🔤 Step 3: Generating query embedding...")
        query_embedding = await self.embedding_service.generate_embedding(query)
        print(f"   Embedding generated (dimension: {len(query_embedding)})")
        
        # Step 4: Vector similarity search
        print("\n🔍 Step 4: Searching content database...")
        search_results = await self.vector_store.search_similar_content(
            query_embedding, top_k=3
        )
        
        print(f"   Found {len(search_results)} relevant papers:")
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result.metadata.get('title', 'Unknown')[:60]}...")
            print(f"      Similarity: {result.similarity_score:.3f}, Creator: {result.creator_id}")
        
        # Step 5: Charge user for context usage
        print(f"\n💳 Step 5: Charging context usage...")
        charge_success = await self.ftns_service.charge_context_usage(user_id, context_cost)
        
        if not charge_success:
            print("❌ Failed to charge for context usage")
            return {"error": "Payment processing failed"}
        
        # Step 6: Distribute creator royalties
        print("\n👨‍🎨 Step 6: Distributing creator royalties...")
        royalty_data = []
        for result in search_results:
            royalty_data.append({
                "creator_id": result.creator_id,
                "royalty_rate": result.royalty_rate,
                "similarity_score": result.similarity_score
            })
        
        await self.ftns_service.distribute_royalties(royalty_data, context_cost * 0.3)  # 30% goes to creators
        
        # Step 7: Compile final response
        print("\n📝 Step 7: Compiling response...")
        
        # Generate response summary
        top_result = search_results[0] if search_results else None
        if top_result:
            response_text = f"""Based on your query about "{query}", I found highly relevant research in our decentralized knowledge base.

Most relevant result:
📄 {top_result.metadata.get('title', 'Unknown')}
👥 Authors: {', '.join(top_result.metadata.get('authors', ['Unknown']))}
🎯 Relevance: {top_result.similarity_score:.1%}
📊 Quality Score: {top_result.quality_score:.2f}
📈 Citations: {top_result.citation_count}

Abstract: {top_result.metadata.get('abstract', 'Not available')}

This research is part of PRSM's legally compliant, creator-compensated knowledge network. Your query contributed {context_cost:.4f} FTNS tokens to support the creators and maintain the decentralized research infrastructure."""
        else:
            response_text = "No relevant research found for your query."
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        self.demo_stats["queries_processed"] += 1
        self.demo_stats["total_processing_time"] += processing_time
        self.demo_stats["average_response_time"] = (
            self.demo_stats["total_processing_time"] / self.demo_stats["queries_processed"]
        )
        
        print(f"\n✅ Query processed successfully in {processing_time:.2f}s")
        
        return {
            "success": True,
            "response": response_text,
            "processing_time": processing_time,
            "context_cost": context_cost,
            "results_found": len(search_results),
            "intent_category": intent_category,
            "complexity": complexity_estimate
        }
    
    def display_system_status(self):
        """Display comprehensive system status"""
        print("\n📊 PRSM SYSTEM STATUS")
        print("=" * 60)
        
        # Vector store stats
        print("🗄️  Vector Store Performance:")
        if hasattr(self.vector_store, 'performance_tracker'):
            perf_stats = self.vector_store.performance_tracker.get_performance_stats()
            print(f"   Operations: {perf_stats.get('total_operations', 0)}")
            print(f"   Success rate: {perf_stats.get('success_rate', 0)*100:.1f}%")
            print(f"   Avg duration: {perf_stats.get('average_duration', 0)*1000:.2f}ms")
        
        # Embedding service stats
        embedding_stats = self.embedding_service.get_usage_stats()
        print(f"\n🔤 Embedding Service Usage:")
        print(f"   API calls: {embedding_stats['api_calls']}")
        print(f"   Total cost: ${embedding_stats['total_cost_usd']:.4f}")
        print(f"   Avg per call: ${embedding_stats['average_cost_per_call']:.6f}")
        
        # FTNS economics
        economics = self.ftns_service.get_economics_summary()
        print(f"\n💰 FTNS Token Economics:")
        print(f"   Active users: {len(economics['user_balances'])}")
        print(f"   Total transactions: {economics['total_transactions']}")
        print(f"   Transaction volume: {economics['total_volume']:.4f} FTNS")
        
        print(f"\n   Creator Earnings:")
        for creator_id, earnings in economics['creator_earnings'].items():
            print(f"     {creator_id}: {earnings:.6f} FTNS")
        
        # Demo performance
        print(f"\n⚡ Demo Performance:")
        print(f"   Queries processed: {self.demo_stats['queries_processed']}")
        print(f"   Avg response time: {self.demo_stats['average_response_time']:.2f}s")
        print(f"   Total processing time: {self.demo_stats['total_processing_time']:.2f}s")


async def run_interactive_demo():
    """Run interactive command-line demo"""
    demo = PRSMIntegrationDemo()
    await demo.initialize()
    
    print("\n🎯 INTERACTIVE PRSM DEMO")
    print("=" * 60)
    print("Try these example queries:")
    print("• 'How can AI governance be democratized?'")
    print("• 'What are the legal requirements for AI training data?'")
    print("• 'How do economic incentives work in distributed AI networks?'")
    print("• 'Show me climate change research datasets'")
    print("\nType 'status' to see system metrics, 'quit' to exit")
    print("=" * 60)
    
    current_user = "demo_user"
    
    while True:
        try:
            query = input(f"\n🔍 [{current_user}] Enter your research query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'status':
                demo.display_system_status()
                continue
            elif query.lower().startswith('user '):
                # Allow switching users
                new_user = query[5:].strip()
                if new_user:
                    current_user = new_user
                    print(f"✅ Switched to user: {current_user}")
                continue
            elif not query:
                continue
            
            # Process the query
            result = await demo.process_query(current_user, query)
            
            if result.get("success"):
                print(f"\n📄 PRSM RESPONSE:")
                print("=" * 60)
                print(result["response"])
                print("\n📊 Query Metrics:")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Context cost: {result['context_cost']:.4f} FTNS")
                print(f"   Results found: {result['results_found']}")
                print(f"   Intent: {result['intent_category']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
    
    print("\n👋 Thanks for trying the PRSM integration demo!")
    demo.display_system_status()


async def run_automated_demo():
    """Run automated demo with predefined queries"""
    demo = PRSMIntegrationDemo()
    await demo.initialize()
    
    print("\n🎯 AUTOMATED PRSM DEMO")
    print("=" * 60)
    
    demo_queries = [
        ("demo_user", "How can AI governance be democratized through blockchain technology?"),
        ("researcher_alice", "What are the legal requirements for content provenance in AI training?"),
        ("data_scientist_bob", "Show me research on economic incentives in distributed AI networks"),
        ("demo_user", "Find climate change datasets with high quality scores")
    ]
    
    for user_id, query in demo_queries:
        print(f"\n⏳ Processing query {demo_queries.index((user_id, query)) + 1}/{len(demo_queries)}...")
        result = await demo.process_query(user_id, query)
        
        if result.get("success"):
            print("✅ Query completed successfully")
        else:
            print(f"❌ Query failed: {result.get('error')}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Final system status
    print("\n🏁 DEMO COMPLETED - FINAL SYSTEM STATUS")
    demo.display_system_status()


def main():
    """Main demo launcher"""
    print("🚀 PRSM INTEGRATION DEMO LAUNCHER")
    print("=" * 60)
    print("This demo showcases the complete PRSM pipeline:")
    print("• NWTN Orchestrator for query processing")
    print("• Vector similarity search with provenance")
    print("• FTNS token economics and creator royalties")
    print("• Performance monitoring and optimization")
    print()
    
    mode = input("Choose demo mode (1=Interactive, 2=Automated): ").strip()
    
    if mode == "1":
        print("\n🎮 Starting interactive demo...")
        asyncio.run(run_interactive_demo())
    elif mode == "2":
        print("\n🤖 Starting automated demo...")
        asyncio.run(run_automated_demo())
    else:
        print("❌ Invalid choice. Exiting.")


if __name__ == "__main__":
    main()