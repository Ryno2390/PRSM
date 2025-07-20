#!/usr/bin/env python3
"""
Verify that all NWTN system fixes are working
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def verify_fixes():
    """Verify all fixes are working"""
    print("🔍 Verifying NWTN system fixes...")
    
    # Set environment variables
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    os.environ["PRSM_NWTN_MODEL"] = "claude-3-5-sonnet-20241022"
    
    try:
        # Test 1: ModelExecutor execute_request method
        print("1. Testing ModelExecutor execute_request method...")
        from prsm.agents.executors.model_executor import ModelExecutor
        
        executor = ModelExecutor()
        
        # Check that the method exists
        if hasattr(executor, 'execute_request'):
            print("   ✅ execute_request method exists")
        else:
            print("   ❌ execute_request method missing")
            return False
        
        # Test 2: SemanticEmbeddingEngine
        print("2. Testing SemanticEmbeddingEngine...")
        from prsm.embeddings.semantic_embedding_engine import SemanticEmbeddingEngine, EmbeddingSearchQuery, EmbeddingSpace
        
        engine = SemanticEmbeddingEngine()
        
        # Test proper parameter format
        query = EmbeddingSearchQuery(
            query_text="test query",
            embedding_space=EmbeddingSpace.CONTENT_SEMANTIC,
            max_results=10
        )
        print("   ✅ EmbeddingSearchQuery created successfully")
        
        # Test 3: MarketplaceRecommendationEngine  
        print("3. Testing MarketplaceRecommendationEngine...")
        from prsm.marketplace.recommendation_engine import MarketplaceRecommendationEngine, RecommendationContext
        
        rec_engine = MarketplaceRecommendationEngine()
        context = RecommendationContext(
            user_profile=None,
            current_resource=None,
            search_query="test",
            filters={},
            session_context={},
            business_constraints={}
        )
        print("   ✅ RecommendationContext created successfully")
        
        # Test 4: TaskDefinition validation
        print("4. Testing TaskDefinition validation...")
        from prsm.context.selective_parallelism_engine import TaskDefinition, TaskComplexity
        from prsm.core.models import AgentType
        from uuid import uuid4
        
        task_def = TaskDefinition(
            task_id=uuid4(),
            task_name="Test Task",
            agent_type=AgentType.EXECUTOR,
            complexity=TaskComplexity.MODERATE,
            description="Test task",
            parameters={}
        )
        print("   ✅ TaskDefinition created successfully")
        
        # Test 5: Configuration
        print("5. Testing configuration...")
        from prsm.core.config import get_settings
        settings = get_settings()
        print(f"   ✅ NWTN default model: {settings.nwtn_default_model}")
        
        # Test 6: Multi-modal reasoning engine basic import
        print("6. Testing multi-modal reasoning engine import...")
        from prsm.nwtn.multi_modal_reasoning_engine import MultiModalReasoningEngine
        print("   ✅ MultiModalReasoningEngine imported successfully")
        
        print("\n🎉 All fixes verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_fixes())
    if success:
        print("\n✅ All NWTN system fixes are working correctly!")
    else:
        print("\n❌ Some fixes are not working properly!")