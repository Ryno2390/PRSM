#!/usr/bin/env python3
"""
NWTN Clean Demo - Complete Working Pipeline
==========================================
Clean test of the complete NWTN pipeline with fixed routing
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput

print("🚀 NWTN PIPELINE - CLEAN DEMONSTRATION")
print("=" * 60)

async def test_nwtn_clean():
    """Clean test of NWTN pipeline with natural language response"""
    
    # Clear, interesting test prompt
    test_prompt = "Explain how machine learning algorithms can be used to predict climate change patterns"
    
    print("📝 USER QUERY:")
    print("-" * 20)
    print(f'"{test_prompt}"')
    print()
    print("🔄 Initializing NWTN pipeline...")
    
    try:
        # Load Claude API key
        api_key_path = "/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt"
        with open(api_key_path, 'r') as f:
            claude_api_key = f.read().strip()
        
        # Create UserInput with optimal settings for demonstration
        user_input = UserInput(
            user_id="demo_user",
            prompt=test_prompt,
            context_allocation=300,  # Moderate allocation
            preferences={
                "reasoning_depth": "moderate",
                "response_length": "comprehensive", 
                "enable_cross_domain": True,
                "api_key": claude_api_key,
                "disable_database_persistence": True,
                "preferred_model": "claude-3-5-sonnet-20241022"
            }
        )
        
        print("✅ NWTN configuration ready")
        print("   • Model: claude-3-5-sonnet-20241022")
        print("   • Context allocation: 300 tokens")
        print("   • Reasoning depth: moderate")
        print()
        
        # Initialize and run NWTN
        orchestrator = EnhancedNWTNOrchestrator()
        
        print("🚀 EXECUTING NWTN PIPELINE...")
        print("⏳ Processing (this may take a few minutes for complete analysis)...")
        print()
        
        response = await orchestrator.process_query(user_input=user_input)
        
        print("✅ NWTN PROCESSING COMPLETE!")
        print("=" * 60)
        print()
        
        # Display the response exactly as user would see it
        if response and hasattr(response, 'final_answer') and response.final_answer:
            print("🎯 NATURAL LANGUAGE RESPONSE:")
            print("=" * 40)
            print()
            print(response.final_answer)
            print()
            print("=" * 60)
            
            # Show processing summary
            print("📊 PROCESSING SUMMARY:")
            print("-" * 25)
            print(f"✅ Response generated: {len(response.final_answer)} characters")
            print(f"✅ Context used: {response.context_used} tokens")
            print(f"✅ Safety validated: {response.safety_validated}")
            if response.confidence_score:
                print(f"✅ Confidence score: {response.confidence_score:.2f}")
            print()
            
            return True, response.final_answer
                
        else:
            print("❌ ERROR: No natural language response generated")
            print("The NWTN pipeline completed but did not produce a final answer.")
            return False, None
        
    except Exception as e:
        print(f"❌ NWTN pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("🎬 Starting NWTN demonstration...")
    print()
    
    success, response = asyncio.run(test_nwtn_clean())
    
    print()
    if success:
        print("🎉 NWTN DEMONSTRATION: SUCCESS ✅")
        print()
        print("The NWTN pipeline successfully processed your query and generated")
        print("a comprehensive natural language response using:")
        print("• Semantic retrieval from scientific papers")
        print("• Multi-agent reasoning and analysis")
        print("• Claude API synthesis")
        print("• Response validation and compilation")
    else:
        print("💥 NWTN DEMONSTRATION: FAILED ❌")
        print("Check the error output above for details.")