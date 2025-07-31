#!/usr/bin/env python3
"""
Test Orchestrator Response Compilation
=====================================
Tests the _compile_final_response method directly to verify Claude response extraction
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import PRSMSession, ClarifiedPrompt, UserInput
from prsm.agents.executors.model_executor import ExecutionResult

print("🧪 ORCHESTRATOR RESPONSE COMPILATION TEST")
print("=" * 50)

async def test_response_compilation():
    """Test response compilation with mock Claude results"""
    
    try:
        # Create test session
        session = PRSMSession(
            user_id="test-user",
            nwtn_context_allocation=100,
            context_used=0,
            reasoning_trace=[],
            safety_flags=[],
            metadata={}
        )
        
        print("✅ Test session created")
        
        # Mock Claude response result (simulating what ModelExecutor returns)
        claude_response_content = "The answer to 2+2 is 4."
        mock_claude_result = ExecutionResult(
            model_id="claude-3-5-sonnet-20241022",
            result={
                "type": "model_response",
                "content": claude_response_content,
                "model_id": "claude-3-5-sonnet-20241022",
                "validation_passed": True,
                "safety_score": 0.9,
                "timestamp": 1234567890
            },
            execution_time=1.5,
            success=True,
            error=None
        )
        
        print("✅ Mock Claude result created")
        print(f"   Content: '{claude_response_content}'")
        
        # Mock agent results with successful executor
        agent_results = {
            "architect": {"status": "completed", "result": "Task decomposed successfully"},
            "router": {"status": "completed", "result": "Model selected: claude-3-5-sonnet-20241022"},
            "prompter": {"status": "completed", "result": "Prompt optimized"},
            "executor": {
                "status": "completed", 
                "success": True,
                "execution_results": [mock_claude_result]
            },
            "compiler": {"status": "completed", "result": "Ready to compile"}
        }
        
        print("✅ Mock agent results created")
        
        # Initialize orchestrator
        orchestrator = EnhancedNWTNOrchestrator()
        print("✅ Enhanced orchestrator initialized")
        
        # Test the response compilation method directly
        print("🚀 Testing _compile_final_response method...")
        
        compiled_response = await orchestrator._compile_final_response(agent_results, session)
        
        print("🎉 RESPONSE COMPILATION COMPLETE!")
        print("=" * 40)
        
        if compiled_response:
            print("📋 COMPILED RESPONSE:")
            print("-" * 20)
            print(f"Response type: {type(compiled_response)}")
            print(f"Response length: {len(compiled_response)} characters")
            print()
            print("📝 FULL RESPONSE:")
            print("-" * 15)
            print(compiled_response)
            print()
            
            # Check if Claude's actual response is in the compiled response
            if claude_response_content in compiled_response:
                print("✅ SUCCESS: Claude's actual response found in compiled output!")
                print(f"   Original: '{claude_response_content}'")
                return True
            else:
                print("❌ FAILURE: Claude's actual response NOT found in compiled output")
                print(f"   Expected: '{claude_response_content}'")
                print(f"   Got response without Claude content")
                return False
        else:
            print("❌ CRITICAL FAILURE: No compiled response returned")
            return False
        
    except Exception as e:
        print(f"❌ Response compilation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_response_compilation())
    if success:
        print("\n🎉 ORCHESTRATOR RESPONSE COMPILATION TEST: PASSED ✅")
    else:
        print("\n💥 ORCHESTRATOR RESPONSE COMPILATION TEST: FAILED ❌")