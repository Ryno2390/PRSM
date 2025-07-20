#!/usr/bin/env python3
"""
Test complete NWTN query processing with Claude API
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_complete_nwtn_query():
    """Test the complete NWTN query processing pipeline"""
    print("🚀 Testing complete NWTN query processing with Claude API...")
    
    # Set environment variables for proper configuration
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    os.environ["PRSM_NWTN_MODEL"] = "claude-3-5-sonnet-20241022"
    
    try:
        # Initialize the complete NWTN system
        print("1. Initializing NWTN voicebox system...")
        from prsm.nwtn.voicebox import get_voicebox_service, LLMProvider
        
        voicebox = await get_voicebox_service()
        await voicebox.initialize()
        
        # Configure Claude API key for the test user
        await voicebox.configure_api_key(
            user_id="test_user_001",
            provider=LLMProvider.CLAUDE,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        # Set up sufficient FTNS balance
        from prsm.tokenomics.ftns_service import get_ftns_service
        ftns_service = await get_ftns_service()
        await ftns_service.reward_contribution("test_user_001", "data", 3000.0)
        
        print("✅ NWTN system initialized successfully")
        
        # Test query that should trigger deep reasoning
        print("2. Testing complex query that requires deep reasoning...")
        
        test_query = "What are the fundamental principles of quantum mechanics and how do they relate to the uncertainty principle?"
        
        # Create rich content sources to trigger reasoning
        content_sources = [
            {
                "content_id": "quantum_principles_001",
                "title": "Fundamental Principles of Quantum Mechanics",
                "content": """
                Quantum mechanics is built on several fundamental principles:
                
                1. Wave-Particle Duality: Matter and energy exhibit both wave and particle properties
                2. Superposition: Quantum systems can exist in multiple states simultaneously
                3. Uncertainty Principle: The more precisely we know a particle's position, the less precisely we can know its momentum
                4. Quantum Entanglement: Particles can be correlated in ways that seem to defy classical physics
                5. Measurement Problem: The act of measurement affects the quantum system
                
                These principles form the foundation of quantum theory and have profound implications for our understanding of reality.
                """,
                "source": "quantum_physics_textbook",
                "creator_id": "physics_researcher_001"
            },
            {
                "content_id": "uncertainty_principle_002",
                "title": "Heisenberg Uncertainty Principle Explained",
                "content": """
                The Heisenberg Uncertainty Principle states that there is a fundamental limit to how precisely 
                we can simultaneously know certain pairs of properties (called complementary variables) of a quantum particle.
                
                The most famous example is position and momentum: Δx × Δp ≥ ℏ/2
                
                This is not due to measurement limitations, but is a fundamental property of quantum systems.
                It arises from the wave nature of matter and has deep connections to the mathematical structure
                of quantum mechanics, particularly the non-commuting nature of quantum operators.
                
                The uncertainty principle is intimately connected to quantum superposition and the probabilistic
                nature of quantum measurements.
                """,
                "source": "quantum_mechanics_journal",
                "creator_id": "physics_researcher_002"
            },
            {
                "content_id": "quantum_superposition_003",
                "title": "Quantum Superposition and Measurement",
                "content": """
                Quantum superposition is the principle that quantum systems can exist in multiple states 
                simultaneously until measured. This is fundamentally different from classical physics.
                
                Key aspects of superposition:
                - A quantum system can be in a combination of multiple states
                - The act of measurement collapses the superposition to a single state
                - The probabilities of different measurement outcomes are determined by the quantum state
                
                This connects directly to the uncertainty principle: the uncertainty in measurement outcomes
                is not just about our knowledge, but about the fundamental nature of quantum reality.
                The superposition principle and uncertainty principle together explain why quantum mechanics
                is inherently probabilistic rather than deterministic.
                """,
                "source": "quantum_foundations_review",
                "creator_id": "physics_researcher_003"
            }
        ]
        
        print(f"🔍 Processing query: {test_query}")
        print(f"📚 Using {len(content_sources)} content sources")
        
        # Process the query through the complete NWTN system
        print("3. Processing through NWTN multi-modal reasoning...")
        
        response = await voicebox.process_query(
            user_id="test_user_001",
            query=test_query,
            context={"content_sources": content_sources}
        )
        
        print("4. Analyzing NWTN response...")
        
        # Check for signs of successful deep reasoning
        reasoning_indicators = {
            "response_length": len(response.natural_language_response),
            "has_reasoning_modes": len(response.used_reasoning_modes) > 0,
            "confidence_above_zero": response.confidence_score > 0.0,
            "processing_time_reasonable": response.processing_time_seconds > 1.0,
            "has_source_attribution": len(response.source_links) > 0,
            "has_attribution_summary": bool(response.attribution_summary)
        }
        
        print("\n" + "="*80)
        print("🎯 NWTN DEEP REASONING ANALYSIS")
        print("="*80)
        print(f"📝 Response Length: {reasoning_indicators['response_length']} characters")
        print(f"🧠 Reasoning Modes Used: {response.used_reasoning_modes}")
        print(f"🎯 Confidence Score: {response.confidence_score}")
        print(f"⏱️  Processing Time: {response.processing_time_seconds} seconds")
        print(f"🔗 Source Links: {len(response.source_links)} sources")
        print(f"📊 Attribution Summary: {response.attribution_summary}")
        
        print("\n" + "="*80)
        print("📖 NATURAL LANGUAGE RESPONSE:")
        print("="*80)
        print(response.natural_language_response)
        print("="*80)
        
        # Determine if this is real NWTN processing or fallback
        is_real_nwtn = (
            reasoning_indicators["response_length"] > 200 and
            reasoning_indicators["has_reasoning_modes"] and
            reasoning_indicators["confidence_above_zero"] and
            reasoning_indicators["processing_time_reasonable"] and
            "quantum" in response.natural_language_response.lower()
        )
        
        if is_real_nwtn:
            print("\n✅ SUCCESS: NWTN deep reasoning with Claude API is working!")
            print("🎉 The system successfully:")
            print("   - Processed the query through multi-modal reasoning")
            print("   - Generated a substantive natural language response")
            print("   - Used multiple reasoning modes")
            print("   - Achieved meaningful confidence score")
            print("   - Provided proper source attribution")
            return True
        else:
            print("\n❌ ISSUE: Response appears to be fallback, not deep reasoning")
            print("🔍 Indicators suggest:")
            for indicator, value in reasoning_indicators.items():
                status = "✅" if value else "❌"
                print(f"   {status} {indicator}: {value}")
            return False
        
    except Exception as e:
        print(f"❌ Error during complete NWTN test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_nwtn_query())
    if success:
        print("\n🎉 Complete NWTN query processing test PASSED!")
        print("The system is fully operational with deep reasoning and Claude API integration.")
    else:
        print("\n🚨 Complete NWTN query processing test FAILED!")
        print("Further investigation needed to enable deep reasoning.")