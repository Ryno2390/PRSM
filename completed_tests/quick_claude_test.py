#!/usr/bin/env python3
"""
Quick Claude API Integration Test
==============================

Tests NWTN Voicebox with Claude API integration using a simple query.
"""

import asyncio
import os
from prsm.nwtn.voicebox import NWTNVoicebox

async def test_claude_integration():
    """Test Claude API integration with a simple query"""
    print("🎯 NWTN CLAUDE API INTEGRATION TEST")
    print("=" * 50)
    
    # Initialize voicebox
    voicebox = NWTNVoicebox()
    await voicebox.initialize()
    
    print(f"✅ Voicebox initialized")
    print(f"🔧 Default API config: {voicebox.default_api_config is not None}")
    
    if voicebox.default_api_config:
        print(f"🤖 Provider: {voicebox.default_api_config.provider}")
        print(f"📡 Model: {voicebox.default_api_config.model_name}")
    
    # Simple test query
    query = "What is quantum computing?"
    
    print(f"\n🧠 Testing query: {query}")
    print("-" * 30)
    
    try:
        # Test translation directly (bypassing full NWTN processing for speed)
        from prsm.nwtn.multi_modal_reasoning_engine import IntegratedReasoningResult
        
        # Create mock reasoning result
        mock_result = IntegratedReasoningResult(
            query=query,
            components=[],
            reasoning_results=[],
            integrated_conclusion="Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.",
            overall_confidence=0.8,
            reasoning_consensus=0.9,
            cross_validation_score=0.85,
            reasoning_path=["deductive", "analogical"],
            multi_modal_evidence=["Quantum bits (qubits) can exist in multiple states simultaneously", "Quantum algorithms can solve certain problems exponentially faster"],
            identified_uncertainties=["Hardware limitations", "Error rates"],
            reasoning_completeness=0.9,
            logical_consistency=0.95,
            empirical_grounding=0.8
        )
        
        # Test natural language translation
        from prsm.nwtn.voicebox import QueryAnalysis, QueryComplexity, ClarificationStatus
        
        # Create mock analysis
        mock_analysis = QueryAnalysis(
            query_id="test_query",
            original_query=query,
            complexity=QueryComplexity.SIMPLE,
            estimated_reasoning_modes=["deductive", "analogical"],
            domain_hints=["physics"],
            clarification_status=ClarificationStatus.CLEAR,
            clarification_questions=[],
            estimated_cost_ftns=10.0,
            analysis_confidence=0.8,
            requires_breakthrough_mode=False
        )
        
        natural_response = await voicebox._translate_to_natural_language(
            user_id="test_user",
            original_query=query,
            reasoning_result=mock_result,
            analysis=mock_analysis
        )
        
        print("✨ NATURAL LANGUAGE RESPONSE:")
        print("=" * 40)
        print(natural_response)
        print("=" * 40)
        
        # Check if it's using Claude API or fallback
        if "Based on NWTN's multi-modal reasoning analysis" in natural_response and "confidence:" in natural_response:
            print("📋 Using fallback structured response")
        else:
            print("🎉 Using Claude API natural language translation!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_claude_integration())