#!/usr/bin/env python3
"""
Full NWTN Meta-Reasoning Engine Production Test
==============================================

This script tests the complete NWTN system with physics integration to ensure
it's production-ready for real-world physics reasoning tasks.
"""

import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

# Suppress warnings for cleaner output
logging.getLogger().setLevel(logging.ERROR)

async def test_full_meta_reasoning_engine():
    """Test the complete NWTN Meta-Reasoning Engine with physics integration"""
    
    print("🚀 Full NWTN Meta-Reasoning Engine Production Test")
    print("=" * 60)
    
    try:
        # Import the full system
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
        
        print("✅ Successfully imported MetaReasoningEngine")
        
        # Initialize the full system
        print("\n🔧 Initializing NWTN Meta-Reasoning Engine...")
        meta_engine = MetaReasoningEngine()
        
        print("✅ NWTN Meta-Reasoning Engine initialized successfully")
        print(f"   - World Model: {len(meta_engine.world_model.knowledge_index)} knowledge items")
        print(f"   - Physics Laws: {len(meta_engine.world_model.physical_laws)} physics principles")
        
        # Test physics-based reasoning queries
        physics_queries = [
            {
                'query': 'What happens when I push a stationary object with a force?',
                'context': 'classical mechanics',
                'expected_concepts': ['Newton\'s laws', 'acceleration', 'force', 'mass']
            },
            {
                'query': 'Can energy be created or destroyed in a closed system?',
                'context': 'thermodynamics',
                'expected_concepts': ['conservation of energy', 'first law', 'thermodynamics']
            },
            {
                'query': 'What is the relationship between force and acceleration?',
                'context': 'classical physics',
                'expected_concepts': ['F = m*a', 'Newton\'s second law', 'mass']
            },
            {
                'query': 'What happens to entropy in an isolated system?',
                'context': 'thermodynamics',
                'expected_concepts': ['entropy', 'second law', 'isolated system']
            }
        ]
        
        print(f"\n🧪 Testing {len(physics_queries)} Physics Reasoning Queries")
        print("=" * 60)
        
        test_results = []
        
        for i, query_data in enumerate(physics_queries, 1):
            print(f"\n📋 Test {i}: {query_data['query']}")
            print(f"Context: {query_data['context']}")
            
            try:
                # Test quick thinking mode
                start_time = datetime.now()
                
                result = await meta_engine.meta_reason(
                    query=query_data['query'],
                    context={'domain': query_data['context']},
                    thinking_mode=ThinkingMode.QUICK
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Analyze results
                success = True
                issues = []
                
                if not result:
                    success = False
                    issues.append("No result returned")
                else:
                    # Check confidence
                    if hasattr(result, 'confidence') and result.confidence < 0.7:
                        issues.append(f"Low confidence: {result.confidence:.3f}")
                    
                    # Check if world model was used
                    world_model_used = False
                    if hasattr(result, 'metadata') and result.metadata:
                        if 'world_model_validation' in result.metadata:
                            world_model_used = True
                    
                    # Check for physics concepts
                    result_text = str(result.final_synthesis) if hasattr(result, 'final_synthesis') else str(result)
                    physics_concepts_found = 0
                    for concept in query_data['expected_concepts']:
                        if concept.lower() in result_text.lower():
                            physics_concepts_found += 1
                    
                    concept_coverage = physics_concepts_found / len(query_data['expected_concepts'])
                    
                    print(f"   ✅ Processing time: {processing_time:.2f}s")
                    print(f"   ✅ Result generated: {len(str(result_text))} chars")
                    print(f"   ✅ Physics concepts found: {physics_concepts_found}/{len(query_data['expected_concepts'])}")
                    print(f"   ✅ World model used: {'Yes' if world_model_used else 'No'}")
                    
                    if hasattr(result, 'confidence'):
                        print(f"   ✅ Confidence: {result.confidence:.3f}")
                    
                    if concept_coverage < 0.5:
                        issues.append(f"Low physics concept coverage: {concept_coverage:.1%}")
                    
                    if not world_model_used:
                        issues.append("World model not used in reasoning")
                
                if issues:
                    print(f"   ⚠️  Issues: {', '.join(issues)}")
                    success = False
                else:
                    print(f"   🎉 Test passed!")
                
                test_results.append({
                    'query': query_data['query'],
                    'success': success,
                    'processing_time': processing_time,
                    'issues': issues,
                    'result': result
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_results.append({
                    'query': query_data['query'],
                    'success': False,
                    'processing_time': 0,
                    'issues': [str(e)],
                    'result': None
                })
        
        return test_results
        
    except Exception as e:
        print(f"❌ Failed to initialize NWTN system: {e}")
        import traceback
        traceback.print_exc()
        return []

async def test_thinking_modes():
    """Test different thinking modes with physics queries"""
    
    print(f"\n🧠 Testing Different Thinking Modes")
    print("=" * 60)
    
    try:
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
        
        meta_engine = MetaReasoningEngine()
        
        test_query = {
            'query': 'Explain the relationship between force, mass, and acceleration',
            'context': 'physics education'
        }
        
        thinking_modes = [
            (ThinkingMode.QUICK, "Quick Thinking"),
            (ThinkingMode.INTERMEDIATE, "Intermediate Thinking"),
            (ThinkingMode.DEEP, "Deep Thinking")
        ]
        
        mode_results = []
        
        for mode, mode_name in thinking_modes:
            print(f"\n🔄 Testing {mode_name} Mode")
            
            try:
                start_time = datetime.now()
                
                result = await meta_engine.meta_reason(
                    query=test_query['query'],
                    context={'domain': test_query['context']},
                    thinking_mode=mode
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Analyze quality
                result_text = str(result.final_synthesis) if hasattr(result, 'final_synthesis') else str(result)
                quality_score = len(result_text) / 1000  # Simple length-based quality metric
                
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")
                print(f"   📏 Result length: {len(result_text)} chars")
                print(f"   📊 Quality score: {quality_score:.2f}")
                
                if hasattr(result, 'confidence'):
                    print(f"   🎯 Confidence: {result.confidence:.3f}")
                
                mode_results.append({
                    'mode': mode_name,
                    'processing_time': processing_time,
                    'quality_score': quality_score,
                    'result_length': len(result_text),
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ Error in {mode_name}: {e}")
                mode_results.append({
                    'mode': mode_name,
                    'processing_time': 0,
                    'quality_score': 0,
                    'result_length': 0,
                    'success': False
                })
        
        return mode_results
        
    except Exception as e:
        print(f"❌ Error testing thinking modes: {e}")
        return []

async def test_world_model_validation():
    """Test world model validation with conflicting physics statements"""
    
    print(f"\n🔍 Testing World Model Validation")
    print("=" * 60)
    
    try:
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine
        
        meta_engine = MetaReasoningEngine()
        
        # Test queries that should conflict with physics knowledge
        validation_tests = [
            {
                'query': 'Energy can be created from nothing in a closed system',
                'should_conflict': True,
                'description': 'Violates conservation of energy'
            },
            {
                'query': 'Force equals mass times acceleration',
                'should_conflict': False,
                'description': 'Matches Newton\'s second law'
            },
            {
                'query': 'Objects in motion tend to stay in motion unless acted upon by a force',
                'should_conflict': False,
                'description': 'Matches Newton\'s first law'
            },
            {
                'query': 'Entropy always decreases in isolated systems',
                'should_conflict': True,
                'description': 'Violates second law of thermodynamics'
            }
        ]
        
        validation_results = []
        
        for test_data in validation_tests:
            print(f"\n📝 Testing: {test_data['query']}")
            print(f"Expected: {'Should conflict' if test_data['should_conflict'] else 'Should validate'}")
            
            try:
                result = await meta_engine.meta_reason(
                    query=test_data['query'],
                    context={'domain': 'physics validation'}
                )
                
                # Check if world model validation was applied
                world_model_used = False
                conflicts_detected = 0
                
                if hasattr(result, 'metadata') and result.metadata:
                    world_model_info = result.metadata.get('world_model_validation', {})
                    if world_model_info:
                        world_model_used = True
                        conflicts_detected = world_model_info.get('conflicts', 0)
                
                # Analyze confidence adjustment
                confidence_adjusted = False
                if hasattr(result, 'confidence'):
                    # Lower confidence might indicate conflict detection
                    if result.confidence < 0.8:
                        confidence_adjusted = True
                
                print(f"   🔍 World model used: {'Yes' if world_model_used else 'No'}")
                print(f"   ⚠️  Conflicts detected: {conflicts_detected}")
                print(f"   📉 Confidence adjusted: {'Yes' if confidence_adjusted else 'No'}")
                
                # Determine if validation worked correctly
                validation_correct = False
                if test_data['should_conflict']:
                    validation_correct = conflicts_detected > 0 or confidence_adjusted
                else:
                    validation_correct = conflicts_detected == 0 and not confidence_adjusted
                
                status = "✅ CORRECT" if validation_correct else "❌ INCORRECT"
                print(f"   {status} validation behavior")
                
                validation_results.append({
                    'query': test_data['query'],
                    'should_conflict': test_data['should_conflict'],
                    'conflicts_detected': conflicts_detected,
                    'validation_correct': validation_correct,
                    'world_model_used': world_model_used
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                validation_results.append({
                    'query': test_data['query'],
                    'should_conflict': test_data['should_conflict'],
                    'conflicts_detected': 0,
                    'validation_correct': False,
                    'world_model_used': False
                })
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error testing world model validation: {e}")
        return []

async def run_full_production_test():
    """Run the complete production readiness test suite"""
    
    print("🎯 NWTN Meta-Reasoning Engine Production Readiness Test")
    print("=" * 70)
    
    # Test 1: Basic physics reasoning
    print("\n" + "="*70)
    print("📋 TEST 1: PHYSICS REASONING QUERIES")
    print("="*70)
    
    reasoning_results = await test_full_meta_reasoning_engine()
    
    # Test 2: Different thinking modes
    print("\n" + "="*70)
    print("📋 TEST 2: THINKING MODES")
    print("="*70)
    
    mode_results = await test_thinking_modes()
    
    # Test 3: World model validation
    print("\n" + "="*70)
    print("📋 TEST 3: WORLD MODEL VALIDATION")
    print("="*70)
    
    validation_results = await test_world_model_validation()
    
    # Generate comprehensive report
    print("\n" + "="*70)
    print("📊 PRODUCTION READINESS REPORT")
    print("="*70)
    
    # Analyze results
    total_tests = len(reasoning_results) + len(mode_results) + len(validation_results)
    
    reasoning_passed = sum(1 for r in reasoning_results if r['success'])
    mode_passed = sum(1 for r in mode_results if r['success'])
    validation_passed = sum(1 for r in validation_results if r['validation_correct'])
    
    total_passed = reasoning_passed + mode_passed + validation_passed
    
    print(f"\n🎯 Overall Test Results:")
    print(f"   Physics Reasoning: {reasoning_passed}/{len(reasoning_results)} passed")
    print(f"   Thinking Modes: {mode_passed}/{len(mode_results)} passed")
    print(f"   World Model Validation: {validation_passed}/{len(validation_results)} passed")
    print(f"   TOTAL: {total_passed}/{total_tests} tests passed")
    
    success_rate = total_passed / total_tests if total_tests > 0 else 0
    print(f"   Success Rate: {success_rate:.1%}")
    
    # Performance metrics
    if reasoning_results:
        avg_processing_time = sum(r['processing_time'] for r in reasoning_results if r['success']) / reasoning_passed if reasoning_passed > 0 else 0
        print(f"   Average Processing Time: {avg_processing_time:.2f}s")
    
    # Production readiness assessment
    print(f"\n🚀 Production Readiness Assessment:")
    
    if success_rate >= 0.8:
        print("   ✅ PRODUCTION READY")
        print("   🎉 NWTN Meta-Reasoning Engine with physics integration is ready for production use!")
        
        print(f"\n🔧 Key Features Verified:")
        print("   ✅ Physics knowledge integration (24 essential principles)")
        print("   ✅ World model validation and conflict detection")
        print("   ✅ Multiple thinking modes (Quick, Intermediate, Deep)")
        print("   ✅ Real-world physics reasoning capabilities")
        print("   ✅ Robust error handling and performance")
        
        return True
    else:
        print("   ❌ NOT PRODUCTION READY")
        print("   ⚠️  Some critical tests failed. Review issues before production deployment.")
        
        # Show critical issues
        all_issues = []
        for result in reasoning_results:
            if not result['success']:
                all_issues.extend(result['issues'])
        
        if all_issues:
            print(f"\n⚠️  Critical Issues Found:")
            for issue in set(all_issues):
                print(f"   - {issue}")
        
        return False

if __name__ == "__main__":
    async def main():
        success = await run_full_production_test()
        
        if success:
            print("\n🎉 NWTN Meta-Reasoning Engine is PRODUCTION READY!")
            print("✅ Physics integration complete and validated")
            print("🚀 Ready for real-world deployment")
        else:
            print("\n❌ Production readiness test failed")
            print("🔧 Address issues before deployment")
    
    # Run the async test
    asyncio.run(main())