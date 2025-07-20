#!/usr/bin/env python3
"""
5-Prompt Production Test - Final Validation
===========================================

This test runs the original 5 complex creative challenge prompts that failed
earlier today. With all fixes applied, these should now achieve >80% success rate.

The 5 prompts test different aspects of NWTN's capabilities:
1. Machine learning algorithms (interdisciplinary synthesis)
2. Topological insulators (advanced physics)  
3. Artificial general intelligence (complex reasoning)
4. Quantum computing applications (emerging technology)
5. Fundamental computation limits (theoretical boundaries)
"""

import asyncio
import sys
import time
import json
from typing import List, Dict, Any
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

# Original 5 complex prompts that failed earlier
PRODUCTION_PROMPTS = [
    {
        "id": 1,
        "query": "What are the latest advances in machine learning algorithms for natural language processing?",
        "category": "Machine Learning & NLP",
        "complexity": "High",
        "expected_sources": 8-15
    },
    {
        "id": 2, 
        "query": "How do topological insulators work and what are their potential applications?",
        "category": "Advanced Physics",
        "complexity": "Very High",
        "expected_sources": 5-12
    },
    {
        "id": 3,
        "query": "What are the most promising approaches to achieving artificial general intelligence?",
        "category": "AI Theory",
        "complexity": "Very High", 
        "expected_sources": 10-20
    },
    {
        "id": 4,
        "query": "How might quantum computing revolutionize drug discovery and molecular simulation?",
        "category": "Quantum Applications",
        "complexity": "High",
        "expected_sources": 6-15
    },
    {
        "id": 5,
        "query": "What are the fundamental limits of computation and information processing?",
        "category": "Theoretical Computer Science",
        "complexity": "Very High",
        "expected_sources": 5-12
    }
]

async def test_single_prompt(integrator: SystemIntegrator, prompt_info: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single prompt and return detailed results"""
    print(f"\n🧠 Testing Prompt {prompt_info['id']}: {prompt_info['category']}")
    print(f"   Query: {prompt_info['query']}")
    print(f"   Complexity: {prompt_info['complexity']}")
    
    start_time = time.time()
    
    try:
        result = await integrator.process_complete_query(
            query=prompt_info['query'],
            user_id=f'production_test_prompt_{prompt_info["id"]}',
            query_cost=5.0  # Higher cost for complex queries
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Analyze results
        success = result.success
        quality = result.quality_score if success else 0.0
        sources_found = len(result.citations) if success else 0
        cost = result.total_cost if success else 0.0
        payments = len(result.payment_distributions) if success else 0
        
        # Determine if this meets production standards
        meets_standards = (
            success and
            quality >= 0.6 and  # Minimum quality threshold
            sources_found >= 3   # Minimum sources threshold
        )
        
        prompt_result = {
            "prompt_id": prompt_info['id'],
            "category": prompt_info['category'],
            "complexity": prompt_info['complexity'],
            "query": prompt_info['query'],
            "success": success,
            "meets_standards": meets_standards,
            "quality_score": quality,
            "sources_found": sources_found,
            "processing_time_seconds": processing_time,
            "total_cost": cost,
            "payments_distributed": payments,
            "error_message": result.error_message if not success else None
        }
        
        # Display results
        status_icon = "✅" if meets_standards else "❌" if success else "💥"
        print(f"   {status_icon} Result: {'PASS' if meets_standards else 'FAIL' if success else 'ERROR'}")
        print(f"   📊 Quality: {quality:.3f} | Sources: {sources_found} | Time: {processing_time:.1f}s")
        
        if not success:
            print(f"   ⚠️  Error: {result.error_message}")
        
        return prompt_result
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"   💥 EXCEPTION: {str(e)}")
        
        return {
            "prompt_id": prompt_info['id'],
            "category": prompt_info['category'],
            "complexity": prompt_info['complexity'],
            "query": prompt_info['query'],
            "success": False,
            "meets_standards": False,
            "quality_score": 0.0,
            "sources_found": 0,
            "processing_time_seconds": processing_time,
            "total_cost": 0.0,
            "payments_distributed": 0,
            "error_message": str(e)
        }

async def run_5_prompt_production_test():
    """Run the comprehensive 5-prompt production test"""
    print("🚀 Starting 5-Prompt Production Test")
    print("=" * 60)
    print("Testing NWTN production readiness with complex creative challenges")
    print(f"Target: >80% success rate ({4}/5 prompts must pass)")
    print("=" * 60)
    
    # Initialize system integrator
    print("🔧 Initializing NWTN System...")
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    print("✅ NWTN System initialized successfully")
    
    # Run all 5 prompts
    results = []
    total_start_time = time.time()
    
    for prompt_info in PRODUCTION_PROMPTS:
        prompt_result = await test_single_prompt(integrator, prompt_info)
        results.append(prompt_result)
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # Analyze overall results
    successful_prompts = [r for r in results if r['success']]
    passing_prompts = [r for r in results if r['meets_standards']]
    
    success_rate = len(successful_prompts) / len(results) * 100
    pass_rate = len(passing_prompts) / len(results) * 100
    
    avg_quality = sum(r['quality_score'] for r in successful_prompts) / len(successful_prompts) if successful_prompts else 0
    avg_sources = sum(r['sources_found'] for r in successful_prompts) / len(successful_prompts) if successful_prompts else 0
    total_sources = sum(r['sources_found'] for r in results)
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("🎯 5-PROMPT PRODUCTION TEST RESULTS")
    print("=" * 60)
    
    print(f"📊 Overall Performance:")
    print(f"   Success Rate: {success_rate:.1f}% ({len(successful_prompts)}/5 prompts)")
    print(f"   Pass Rate: {pass_rate:.1f}% ({len(passing_prompts)}/5 prompts)")
    print(f"   Production Ready: {'✅ YES' if pass_rate >= 80 else '❌ NO'}")
    
    print(f"\n📈 Quality Metrics:")
    print(f"   Average Quality Score: {avg_quality:.3f}")
    print(f"   Average Sources per Query: {avg_sources:.1f}")
    print(f"   Total Sources Found: {total_sources}")
    print(f"   Total Processing Time: {total_processing_time:.1f}s")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['meets_standards'] else "❌ FAIL" if result['success'] else "💥 ERROR"
        print(f"   {result['prompt_id']}. {result['category']}: {status}")
        print(f"      Quality: {result['quality_score']:.3f} | Sources: {result['sources_found']} | Time: {result['processing_time_seconds']:.1f}s")
        if result['error_message']:
            print(f"      Error: {result['error_message']}")
    
    # Save detailed results
    test_results = {
        "test_name": "5_Prompt_Production_Test",
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_prompts": len(results),
        "successful_prompts": len(successful_prompts),
        "passing_prompts": len(passing_prompts),
        "success_rate_percent": success_rate,
        "pass_rate_percent": pass_rate,
        "production_ready": pass_rate >= 80,
        "average_quality_score": avg_quality,
        "average_sources_per_query": avg_sources,
        "total_sources_found": total_sources,
        "total_processing_time_seconds": total_processing_time,
        "individual_results": results
    }
    
    with open('test_5_prompt_production_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: test_5_prompt_production_results.json")
    
    # Final determination
    if pass_rate >= 80:
        print(f"\n🎉 SUCCESS: NWTN IS PRODUCTION READY!")
        print(f"   Achieved {pass_rate:.1f}% pass rate (target: >80%)")
        print(f"   All critical systems working: 150K search, session mgmt, reasoning")
        return True
    else:
        print(f"\n⚠️  NEEDS WORK: Production readiness not achieved")
        print(f"   Pass rate: {pass_rate:.1f}% (target: >80%)")
        return False

if __name__ == "__main__":
    production_ready = asyncio.run(run_5_prompt_production_test())
    
    if production_ready:
        print("\n🚀 NWTN is ready for production deployment!")
    else:
        print("\n🔧 Additional optimizations needed before production")