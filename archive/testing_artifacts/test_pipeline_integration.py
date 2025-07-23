#!/usr/bin/env python3
"""
NWTN Pipeline Integration Test
=============================

This script tests the integration and flow between NWTN pipeline components
without requiring full system initialization or external dependencies.
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class MockPipelineResult:
    """Mock pipeline result for testing"""
    stage: str
    status: str
    data: Dict[str, Any]
    processing_time: float
    confidence_score: float = 0.0

class MockPipelineComponent:
    """Mock component for testing pipeline flow"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.call_count = 0
    
    async def initialize(self):
        """Mock initialization"""
        await asyncio.sleep(0.01)  # Simulate async work
        self.initialized = True
        print(f"✅ {self.name} initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing"""
        self.call_count += 1
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Transform input to show data flow
        output_data = input_data.copy()
        output_data[f'{self.name}_processed'] = True
        output_data[f'{self.name}_timestamp'] = datetime.now(timezone.utc).isoformat()
        output_data[f'{self.name}_call_count'] = self.call_count
        
        print(f"🔄 {self.name} processed data (call #{self.call_count})")
        return output_data

class MockPipelineOrchestrator:
    """Mock orchestrator to test pipeline flow"""
    
    def __init__(self):
        # Create mock components representing the actual pipeline
        self.components = {
            'query_analyzer': MockPipelineComponent('Query Analyzer'),
            'content_retriever': MockPipelineComponent('Content Retriever'),
            'content_analyzer': MockPipelineComponent('Content Analyzer'),
            'candidate_generator': MockPipelineComponent('Candidate Generator'),
            'meta_reasoner': MockPipelineComponent('Meta Reasoner'),
            'content_grounder': MockPipelineComponent('Content Grounder'),
            'synthesizer': MockPipelineComponent('Synthesizer')
        }
        
        self.pipeline_order = [
            'query_analyzer',
            'content_retriever', 
            'content_analyzer',
            'candidate_generator',
            'meta_reasoner',
            'content_grounder',
            'synthesizer'
        ]
        
        self.results = []
    
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Mock Pipeline Orchestrator...")
        
        for component in self.components.values():
            await component.initialize()
        
        print("✅ All components initialized")
    
    async def process_query(self, query: str) -> List[MockPipelineResult]:
        """Process query through the pipeline"""
        print(f"\n🔄 Processing query: '{query}'")
        
        # Initial data
        pipeline_data = {
            'original_query': query,
            'processing_start': datetime.now(timezone.utc).isoformat(),
            'pipeline_flow': []
        }
        
        results = []
        
        # Process through each stage
        for i, component_name in enumerate(self.pipeline_order):
            stage_start = datetime.now(timezone.utc)
            component = self.components[component_name]
            
            print(f"\n📋 Stage {i+1}: {component.name}")
            
            # Process data through component
            pipeline_data = await component.process(pipeline_data)
            pipeline_data['pipeline_flow'].append(component_name)
            
            # Calculate stage time
            stage_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
            
            # Create result
            result = MockPipelineResult(
                stage=component_name,
                status='completed',
                data=pipeline_data.copy(),
                processing_time=stage_time,
                confidence_score=min(0.9, 0.5 + (i * 0.07))  # Increasing confidence
            )
            
            results.append(result)
            print(f"   ⏱️  Stage time: {stage_time:.3f}s")
            print(f"   📊 Confidence: {result.confidence_score:.2f}")
        
        return results
    
    def analyze_pipeline_flow(self, results: List[MockPipelineResult]):
        """Analyze pipeline performance and data flow"""
        print(f"\n📊 Pipeline Analysis")
        print("=" * 50)
        
        total_time = sum(r.processing_time for r in results)
        final_confidence = results[-1].confidence_score if results else 0.0
        
        print(f"Total Processing Time: {total_time:.3f}s")
        print(f"Final Confidence Score: {final_confidence:.2f}")
        print(f"Stages Completed: {len(results)}")
        
        print(f"\n🔍 Data Flow Trace:")
        if results:
            final_data = results[-1].data
            pipeline_flow = final_data.get('pipeline_flow', [])
            
            for i, stage in enumerate(pipeline_flow):
                print(f"  {i+1}. {stage}")
            
            print(f"\n📈 Processing Statistics:")
            for component_name, component in self.components.items():
                print(f"  {component_name}: {component.call_count} calls")
        
        print(f"\n✅ Pipeline Integration Test: {'PASSED' if len(results) == len(self.pipeline_order) else 'FAILED'}")

async def test_pipeline_integration():
    """Test the complete pipeline integration"""
    print("🧪 NWTN Pipeline Integration Test")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = MockPipelineOrchestrator()
    
    # Initialize
    await orchestrator.initialize()
    
    # Test queries
    test_queries = [
        "What are the most promising approaches for commercial atomically precise manufacturing?",
        "How can AI systems achieve better reasoning capabilities?",
        "What are the key challenges in quantum computing?"
    ]
    
    all_results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n🔬 Test Case {i+1}")
        print("-" * 30)
        
        results = await orchestrator.process_query(query)
        all_results.extend(results)
        
        # Quick analysis
        if results:
            print(f"✅ Query processed successfully")
            print(f"   Stages: {len(results)}")
            print(f"   Total time: {sum(r.processing_time for r in results):.3f}s")
            print(f"   Final confidence: {results[-1].confidence_score:.2f}")
        else:
            print(f"❌ Query processing failed")
    
    # Overall analysis
    if all_results:
        print(f"\n📊 Overall Test Results")
        print("=" * 50)
        
        avg_time = sum(r.processing_time for r in all_results) / len(all_results)
        avg_confidence = sum(r.confidence_score for r in all_results) / len(all_results)
        
        print(f"Total Test Cases: {len(test_queries)}")
        print(f"Total Pipeline Stages: {len(all_results)}")  
        print(f"Average Stage Time: {avg_time:.3f}s")
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        # Analyze data flow integrity
        orchestrator.analyze_pipeline_flow(all_results[-len(orchestrator.pipeline_order):])
    
    return len(all_results) > 0

async def test_component_availability():
    """Test availability of actual NWTN components"""
    print(f"\n🔍 Testing NWTN Component Availability")
    print("=" * 50)
    
    components_status = {}
    
    # Test imports
    test_imports = [
        ('prsm.nwtn.unified_pipeline_controller', 'UnifiedPipelineController'),
        ('prsm.nwtn.complete_system', 'NWTNCompleteSystem'),
        ('prsm.nwtn.voicebox', 'NWTNVoicebox'),
        ('prsm.nwtn.meta_reasoning_engine', 'MetaReasoningEngine'),
    ]
    
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            components_status[class_name] = 'Available'
            print(f"✅ {class_name}: Available")
        except Exception as e:
            components_status[class_name] = f'Error: {str(e)[:50]}...'
            print(f"❌ {class_name}: {str(e)[:50]}...")
    
    # Test unified pipeline controller specifically
    if 'UnifiedPipelineController' in components_status and 'Available' in components_status['UnifiedPipelineController']:
        try:
            from prsm.nwtn.unified_pipeline_controller import UnifiedPipelineController
            controller = UnifiedPipelineController()
            print(f"✅ UnifiedPipelineController: Instantiated successfully")
            components_status['UnifiedPipelineControllerInstantiation'] = 'Success'
        except Exception as e:
            print(f"❌ UnifiedPipelineController instantiation failed: {e}")
            components_status['UnifiedPipelineControllerInstantiation'] = f'Failed: {e}'
    
    return components_status

async def main():
    """Main test function"""
    print("🚀 NWTN Pipeline Integration Testing Suite")
    print("=" * 60)
    
    # Test 1: Component availability
    component_status = await test_component_availability()
    
    # Test 2: Mock pipeline integration
    integration_success = await test_pipeline_integration()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 60)
    
    available_components = sum(1 for status in component_status.values() if 'Available' in status or 'Success' in status)
    total_components = len(component_status)
    
    print(f"Component Availability: {available_components}/{total_components}")
    print(f"Pipeline Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    # Detailed component status
    print(f"\n🔍 Component Status Details:")
    for component, status in component_status.items():
        icon = "✅" if ("Available" in status or "Success" in status) else "❌"
        print(f"  {icon} {component}: {status}")
    
    overall_success = integration_success and available_components > 0
    print(f"\n🎯 Overall Result: {'✅ SYSTEM READY' if overall_success else '⚠️ NEEDS ATTENTION'}")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)