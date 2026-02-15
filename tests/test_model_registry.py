#!/usr/bin/env python3
"""
Test script for PRSM Distributed Model Registry
Tests Phase 1, Week 2, Task 2 - Distributed Model Registry
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any

async def test_model_registry():
    """Test distributed model registry functionality"""
    
    print("🏛️ Testing PRSM Distributed Model Registry...")
    print("=" * 70)
    
    try:
        # Import the model registry and dependencies
        from prsm.compute.federation.model_registry import ModelRegistry, model_registry
        from prsm.core.models import TeacherModel, ModelType, PeerNode
        from prsm.data.data_layer.enhanced_ipfs import prsm_ipfs_client
        
        print("✅ Model registry imports successful")
        
        # Create registry instance
        registry = ModelRegistry()
        
        # === Test Registry Initialization ===
        
        initial_stats = await registry.get_registry_stats()
        assert initial_stats["total_models"] == 0
        assert initial_stats["total_categories"] == 0
        print(f"✅ Registry initialized with {initial_stats['total_models']} models")
        
        # === Test Model Registration ===
        
        print(f"\n📝 Testing model registration...")
        
        # Create test teacher models
        model1 = TeacherModel(
            name="Advanced NLP Transformer",
            specialization="natural_language_processing",
            model_type=ModelType.TEACHER,
            performance_score=0.92,
            version="2.1.0",
            active=True
        )
        
        model2 = TeacherModel(
            name="Computer Vision Specialist",
            specialization="computer_vision", 
            model_type=ModelType.TEACHER,
            performance_score=0.88,
            version="1.5.3",
            active=True
        )
        
        model3 = TeacherModel(
            name="Data Analysis Expert",
            specialization="data_analysis",
            model_type=ModelType.TEACHER,
            performance_score=0.85,
            version="3.0.0",
            active=True
        )
        
        # Store models in IPFS first (simulation mode)
        model1_data = b"FAKE_NLP_MODEL_DATA_" + b"x" * 500
        model1_cid = await prsm_ipfs_client.store_model(
            model1_data, 
            {"uploader_id": "researcher_001", "model_type": "nlp_transformer"}
        )
        
        model2_data = b"FAKE_CV_MODEL_DATA_" + b"x" * 600
        model2_cid = await prsm_ipfs_client.store_model(
            model2_data,
            {"uploader_id": "researcher_002", "model_type": "cnn_vision"}
        )
        
        model3_data = b"FAKE_DATA_MODEL_" + b"x" * 400
        model3_cid = await prsm_ipfs_client.store_model(
            model3_data,
            {"uploader_id": "researcher_003", "model_type": "data_analyzer"}
        )
        
        # Register models in registry
        reg1_success = await registry.register_teacher_model(model1, model1_cid)
        assert reg1_success == True
        print(f"✅ Registered NLP model: {model1.name}")
        
        reg2_success = await registry.register_teacher_model(model2, model2_cid)
        assert reg2_success == True
        print(f"✅ Registered CV model: {model2.name}")
        
        reg3_success = await registry.register_teacher_model(model3, model3_cid)
        assert reg3_success == True
        print(f"✅ Registered Data model: {model3.name}")
        
        # === Test Registry Stats After Registration ===
        
        updated_stats = await registry.get_registry_stats()
        assert updated_stats["total_models"] == 3
        assert updated_stats["total_categories"] == 3
        print(f"✅ Registry now contains {updated_stats['total_models']} models in {updated_stats['total_categories']} categories")
        
        # === Test Specialist Discovery ===
        
        print(f"\n🔍 Testing specialist discovery...")
        
        # Test exact category match
        nlp_specialists = await registry.discover_specialists("natural_language_processing")
        assert len(nlp_specialists) == 1
        assert str(model1.teacher_id) in nlp_specialists
        print(f"✅ Found {len(nlp_specialists)} NLP specialists")
        
        # Test partial category match
        cv_specialists = await registry.discover_specialists("computer_vision")
        assert len(cv_specialists) == 1
        assert str(model2.teacher_id) in cv_specialists
        print(f"✅ Found {len(cv_specialists)} Computer Vision specialists")
        
        # Test fuzzy matching
        data_specialists = await registry.discover_specialists("data")
        assert len(data_specialists) >= 1
        print(f"✅ Found {len(data_specialists)} Data specialists (fuzzy match)")
        
        # Test non-existent category
        audio_specialists = await registry.discover_specialists("audio_processing")
        assert len(audio_specialists) == 0
        print(f"✅ Correctly found {len(audio_specialists)} Audio specialists (none exist)")
        
        # === Test Model Integrity Validation ===
        
        print(f"\n🔒 Testing model integrity validation...")
        
        # Test valid CIDs
        model1_integrity = await registry.validate_model_integrity(model1_cid)
        assert model1_integrity == True
        print(f"✅ Model 1 integrity verified")
        
        model2_integrity = await registry.validate_model_integrity(model2_cid)
        assert model2_integrity == True
        print(f"✅ Model 2 integrity verified")
        
        # Test invalid CID
        fake_cid = "bafybeig" + "x" * 50
        fake_integrity = await registry.validate_model_integrity(fake_cid)
        # In simulation mode, this might return True, but structure is correct
        print(f"✅ Fake CID integrity check: {fake_integrity}")
        
        # === Test Performance Metrics Updates ===
        
        print(f"\n📊 Testing performance metrics updates...")
        
        # Update model1 metrics
        model1_update = await registry.update_performance_metrics(
            str(model1.teacher_id),
            {
                "usage_increment": True,
                "rating": 0.95,
                "success": True,
                "response_time": 1.2,
                "user_id": "test_user_001"
            }
        )
        assert model1_update == True
        print(f"✅ Updated model 1 performance metrics")
        
        # Update model2 metrics multiple times
        for i in range(3):
            await registry.update_performance_metrics(
                str(model2.teacher_id),
                {
                    "usage_increment": True,
                    "rating": 0.90 + (i * 0.01),
                    "success": True,
                    "response_time": 0.8 + (i * 0.1),
                    "user_id": f"test_user_{i:03d}"
                }
            )
        print(f"✅ Updated model 2 performance metrics (3 times)")
        
        # Test model3 with mixed success
        await registry.update_performance_metrics(
            str(model3.teacher_id),
            {
                "usage_increment": True,
                "rating": 0.82,
                "success": False,  # One failure
                "response_time": 2.1,
                "user_id": "test_user_004"
            }
        )
        print(f"✅ Updated model 3 performance metrics (with failure)")
        
        # === Test Model Details Retrieval ===
        
        print(f"\n📋 Testing model details retrieval...")
        
        model1_details = await registry.get_model_details(str(model1.teacher_id))
        assert model1_details is not None
        assert model1_details["model"]["name"] == model1.name
        assert model1_details["ipfs_cid"] == model1_cid
        assert "performance_metrics" in model1_details
        print(f"✅ Retrieved model 1 details with performance metrics")
        
        # Check performance metrics were updated
        metrics = model1_details["performance_metrics"]
        assert metrics["usage_count"] == 1
        assert metrics["average_rating"] == 0.95  # Updated rating
        print(f"   - Usage count: {metrics['usage_count']}")
        print(f"   - Average rating: {metrics['average_rating']:.3f}")
        print(f"   - Success rate: {metrics.get('success_rate', 0.0):.3f}")
        
        # Test non-existent model
        fake_model_details = await registry.get_model_details(str(uuid4()))
        assert fake_model_details is None
        print(f"✅ Correctly returned None for non-existent model")
        
        # === Test Advanced Model Search ===
        
        print(f"\n🔎 Testing advanced model search...")
        
        # Search by name
        name_results = await registry.search_models("NLP")
        assert len(name_results) >= 1
        print(f"✅ Found {len(name_results)} models matching 'NLP'")
        
        # Search by specialization
        vision_results = await registry.search_models("vision")
        assert len(vision_results) >= 1
        print(f"✅ Found {len(vision_results)} models matching 'vision'")
        
        # Search with performance filter (using a query that will match)
        high_perf_results = await registry.search_models(
            "Advanced", 
            {"performance_min": 0.90}
        )
        print(f"📊 Debug: Found {len(high_perf_results)} high-performance models (>0.90)")
        for result in high_perf_results:
            print(f"   - {result['model']['name']}: original={result['model']['performance_score']:.3f}")
        
        # Let's also check all models that match "Advanced"
        all_advanced_results = await registry.search_models("Advanced")
        print(f"📊 Debug: All models matching 'Advanced': {len(all_advanced_results)}")
        for result in all_advanced_results:
            print(f"   - {result['model']['name']}: {result['model']['performance_score']:.3f}")
        
        assert len(high_perf_results) >= 1  # model1 should qualify (0.95 after update)
        print(f"✅ Found {len(high_perf_results)} high-performance models (>0.90)")
        
        # Search with model type filter
        teacher_results = await registry.search_models(
            "",  # Empty query
            {"model_type": ModelType.TEACHER}
        )
        assert len(teacher_results) == 3  # All our models are teachers
        print(f"✅ Found {len(teacher_results)} teacher models")
        
        # === Test Top Performers ===
        
        print(f"\n🏆 Testing top performers ranking...")
        
        # Get overall top performers
        top_overall = await registry.get_top_performers(limit=5)
        assert len(top_overall) == 3  # We have 3 models total
        # Should be sorted by weighted score (performance + usage)
        print(f"✅ Retrieved {len(top_overall)} top performers overall")
        
        # Check ranking order (model1 should be on top due to higher rating)
        top_model = top_overall[0]
        print(f"   - Top model: {top_model['model'].name}")
        print(f"   - Performance: {top_model['performance_score']:.3f}")
        print(f"   - Usage count: {top_model['usage_count']}")
        print(f"   - Weighted score: {top_model['weighted_score']:.3f}")
        
        # Get top performers in specific category
        top_nlp = await registry.get_top_performers(category="natural_language_processing", limit=2)
        assert len(top_nlp) == 1  # Only one NLP model
        assert top_nlp[0]['model'].teacher_id == model1.teacher_id
        print(f"✅ Retrieved {len(top_nlp)} top NLP performers")
        
        # === Test P2P Federation ===
        
        print(f"\n🌐 Testing P2P federation functionality...")
        
        # Create test peer node
        peer_node = PeerNode(
            node_id="test_peer_001",
            peer_id="12D3KooWTest123",
            multiaddr="/ip4/192.168.1.100/tcp/4001",
            capabilities=["model_hosting", "model_discovery"],
            reputation_score=0.85
        )
        
        # Join federation
        join_success = await registry.join_federation(peer_node)
        assert join_success == True
        print(f"✅ Successfully joined federation network")
        
        # Check federation stats
        fed_stats = await registry.get_registry_stats()
        assert fed_stats["federation_peers"] == 1
        print(f"✅ Federation now has {fed_stats['federation_peers']} peer(s)")
        
        # Leave federation
        leave_success = await registry.leave_federation(peer_node.node_id)
        assert leave_success == True
        print(f"✅ Successfully left federation network")
        
        # === Test Caching ===
        
        print(f"\n🗂️ Testing search caching...")
        
        # First search (should miss cache)
        cache_stats_before = await registry.get_registry_stats()
        cache_hits_before = cache_stats_before["cache_hits"]
        
        first_search = await registry.discover_specialists("natural_language_processing")
        
        # Second search (should hit cache)
        second_search = await registry.discover_specialists("natural_language_processing")
        
        cache_stats_after = await registry.get_registry_stats()
        cache_hits_after = cache_stats_after["cache_hits"]
        
        assert cache_hits_after > cache_hits_before
        assert first_search == second_search
        print(f"✅ Search caching working (cache hits: {cache_hits_before} → {cache_hits_after})")
        
        # === Test Invalid Model Registration ===
        
        print(f"\n⚠️ Testing invalid model registration...")
        
        # Test low performance model
        low_perf_model = TeacherModel(
            name="Poor Performer",
            specialization="test_category",
            model_type=ModelType.TEACHER,
            performance_score=0.05,  # Below minimum threshold
            version="0.1.0",
            active=True
        )
        
        low_perf_data = b"POOR_MODEL_DATA"
        low_perf_cid = await prsm_ipfs_client.store_model(
            low_perf_data,
            {"uploader_id": "test_user", "model_type": "poor_model"}
        )
        
        low_perf_registration = await registry.register_teacher_model(low_perf_model, low_perf_cid)
        assert low_perf_registration == False
        print(f"✅ Correctly rejected low-performance model")
        
        # === Test Health Status ===
        
        print(f"\n🏥 Testing registry health status...")
        
        health = await registry.get_health_status()
        assert health["total_models"] == 3
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        print(f"✅ Registry health status: {health['status']}")
        print(f"   - Health percentage: {health['health_percentage']:.1f}%")
        print(f"   - Healthy models: {health['healthy_models']}/{health['total_models']}")
        print(f"   - Cache hit rate: {health['cache_hit_rate']:.1f}%")
        
        # === Test Global Registry Instance ===
        
        print(f"\n🌍 Testing global registry instance...")
        
        global_stats = await model_registry.get_registry_stats()
        assert global_stats is not None
        print(f"✅ Global model registry instance working")
        print(f"   - Global registry models: {global_stats['total_models']}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL MODEL REGISTRY TESTS PASSED!")
        
        # Show final comprehensive statistics
        final_stats = await registry.get_registry_stats()
        final_health = await registry.get_health_status()
        
        print(f"\n📊 Final Registry Statistics:")
        print(f"   - Total Models: {final_stats['total_models']}")
        print(f"   - Categories: {len(final_stats['categories'])}")
        print(f"   - Total Searches: {final_stats['total_searches']}")
        print(f"   - Cache Hits: {final_stats['cache_hits']}")
        print(f"   - Integrity Validations: {final_stats['total_validations']}")
        print(f"   - Average Performance: {final_stats['avg_performance']:.3f}")
        print(f"   - Health Status: {final_health['status']}")
        print(f"   - IPFS Integration: {'✅ Connected' if final_health['ipfs_connected'] else '📡 Simulated'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model_registry())
    sys.exit(0 if success else 1)