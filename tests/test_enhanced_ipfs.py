#!/usr/bin/env python3
"""
Test script for PRSM Enhanced IPFS Client
Tests Phase 1, Week 2, Task 1 - Enhanced IPFS Client
"""

import asyncio
import json
import sys
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any

async def test_enhanced_ipfs_client():
    """Test enhanced IPFS client functionality"""
    
    print("📦 Testing PRSM Enhanced IPFS Client...")
    print("=" * 60)
    
    try:
        # Import the enhanced IPFS client
        from prsm.data.data_layer.enhanced_ipfs import PRSMIPFSClient, prsm_ipfs_client
        from prsm.core.models import ProvenanceRecord, ModelShard
        
        print("✅ Enhanced IPFS client imports successful")
        
        # Create client instance
        client = PRSMIPFSClient()
        
        # Wait for connection initialization
        await asyncio.sleep(1)
        
        # === Test Client Status ===
        
        status = await client.get_status()
        print(f"✅ IPFS client status retrieved")
        print(f"   - Connected: {status['connected']}")
        print(f"   - API Address: {status['api_address']}")
        print(f"   - Provenance Tracking: {status['provenance_tracking']}")
        print(f"   - Access Rewards: {status['access_rewards']}")
        print(f"   - Max Model Size: {status['max_model_size_mb']}MB")
        
        # === Test Model Storage ===
        
        # Create test model data
        test_model_data = b"FAKE_MODEL_DATA_" + b"x" * 1000  # 1KB model
        model_metadata = {
            "uploader_id": "test_user_123",
            "model_type": "neural_network", 
            "version": "1.0.0",
            "framework": "pytorch",
            "accuracy": 0.95,
            "training_dataset": "test_dataset_v1"
        }
        
        print(f"\n📤 Testing model storage...")
        model_cid = await client.store_model(test_model_data, model_metadata)
        assert model_cid is not None
        print(f"✅ Model stored successfully with CID: {model_cid}")
        
        # === Test Dataset Storage ===
        
        # Create test dataset
        test_dataset = json.dumps({
            "data": [{"x": i, "y": i*2} for i in range(100)],
            "metadata": {"rows": 100, "features": 2}
        }).encode('utf-8')
        
        dataset_provenance = {
            "uploader_id": "test_user_123",
            "data_type": "training_data",
            "source": "synthetic_generator",
            "collection_date": "2025-06-03",
            "preprocessing_steps": ["normalization", "train_test_split"],
            "quality_score": 0.9
        }
        
        print(f"\n📤 Testing dataset storage...")
        dataset_cid = await client.store_dataset(test_dataset, dataset_provenance)
        assert dataset_cid is not None
        print(f"✅ Dataset stored successfully with CID: {dataset_cid}")
        
        # === Test Content Retrieval with Provenance ===
        
        print(f"\n📥 Testing model retrieval with provenance...")
        retrieved_model, model_provenance = await client.retrieve_with_provenance(model_cid)
        assert retrieved_model is not None
        assert model_provenance is not None
        print(f"✅ Model retrieved: {len(retrieved_model)} bytes")
        print(f"✅ Model provenance: {model_provenance.get('storage_type', 'unknown')} type")
        
        # Verify model data integrity
        assert retrieved_model == test_model_data
        print(f"✅ Model data integrity verified")
        
        print(f"\n📥 Testing dataset retrieval with provenance...")
        retrieved_dataset, dataset_meta = await client.retrieve_with_provenance(dataset_cid)
        assert retrieved_dataset is not None
        assert dataset_meta is not None
        print(f"✅ Dataset retrieved: {len(retrieved_dataset)} bytes")
        print(f"✅ Dataset provenance: {dataset_meta.get('storage_type', 'unknown')} type")
        
        # Verify dataset integrity
        assert retrieved_dataset == test_dataset
        print(f"✅ Dataset data integrity verified")
        
        # === Test Access Tracking ===
        
        print(f"\n📊 Testing access tracking...")
        
        # Track multiple accesses
        await client.track_access(model_cid, "user_001")
        await client.track_access(model_cid, "user_002")
        await client.track_access(model_cid, "user_001")  # Repeat access
        await client.track_access(dataset_cid, "user_003")
        
        print(f"✅ Access tracking completed for multiple users")
        
        # === Test Usage Metrics ===
        
        print(f"\n📈 Testing usage metrics calculation...")
        
        model_metrics = await client.calculate_usage_metrics(model_cid)
        assert model_metrics is not None
        assert model_metrics["cid"] == model_cid
        print(f"✅ Model metrics calculated:")
        print(f"   - Total accesses: {model_metrics['total_accesses']}")
        print(f"   - Unique accessors: {model_metrics['unique_accessors']}")
        print(f"   - Access frequency: {model_metrics['access_frequency']:.2f}/day")
        
        dataset_metrics = await client.calculate_usage_metrics(dataset_cid)
        print(f"✅ Dataset metrics calculated:")
        print(f"   - Total accesses: {dataset_metrics['total_accesses']}")
        print(f"   - Unique accessors: {dataset_metrics['unique_accessors']}")
        
        # === Test Model Integrity Verification ===
        
        print(f"\n🔒 Testing model integrity verification...")
        
        model_integrity = await client.verify_model_integrity(model_cid)
        assert model_integrity == True
        print(f"✅ Model integrity verification passed")
        
        dataset_integrity = await client.verify_model_integrity(dataset_cid)
        assert dataset_integrity == True
        print(f"✅ Dataset integrity verification passed")
        
        # === Test Model Shard Registration ===
        
        print(f"\n🔗 Testing model shard registration...")
        
        shard = await client.register_model_shard(
            model_cid=model_cid,
            shard_index=0,
            total_shards=3,
            hosted_by=["node_001", "node_002"]
        )
        
        assert shard is not None
        assert shard.model_cid == model_cid
        assert shard.shard_index == 0
        assert shard.total_shards == 3
        assert len(shard.hosted_by) == 2
        print(f"✅ Model shard registered: {shard.shard_id}")
        print(f"   - Verification hash: {shard.verification_hash[:16]}...")
        print(f"   - Size: {shard.size_bytes} bytes")
        
        # === Test Large Model Size Validation ===
        
        print(f"\n⚠️  Testing large model size validation...")
        
        try:
            # Create oversized model (assuming 1000MB limit)
            large_model_data = b"x" * (1001 * 1024 * 1024)  # 1001MB
            await client.store_model(large_model_data, {"uploader_id": "test"})
            assert False, "Should have raised ValueError for oversized model"
        except ValueError as e:
            print(f"✅ Large model correctly rejected: {e}")
        
        # === Test Provenance Cache ===
        
        print(f"\n📋 Testing provenance cache...")
        
        assert len(client.provenance_cache) >= 2  # At least model and dataset
        print(f"✅ Provenance cache contains {len(client.provenance_cache)} records")
        
        # Check model provenance record
        if model_cid in client.provenance_cache:
            model_record = client.provenance_cache[model_cid]
            assert model_record.uploader_id == "test_user_123"
            assert model_record.access_count >= 3  # We tracked 3 accesses
            print(f"✅ Model provenance record verified: {model_record.access_count} accesses")
        
        # === Test Access Log ===
        
        print(f"\n📝 Testing access log...")
        
        assert len(client.access_log) >= 2  # At least model and dataset
        total_logged_accesses = sum(len(logs) for logs in client.access_log.values())
        print(f"✅ Access log contains {total_logged_accesses} total access records")
        
        # Verify model access logs
        if model_cid in client.access_log:
            model_accesses = client.access_log[model_cid]
            assert len(model_accesses) == 3  # We tracked 3 accesses
            
            # Check access record structure
            for access in model_accesses:
                assert "accessor_id" in access
                assert "timestamp" in access
                assert "access_type" in access
            
            print(f"✅ Model access log verified: {len(model_accesses)} records")
        
        # === Test Error Handling ===
        
        print(f"\n❌ Testing error handling...")
        
        # Test retrieval of non-existent CID
        try:
            fake_cid = "bafybeig" + "x" * 50
            content, metadata = await client.retrieve_with_provenance(fake_cid)
            # In simulation mode, this might succeed with fake content
            print(f"ℹ️  Non-existent CID handled (simulation mode)")
        except Exception as e:
            print(f"✅ Non-existent CID properly handled: {type(e).__name__}")
        
        # Test metrics for non-existent CID
        fake_metrics = await client.calculate_usage_metrics("fake_cid_123")
        assert fake_metrics["total_accesses"] == 0
        print(f"✅ Metrics for non-existent CID handled correctly")
        
        # === Test Global Client Instance ===
        
        print(f"\n🌐 Testing global client instance...")
        
        global_status = await prsm_ipfs_client.get_status()
        assert global_status is not None
        print(f"✅ Global IPFS client instance working")
        print(f"   - Tracked content: {global_status['tracked_content']}")
        print(f"   - Total accesses: {global_status['total_accesses']}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL ENHANCED IPFS CLIENT TESTS PASSED!")
        
        # Show final statistics
        final_status = await client.get_status()
        print(f"\n📊 Final Statistics:")
        print(f"   - Connection Status: {'✅ Connected' if final_status['connected'] else '📡 Simulated'}")
        print(f"   - Content Stored: 2 items (1 model, 1 dataset)")
        print(f"   - Provenance Records: {final_status['tracked_content']}")
        print(f"   - Total Access Events: {final_status['total_accesses']}")
        print(f"   - Data Integrity: 100% verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced IPFS client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_ipfs_client())
    sys.exit(0 if success else 1)