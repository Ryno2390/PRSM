#!/usr/bin/env python3
"""
Test Post-Quantum Cryptography Implementation
Validates CRYSTALS-Dilithium / ML-DSA functionality in PRSM
"""

import sys
from pathlib import Path

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

# Import directly from the module file to avoid __init__.py issues
from prsm.core.cryptography.post_quantum import (
    PostQuantumCrypto, SecurityLevel, generate_pq_keypair, 
    sign_with_pq, verify_pq_signature, PostQuantumKeyPair, PostQuantumSignature
)


def test_key_generation():
    """Test post-quantum key generation for all security levels"""
    print("1. 🔑 Testing Key Generation for All Security Levels:")
    print("-" * 50)
    
    for level in SecurityLevel:
        keypair = generate_pq_keypair(level)
        print(f"   {level.value}:")
        print(f"     Public Key:  {len(keypair.public_key):4d} bytes")
        print(f"     Private Key: {len(keypair.private_key):4d} bytes")
        print(f"     Key ID:      {keypair.key_id}")
        
        # Verify keypair structure
        assert isinstance(keypair, PostQuantumKeyPair)
        assert len(keypair.public_key) > 0
        assert len(keypair.private_key) > 0
        assert keypair.security_level == level
    
    print("   ✅ All security levels working correctly")
    return True


def test_signing_and_verification():
    """Test post-quantum signing and verification"""
    print("\n2. 🔐 Testing Signing & Verification:")
    print("-" * 50)
    
    # Test with different security levels
    test_message = "PRSM post-quantum cryptography test message for blockchain signatures"
    
    for level in SecurityLevel:
        print(f"\n   Testing {level.value}:")
        
        # Generate keypair
        keypair = generate_pq_keypair(level)
        
        # Sign message
        signature = sign_with_pq(test_message, keypair)
        print(f"     Signature Size: {len(signature.signature)} bytes")
        print(f"     Signature Type: {signature.signature_type.value}")
        
        # Verify signature
        is_valid = verify_pq_signature(test_message, signature, keypair.public_key)
        print(f"     Valid Signature: {is_valid}")
        assert is_valid, f"Signature verification failed for {level.value}"
        
        # Test with wrong message
        wrong_message = "Different message that should fail verification"
        is_invalid = verify_pq_signature(wrong_message, signature, keypair.public_key)
        print(f"     Wrong Msg Rejected: {not is_invalid}")
        assert not is_invalid, f"Invalid signature not rejected for {level.value}"
        
        # Test with wrong public key
        other_keypair = generate_pq_keypair(level)
        is_wrong_key = verify_pq_signature(test_message, signature, other_keypair.public_key)
        print(f"     Wrong Key Rejected: {not is_wrong_key}")
        assert not is_wrong_key, f"Wrong key not rejected for {level.value}"
    
    print("\n   ✅ All signing and verification tests passed")
    return True


def test_serialization():
    """Test serialization and deserialization of keys and signatures"""
    print("\n3. 💾 Testing Serialization & Deserialization:")
    print("-" * 50)
    
    # Test keypair serialization
    original_keypair = generate_pq_keypair(SecurityLevel.LEVEL_1)
    keypair_dict = original_keypair.to_dict()
    restored_keypair = PostQuantumKeyPair.from_dict(keypair_dict)
    
    print(f"   Original Key ID:  {original_keypair.key_id}")
    print(f"   Restored Key ID:  {restored_keypair.key_id}")
    assert original_keypair.key_id == restored_keypair.key_id
    assert original_keypair.public_key == restored_keypair.public_key
    assert original_keypair.private_key == restored_keypair.private_key
    print("     ✅ Keypair serialization working")
    
    # Test signature serialization
    message = "Test message for serialization"
    original_signature = sign_with_pq(message, original_keypair)
    signature_dict = original_signature.to_dict()
    restored_signature = PostQuantumSignature.from_dict(signature_dict)
    
    assert original_signature.signature == restored_signature.signature
    assert original_signature.message_hash == restored_signature.message_hash
    print("     ✅ Signature serialization working")
    
    # Verify restored signature works
    is_valid = verify_pq_signature(message, restored_signature, restored_keypair.public_key)
    print(f"     Restored Sig Valid: {is_valid}")
    assert is_valid
    
    print("   ✅ Serialization tests passed")
    return True


def test_performance_benchmark():
    """Test performance benchmarking functionality"""
    print("\n4. ⚡ Testing Performance Benchmarking:")
    print("-" * 50)
    
    pq_crypto = PostQuantumCrypto()
    
    # Run quick benchmark (3 iterations for speed)
    benchmark_results = pq_crypto.benchmark_all_security_levels(iterations=3)
    
    for level, results in benchmark_results.items():
        print(f"\n   {level} Performance:")
        print(f"     Key Generation: {results['keygen_performance']['mean_ms']:.2f}ms avg")
        print(f"     Signing:        {results['signing_performance']['mean_ms']:.2f}ms avg")
        print(f"     Verification:   {results['verification_performance']['mean_ms']:.2f}ms avg")
        print(f"     Public Key:     {results['key_sizes']['public_key_bytes']} bytes")
        print(f"     Private Key:    {results['key_sizes']['private_key_bytes']} bytes")
        print(f"     Signature:      {results['signature_size_bytes']} bytes")
    
    print("\n   ✅ Performance benchmarking working")
    return True


def test_security_information():
    """Test security information and metadata"""
    print("\n5. 🛡️ Testing Security Information:")
    print("-" * 50)
    
    pq_crypto = PostQuantumCrypto()
    
    for level in SecurityLevel:
        security_info = pq_crypto.get_security_info(level)
        print(f"\n   {level.value}:")
        print(f"     Algorithm:    {security_info['algorithm']}")
        print(f"     Standard:     {security_info['standard']}")
        print(f"     Category:     {security_info['nist_category']}")
        print(f"     Description:  {security_info['description']}")
        print(f"     Attack Cost:  {security_info['quantum_attack_cost']}")
        
        # Verify expected values
        assert security_info['algorithm'] == "ML-DSA (CRYSTALS-Dilithium)"
        assert security_info['standard'] == "NIST FIPS 204"
        assert "post-quantum security" in security_info['description']
    
    print("\n   ✅ Security information correct")
    return True


def main():
    """Run all post-quantum cryptography tests"""
    print("🔐 PRSM Post-Quantum Cryptography Test Suite")
    print("=" * 60)
    print("Testing CRYSTALS-Dilithium / ML-DSA (NIST FIPS 204) Implementation")
    print()
    
    tests = [
        test_key_generation,
        test_signing_and_verification,
        test_serialization,
        test_performance_benchmark,
        test_security_information
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            success = test()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("🏁 POST-QUANTUM CRYPTOGRAPHY TEST RESULTS:")
    print(f"   Tests Passed: {passed}")
    print(f"   Tests Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All post-quantum cryptography tests PASSED!")
        print("✅ CRYSTALS-Dilithium / ML-DSA implementation is working correctly")
        print("🔒 PRSM is now quantum-resistant!")
        return 0
    else:
        print(f"\n❌ {failed} post-quantum cryptography tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)