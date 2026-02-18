"""
Pytest configuration and shared fixtures for integration tests
"""

import pytest
import asyncio
import tempfile
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Ensure fixtures directory is importable
_fixtures_dir = Path(__file__).parent.parent / "fixtures"
if str(_fixtures_dir.parent) not in sys.path:
    sys.path.insert(0, str(_fixtures_dir.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session."""
    temp_dir = tempfile.mkdtemp(prefix="prsm_integration_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_files(test_data_dir):
    """Create sample files for testing various scenarios."""
    files = {}
    
    # Small text file
    small_file = os.path.join(test_data_dir, "small_document.txt")
    with open(small_file, 'w') as f:
        f.write("This is a small test document for basic functionality testing.")
    files['small'] = small_file
    
    # Medium PDF-like file
    medium_file = os.path.join(test_data_dir, "research_paper.pdf") 
    with open(medium_file, 'wb') as f:
        f.write(b"PDF simulation content for research paper testing." * 1000)
    files['medium'] = medium_file
    
    # Large dataset file
    large_file = os.path.join(test_data_dir, "research_dataset.csv")
    with open(large_file, 'wb') as f:
        f.write(b"dataset_row,value1,value2,value3\n" * 10000)
    files['large'] = large_file
    
    # Proprietary algorithm file (high security)
    proprietary_file = os.path.join(test_data_dir, "proprietary_algorithm.pdf")
    with open(proprietary_file, 'wb') as f:
        f.write(b"CONFIDENTIAL: Proprietary algorithm implementation details." * 500)
    files['proprietary'] = proprietary_file
    
    yield files


@pytest.fixture
def mock_network_peers():
    """Mock network peers for testing."""
    return [
        {
            'id': 'stanford-lab-01',
            'name': 'Stanford University Lab 01',
            'address': '192.168.1.100',
            'port': 8000,
            'region': 'us-west',
            'reputation': 4.8,
            'bandwidth': {'upload': 100, 'download': 150},
            'latency': 23,
            'status': 'online',
            'capabilities': ['high_security', 'quantum_computing']
        },
        {
            'id': 'mit-research-03', 
            'name': 'MIT Research Node 03',
            'address': '192.168.1.101',
            'port': 8000,
            'region': 'us-east',
            'reputation': 4.2,
            'bandwidth': {'upload': 75, 'download': 100},
            'latency': 67,
            'status': 'online',
            'capabilities': ['medium_security', 'ai_research']
        },
        {
            'id': 'duke-medical-02',
            'name': 'Duke Medical Center 02',
            'address': '192.168.1.102', 
            'port': 8000,
            'region': 'us-east',
            'reputation': 4.9,
            'bandwidth': {'upload': 120, 'download': 200},
            'latency': 12,
            'status': 'online',
            'capabilities': ['high_security', 'medical_research', 'hipaa_compliant']
        },
        {
            'id': 'oxford-quantum-lab',
            'name': 'Oxford Quantum Computing Lab',
            'address': '192.168.2.100',
            'port': 8000,
            'region': 'europe',
            'reputation': 4.7,
            'bandwidth': {'upload': 85, 'download': 120},
            'latency': 145,
            'status': 'online',
            'capabilities': ['high_security', 'quantum_computing']
        },
        {
            'id': 'eth-zurich-main',
            'name': 'ETH Zurich Main Node',
            'address': '192.168.2.101',
            'port': 8000,
            'region': 'europe',
            'reputation': 4.6,
            'bandwidth': {'upload': 95, 'download': 140},
            'latency': 134,
            'status': 'online',
            'capabilities': ['medium_security', 'ai_research']
        },
        {
            'id': 'tokyo-tech-ai',
            'name': 'Tokyo Tech AI Research',
            'address': '192.168.3.100',
            'port': 8000,
            'region': 'asia',
            'reputation': 4.5,
            'bandwidth': {'upload': 80, 'download': 110},
            'latency': 98,
            'status': 'online',
            'capabilities': ['medium_security', 'ai_research']
        },
        {
            'id': 'backup-storage-cluster',
            'name': 'Backup Storage Cluster',
            'address': '192.168.0.10',
            'port': 8000,
            'region': 'local',
            'reputation': 5.0,
            'bandwidth': {'upload': 200, 'download': 200},
            'latency': 5,
            'status': 'online',
            'capabilities': ['high_security', 'backup_storage', 'high_availability']
        }
    ]


@pytest.fixture 
def mock_users():
    """Mock users for testing collaboration scenarios."""
    return [
        {
            'id': 'dr.chen@unc.edu',
            'name': 'Dr. Sarah Chen',
            'role': 'principal_investigator',
            'institution': 'University of North Carolina',
            'department': 'Computer Science',
            'clearance_level': 'high',
            'specializations': ['quantum_computing', 'machine_learning'],
            'active_projects': ['quantum_ml_initiative', 'nsf_quantum_grant']
        },
        {
            'id': 'michael.j@sas.com',
            'name': 'Michael Johnson',
            'role': 'industry_partner',
            'institution': 'SAS Institute',
            'department': 'Advanced Analytics Research',
            'clearance_level': 'medium',
            'specializations': ['commercial_ai', 'data_analytics'],
            'active_projects': ['quantum_ml_initiative']
        },
        {
            'id': 'alex.r@duke.edu',
            'name': 'Dr. Alex Rodriguez',
            'role': 'collaborator',
            'institution': 'Duke University',
            'department': 'Medical Informatics',
            'clearance_level': 'restricted',
            'specializations': ['medical_ai', 'privacy_preservation'],
            'active_projects': ['medical_ai_collaboration']
        },
        {
            'id': 'supervisor@unc.edu',
            'name': 'Dr. Patricia Williams',
            'role': 'tech_transfer_supervisor',
            'institution': 'University of North Carolina',
            'department': 'Technology Transfer Office',
            'clearance_level': 'high',
            'specializations': ['ip_management', 'university_industry_partnerships'],
            'active_projects': ['quantum_ml_initiative', 'ip_evaluation_platform']
        },
        {
            'id': 'pi.duke@duke.edu',
            'name': 'Dr. Robert Kim',
            'role': 'co_principal_investigator',
            'institution': 'Duke University',
            'department': 'Physics',
            'clearance_level': 'high',
            'specializations': ['quantum_physics', 'quantum_algorithms'],
            'active_projects': ['nsf_quantum_grant']
        }
    ]


@pytest.fixture
def mock_security_config():
    """Mock security configuration for testing."""
    return {
        'post_quantum': {
            'key_encapsulation': 'kyber-1024',
            'digital_signature': 'ml-dsa-87',
            'key_rotation_interval': 3600,  # 1 hour
            'minimum_key_strength': 256
        },
        'access_control': {
            'multi_signature_threshold': {
                'high_security': 2,
                'medium_security': 1,
                'standard_security': 1
            },
            'approval_timeout': 86400,  # 24 hours
            'session_timeout': 3600     # 1 hour
        },
        'sharding': {
            'security_levels': {
                'high': {'shard_count': 7, 'threshold': 4},
                'medium': {'shard_count': 5, 'threshold': 3},
                'standard': {'shard_count': 3, 'threshold': 2}
            },
            'encryption_algorithm': 'aes-256-gcm',
            'compression_enabled': True
        },
        'network': {
            'min_peer_reputation': 4.0,
            'max_latency_threshold': 200,  # ms
            'bandwidth_requirements': {
                'minimum': 10,   # MB/s
                'optimal': 50    # MB/s
            }
        }
    }


@pytest.fixture
def mock_workspace_configs():
    """Mock workspace configurations for different collaboration types."""
    return {
        'university_industry': {
            'name': 'University-Industry Partnership',
            'security_level': 'high',
            'shard_count': 7,
            'approval_requirements': {
                'file_access': 2,
                'data_export': 3,
                'ip_evaluation': 2
            },
            'participant_roles': ['principal_investigator', 'industry_partner', 'tech_transfer_supervisor'],
            'compliance_requirements': ['ip_protection', 'confidentiality_agreement']
        },
        'research_consortium': {
            'name': 'Multi-Institutional Research Consortium',
            'security_level': 'medium',
            'shard_count': 5,
            'approval_requirements': {
                'file_access': 1,
                'data_export': 2,
                'publication_approval': 3
            },
            'participant_roles': ['principal_investigator', 'co_principal_investigator', 'researcher'],
            'compliance_requirements': ['grant_requirements', 'institutional_approval']
        },
        'clinical_trial': {
            'name': 'Clinical Trial Collaboration',
            'security_level': 'high',
            'shard_count': 7,
            'approval_requirements': {
                'patient_data_access': 3,
                'data_export': 4,
                'protocol_modification': 3
            },
            'participant_roles': ['principal_investigator', 'clinical_researcher', 'data_manager', 'irb_representative'],
            'compliance_requirements': ['hipaa', 'irb_approval', 'fda_regulations']
        }
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        'file_operations': {
            'small_file_sharding': {'max_time': 1.0, 'target_time': 0.5},     # seconds
            'medium_file_sharding': {'max_time': 5.0, 'target_time': 2.0},    # seconds
            'large_file_sharding': {'max_time': 15.0, 'target_time': 8.0},    # seconds
            'file_reconstruction': {'max_time': 3.0, 'target_time': 1.0}      # seconds
        },
        'network_operations': {
            'peer_discovery': {'max_time': 2.0, 'target_time': 1.0},          # seconds
            'shard_distribution': {'max_time': 10.0, 'target_time': 5.0},     # seconds
            'integrity_validation': {'max_time': 5.0, 'target_time': 2.0}     # seconds
        },
        'security_operations': {
            'key_generation': {'max_time': 2.0, 'target_time': 1.0},          # seconds
            'access_authorization': {'max_time': 1.0, 'target_time': 0.3},    # seconds
            'signature_verification': {'max_time': 0.5, 'target_time': 0.1}   # seconds
        },
        'ui_operations': {
            'dashboard_load': {'max_time': 3.0, 'target_time': 1.5},          # seconds
            'real_time_update': {'max_time': 1.0, 'target_time': 0.3},        # seconds
            'user_interaction': {'max_time': 0.5, 'target_time': 0.2}         # seconds
        }
    }


# Utility functions for tests
def create_test_file(path, content, size_mb=None):
    """Create a test file with specified content or size."""
    with open(path, 'wb') as f:
        if size_mb:
            # Create file of specific size
            chunk = b'A' * 1024  # 1KB chunk
            for _ in range(size_mb * 1024):
                f.write(chunk)
        else:
            # Write specific content
            if isinstance(content, str):
                content = content.encode()
            f.write(content)


def assert_performance(actual_time, benchmark, operation_name):
    """Assert that operation performance meets benchmarks."""
    max_time = benchmark['max_time']
    target_time = benchmark['target_time']
    
    # Must be under maximum time
    assert actual_time <= max_time, f"{operation_name} took {actual_time:.2f}s, max allowed: {max_time}s"
    
    # Log performance relative to target
    if actual_time <= target_time:
        print(f"✅ {operation_name}: {actual_time:.2f}s (target: {target_time}s)")
    else:
        print(f"⚠️  {operation_name}: {actual_time:.2f}s (target: {target_time}s, max: {max_time}s)")


def simulate_network_delay(latency_ms=50):
    """Simulate network delay for testing."""
    import time
    time.sleep(latency_ms / 1000.0)


async def async_simulate_network_delay(latency_ms=50):
    """Async version of network delay simulation."""
    await asyncio.sleep(latency_ms / 1000.0)


class APITestDataFactory:
    """Factory for creating API test data"""

    @staticmethod
    def create_nwtn_query_request(query: str = "Test reasoning query", mode: str = "adaptive",
                                  max_depth: int = 3, **kwargs) -> Dict[str, Any]:
        return {"query": query, "mode": mode, "max_depth": max_depth,
                "timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}

    @staticmethod
    def create_ftns_transfer_request(recipient: str = "test_recipient", amount: float = 10.0,
                                     description: str = "Test transfer", **kwargs) -> Dict[str, Any]:
        return {"recipient": recipient, "amount": amount, "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}

    @staticmethod
    def create_marketplace_item_request(title: str = "Test Item", description: str = "Test marketplace item",
                                        price: float = 50.0, category: str = "tools", **kwargs) -> Dict[str, Any]:
        return {"title": title, "description": description, "price": price, "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(), **kwargs}

    @staticmethod
    def create_user_registration_request(username: str = "testuser", email: str = "test@example.com",
                                          password: str = "testpassword123", **kwargs) -> Dict[str, Any]:
        return {"username": username, "email": email, "password": password, **kwargs}


@pytest.fixture
def api_data_factory():
    """API test data factory fixture"""
    return APITestDataFactory()


@pytest.fixture
def user_headers():
    """Regular user authorization headers for API testing"""
    import jwt as pyjwt
    payload = {
        "sub": "test_regular_user",
        "username": "user",
        "email": "user@test.com",
        "role": "user",
        "permissions": ["read", "write"],
        "token_type": "access",
        "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        "iat": datetime.now(timezone.utc).timestamp(),
        "jti": "test_jti_user"
    }
    token = pyjwt.encode(payload, "test-secret-key-for-testing-only-minimum-32-chars-required-here", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def expired_token_headers():
    """Expired token headers for testing authentication failures"""
    import jwt as pyjwt
    payload = {
        "sub": "test_regular_user",
        "username": "user",
        "email": "user@test.com",
        "role": "user",
        "permissions": ["read", "write"],
        "token_type": "access",
        "exp": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp(),
        "iat": (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp(),
        "jti": "test_jti_expired"
    }
    token = pyjwt.encode(payload, "test-secret-key-for-testing-only-minimum-32-chars-required-here", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


class _APIPerformanceMonitor:
    def __init__(self):
        self.requests = []

    def record_request(self, start_time, end_time, response_size):
        self.requests.append({"start": start_time, "end": end_time, "size": response_size})

    def get_stats(self):
        if not self.requests:
            return {"avg_time": 0, "total_requests": 0}
        times = [r["end"] - r["start"] for r in self.requests]
        return {"avg_time": sum(times) / len(times), "total_requests": len(self.requests)}


@pytest.fixture
def api_performance_monitor():
    """API performance monitoring fixture"""
    return _APIPerformanceMonitor()


class _SecurityTestHelper:
    sql_injection_payloads = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
    ]
    xss_payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
    ]

    def get_sql_injection_payloads(self):
        return self.sql_injection_payloads

    def get_xss_payloads(self):
        return self.xss_payloads

    def get_malformed_json_payloads(self):
        return ["{invalid}", "{'bad': }", "{\"unterminated\": "]


@pytest.fixture
def security_test_helper():
    """Security testing helper fixture"""
    return _SecurityTestHelper()


class _RateLimitTester:
    async def send_requests(self, client, url, headers, count=10):
        results = []
        for _ in range(count):
            resp = await client.get(url, headers=headers)
            results.append(resp.status_code)
        return results


@pytest.fixture
def rate_limit_tester():
    """Rate limit testing fixture"""
    return _RateLimitTester()


@pytest.fixture
def api_response_schemas():
    """Expected API response schemas for validation"""
    return {
        "nwtn_response": {
            "required_fields": ["response", "session_id", "timestamp"],
        },
        "error_response": {
            "required_fields": ["detail"],
        },
    }