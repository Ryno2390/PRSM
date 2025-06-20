# PRSM IPFS Integration Guide

## Overview

This comprehensive guide covers PRSM's IPFS (InterPlanetary File System) integration, providing developers and operators with everything needed to understand, deploy, and maintain the distributed storage layer that powers PRSM's decentralized AI model ecosystem.

## 🎯 IPFS Integration Status

**✅ PRODUCTION READY** - PRSM's IPFS integration is fully implemented and tested for production deployment.

### Key Capabilities

- **🌐 Multi-node architecture** with intelligent failover
- **🔒 Enterprise security** with content integrity verification  
- **⚡ High performance** with optimized caching and concurrency
- **📊 Comprehensive monitoring** with Prometheus metrics
- **💰 Token integration** with FTNS rewards and royalties
- **🔄 Automatic recovery** with exponential backoff retry logic

## 📚 Architecture Overview

### IPFS Components in PRSM

```
┌─────────────────────────────────────────────────────────────────┐
│                          PRSM IPFS Stack                        │
├─────────────────────────────────────────────────────────────────┤
│  API Layer          │  Enhanced Operations  │  Core Client      │
│  ├─ Upload/Download  │  ├─ Model Storage     │  ├─ Multi-node    │
│  ├─ Health Checks    │  ├─ Dataset Storage   │  ├─ Failover      │
│  └─ Status Endpoints │  ├─ Provenance Track  │  ├─ Health Monitor│
│                      │  └─ Usage Analytics   │  └─ Content Cache │
├─────────────────────────────────────────────────────────────────┤
│  Integration Layer                                              │
│  ├─ Database (Metadata Storage)  ├─ FTNS Tokens (Rewards)      │
│  ├─ Model Registry (Discovery)   ├─ P2P Network (Federation)   │
│  ├─ Safety Systems (Validation)  └─ Monitoring (Metrics)       │
├─────────────────────────────────────────────────────────────────┤
│                         IPFS Network                            │
│  ├─ Primary Node (localhost:5001)                              │
│  ├─ Gateway Nodes (ipfs.io, pinata.cloud, dweb.link)          │
│  └─ P2P Swarm (Distributed Content Discovery)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Core IPFS Client** (`prsm/core/ipfs_client.py`)
   - Multi-node management with health monitoring
   - Intelligent failover and load balancing
   - Content upload/download with progress tracking
   - Gateway fallback for reliability

2. **Enhanced IPFS Client** (`prsm/data_layer/enhanced_ipfs.py`)
   - PRSM-specific model and dataset operations
   - Provenance tracking with comprehensive metadata
   - FTNS token integration for rewards
   - Usage analytics and access tracking

3. **Integration Points**
   - API endpoints for external access
   - Database persistence for metadata
   - Model registry for discovery
   - Token economy for incentives

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- IPFS node (optional - will use gateways if unavailable)
- Redis (for caching)
- PostgreSQL (for metadata)

### Installation

```bash
# Install PRSM with IPFS support
pip install -e .

# Install IPFS (optional)
# macOS
brew install ipfs

# Ubuntu/Debian  
wget https://dist.ipfs.io/go-ipfs/v0.17.0/go-ipfs_v0.17.0_linux-amd64.tar.gz
tar -xzf go-ipfs_v0.17.0_linux-amd64.tar.gz
sudo mv go-ipfs/ipfs /usr/local/bin/
```

### Basic Usage

```python
import asyncio
from prsm.core.ipfs_client import get_ipfs_client, init_ipfs
from prsm.data_layer.enhanced_ipfs import get_ipfs_client as get_enhanced_client

async def main():
    # Initialize IPFS clients
    await init_ipfs()
    
    core_client = get_ipfs_client()
    enhanced_client = get_enhanced_client()
    
    # Upload content
    content = b"Hello PRSM IPFS!"
    result = await core_client.upload_content(content, filename="test.txt")
    
    if result.success:
        print(f"Content uploaded: {result.cid}")
        
        # Download content
        download_result = await core_client.download_content(result.cid)
        if download_result.success:
            downloaded_content = download_result.metadata["content"]
            print(f"Content downloaded: {downloaded_content.decode()}")

asyncio.run(main())
```

## 🔧 Configuration

### Environment Variables

```bash
# Core IPFS Settings
PRSM_IPFS_HOST=localhost
PRSM_IPFS_PORT=5001
PRSM_IPFS_TIMEOUT=60
PRSM_IPFS_GATEWAY=http://localhost:8080

# Enhanced IPFS Settings  
PRSM_IPFS_PROVENANCE=true
PRSM_IPFS_REWARDS=true
PRSM_MAX_MODEL_SIZE_MB=1000

# Performance Settings
PRSM_IPFS_MAX_CONNECTIONS=100
PRSM_IPFS_CACHE_SIZE_MB=100
PRSM_IPFS_RETRY_ATTEMPTS=3
```

### Configuration File

```python
# config.py
class PRSMSettings(BaseSettings):
    # IPFS Configuration
    ipfs_host: str = Field(default="localhost", env="PRSM_IPFS_HOST")
    ipfs_port: int = Field(default=5001, env="PRSM_IPFS_PORT") 
    ipfs_timeout: int = Field(default=60, env="PRSM_IPFS_TIMEOUT")
    ipfs_gateway_url: str = Field(default="http://localhost:8080", env="PRSM_IPFS_GATEWAY")
    
    @property
    def ipfs_config(self) -> Dict[str, Any]:
        return {
            "host": self.ipfs_host,
            "port": self.ipfs_port,
            "timeout": self.ipfs_timeout,
        }
```

## 📝 API Documentation

### Core Client API

#### Upload Content

```python
async def upload_content(
    content: Union[bytes, str, Path],
    filename: Optional[str] = None,
    pin: bool = True,
    progress_callback: Optional[Callable[[IPFSUploadProgress], None]] = None
) -> IPFSResult
```

**Parameters:**
- `content`: Content to upload (bytes, string, or file path)
- `filename`: Optional filename for the content
- `pin`: Whether to pin the content (prevents garbage collection)
- `progress_callback`: Optional callback for upload progress

**Returns:** `IPFSResult` with success status, CID, and metadata

#### Download Content

```python
async def download_content(
    cid: str,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verify_integrity: bool = True
) -> IPFSResult
```

**Parameters:**
- `cid`: Content identifier to download
- `output_path`: Optional file path to save content
- `progress_callback`: Optional callback for download progress  
- `verify_integrity`: Whether to verify content integrity

**Returns:** `IPFSResult` with downloaded content or file information

### Enhanced Client API

#### Store Model

```python
async def store_model(
    model_data: bytes, 
    metadata: Dict[str, Any]
) -> str
```

**Parameters:**
- `model_data`: Serialized model data
- `metadata`: Model metadata including uploader_id, model_type, etc.

**Returns:** CID of stored model with provenance tracking

#### Retrieve with Provenance

```python
async def retrieve_with_provenance(
    cid: str
) -> Tuple[bytes, Dict[str, Any]]
```

**Parameters:**
- `cid`: Content identifier to retrieve

**Returns:** Tuple of (content_bytes, provenance_metadata)

#### Track Access

```python
async def track_access(
    cid: str, 
    accessor_id: str
) -> None
```

**Parameters:**
- `cid`: Content identifier that was accessed
- `accessor_id`: ID of user/system accessing content

**Effects:** Updates access logs and triggers FTNS royalty payments

## 🧪 Testing

### Running Tests

```bash
# Run all IPFS tests
python tests/test_enhanced_ipfs.py
python tests/test_core_ipfs_client.py
python tests/test_ipfs_performance_optimization.py
python tests/test_ipfs_system_integration.py

# Run with pytest (if available)
pytest tests/test_*ipfs*.py -v
```

### Test Coverage

- ✅ **Enhanced IPFS Client**: Model/dataset operations, provenance tracking
- ✅ **Core IPFS Client**: Multi-node failover, health monitoring, content operations
- ✅ **Performance Testing**: Throughput benchmarks, concurrency testing, optimization
- ✅ **System Integration**: Database, API, FTNS, monitoring integration

### Sample Test Output

```
🌐 Testing PRSM Core IPFS Client...
✅ Core IPFS client imports successful
✅ IPFS client initialized
   - Connected: True
   - Total nodes: 5
   - Gateway nodes: 4
   - Primary node: http://localhost:5001
✅ Health check completed: 3/5 nodes healthy
✅ Content upload/download working
✅ Error handling robust
🏆 Overall Status: PRODUCTION READY
```

## 📊 Monitoring and Observability

### Health Endpoints

```python
# Health check endpoint
GET /api/v1/ipfs/health
{
    "status": "healthy",
    "nodes": {
        "total": 5,
        "healthy": 3,
        "unhealthy": 2
    },
    "performance": {
        "avg_response_time": 0.085,
        "cache_hit_rate": 0.87,
        "operations_per_second": 45.2
    }
}
```

### Prometheus Metrics

```
# IPFS node health (0=unhealthy, 1=healthy)
ipfs_node_health{node="localhost:5001"} 0
ipfs_node_health{node="ipfs.io"} 1

# Operations counter
ipfs_operations_total{operation="upload",status="success"} 1247
ipfs_operations_total{operation="download",status="success"} 892

# Response time histogram
ipfs_operation_duration_seconds{operation="upload"} 0.125
ipfs_operation_duration_seconds{operation="download"} 0.067

# Cache performance
ipfs_cache_hits_total 1840
ipfs_cache_misses_total 287
```

### Grafana Dashboard

Key metrics to monitor:
- Node health status and response times
- Upload/download success rates and throughput
- Cache hit rates and memory usage
- Error rates and retry statistics
- Content size distribution and storage utilization

## 🔒 Security Considerations

### Content Validation

```python
# Automatic content validation
SECURITY_SETTINGS = {
    "max_file_size": 1024 * 1024 * 1024,  # 1GB limit
    "allowed_types": ["model", "dataset", "research"],
    "virus_scanning": True,
    "integrity_verification": True,
    "access_control": True
}
```

### Access Control

- API authentication required for uploads
- Rate limiting on IPFS operations
- Content type validation
- Size limits enforcement
- Provenance tracking for audit trails

### Network Security

- IPFS API restricted to local network
- Gateway access through load balancer
- SSL/TLS encryption for external access
- Firewall rules for port access

## 🎯 FTNS Token Integration

### Rewards System

IPFS operations are integrated with PRSM's FTNS token economy:

```python
# Automatic rewards for content contribution
await ftns_service.reward_contribution(
    user_id="user123",
    contribution_type="model",
    contribution_value=1.0,  # FTNS tokens
    metadata={
        "ipfs_cid": "bafybeig...",
        "model_type": "neural_network",
        "size_mb": 50.0
    }
)

# Royalty payments for content access
royalty = await ftns_service.calculate_royalties(
    content_cid="bafybeig...",
    access_count=10
)
```

### Revenue Model

- **Upload Rewards**: Users earn FTNS for contributing models/datasets
- **Access Royalties**: Content creators earn from usage
- **Storage Incentives**: Node operators rewarded for hosting
- **Quality Bonuses**: Higher rewards for well-performing models

## 🚀 Performance Optimization

### Best Practices

1. **Content Strategy**
   - Pin important models and datasets
   - Use appropriate chunk sizes (1MB recommended)
   - Enable compression for large files
   - Implement content deduplication

2. **Network Optimization**
   - Configure multiple gateway nodes
   - Use connection pooling
   - Enable DNS caching
   - Implement circuit breakers

3. **Caching Strategy**
   - Cache metadata in Redis
   - Use content addressing for deduplication
   - Implement TTL-based expiration
   - Monitor cache hit rates

4. **Concurrency Settings**
   - Optimal concurrency: 10 operations
   - Connection pool size: 100
   - Timeout settings: 60s
   - Retry attempts: 3 with exponential backoff

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failures**
   ```
   Error: Cannot connect to IPFS node
   Solution: Check IPFS daemon status, verify port 5001 accessibility
   ```

2. **Slow Operations**
   ```
   Issue: Upload/download timeouts
   Solution: Increase timeout settings, check network connectivity
   ```

3. **High Error Rates**
   ```
   Issue: Failed operations
   Solution: Verify node health, check gateway availability
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("prsm.core.ipfs_client").setLevel(logging.DEBUG)

# Check node status
client = get_ipfs_client()
statuses = await client.get_node_status()
for status in statuses:
    print(f"Node {status['url']}: {status['healthy']}")
```

## 📋 Production Deployment

### Docker Compose

```yaml
services:
  prsm-api:
    build: .
    environment:
      - PRSM_IPFS_HOST=ipfs
      - PRSM_IPFS_PORT=5001
    depends_on:
      - ipfs
      - redis
      - postgres

  ipfs:
    image: ipfs/go-ipfs:latest
    ports:
      - "4001:4001"  # P2P
      - "5001:5001"  # API
      - "8080:8080"  # Gateway
    volumes:
      - ipfs_data:/data/ipfs
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prsm-ipfs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ipfs
  template:
    metadata:
      labels:
        app: ipfs
    spec:
      containers:
      - name: ipfs
        image: ipfs/go-ipfs:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
```

### Scaling Considerations

- **Horizontal Scaling**: Deploy multiple IPFS nodes with load balancing
- **Vertical Scaling**: Increase memory/CPU for high-throughput scenarios  
- **Storage Scaling**: Use distributed storage for large content repositories
- **Geographic Distribution**: Deploy nodes in multiple regions

## 📚 Additional Resources

### Documentation
- [IPFS Production Optimization Guide](./IPFS_PRODUCTION_OPTIMIZATION.md)
- [PRSM Architecture Documentation](./architecture.md)
- [API Documentation](../prsm/api/README.md)

### External Resources
- [IPFS Documentation](https://docs.ipfs.io/)
- [Go-IPFS GitHub](https://github.com/ipfs/go-ipfs)
- [IPFS Best Practices](https://docs.ipfs.io/concepts/usage-ideas-examples/)

### Support
- GitHub Issues: [PRSM Repository](https://github.com/Ryno2390/PRSM/issues)
- Documentation: [PRSM Docs](../docs/)
- Community: [PRSM Discord](https://discord.gg/prsm)

---

**Last Updated**: June 2025  
**Version**: 1.0  
**Status**: Production Ready ✅

**Test Results Summary**:
- ✅ Core Functionality: All tests passing
- ✅ Performance: Optimized for production workloads  
- ✅ Security: Enterprise-grade validation and access control
- ✅ Integration: Seamless with all PRSM components
- ✅ Monitoring: Comprehensive observability and alerting