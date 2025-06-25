# P2P Network API

Manage peer-to-peer networking, node discovery, and distributed communication in the PRSM network.

## 🎯 Overview

The P2P Network API enables decentralized communication between PRSM nodes, allowing for distributed model inference, data sharing, and collaborative computation without centralized coordination.

## 📋 Base URL

```
https://api.prsm.ai/v1/network
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/network
```

## 🚀 Quick Start

### Join Network

```python
import prsm

client = prsm.Client(api_key="your-api-key")

# Join the PRSM network
node = client.network.join(
    node_id="node_123",
    capabilities=["inference", "storage", "compute"],
    location="us-west-2"
)

print(f"Joined network as node: {node.id}")
```

## 📊 Endpoints

### POST /network/join
Join the PRSM peer-to-peer network.

**Request Body:**
```json
{
  "node_id": "node_abc123",
  "capabilities": ["inference", "storage", "compute"],
  "location": "us-west-2",
  "public_key": "-----BEGIN PUBLIC KEY-----...",
  "resources": {
    "cpu_cores": 8,
    "ram_gb": 32,
    "gpu_count": 2,
    "storage_gb": 1000
  },
  "metadata": {
    "version": "1.0.0",
    "client": "prsm-python-sdk"
  }
}
```

**Response:**
```json
{
  "node_id": "node_abc123",
  "network_id": "prsm_mainnet",
  "peer_count": 1247,
  "status": "connected",
  "assigned_region": "us-west-2",
  "bootstrap_peers": [
    {
      "node_id": "bootstrap_1",
      "address": "bootstrap1.prsm.ai:4001",
      "public_key": "..."
    }
  ],
  "network_config": {
    "protocol_version": "1.0",
    "heartbeat_interval": 30,
    "discovery_interval": 300
  }
}
```

### GET /network/peers
Discover and list available peers in the network.

**Query Parameters:**
- `capability`: Filter by capability (inference, storage, compute)
- `location`: Filter by geographic location
- `limit`: Maximum number of peers to return (default: 50)
- `quality_threshold`: Minimum peer quality score (0.0-1.0)

**Response:**
```json
{
  "peers": [
    {
      "node_id": "peer_xyz789",
      "capabilities": ["inference", "compute"],
      "location": "us-west-2",
      "quality_score": 0.95,
      "latency_ms": 15,
      "availability": 0.99,
      "resources": {
        "cpu_usage": 0.3,
        "ram_usage": 0.6,
        "gpu_usage": 0.2
      },
      "pricing": {
        "compute_per_hour": 0.10,
        "storage_per_gb": 0.05,
        "inference_per_1k_tokens": 0.002
      }
    }
  ],
  "total_peers": 1247,
  "network_health": 0.97
}
```

### POST /network/connect
Establish a direct connection to a specific peer.

**Request Body:**
```json
{
  "peer_id": "peer_xyz789",
  "connection_type": "direct",
  "encryption": true,
  "timeout_ms": 5000
}
```

### POST /network/broadcast
Broadcast a message to the entire network or specific peers.

**Request Body:**
```json
{
  "message_type": "inference_request",
  "payload": {
    "task_id": "task_123",
    "model": "gpt-3.5-turbo",
    "input": "Explain quantum computing"
  },
  "target_peers": ["peer_1", "peer_2"],
  "ttl": 300,
  "priority": "high"
}
```

### POST /network/route
Route a message to a specific peer through the network.

**Request Body:**
```json
{
  "destination": "peer_xyz789",
  "message": {
    "type": "inference_request",
    "payload": {...}
  },
  "routing_strategy": "shortest_path",
  "max_hops": 5
}
```

## 🔗 Connection Management

### Connection Types

**Direct Connection:**
```python
# Establish direct peer connection
connection = client.network.connect(
    peer_id="peer_xyz789",
    connection_type="direct",
    encryption=True
)
```

**Relay Connection:**
```python
# Connect through relay nodes
connection = client.network.connect(
    peer_id="peer_distant",
    connection_type="relay",
    max_hops=3
)
```

**Mesh Connection:**
```python
# Join mesh network
mesh = client.network.join_mesh(
    mesh_id="research_cluster",
    peers=["peer_1", "peer_2", "peer_3"]
)
```

## 🎛️ Node Discovery

### Capability-Based Discovery

```python
# Find peers with specific capabilities
peers = client.network.discover(
    capabilities=["inference", "gpu"],
    filters={
        "min_gpu_memory": "8GB",
        "max_latency": 100,
        "min_quality": 0.9
    }
)
```

### Geographic Discovery

```python
# Find nearby peers
local_peers = client.network.discover_nearby(
    radius_km=100,
    limit=10
)
```

### Resource-Based Discovery

```python
# Find peers with available resources
available_peers = client.network.discover_available(
    required_resources={
        "cpu_cores": 4,
        "ram_gb": 16,
        "gpu_memory_gb": 8
    }
)
```

## 💬 Messaging Protocol

### Message Types

**Inference Request:**
```python
message = {
    "type": "inference_request",
    "task_id": "task_123",
    "model": "gpt-3.5-turbo",
    "input": "Question text",
    "max_tokens": 150,
    "requester": "node_abc123"
}

client.network.send_message("peer_xyz789", message)
```

**Data Sharing:**
```python
message = {
    "type": "data_share",
    "data_id": "dataset_456",
    "chunk_hash": "sha256:...",
    "chunk_size": 1024,
    "total_chunks": 100
}
```

**Resource Availability:**
```python
message = {
    "type": "resource_update",
    "node_id": "node_abc123",
    "available_resources": {
        "cpu_cores": 6,
        "ram_gb": 24,
        "gpu_memory_gb": 6
    },
    "pricing": {
        "compute_per_hour": 0.08
    }
}
```

## 🔐 Security and Encryption

### End-to-End Encryption

```python
# Configure encryption for peer communication
client.network.configure_encryption(
    algorithm="AES-256-GCM",
    key_exchange="ECDH",
    verify_certificates=True
)
```

### Identity Verification

```python
# Verify peer identity
is_trusted = client.network.verify_peer(
    peer_id="peer_xyz789",
    public_key="...",
    certificate_chain=[...]
)
```

### Secure Channels

```python
# Establish secure channel
channel = client.network.create_secure_channel(
    peer_id="peer_xyz789",
    encryption_level="high",
    authentication_required=True
)
```

## 📊 Network Monitoring

### Network Health

```python
# Get network health metrics
health = client.network.health()
print(f"Network uptime: {health.uptime_percentage}%")
print(f"Average latency: {health.avg_latency_ms}ms")
print(f"Active peers: {health.active_peer_count}")
```

### Connection Statistics

```python
# Monitor connection performance
stats = client.network.connection_stats()
for peer_id, metrics in stats.items():
    print(f"Peer {peer_id}:")
    print(f"  Latency: {metrics.latency_ms}ms")
    print(f"  Bandwidth: {metrics.bandwidth_mbps}Mbps")
    print(f"  Packet loss: {metrics.packet_loss}%")
```

### Traffic Analysis

```python
# Analyze network traffic
traffic = client.network.traffic_analysis(
    timeframe="last_hour"
)
print(f"Messages sent: {traffic.messages_sent}")
print(f"Messages received: {traffic.messages_received}")
print(f"Bandwidth used: {traffic.bandwidth_used_mb}MB")
```

## 🌐 Network Topology

### Topology Discovery

```python
# Get network topology
topology = client.network.topology()
print(f"Network diameter: {topology.diameter}")
print(f"Clustering coefficient: {topology.clustering}")
print(f"Connected components: {topology.components}")
```

### Route Optimization

```python
# Optimize routing paths
optimized_routes = client.network.optimize_routes(
    destination_peers=["peer_1", "peer_2", "peer_3"],
    optimization_goal="latency"  # or "bandwidth", "cost"
)
```

## 🔄 Load Balancing

### Request Distribution

```python
# Distribute requests across peers
response = client.network.distribute_request(
    request_type="inference",
    payload={
        "model": "gpt-3.5-turbo",
        "input": "Question"
    },
    distribution_strategy="round_robin",  # or "random", "capability_based"
    peer_count=3
)
```

### Capacity Management

```python
# Monitor and manage peer capacity
capacity = client.network.peer_capacity()
for peer_id, usage in capacity.items():
    if usage.cpu_percentage > 0.8:
        client.network.reduce_load(peer_id)
```

## 🛠️ Fault Tolerance

### Automatic Failover

```python
# Configure automatic failover
client.network.configure_failover(
    strategy="immediate",
    backup_peer_count=2,
    health_check_interval=10
)
```

### Network Partitioning

```python
# Handle network partitions
partition_handler = client.network.partition_handler()
partition_handler.on_partition_detected(lambda: print("Network partition detected"))
partition_handler.on_partition_healed(lambda: print("Network partition healed"))
```

## 📈 Performance Optimization

### Bandwidth Optimization

```python
# Optimize bandwidth usage
client.network.optimize_bandwidth(
    compression_enabled=True,
    message_batching=True,
    adaptive_quality=True
)
```

### Latency Reduction

```python
# Reduce network latency
client.network.optimize_latency(
    connection_pooling=True,
    route_caching=True,
    predict_routing=True
)
```

## 🧪 Testing and Simulation

### Network Simulation

```python
# Simulate network conditions
simulator = client.network.simulator()
simulator.simulate_conditions(
    latency_range=(10, 500),
    packet_loss_rate=0.01,
    bandwidth_limit="100Mbps"
)
```

### Load Testing

```python
# Test network under load
load_test = client.network.load_test(
    concurrent_connections=100,
    message_rate=1000,
    duration_seconds=300
)
```

## 📚 Advanced Features

### Custom Protocols

```python
# Implement custom protocol
class CustomProtocol(prsm.NetworkProtocol):
    def handle_message(self, message):
        # Custom message handling logic
        pass
    
    def send_custom_message(self, peer_id, data):
        # Custom message sending logic
        pass

client.network.register_protocol("custom", CustomProtocol())
```

### Network Plugins

```python
# Load network plugins
plugin = client.network.load_plugin("advanced_routing")
plugin.configure({
    "algorithm": "dijkstra",
    "weight_function": "latency_bandwidth_hybrid"
})
```

## 📞 Support

- **Network Issues**: network-support@prsm.ai
- **Connection Problems**: connectivity@prsm.ai
- **Security Concerns**: security@prsm.ai
- **Performance**: performance@prsm.ai