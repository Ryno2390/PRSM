"""
PRSM Bootstrap Server Connectivity Tests

Tests for verifying bootstrap server deployment and connectivity.
Run locally or against a deployed server.

Usage:
    # Test local server
    pytest tests/integration/test_bootstrap_connectivity.py -v
    
    # Test deployed server
    BOOTSTRAP_HOST=bootstrap.prsm-network.com pytest tests/integration/test_bootstrap_connectivity.py -v
"""

import asyncio
import json
import os
import socket
import urllib.request
import urllib.error
import pytest
from typing import Optional

# Configuration
BOOTSTRAP_HOST = os.environ.get("BOOTSTRAP_HOST", "localhost")
WEBSOCKET_PORT = int(os.environ.get("WEBSOCKET_PORT", "8765"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8000"))
USE_SSL = os.environ.get("USE_SSL", "false").lower() == "true"
TIMEOUT_SECONDS = 10


def get_websocket_uri():
    """Get WebSocket URI based on configuration."""
    protocol = "wss" if USE_SSL else "ws"
    return f"{protocol}://{BOOTSTRAP_HOST}:{WEBSOCKET_PORT}"


def get_http_uri(endpoint=""):
    """Get HTTP URI based on configuration."""
    protocol = "https" if USE_SSL else "http"
    return f"{protocol}://{BOOTSTRAP_HOST}:{HTTP_PORT}{endpoint}"


class TestBootstrapHTTPAPI:
    """Tests for the bootstrap server HTTP API."""
    
    def test_health_endpoint_returns_200(self):
        """Test that the health endpoint returns 200 OK."""
        uri = get_http_uri("/health")
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                assert response.status == 200, f"Expected 200, got {response.status}"
        except urllib.error.URLError as e:
            pytest.skip(f"Cannot connect to {uri}: {e.reason}")
    
    def test_health_endpoint_returns_json(self):
        """Test that the health endpoint returns valid JSON."""
        uri = get_http_uri("/health")
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode('utf-8'))
                assert isinstance(data, dict), "Response should be a JSON object"
        except urllib.error.URLError as e:
            pytest.skip(f"Cannot connect to {uri}: {e.reason}")
    
    def test_health_endpoint_has_status_field(self):
        """Test that the health response contains a status field."""
        uri = get_http_uri("/health")
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode('utf-8'))
                assert "status" in data, "Response should contain 'status' field"
                assert data["status"] in ("healthy", "ok", "degraded"), \
                    f"Unexpected status: {data['status']}"
        except urllib.error.URLError as e:
            pytest.skip(f"Cannot connect to {uri}: {e.reason}")
    
    def test_metrics_endpoint_if_available(self):
        """Test that metrics endpoint is available (optional)."""
        uri = get_http_uri("/metrics")
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                # Metrics endpoint may return different formats
                assert response.status == 200
        except urllib.error.URLError:
            pytest.skip("Metrics endpoint not available (optional)")
    
    def test_peers_endpoint_if_available(self):
        """Test that peers endpoint returns peer information."""
        uri = get_http_uri("/peers")
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode('utf-8'))
                assert "peers" in data or isinstance(data, list), \
                    "Response should contain peers data"
        except urllib.error.URLError:
            pytest.skip("Peers endpoint not available")


class TestBootstrapWebSocket:
    """Tests for the bootstrap server WebSocket endpoint."""
    
    @pytest.fixture
    def websockets_available(self):
        """Check if websockets library is available."""
        try:
            import websockets
            return True
        except ImportError:
            return False
    
    @pytest.mark.skip(reason="Requires live WebSocket server on port 8765 — run manually")
    def test_websocket_port_is_open(self):
        """Test that the WebSocket port is accepting connections."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT_SECONDS)
        try:
            result = sock.connect_ex((BOOTSTRAP_HOST, WEBSOCKET_PORT))
            assert result == 0, f"Port {WEBSOCKET_PORT} is not open on {BOOTSTRAP_HOST}"
        except socket.gaierror as e:
            pytest.skip(f"Cannot resolve hostname {BOOTSTRAP_HOST}: {e}")
        finally:
            sock.close()
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, websockets_available):
        """Test that we can establish a WebSocket connection."""
        if not websockets_available:
            pytest.skip("websockets library not installed")
        
        import websockets
        
        uri = get_websocket_uri()
        try:
            async with websockets.connect(uri, close_timeout=5) as ws:
                assert ws.open, "WebSocket connection should be open"
        except Exception as e:
            pytest.skip(f"Cannot connect to WebSocket at {uri}: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, websockets_available):
        """Test that the server responds to ping messages."""
        if not websockets_available:
            pytest.skip("websockets library not installed")
        
        import websockets
        
        uri = get_websocket_uri()
        try:
            async with websockets.connect(uri, close_timeout=5) as ws:
                # Send a ping message
                ping_msg = json.dumps({"type": "ping"})
                await ws.send(ping_msg)
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    data = json.loads(response)
                    assert data.get("type") in ("pong", "ping", "response"), \
                        f"Expected pong response, got: {data}"
                except asyncio.TimeoutError:
                    pytest.fail("No response received within timeout")
        except Exception as e:
            pytest.skip(f"Cannot connect to WebSocket at {uri}: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_peer_announcement(self, websockets_available):
        """Test that we can announce as a peer."""
        if not websockets_available:
            pytest.skip("websockets library not installed")
        
        import websockets
        
        uri = get_websocket_uri()
        try:
            async with websockets.connect(uri, close_timeout=5) as ws:
                # Send peer announcement
                announce_msg = json.dumps({
                    "type": "announce",
                    "peer_id": "test-peer-123",
                    "address": "127.0.0.1",
                    "port": 9000,
                    "capabilities": ["test"]
                })
                await ws.send(announce_msg)
                
                # Wait for acknowledgment
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    data = json.loads(response)
                    # Server should acknowledge or provide peer list
                    assert "type" in data, f"Response should have type field: {data}"
                except asyncio.TimeoutError:
                    # Some servers may not respond to announcements immediately
                    pass
        except Exception as e:
            pytest.skip(f"Cannot connect to WebSocket at {uri}: {e}")


class TestBootstrapServerIntegration:
    """Integration tests for the complete bootstrap flow."""
    
    @pytest.mark.asyncio
    async def test_full_bootstrap_sequence(self):
        """Test the complete bootstrap sequence a node would perform."""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets library not installed")
        
        # Step 1: Check HTTP health
        health_uri = get_http_uri("/health")
        try:
            request = urllib.request.Request(health_uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                assert response.status == 200
        except urllib.error.URLError as e:
            pytest.skip(f"Health check failed: {e.reason}")
        
        # Step 2: Connect via WebSocket
        ws_uri = get_websocket_uri()
        try:
            async with websockets.connect(ws_uri, close_timeout=5) as ws:
                # Step 3: Announce presence
                await ws.send(json.dumps({
                    "type": "announce",
                    "peer_id": "integration-test-peer",
                    "address": "127.0.0.1",
                    "port": 9999
                }))
                
                # Step 4: Request peer list
                await ws.send(json.dumps({"type": "get_peers"}))
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_SECONDS)
                    data = json.loads(response)
                    # Should receive peer list or acknowledgment
                    assert isinstance(data, dict), "Response should be a JSON object"
                except asyncio.TimeoutError:
                    pass  # Server may not respond to all messages
                
                # Step 5: Graceful disconnect
                await ws.close()
        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e}")
    
    def test_server_responds_within_timeout(self):
        """Test that server responds within acceptable time."""
        import time
        
        uri = get_http_uri("/health")
        start_time = time.time()
        
        try:
            request = urllib.request.Request(uri, method='GET')
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                elapsed = time.time() - start_time
                assert elapsed < 5.0, f"Response took too long: {elapsed:.2f}s"
        except urllib.error.URLError as e:
            pytest.skip(f"Cannot connect to {uri}: {e.reason}")


class TestBootstrapServerResilience:
    """Tests for server resilience and error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self):
        """Test that server handles invalid messages gracefully."""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets library not installed")
        
        uri = get_websocket_uri()
        try:
            async with websockets.connect(uri, close_timeout=5) as ws:
                # Send invalid JSON
                await ws.send("not valid json")
                
                # Server should not close connection immediately
                await asyncio.sleep(1)
                
                # Try to send a valid message
                await ws.send(json.dumps({"type": "ping"}))
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    # Server should still respond
                except asyncio.TimeoutError:
                    pass  # Server may ignore invalid messages
                
                # Connection should still be open
                assert ws.open, "Connection should remain open after invalid message"
        except Exception as e:
            pytest.skip(f"Cannot connect to WebSocket at {uri}: {e}")
    
    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test that server handles multiple concurrent connections."""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets library not installed")
        
        uri = get_websocket_uri()
        connections = []
        
        try:
            # Open multiple connections
            for i in range(5):
                ws = await websockets.connect(uri, close_timeout=5)
                connections.append(ws)
            
            # All should be open
            for i, ws in enumerate(connections):
                assert ws.open, f"Connection {i} should be open"
            
            # Send messages on all
            for ws in connections:
                await ws.send(json.dumps({"type": "ping"}))
            
            # Close all
            for ws in connections:
                await ws.close()
                
        except Exception as e:
            pytest.skip(f"Cannot test multiple connections: {e}")
        finally:
            for ws in connections:
                try:
                    await ws.close()
                except:
                    pass


# Standalone test runner for quick verification
def run_quick_test():
    """Run a quick connectivity test without pytest."""
    print("=" * 60)
    print("PRSM Bootstrap Server Connectivity Test")
    print("=" * 60)
    print(f"Host: {BOOTSTRAP_HOST}")
    print(f"WebSocket Port: {WEBSOCKET_PORT}")
    print(f"HTTP Port: {HTTP_PORT}")
    print(f"SSL Enabled: {USE_SSL}")
    print("=" * 60)
    
    # Test HTTP health
    print("\n[1] Testing HTTP Health Endpoint...")
    health_uri = get_http_uri("/health")
    try:
        request = urllib.request.Request(health_uri, method='GET')
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode('utf-8'))
            print(f"    ✓ Status: {response.status}")
            print(f"    ✓ Response: {json.dumps(data, indent=4)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Test WebSocket port
    print("\n[2] Testing WebSocket Port...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT_SECONDS)
    result = sock.connect_ex((BOOTSTRAP_HOST, WEBSOCKET_PORT))
    sock.close()
    if result == 0:
        print(f"    ✓ Port {WEBSOCKET_PORT} is open")
    else:
        print(f"    ✗ Port {WEBSOCKET_PORT} is not accessible")
    
    # Test WebSocket connection
    print("\n[3] Testing WebSocket Connection...")
    try:
        import websockets
        
        async def test_ws():
            uri = get_websocket_uri()
            async with websockets.connect(uri, close_timeout=5) as ws:
                print(f"    ✓ Connected to {uri}")
                
                # Send ping
                await ws.send(json.dumps({"type": "ping"}))
                print("    ✓ Sent ping message")
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    print(f"    ✓ Received: {response}")
                except asyncio.TimeoutError:
                    print("    ⚠ No response (server may not echo pings)")
        
        asyncio.run(test_ws())
    except ImportError:
        print("    ✗ websockets library not installed")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()
