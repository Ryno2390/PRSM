#!/usr/bin/env python3
"""
PRSM Bootstrap Server Health Check

This script is used by Docker healthcheck to verify the bootstrap server is healthy.
It checks both the WebSocket endpoint and the HTTP API.
"""

import sys
import json
import urllib.request
import urllib.error
import socket

# Configuration
HTTP_PORT = 8000
WEBSOCKET_PORT = 8765
HEALTH_ENDPOINT = f"http://localhost:{HTTP_PORT}/health"
TIMEOUT_SECONDS = 5


def check_http_health():
    """Check the HTTP health endpoint."""
    try:
        request = urllib.request.Request(HEALTH_ENDPOINT, method='GET')
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                # Check if status is healthy
                if data.get('status') in ('healthy', 'ok'):
                    return True, "HTTP health check passed"
                return False, f"Unhealthy status: {data.get('status')}"
            return False, f"Unexpected status code: {response.status}"
    except urllib.error.URLError as e:
        return False, f"HTTP connection failed: {e.reason}"
    except json.JSONDecodeError:
        return False, "Invalid JSON response from health endpoint"
    except Exception as e:
        return False, f"HTTP health check error: {str(e)}"


def check_port_listening(port):
    """Check if a port is listening."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(TIMEOUT_SECONDS)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            return True, f"Port {port} is listening"
        return False, f"Port {port} is not responding"
    except socket.error as e:
        return False, f"Port check error: {str(e)}"


def main():
    """Run all health checks."""
    checks = []
    all_passed = True
    
    # Check HTTP API health endpoint
    http_passed, http_message = check_http_health()
    checks.append(('HTTP API', http_passed, http_message))
    if not http_passed:
        all_passed = False
    
    # Check WebSocket port is listening
    ws_passed, ws_message = check_port_listening(WEBSOCKET_PORT)
    checks.append(('WebSocket Port', ws_passed, ws_message))
    if not ws_passed:
        all_passed = False
    
    # Check HTTP API port is listening
    api_passed, api_message = check_port_listening(HTTP_PORT)
    checks.append(('HTTP API Port', api_passed, api_message))
    if not api_passed:
        all_passed = False
    
    # Print results
    print("PRSM Bootstrap Server Health Check")
    print("=" * 40)
    for name, passed, message in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status} - {message}")
    
    print("=" * 40)
    if all_passed:
        print("Overall Status: HEALTHY")
        sys.exit(0)
    else:
        print("Overall Status: UNHEALTHY")
        sys.exit(1)


if __name__ == "__main__":
    main()
