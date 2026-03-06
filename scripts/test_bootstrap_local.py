#!/usr/bin/env python3
"""
PRSM Bootstrap Server Local Test

Starts a local bootstrap server for testing without Docker.
This is useful for development and testing the bootstrap functionality.

Usage:
    python scripts/test_bootstrap_local.py
    
    # In another terminal, test connectivity:
    python tests/integration/test_bootstrap_connectivity.py
"""

import asyncio
import os
import sys
import tempfile
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def run_local_bootstrap_server():
    """Run a local bootstrap server for testing."""
    from prsm.bootstrap.server import BootstrapServer
    from prsm.bootstrap.config import BootstrapConfig
    
    # Create temporary directory for peer data
    temp_dir = tempfile.mkdtemp(prefix="prsm_bootstrap_")
    peer_db_path = os.path.join(temp_dir, "peers.json")
    
    logger.info(f"Using temporary directory: {temp_dir}")
    
    # Create configuration for local testing
    config = BootstrapConfig(
        host="0.0.0.0",
        port=8765,
        api_port=8000,
        ssl_enabled=False,  # No SSL for local testing
        ssl_cert_path=None,
        ssl_key_path=None,
        max_peers=100,
        peer_timeout=300,
        heartbeat_interval=30,
        peer_db_path=peer_db_path,
        log_level="DEBUG",
        metrics_enabled=True,
        region="local",
        federation_peers=[],
    )
    
    logger.info("=" * 60)
    logger.info("PRSM Bootstrap Server - Local Test Mode")
    logger.info("=" * 60)
    logger.info(f"WebSocket: ws://localhost:{config.port}")
    logger.info(f"HTTP API:  http://localhost:{config.api_port}")
    logger.info(f"Health:    http://localhost:{config.api_port}/health")
    logger.info(f"Metrics:   http://localhost:{config.api_port}/metrics")
    logger.info(f"Peers:     http://localhost:{config.api_port}/peers")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Create and start server
    server = BootstrapServer(config)
    
    try:
        await server.start()
        logger.info("Bootstrap server started successfully")
        
        # Keep running until interrupted
        while server.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop()
        logger.info("Bootstrap server stopped")
        
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not cleanup temp directory: {e}")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("PRSM Bootstrap Server - Local Test")
    print("=" * 60)
    print("\nThis script starts a local bootstrap server for testing.")
    print("No SSL is used, and peer data is stored in a temp directory.")
    print("\nTo test connectivity:")
    print("  curl http://localhost:8000/health")
    print("\nOr run the connectivity tests:")
    print("  python tests/integration/test_bootstrap_connectivity.py")
    print("\n" + "=" * 60 + "\n")
    
    try:
        asyncio.run(run_local_bootstrap_server())
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
