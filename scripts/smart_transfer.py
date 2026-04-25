#!/usr/bin/env python3
"""
Smart Transfer Script
===================

Intelligent file transfer that works around the external drive performance issues
by using rsync with timeouts and resumption capabilities.
"""

import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ryneschultz/Documents/GitHub/PRSM/transfer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_rsync_with_timeout(source: str, dest: str, timeout: int = 3600) -> bool:
    """Run rsync with timeout and progress monitoring"""
    cmd = [
        'rsync', '-avh', '--progress', '--partial', '--timeout=30',
        source, dest
    ]
    
    try:
        logger.info(f"🔄 Starting transfer: {source} -> {dest}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                logger.info(f"✅ Transfer completed successfully")
                return True
            else:
                logger.error(f"❌ Transfer failed: {stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"⏱️ Transfer timed out after {timeout}s, terminating...")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            return False
            
    except Exception as e:
        logger.error(f"❌ Transfer error: {e}")
        return False

def main():
    """Main transfer execution"""
    logger.info("🚀 Starting smart transfer from external drive to local storage")
    
    # Define transfer tasks
    transfers = [
        {
            'source': '/Volumes/My Passport/PRSM_Storage/03_NWTN_READY/embeddings/',
            'dest': '/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings/',
            'description': 'NWTN embeddings (49,631 files)'
        },
        {
            'source': '/Volumes/My Passport/PRSM_Storage/03_NWTN_READY/content_hashes/',
            'dest': '/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/content_hashes/',
            'description': 'Content hashes (49,631 files)'
        },
        {
            'source': '/Volumes/My Passport/PRSM_Storage/03_NWTN_READY/provenance/',
            'dest': '/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/provenance/',
            'description': 'Provenance records (49,631 files)'
        }
    ]
    
    # Execute transfers with retry logic
    for i, transfer in enumerate(transfers, 1):
        logger.info(f"📦 Transfer {i}/3: {transfer['description']}")
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info(f"🔄 Attempt {attempt}/{max_retries}")
            
            if run_rsync_with_timeout(transfer['source'], transfer['dest'], timeout=1800):  # 30 min timeout
                logger.info(f"✅ Transfer {i} completed successfully")
                break
            else:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Attempt {attempt} failed, retrying in 30 seconds...")
                    time.sleep(30)
                else:
                    logger.error(f"❌ Transfer {i} failed after {max_retries} attempts")
    
    logger.info("🎉 Smart transfer process completed")

if __name__ == "__main__":
    main()