import hashlib
import json
from typing import Dict, Any
from datetime import datetime

def generate_digital_twin_hash(metadata: Dict[str, Any]) -> str:
    """
    Generates a SHA-256 hash from hardware sensor metadata for verifiable discovery trails.
    
    This creates a cryptographic binding between the physical hardware state
    and the digital validation report, preventing 'Potemkin' simulations.
    
    Args:
        metadata: Dictionary containing hardware sensor metadata. 
                  Must include 'uuid', 'timestamp', and 'calibration_state'.
    
    Returns:
        str: Hexadecimal string of the SHA-256 hash.
        
    Raises:
        ValueError: If required keys are missing from metadata.
    """
    required_keys = ['uuid', 'timestamp', 'calibration_state']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: {key}")

    # Ensure consistent ordering and serialization for deterministic hashing
    processed_metadata = {}
    for k, v in metadata.items():
        # Handle datetime objects standardizing to ISO format
        if isinstance(v, datetime):
            processed_metadata[k] = v.isoformat()
        elif hasattr(v, "isoformat"): # Handle other date-like objects
             processed_metadata[k] = v.isoformat()
        else:
            processed_metadata[k] = v

    # Serialize to JSON with sorted keys to ensure the hash is deterministic
    # separators=(',', ':') removes whitespace to make it compact and consistent
    serialized_data = json.dumps(processed_metadata, sort_keys=True, separators=(',', ':'))
    
    # Generate SHA-256 Hash
    return hashlib.sha256(serialized_data.encode('utf-8')).hexdigest()
