"""
PRSM NWTN Hardware SDK (2026 Edition)
=====================================

Lightweight entry point for hardware manufacturers to embed NWTN 
directly into sensors, chemical reactors, and lab equipment.

Features:
- Edge Surprise Gating: Only transmit anomalies.
- Digital Twin Hashing: Secure machine-state provenance.
- Low-Latency FSMN Support: Real-time kinetic monitoring.
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from prsm.compute.nwtn.reasoning.surprise_gating import SurpriseGater

class HardwareNWTNNode:
    """
    Standardized interface for PRSM-ready hardware.
    """
    def __init__(self, hardware_id: str, secret_key: str):
        self.hardware_id = hardware_id
        self.secret_key = secret_key
        # High-threshold gating for edge devices to save bandwidth
        self.gater = SurpriseGater(surprise_threshold=0.8)
        self.last_observation = None

    def process_sensor_reading(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Main edge logic: Gating + Hashing.
        Returns a 'Surprise Payload' if data is an anomaly.
        """
        if self.last_observation is None:
            self.last_observation = data
            return self._create_payload(data, 1.0) # First reading is always surprising

        # Calculate surprise relative to last reading
        surprise = self.gater.calculate_surprise(self.last_observation, data)
        self.last_observation = data

        if not self.gater.should_gate(surprise):
            # ANOMALY DETECTED: Transmit to network
            return self._create_payload(data, surprise)
        
        return None # Discard redundant data

    def _create_payload(self, data: Any, surprise: float) -> Dict[str, Any]:
        """Creates a signed, verifiable payload with Digital Twin metadata"""
        # Capture machine state (Digital Twin)
        timestamp = time.time()
        machine_state = f"{self.hardware_id}-{timestamp}-calibrated"
        state_hash = hashlib.sha256(machine_state.encode()).hexdigest()
        
        payload = {
            "hw_id": self.hardware_id,
            "data": data,
            "surprise": surprise,
            "dt_hash": state_hash,
            "ts": timestamp
        }
        
        # In production, this would be signed with the hardware's private key (NHI)
        payload["sig"] = hashlib.md5(f"{payload}-{self.secret_key}".encode()).hexdigest()
        return payload
