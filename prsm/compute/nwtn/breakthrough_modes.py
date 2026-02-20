#!/usr/bin/env python3
"""
NWTN Breakthrough Modes
=======================

Breakthrough mode definitions for NWTN reasoning system.

Modes:
- CONSERVATIVE: Established consensus, proven approaches, high confidence threshold
- BALANCED: Balanced approach between conservative and revolutionary
- REVOLUTIONARY: Novel connections, speculative breakthroughs, lower confidence threshold
"""

from enum import Enum


class BreakthroughMode(str, Enum):
    """Modes for breakthrough reasoning"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    REVOLUTIONARY = "revolutionary"


__all__ = ["BreakthroughMode"]
