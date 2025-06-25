"""
SEAL (Safety Enhanced AI Learning) Package

Advanced safety and alignment framework for RLT teachers,
implementing SEAL protocols for enhanced safety and reliability.
"""

from .seal_rlt_enhanced_teacher import SEALRLTEnhancedTeacher, SafetyProtocol, TeachingSafeguards

__all__ = [
    'SEALRLTEnhancedTeacher',
    'SafetyProtocol',
    'TeachingSafeguards'
]