"""
PRSM Governance System

Comprehensive governance infrastructure for community decision-making,
token-weighted voting, and proposal management.

v1.6.0 scope alignment: AI-improvement proposal governance (proposals.py,
which depended on the deleted ImprovementProposalEngine and SafetyGovernance)
has been removed. The surviving modules cover FTNS-token-weighted voting and
node-operator governance mechanics only.
"""

from .voting import TokenWeightedVoting

__all__ = [
    "TokenWeightedVoting",
]
