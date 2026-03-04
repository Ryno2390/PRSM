"""
PRSM Web Dashboard
==================

Browser-based dashboard for researchers to interact with PRSM.
Provides real-time node status, job submission, transaction history,
and FTNS balance/staking views.
"""

from prsm.dashboard.app import DashboardServer, create_dashboard_app

__all__ = ["DashboardServer", "create_dashboard_app"]
