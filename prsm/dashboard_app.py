"""
PRSM Dashboard
==============

Streamlit dashboard for monitoring PRSM node status.
Launch with: prsm dashboard

For full Streamlit UI, install streamlit: pip install streamlit
This module also provides a CLI fallback via show_ring_status().

Ring 1-10 status is also available at the /rings/status API endpoint.
"""


def show_ring_status():
    """Display Ring 1-10 status (works without Streamlit for CLI fallback)."""
    try:
        from prsm.observability.dashboard_metrics import DashboardMetrics
        metrics = DashboardMetrics()
        summary = metrics.get_summary()

        print(f"PRSM Node Dashboard")
        print(f"  Rings: {summary['rings_initialized']}/10 initialized")
        print()
        for ring in summary.get("rings", []):
            status = "OK" if ring["initialized"] else "--"
            print(f"  [{status}] Ring {ring['ring']}: {ring['name']}")

        pricing = summary.get("pricing", {})
        if pricing:
            print(f"\n  Spot: {pricing.get('spot_multiplier', '1.0')}x | Util: {pricing.get('utilization', 0):.0%}")

        forge = summary.get("forge", {})
        if forge:
            print(f"  Traces: {forge.get('traces_collected', 0)}")

    except Exception as e:
        print(f"Dashboard error: {e}")


if __name__ == "__main__":
    show_ring_status()
