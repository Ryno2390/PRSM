import streamlit as st
import requests
import base64
import json
import time
from pathlib import Path
from datetime import datetime

PRSM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MOCKUP_DIR = PRSM_ROOT / "PRSM_ui_mockup"
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PRSM | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background-color: #0a0a0f;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #2a2a4e;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #00ff88;
        }
        .metric-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
        }
        .status-online {
            color: #00ff88;
        }
        .status-offline {
            color: #ff4444;
        }
    </style>
""", unsafe_allow_html=True)

class PRSMClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 5
    
    def get_status(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return resp.json()
        except:
            return None
    
    def get_peers(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/peers", timeout=self.timeout)
            return resp.json()
        except:
            return None
    
    def get_balance(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/balance", timeout=self.timeout)
            return resp.json()
        except:
            return None
    
    def get_storage(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/storage/stats", timeout=self.timeout)
            return resp.json()
        except:
            return None
    
    def get_compute(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/compute/stats", timeout=self.timeout)
            return resp.json()
        except:
            return None


def render_header(status: dict):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Node Status</div>
                <div class="metric-value" style="color: {'#00ff88' if status.get('started') else '#ff4444'}">
                    {'● Online' if status.get('started') else '○ Offline'}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Node ID</div>
                <div class="metric-value" style="font-size: 16px;">
                    {status.get('node_id', 'N/A')[:16]}...
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        uptime = status.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Uptime</div>
                <div class="metric-value" style="font-size: 24px;">
                    {hours}h {minutes}m
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Display Name</div>
                <div class="metric-value" style="font-size: 20px;">
                    {status.get('display_name', 'Unknown')}
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_peers(peers: dict):
    st.subheader("🌐 Network Peers")
    
    col1, col2 = st.columns(2)
    with col1:
        connected = peers.get('connected_count', 0) if peers else 0
        st.metric("Connected Peers", connected)
    with col2:
        known = peers.get('known_count', 0) if peers else 0
        st.metric("Known Peers", known)
    
    if peers and peers.get('connected'):
        with st.expander("View Connected Peers"):
            for peer in peers['connected'][:10]:
                st.text(f"{peer.get('display_name', 'Unknown')} - {peer.get('address', 'N/A')}")


def render_balance(balance: dict):
    st.subheader("💰 FTNS Token Balance")
    
    if balance:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Balance", f"{balance.get('balance', 0):.4f} FTNS")
        with col2:
            st.metric("Wallet ID", balance.get('wallet_id', 'N/A')[:20] + "...")
        
        if balance.get('recent_transactions'):
            with st.expander("Recent Transactions"):
                for tx in balance['recent_transactions'][:5]:
                    st.text(f"{tx['type']}: {tx['amount']:.4f} FTNS - {tx['description'][:40]}")
    else:
        st.info("No balance data available. Connect to the network to see your FTNS balance.")


def render_compute(compute_stats: dict, compute_requester_stats: dict):
    st.subheader("💻 Compute")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if compute_stats:
            st.metric("Jobs Completed", compute_stats.get('jobs_completed', 0))
        else:
            st.metric("Jobs Completed", "N/A")
    
    with col2:
        if compute_stats:
            st.metric("Jobs In Queue", compute_stats.get('jobs_queued', 0))
        else:
            st.metric("Jobs In Queue", "N/A")
    
    with col3:
        if compute_requester_stats:
            st.metric("Jobs Submitted", compute_requester_stats.get('jobs_submitted', 0))
        else:
            st.metric("Jobs Submitted", "N/A")


def render_storage(storage_stats: dict):
    st.subheader("💾 Storage")
    
    if storage_stats:
        col1, col2, col3 = st.columns(3)
        
        pledged = storage_stats.get('pledged_gb', 0)
        used = storage_stats.get('used_gb', 0)
        
        with col1:
            st.metric("Pledged Storage", f"{pledged:.2f} GB")
        with col2:
            st.metric("Used Storage", f"{used:.2f} GB")
        with col3:
            st.metric("Pinned Content", storage_stats.get('pinned_count', 0))
        
        st.progress(used / pledged if pledged > 0 else 0, "Storage Usage")
    else:
        st.info("No storage provider initialized. Enable storage role to contribute storage.")


def render_content(content_stats: dict):
    st.subheader("📄 Content Index")
    
    if content_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Content", content_stats.get('total_cids', 0))
        with col2:
            st.metric("Storage Providers", content_stats.get('provider_count', 0))
    else:
        st.info("No content indexed yet.")


def render_fallback():
    st.warning("⚠️ PRSM Node is not running. Start your node to see live data.")
    st.info("Run: `python -m prsm.cli start` to start the PRSM node")
    
    if (MOCKUP_DIR / "index.html").exists():
        with st.expander("View UI Mockup (Static)"):
            with open(MOCKUP_DIR / "index.html", 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=True)


def main():
    st.title("🧠 PRSM Dashboard")
    st.caption("Protocol for Recursive Scientific Modeling")
    
    client = PRSMClient(API_BASE_URL)
    
    status = client.get_status()
    
    if status:
        render_header(status)
        
        peers = client.get_peers()
        balance = client.get_balance()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Network", "Wallet", "Compute", "Storage"])
        
        with tab1:
            render_peers(peers)
        
        with tab2:
            render_balance(balance)
        
        with tab3:
            compute_stats = status.get('compute')
            compute_requester_stats = status.get('compute_requester')
            render_compute(compute_stats, compute_requester_stats)
        
        with tab4:
            storage_stats = status.get('storage')
            render_storage(storage_stats)
        
        content_stats = status.get('content')
        with st.expander("Content Index Details"):
            render_content(content_stats)
        
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        time.sleep(30)
        st.rerun()
    else:
        render_fallback()


if __name__ == "__main__":
    main()
