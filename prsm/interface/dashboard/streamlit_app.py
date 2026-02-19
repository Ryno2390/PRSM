import streamlit as st
import requests
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
            color: #e0e0e0;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #2a2a4e;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #00ff88;
        }
        .metric-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stMetric {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #1a1a2e;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00ff88;
            color: #000;
        }
        div[data-testid="stExpander"] {
            background-color: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


class PRSMClient:
    """Client for interacting with PRSM node API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 5
    
    def _get(self, endpoint: str) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
            return resp.json()
        except Exception:
            return None
    
    def get_status(self) -> dict | None:
        return self._get("/status")
    
    def get_peers(self) -> dict | None:
        return self._get("/peers")
    
    def get_balance(self) -> dict | None:
        return self._get("/balance")
    
    def get_transactions(self, limit: int = 20) -> dict | None:
        return self._get(f"/transactions?limit={limit}")
    
    def get_storage_stats(self) -> dict | None:
        return self._get("/storage/stats")
    
    def get_compute_stats(self) -> dict | None:
        return self._get("/compute/stats")
    
    def get_agents(self) -> dict | None:
        return self._get("/agents")
    
    def get_content_search(self, query: str, limit: int = 10) -> dict | None:
        return self._get(f"/content/search?q={query}&limit={limit}")
    
    def health_check(self) -> bool:
        return self._get("/health") is not None


def render_mockup_viewer():
    """Render the full UI mockup in an iframe."""
    st.subheader("🎨 PRSM UI Mockup")
    
    if not (MOCKUP_DIR / "index.html").exists():
        st.error(f"Mockup file not found at {MOCKUP_DIR}")
        return
    
    # Read and display the mockup HTML
    html_path = MOCKUP_DIR / "index.html"
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.markdown(f"📄 Loading mockup from `{html_path.name}` ({len(html_content)//1024}KB)...")
        
        # Render in an iframe
        st.components.v1.html(
            html_content, 
            height=2000, 
            scrolling=True
        )
    except Exception as e:
        st.error(f"Error loading mockup: {e}")


def render_live_dashboard():
    """Render the live dashboard with real node data."""
    client = PRSMClient(API_BASE_URL)
    
    status = client.get_status()
    
    if not status:
        st.warning("⚠️ Could not connect to PRSM node. Showing mockup instead.")
        render_mockup_viewer()
        return
    
    # Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        is_online = status.get('started', False)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Node Status</div>
                <div class="metric-value" style="color: {'#00ff88' if is_online else '#ff4444'}">
                    {'● Online' if is_online else '○ Offline'}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        node_id = status.get('node_id', 'N/A')
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Node ID</div>
                <div class="metric-value" style="font-size: 14px; word-break: break-all;">
                    {node_id[:20]}...
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
                <div class="metric-value" style="font-size: 20px;">
                    {hours}h {minutes}m
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ledger_type = status.get('ledger_type', 'legacy')
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ledger</div>
                <div class="metric-value" style="font-size: 18px; color: #00bfff;">
                    {ledger_type.upper()}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Network", 
        "💰 Wallet", 
        "💻 Compute", 
        "💾 Storage",
        "🔍 Content"
    ])
    
    with tab1:
        st.subheader("🌐 Network Peers")
        peers = client.get_peers()
        
        if peers:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Connected", peers.get('connected_count', 0))
            with col2:
                st.metric("Known", peers.get('known_count', 0))
            
            if peers.get('connected'):
                with st.expander("🔗 Connected Peers", expanded=True):
                    for peer in peers['connected']:
                        st.write(f"• **{peer.get('display_name', 'Unknown')}** - {peer.get('address', 'N/A')[:40]}")
        else:
            st.info("No peer data available.")
    
    with tab2:
        st.subheader("💰 FTNS Token Balance")
        balance = client.get_balance()
        
        if balance:
            st.metric("Balance", f"{balance.get('balance', 0):.4f} FTNS")
            
            # Transaction history
            st.subheader("📜 Transaction History")
            txs = client.get_transactions(limit=10)
            
            if txs and txs.get('transactions'):
                for tx in txs['transactions']:
                    with st.expander(f"{tx['type']}: {tx['amount']:.4f} FTNS"):
                        st.write(f"**From:** {tx.get('from', 'N/A')[:20]}...")
                        st.write(f"**To:** {tx.get('to', 'N/A')[:20]}...")
                        ts = tx.get('timestamp', 0)
                        st.write(f"**Time:** {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")
            else:
                st.info("No transactions yet.")
            
            # DAG Stats
            dag_stats = status.get('dag_stats')
            if dag_stats:
                st.subheader("🔗 DAG Ledger Stats")
                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                with col_d1:
                    st.metric("Total Txs", dag_stats.get('total_transactions', 0))
                with col_d2:
                    st.metric("Tips", dag_stats.get('tips', 0))
                with col_d3:
                    st.metric("Confirmed", dag_stats.get('confirmed', 0))
                with col_d4:
                    confirm_level = dag_stats.get('avg_confirmation_level', 0)
                    st.metric("Avg Confirm", f"{confirm_level:.1%}")
        else:
            st.info("No balance data.")
    
    with tab3:
        st.subheader("💻 Compute Network")
        compute_stats = status.get('compute')
        compute_requester_stats = status.get('compute_requester')
        
        if compute_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jobs Completed", compute_stats.get('jobs_completed', 0))
            with col2:
                st.metric("Jobs In Queue", compute_stats.get('jobs_queued', 0))
            with col3:
                st.metric("Active Workers", compute_stats.get('active_workers', 0))
        else:
            st.info("Compute provider not initialized.")
    
    with tab4:
        st.subheader("💾 Storage Network")
        storage_stats = status.get('storage')
        
        if storage_stats:
            pledged = storage_stats.get('pledged_gb', 0)
            used = storage_stats.get('used_gb', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pledged", f"{pledged:.2f} GB")
            with col2:
                st.metric("Used", f"{used:.2f} GB")
            with col3:
                st.metric("Pinned", storage_stats.get('pinned_count', 0))
            
            if pledged > 0:
                st.progress(min(used / pledged, 1.0), "Storage Usage")
        else:
            st.info("Storage provider not initialized.")
        
        content_stats = status.get('content_index')
        if content_stats:
            st.subheader("📄 Content Index")
            st.metric("Indexed Content", content_stats.get('total_cids', 0))
    
    with tab5:
        st.subheader("🔍 Content Search")
        query = st.text_input("Search for content", placeholder="Enter keywords...")
        
        if query:
            results = client.get_content_search(query, limit=10)
            
            if results and results.get('results'):
                st.write(f"Found **{results.get('count', 0)}** results")
                
                for result in results['results']:
                    with st.expander(f"📄 {result.get('filename', 'Unknown')}"):
                        st.write(f"**CID:** {result.get('cid', 'N/A')[:30]}...")
                        st.write(f"**Size:** {result.get('size_bytes', 0) / 1024:.1f} KB")
            else:
                st.info("No content found.")
    
    # Agents section
    with st.expander("🤖 Agent Registry"):
        agents = client.get_agents()
        
        if agents and agents.get('agents'):
            st.write(f"**{agents.get('count', 0)}** registered agents")
            
            for agent in agents['agents']:
                with st.container():
                    st.write(f"• **{agent.get('agent_name', 'Unknown')}** - {agent.get('status', 'N/A')}")
        else:
            st.info("No agents registered.")
    
    # Footer
    st.divider()
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    with col_f2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Auto-refresh
    time.sleep(30)
    st.rerun()


def main():
    """Main dashboard application."""
    st.title("🧠 PRSM Dashboard")
    st.caption("Protocol for Recursive Scientific Modeling")
    
    # Sidebar with mode selection
    with st.sidebar:
        st.header("⚙️ Dashboard Mode")
        
        mode = st.radio(
            "Select Mode:",
            ["🟢 Live Node", "🎨 Mockup Preview"],
            help="Live Node requires a running PRSM node. Mockup shows the UI design."
        )
        
        st.divider()
        
        st.header("🔗 Connection")
        st.write(f"**API URL:** `{API_BASE_URL}`")
        
        # Check connection
        client = PRSMClient(API_BASE_URL)
        if client.health_check():
            st.success("✅ API is responding")
        else:
            st.warning("⚠️ API not responding")
        
        st.divider()
        
        st.header("📋 Quick Links")
        st.markdown("""
        - [GitHub](https://github.com/Ryno2390/PRSM)
        - [Documentation](https://prsm.readthedocs.io)
        """)
    
    # Render based on mode selection
    if mode == "🎨 Mockup Preview":
        render_mockup_viewer()
    else:
        render_live_dashboard()


if __name__ == "__main__":
    main()
