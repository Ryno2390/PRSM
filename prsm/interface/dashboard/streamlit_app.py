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


def render_header(status: dict):
    """Render the header with node status."""
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


def render_network_tab(client: PRSMClient):
    """Render the network/peers tab."""
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
                    with st.container():
                        col_p1, col_p2 = st.columns([3, 2])
                        with col_p1:
                            st.write(f"**{peer.get('display_name', 'Unknown')}**")
                        with col_p2:
                            st.caption(peer.get('address', 'N/A')[:40])
        
        if peers.get('known'):
            with st.expander("📋 Known Peers"):
                for peer in peers['known']:
                    st.text(f"• {peer.get('display_name', 'Unknown')} - {peer.get('address', 'N/A')[:40]}")
    else:
        st.info("No peer data available. Connect to the network to see peers.")


def render_wallet_tab(client: PRSMClient):
    """Render the wallet/balance tab."""
    st.subheader("💰 FTNS Token Balance")
    
    balance = client.get_balance()
    
    if balance:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Balance", f"{balance.get('balance', 0):.4f} FTNS")
        with col2:
            wallet_id = balance.get('wallet_id', 'N/A')
            st.metric("Wallet", f"{wallet_id[:16]}...")
        
        # Transaction history
        st.subheader("📜 Transaction History")
        txs = client.get_transactions(limit=10)
        
        if txs and txs.get('transactions'):
            for tx in txs['transactions']:
                with st.expander(f"{tx['type']}: {tx['amount']:.4f} FTNS"):
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.write(f"**From:** {tx.get('from', 'N/A')[:20]}...")
                        st.write(f"**To:** {tx.get('to', 'N/A')[:20]}...")
                    with col_t2:
                        ts = tx.get('timestamp', 0)
                        st.write(f"**Time:** {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")
                        st.write(f"**ID:** {tx.get('tx_id', 'N/A')[:16]}...")
                    st.write(f"**Description:** {tx.get('description', 'N/A')}")
        else:
            st.info("No transactions yet.")
        
        # DAG Stats
        status = client.get_status()
        dag_stats = status.get('dag_stats') if status else None
        
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
        st.info("No balance data. Start the PRSM node to see wallet info.")


def render_compute_tab(client: PRSMClient):
    """Render the compute tab."""
    st.subheader("💻 Compute Network")
    
    status = client.get_status()
    compute_stats = status.get('compute') if status else None
    compute_requester_stats = status.get('compute_requester') if status else None
    
    if compute_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jobs Completed", compute_stats.get('jobs_completed', 0))
        with col2:
            st.metric("Jobs In Queue", compute_stats.get('jobs_queued', 0))
        with col3:
            st.metric("Active Workers", compute_stats.get('active_workers', 0))
    
    if compute_requester_stats:
        st.subheader("📤 Your Submitted Jobs")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jobs Submitted", compute_requester_stats.get('jobs_submitted', 0))
        with col2:
            st.metric("Jobs Completed", compute_requester_stats.get('jobs_completed', 0))
    
    if not compute_stats and not compute_requester_stats:
        st.info("Compute provider not initialized. Enable compute role to participate in the network.")


def render_storage_tab(client: PRSMClient):
    """Render the storage tab."""
    st.subheader("💾 Storage Network")
    
    status = client.get_status()
    storage_stats = status.get('storage') if status else None
    
    if storage_stats:
        col1, col2, col3 = st.columns(3)
        
        pledged = storage_stats.get('pledged_gb', 0)
        used = storage_stats.get('used_gb', 0)
        
        with col1:
            st.metric("Pledged", f"{pledged:.2f} GB")
        with col2:
            st.metric("Used", f"{used:.2f} GB")
        with col3:
            st.metric("Pinned Content", storage_stats.get('pinned_count', 0))
        
        if pledged > 0:
            usage_pct = min(used / pledged, 1.0)
            st.progress(usage_pct, "Storage Usage")
    
    # Content Index
    content_stats = status.get('content_index') if status else None
    
    if content_stats:
        st.subheader("📄 Content Index")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Content", content_stats.get('total_cids', 0))
        with col2:
            st.metric("Providers", content_stats.get('provider_count', 0))
    
    if not storage_stats and not content_stats:
        st.info("Storage provider not initialized. Enable storage role to contribute.")


def render_agents_tab(client: PRSMClient):
    """Render the agents tab."""
    st.subheader("🤖 Agent Management")
    
    agents = client.get_agents()
    
    if agents and agents.get('agents'):
        st.write(f"**{agents.get('count', 0)}** registered agents")
        
        for agent in agents['agents']:
            with st.expander(f"🤖 {agent.get('agent_name', 'Unknown')}"):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.write(f"**Agent ID:** {agent.get('agent_id', 'N/A')[:20]}...")
                    st.write(f"**Status:** {agent.get('status', 'N/A')}")
                with col_a2:
                    st.write(f"**Capabilities:** {', '.join(agent.get('capabilities', []))}")
                    st.write(f"**Version:** {agent.get('version', 'N/A')}")
    else:
        st.info("No agents registered. Agents will appear here when discovered on the network.")


def render_content_search(client: PRSMClient):
    """Render content search functionality."""
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
                    st.write(f"**Creator:** {result.get('creator_id', 'N/A')[:20]}...")
                    if result.get('royalty_rate'):
                        st.write(f"**Royalty Rate:** {result.get('royalty_rate'):.4f} FTNS")
        else:
            st.info("No content found matching your query.")


def render_fallback():
    """Render fallback UI when node is not running."""
    st.warning("⚠️ **PRSM Node is Not Running**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### Start Your Node
        
        Run the following command in your terminal:
        
        ```bash
        python -m prsm.cli start
        ```
        
        This will start the PRSM node and API server on port 8000.
        """)
    
    with col2:
        st.info("""
        ### Configuration
        
        The dashboard expects the API at `http://127.0.0.1:8000`
        
        To use a different port, update `API_BASE_URL` in the dashboard code.
        """)
    
    # Show mockup as fallback
    if (MOCKUP_DIR / "index.html").exists():
        st.subheader("📱 UI Mockup Preview")
        with st.expander("View Static Mockup"):
            with open(MOCKUP_DIR / "index.html", 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=True)


def main():
    """Main dashboard application."""
    st.title("🧠 PRSM Dashboard")
    st.caption("Protocol for Recursive Scientific Modeling")
    
    client = PRSMClient(API_BASE_URL)
    
    # Check if node is running
    if not client.health_check():
        render_fallback()
        return
    
    status = client.get_status()
    
    if not status:
        render_fallback()
        return
    
    # Render header
    render_header(status)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Network", 
        "💰 Wallet", 
        "💻 Compute", 
        "💾 Storage",
        "🔍 Content"
    ])
    
    with tab1:
        render_network_tab(client)
    
    with tab2:
        render_wallet_tab(client)
    
    with tab3:
        render_compute_tab(client)
    
    with tab4:
        render_storage_tab(client)
    
    with tab5:
        render_content_search(client)
    
    # Agents section (collapsible)
    with st.expander("🤖 Agent Registry"):
        render_agents_tab(client)
    
    # Footer with refresh
    st.divider()
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    with col_f2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
