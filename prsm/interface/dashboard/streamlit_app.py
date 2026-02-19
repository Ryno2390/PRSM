import streamlit as st
import requests
import json
import time
from pathlib import Path
from datetime import datetime
import webbrowser

PRSM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MOCKUP_DIR = PRSM_ROOT / "PRSM_ui_mockup"
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PRSM | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #0a0a0f;
            color: #e0e0e0;
        }
        .stSidebar {
            background-color: #12121a;
        }
        div[data-testid="stMetric"] {
            background-color: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a2e;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
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
        .mockup-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            display: inline-block;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)


class PRSMClient:
    """Client for interacting with PRSM API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 3
    
    def _get(self, endpoint: str) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
            return resp.json()
        except:
            return None
    
    def health_check(self) -> bool:
        return self._get("/health") is not None
    
    def get_status(self) -> dict | None:
        return self._get("/status")
    
    def get_peers(self) -> dict | None:
        return self._get("/peers")
    
    def get_balance(self) -> dict | None:
        return self._get("/balance")


def render_mockup():
    """Render a link to the mockup instead of embedding it."""
    st.header("🎨 PRSM UI Mockup")
    
    html_path = MOCKUP_DIR / "index.html"
    
    if not html_path.exists():
        st.error(f"Mockup file not found!")
        return
    
    # Show file info
    file_size = html_path.stat().st_size // 1024
    st.info(f"📄 Mockup file: {file_size}KB")
    
    # Show preview images/assets that exist
    assets_dir = MOCKUP_DIR / "assets"
    if assets_dir.exists():
        images = list(assets_dir.glob("*.png")) + list(assets_dir.glob("*.jpg"))
        if images:
            st.subheader("🖼️ Preview Assets")
            for img in images[:4]:  # Show up to 4 images
                try:
                    st.image(str(img), use_container_width=True)
                except:
                    pass
    
    # Show the mockup as a download/link option
    st.subheader("📥 Open Mockup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show the raw HTML in a code block for reference
        with open(html_path, 'r', encoding='utf-8') as f:
            html_preview = f.read()[:2000]  # First 2000 chars
        
        with st.expander("📝 HTML Preview (first 2000 chars)"):
            st.code(html_preview, language="html")
    
    with col2:
        st.markdown("### 🚀 Open Full Mockup")
        st.markdown("The mockup is a complex HTML file that doesn't render well in Streamlit.")
        
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <a href="file://{}" target="_blank">
                <div class="mockup-button">📱 Open in Browser</div>
            </a>
        </div>
        """.format(html_path.absolute()), unsafe_allow_html=True)
        
        st.caption("Click above to open the full mockup in your browser")


def render_live_dashboard():
    """Render the live dashboard with real data."""
    client = PRSMClient(API_BASE_URL)
    
    # Check connection
    if not client.health_check():
        st.error("❌ Cannot connect to PRSM API. Make sure the API server is running.")
        st.info("Try running: python -m prsm.cli serve")
        return
    
    # Header metrics
    status = client.get_status()
    if not status:
        st.warning("Could not get node status")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Online" if status.get('started') else "🔴 Offline")
    with col2:
        node_id = status.get('node_id', 'N/A')
        st.metric("Node ID", node_id[:12] + "...")
    with col3:
        uptime = int(status.get('uptime_seconds', 0))
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        st.metric("Uptime", f"{hours}h {minutes}m")
    with col4:
        ledger = status.get('ledger_type', 'legacy').upper()
        st.metric("Ledger", ledger)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network", "💰 Wallet", "💻 Compute", "💾 Storage"])
    
    with tab1:
        peers = client.get_peers()
        if peers:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Connected", peers.get('connected_count', 0))
            with col_b:
                st.metric("Known", peers.get('known_count', 0))
            
            if peers.get('connected'):
                with st.expander("Connected Peers"):
                    for p in peers['connected']:
                        st.write(f"• {p.get('display_name', 'Unknown')}")
        else:
            st.info("No peer data")
    
    with tab2:
        balance = client.get_balance()
        if balance:
            st.metric("FTNS Balance", f"{balance.get('balance', 0):.4f}")
            
            dag_stats = status.get('dag_stats')
            if dag_stats:
                st.subheader("🔗 DAG Stats")
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Total Txs", dag_stats.get('total_transactions', 0))
                with col_d2:
                    st.metric("Tips", dag_stats.get('tips', 0))
                with col_d3:
                    conf = dag_stats.get('avg_confirmation_level', 0)
                    st.metric("Confirmation", f"{conf:.1%}")
        else:
            st.info("No balance data")
    
    with tab3:
        compute = status.get('compute')
        if compute:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("Jobs Done", compute.get('jobs_completed', 0))
            with col_c2:
                st.metric("In Queue", compute.get('jobs_queued', 0))
        else:
            st.info("Compute not initialized")
    
    with tab4:
        storage = status.get('storage')
        if storage:
            pledged = storage.get('pledged_gb', 0)
            used = storage.get('used_gb', 0)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Pledged", f"{pledged:.1f} GB")
            with col_s2:
                st.metric("Used", f"{used:.1f} GB")
            with col_s3:
                st.metric("Pinned", storage.get('pinned_count', 0))
            
            if pledged > 0:
                st.progress(used/pledged, "Storage Usage")
        else:
            st.info("Storage not initialized")
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    # Auto-refresh
    time.sleep(30)
    st.rerun()


def main():
    """Main app."""
    st.title("🧠 PRSM Dashboard")
    st.caption("Protocol for Recursive Scientific Modeling")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Mode")
        
        mode = st.radio(
            "Select:",
            ["🟢 Live Dashboard", "🎨 Mockup Preview"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.header("🔗 Connection")
        client = PRSMClient(API_BASE_URL)
        if client.health_check():
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API Not Running")
        
        st.divider()
        
        st.header("📋 Commands")
        st.code("python -m prsm.cli dashboard", language="bash")
        st.code("python -m prsm.cli serve", language="bash")
    
    # Render
    if mode == "🎨 Mockup Preview":
        render_mockup()
    else:
        render_live_dashboard()


if __name__ == "__main__":
    main()
