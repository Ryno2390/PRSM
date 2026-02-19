import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import requests
from pathlib import Path

# --- Configuration ---
PRSM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MOCKUP_DIR = PRSM_ROOT / "PRSM_ui_mockup"
API_BASE_URL = "http://127.0.0.1:8000"

# Force current working directory to PRSM root so relative file access works
os.chdir(str(PRSM_ROOT))

st.set_page_config(
    page_title="PRSM | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar hidden
)

# --- API Client ---
class PRSMClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.timeout = 3
    
    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.status_code == 200
        except:
            return False
    
    def get_status(self) -> dict | None:
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return resp.json()
        except:
            return None

# --- Helper to load and process assets ---
def get_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- Main App Logic ---
def main():
    # Initialize session state for connection panel
    if 'show_status' not in st.session_state:
        st.session_state.show_status = True
    
    # Render connection status as an expander at the top
    with st.expander("🔗 Connection Status", expanded=st.session_state.show_status):
        client = PRSMClient(API_BASE_URL)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if client.health_check():
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Not Running")
        
        if client.health_check():
            status = client.get_status()
            if status:
                with col2:
                    st.metric("Node", status.get('display_name', 'N/A')[:15])
                with col3:
                    st.metric("Ledger", status.get('ledger_type', 'legacy').upper())
                with col4:
                    balance = status.get('ftns_balance', 0)
                    st.metric("Balance", f"{balance:.2f} FTNS")
                with col5:
                    dag_stats = status.get('dag_stats')
                    if dag_stats:
                        st.metric("DAG Txs", dag_stats.get('total_transactions', 0))
        else:
            with col2:
                st.caption("Run: python -m prsm.cli serve")
        
        # Toggle button
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Floating toggle button in top-right corner
    st.markdown("""
        <style>
            .stButton > button {
                position: fixed;
                top: 10px;
                right: 20px;
                z-index: 999;
                opacity: 0.7;
            }
            .stButton > button:hover {
                opacity: 1;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # CSS to make the iframe take up the main area
    st.markdown("""
        <style>
            .stApp {
                background-color: #000000;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            iframe {
                border: none;
                width: 100vw;
                height: 100vh;
            }
        </style>
    """, unsafe_allow_html=True)

    # Load index.html
    index_path = MOCKUP_DIR / "index.html"
    if not index_path.exists():
        st.error(f"Could not find index.html at {index_path}")
        return

    html_content = get_file_content(index_path)

    # 1. Inline CSS
    for css_file in ["style.css", "p2p-dashboard.css", "security-indicators.css", "shard-visualization.css"]:
        css_path = MOCKUP_DIR / "css" / css_file
        if css_path.exists():
            link_tag = f'<link rel="stylesheet" href="css/{css_file}">'
            if link_tag in html_content:
                html_content = html_content.replace(link_tag, f'<style>{get_file_content(css_path)}</style>')
            else:
                html_content = html_content.replace('</head>', f'<style>{get_file_content(css_path)}</style></head>')

    # 2. Inline JS
    js_files = ["api-client.js", "script.js", "p2p-dashboard.js", "security-indicators.js", "shard-visualization.js"]
    js_combined = ""
    for js_file in js_files:
        js_path = MOCKUP_DIR / "js" / js_file
        if js_path.exists():
            js_combined += f"\n// --- {js_file} ---\n" + get_file_content(js_path)
    
    import re
    html_content = re.sub(r'<script src="js/.*"></script>', '', html_content)
    html_content = html_content.replace('</body>', f'<script>{js_combined}</script></body>')

    # 3. Handle Logos
    dark_logo_b64 = get_image_base64(MOCKUP_DIR / "assets" / "PRSM_Logo_Dark.png")
    light_logo_b64 = get_image_base64(MOCKUP_DIR / "assets" / "PRSM_Logo_Light.png")
    
    html_content = html_content.replace('assets/PRSM_Logo_Dark.png', f'data:image/png;base64,{dark_logo_b64}')
    html_content = html_content.replace('assets/PRSM_Logo_Light.png', f'data:image/png;base64,{light_logo_b64}')

    # Render the full HTML
    components.html(html_content, height=2000, scrolling=True)

if __name__ == "__main__":
    main()
