import streamlit as st
import streamlit.components.v1 as components
import os
import base64
from pathlib import Path

# --- Configuration ---
PRSM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MOCKUP_DIR = PRSM_ROOT / "PRSM_ui_mockup"

# Force current working directory to PRSM root so relative file access works
os.chdir(str(PRSM_ROOT))

st.set_page_config(
    page_title="PRSM | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar to give full room to the mockup
)

# --- Helper to load and process assets ---
def get_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- Main App Logic ---
def main():
    # CSS to make the iframe take up the whole page
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

    # We need to replace relative paths with either inline content or Streamlit-accessible paths
    # For a high-fidelity copy, we'll inline the CSS and JS
    
    # 1. Inline CSS
    for css_file in ["style.css", "p2p-dashboard.css", "security-indicators.css", "shard-visualization.css"]:
        css_path = MOCKUP_DIR / "css" / css_file
        if css_path.exists():
            # Replace the link tag with the actual style content
            link_tag = f'<link rel="stylesheet" href="css/{css_file}">'
            if link_tag in html_content:
                html_content = html_content.replace(link_tag, f'<style>{get_file_content(css_path)}</style>')
            else:
                # Fallback: append to head
                html_content = html_content.replace('</head>', f'<style>{get_file_content(css_path)}</style></head>')

    # 2. Inline JS (at the end of body)
    js_files = ["api-client.js", "script.js", "p2p-dashboard.js", "security-indicators.js", "shard-visualization.js"]
    js_combined = ""
    for js_file in js_files:
        js_path = MOCKUP_DIR / "js" / js_file
        if js_path.exists():
            js_combined += f"\n// --- {js_file} ---\n" + get_file_content(js_path)
    
    # Remove original script tags to prevent 404s
    import re
    html_content = re.sub(r'<script src="js/.*"></script>', '', html_content)
    html_content = html_content.replace('</body>', f'<script>{js_combined}</script></body>')

    # 3. Handle Logos (Base64 injection)
    dark_logo_b64 = get_image_base64(MOCKUP_DIR / "assets" / "PRSM_Logo_Dark.png")
    light_logo_b64 = get_image_base64(MOCKUP_DIR / "assets" / "PRSM_Logo_Light.png")
    
    # Global replacement for any asset paths to data URIs
    html_content = html_content.replace('assets/PRSM_Logo_Dark.png', f'data:image/png;base64,{dark_logo_b64}')
    html_content = html_content.replace('assets/PRSM_Logo_Light.png', f'data:image/png;base64,{light_logo_b64}')


    # 4. Inject a bridge to talk to the Python backend if needed
    # (This is where we could use st.query_params or a custom component event)

    # Render the full HTML
    # We use a large height to ensure the mockup fills the view
    # In a real app, you might use a custom component for 100% height
    components.html(html_content, height=2000, scrolling=True)

if __name__ == "__main__":
    main()
