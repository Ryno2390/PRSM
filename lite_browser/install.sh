#!/bin/bash
# LITE Browser Installation Script
# Linked Information Transfer Engine - Native P2P Research Browser

set -e

echo "💡 LITE Browser - Linked Information Transfer Engine"
echo "🔬 Installing Native P2P Research Collaboration Browser..."
echo ""

# Check system requirements
echo "🔍 Checking system requirements..."

# Check for required dependencies
DEPS_MISSING=false

if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Please install cmake first."
    DEPS_MISSING=true
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ C++ compiler not found. Please install g++ or clang++."
    DEPS_MISSING=true
fi

if [ "$DEPS_MISSING" = true ]; then
    echo ""
    echo "📋 Installation Requirements:"
    echo "   • cmake (>= 3.16)"
    echo "   • C++ compiler (g++ or clang++)"
    echo "   • Git"
    echo ""
    echo "🔧 Install on macOS: brew install cmake"
    echo "🔧 Install on Ubuntu: sudo apt install cmake build-essential"
    echo "🔧 Install on CentOS: sudo yum install cmake gcc-c++"
    exit 1
fi

echo "✅ System requirements met"

# Create installation directory
INSTALL_DIR="$HOME/.lite_browser"
echo "📁 Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone or update LITE Browser
if [ -d "lite_browser" ]; then
    echo "🔄 Updating existing LITE Browser installation..."
    cd lite_browser
    git pull origin main
else
    echo "📥 Downloading LITE Browser..."
    git clone https://github.com/prsm-research/lite_browser.git
    cd lite_browser
fi

# Build LITE Browser
echo "🔨 Building LITE Browser..."
mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -eq 0 ]; then
    echo "✅ LITE Browser built successfully"
else
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi

# Install binary to system PATH
echo "📦 Installing LITE Browser..."
sudo mkdir -p /usr/local/bin
sudo cp lite_browser /usr/local/bin/
sudo chmod +x /usr/local/bin/lite_browser

# Create desktop entry (Linux/macOS compatible)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🖥️  Creating desktop entry..."
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/lite-browser.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=LITE Browser
Comment=Linked Information Transfer Engine - P2P Research Browser
Exec=/usr/local/bin/lite_browser
Icon=web-browser
Terminal=false
Categories=Network;WebBrowser;Science;Education;
Keywords=browser;research;p2p;collaboration;academic;
StartupNotify=true
EOF
fi

# Create configuration directory
echo "⚙️  Setting up configuration..."
CONFIG_DIR="$HOME/.config/lite_browser"
mkdir -p "$CONFIG_DIR"

# Create initial configuration
cat > "$CONFIG_DIR/config.json" << EOF
{
  "first_run": true,
  "user_profile": {
    "institution": "",
    "research_areas": [],
    "collaboration_preferences": {
      "allow_industry_collaboration": false,
      "require_multi_sig_approval": true,
      "auto_accept_institutional_invites": false
    }
  },
  "compute_contribution": {
    "enabled": false,
    "max_cpu_usage": 25,
    "max_storage_gb": 10,
    "contribute_during_idle_only": true,
    "exclude_sensitive_files": true
  },
  "ftns_settings": {
    "auto_earn_enabled": false,
    "minimum_payout_threshold": 100,
    "preferred_contribution_types": ["storage", "compute"],
    "institutional_validation_required": true
  },
  "network": {
    "enable_ipv6": true,
    "max_peer_connections": 50,
    "bootstrap_nodes": [
      "unc.edu:8445",
      "mit.edu:8445", 
      "stanford.edu:8445"
    ]
  },
  "security": {
    "post_quantum_enabled": true,
    "strict_certificate_validation": true,
    "encrypt_local_storage": true,
    "auto_update_trust_anchors": true
  }
}
EOF

echo ""
echo "🎉 LITE Browser installation complete!"
echo ""
echo "🚀 Next Steps:"
echo "   1. Run 'lite_browser' to start your first-time setup"
echo "   2. Configure your institutional profile"
echo "   3. Set up compute/storage contribution preferences"
echo "   4. Start earning FTNS tokens through research collaboration!"
echo ""
echo "📚 Documentation: https://docs.prsm.dev/lite-browser"
echo "🆘 Support: https://github.com/prsm-research/lite_browser/issues"
echo ""
echo "💡 Welcome to the future of P2P research collaboration!"

# Offer to run immediately
echo ""
read -p "🚀 Launch LITE Browser now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎯 Launching LITE Browser..."
    /usr/local/bin/lite_browser --first-run
fi