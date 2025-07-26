# LITE Browser - Getting Started Guide

## 🚀 Quick Installation & Setup

### Step 1: Install LITE Browser

Simply run our installation script in your terminal:

```bash
curl -fsSL https://install.prsm.dev/lite | bash
```

Or manually:
```bash
git clone https://github.com/prsm-research/lite_browser.git
cd lite_browser
chmod +x install.sh
./install.sh
```

The installer will:
- ✅ Check system requirements (cmake, C++ compiler)
- 📥 Download and build LITE Browser
- 🔧 Install to `/usr/local/bin/lite_browser`
- ⚙️ Create configuration directories
- 🎯 Optionally launch first-time setup

### Step 2: First-Time Setup

When you first run LITE Browser, you'll go through a guided setup:

```bash
lite_browser
```

#### 👤 **Profile Setup**
- **Full Name**: Your research identity
- **Email**: Must be institutional (.edu, .ac.uk, etc.)
- **Institution**: University or research organization
- **Research Areas**: Select from 17+ categories
- **Collaboration Preferences**: 
  - Allow industry partnerships? 
  - Require approval for sensitive work?
  - Auto-accept invites from your institution?

#### 💻 **Compute & Storage Contribution**
Configure how much of your resources to contribute for FTNS earning:

- **CPU Usage**: 10-75% (recommendation: 25% for light impact)
- **Storage**: 1-1000GB (recommendation: 10GB to start)
- **Bandwidth**: 1-100Mbps (recommendation: 10Mbps)
- **Smart Settings**:
  - ✅ Only contribute when idle
  - ✅ Exclude sensitive research files
  - ✅ Allow data processing tasks
  - ❓ Allow ML training (higher earnings)

**💰 Earnings Estimate**: ~50-200 FTNS tokens/month based on your settings

#### 🪙 **FTNS Token Configuration**
Set up automatic earning from your contributions:

- **Auto-earning**: Enable/disable background token earning
- **Minimum Payout**: 100 FTNS (adjustable 10-1000)
- **Wallet Address**: Leave empty for institutional default
- **Task Preferences**:
  - ✅ Prioritize research over commercial tasks
  - ❓ Accept industry compute tasks (higher pay)
  - ✅ Require institutional validation for high-value work
- **Minimum Rate**: 1.0 FTNS/hour (adjust based on selectivity)

#### 🌐 **Network & Security**
- **IPv6**: Enable for better P2P connectivity
- **Max Connections**: 50 peers (10-100 range)
- **Academic-Only**: Restrict to `.edu` networks
- **Security Level**: 
  - **Standard**: Good for most research
  - **High**: Sensitive data, post-quantum crypto
  - **Maximum**: Classified work, manual trust updates

### Step 3: Start Using LITE Browser

After setup, LITE Browser launches with native support for:

#### 🔬 **Research Network Access** (`lite://`)
```
lite://mit.edu/artificial-intelligence/neural-networks
lite://unc.edu/quantum-computing/error-correction
lite://stanford.edu/climate-science/modeling
```

#### 🧩 **Distributed File Access** (`shard://`)
```
shard://research-data/quantum-dataset.csv
shard://papers/distributed-ml-paper.pdf
shard://simulations/climate-model-results.h5
```

#### 🤝 **Collaboration Sessions** (`collab://`)
```
collab://grant-proposal/nsf-quantum-2024
collab://research-paper/multi-university-ai-study
collab://industry-evaluation/startup-tech-review
```

## 💰 Earning FTNS Tokens

### Automatic Background Earning
Once configured, LITE Browser automatically earns FTNS by:

- **Storage Contribution**: ~0.3 FTNS/GB/month
- **Compute Tasks**: 1-10 FTNS/hour depending on complexity
- **Bandwidth Sharing**: 0.01-0.1 FTNS/GB transferred
- **Peer Discovery**: 0.5 FTNS/hour for network maintenance
- **Research Validation**: 15 FTNS/hour for expert reviews

### Manual Task Acceptance
View and accept higher-paying tasks:

```bash
lite_browser --tasks list
lite_browser --tasks accept storage_task_001
```

### Earnings Dashboard
Monitor your progress:

```bash
lite_browser --earnings status
lite_browser --earnings history
lite_browser --earnings payout
```

**Example Monthly Earnings:**
- **Light User** (10% CPU, 5GB storage): ~25-50 FTNS/month
- **Moderate User** (25% CPU, 20GB storage): ~75-150 FTNS/month  
- **Heavy User** (50% CPU, 100GB storage): ~200-400 FTNS/month
- **Research Validator** (expert tasks): +100-500 FTNS/month

## 🔒 Security & Privacy

### Post-Quantum Cryptography
- All connections use Kyber-1024 key exchange
- ML-DSA digital signatures
- Future-proof against quantum computers

### Institutional Trust
- Multi-signature approval for sensitive actions
- University IT integration
- GDPR and FERPA compliant

### Data Protection
- Zero-knowledge architecture
- Local encryption options
- You control what data is shared

## 🤝 Collaboration Workflows

### Starting a Research Collaboration
1. Navigate to `collab://research-paper/your-project-name`
2. LITE automatically discovers institutional peers
3. Set collaboration preferences (open vs. invitation-only)
4. Begin real-time document editing
5. Request approvals for sensitive actions

### University-Industry Partnerships
1. Enable industry collaboration in settings
2. Join `collab://industry-evaluation/project-name`
3. Multi-signature approval workflow activates
4. Legal and tech transfer offices are notified
5. Proceed only after all approvals received

## 📊 Dashboard & Monitoring

### Real-Time Status
- Current FTNS balance and pending earnings
- Active contributions and resource usage
- Connected peers and collaboration sessions
- Upcoming payouts and task deadlines

### Performance Analytics
- Earnings efficiency (FTNS per resource unit)
- Contribution uptime and reliability scores
- Research collaboration metrics
- Network contribution rankings

## 🆘 Getting Help

### Commands
```bash
lite_browser --help                 # Full command reference
lite_browser --setup               # Re-run first-time setup
lite_browser --status              # Show current status
lite_browser --earnings status     # FTNS earning overview
lite_browser --network peers       # Show connected peers
lite_browser --security check      # Verify security settings
```

### Support Resources
- **Documentation**: https://docs.prsm.dev/lite-browser
- **Community**: https://community.prsm.dev
- **Issues**: https://github.com/prsm-research/lite_browser/issues
- **Security**: security@prsm.dev

### Troubleshooting
- **Institutional Verification Failed**: Contact IT for .edu email setup
- **Low FTNS Earnings**: Increase resource contribution limits
- **Connection Issues**: Check firewall settings for P2P ports
- **Sync Problems**: Verify institutional network access

## 🌟 Welcome to the Future of Research!

LITE Browser represents the first native P2P research collaboration platform. By contributing your compute resources, you're not just earning tokens—you're helping build the infrastructure for the next generation of scientific discovery.

**Key Benefits:**
- 🔬 Direct access to cutting-edge research without HTTP limitations
- 💰 Earn tokens for resources you're already using
- 🤝 Seamless collaboration across institutional boundaries
- 🔒 Post-quantum security for sensitive research
- 🌐 Truly decentralized academic network

Join thousands of researchers already using LITE Browser to accelerate discovery and earn FTNS tokens through meaningful contributions to the scientific community!

---

*LITE Browser - Linking Information, Transferring Knowledge, Engineering the Future* 💡