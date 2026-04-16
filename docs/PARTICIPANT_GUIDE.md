# PRSM Participant Guide — Earn FTNS by Contributing

> **Related docs:** For technical installation / CLI workflow, see [`GETTING_STARTED.md`](GETTING_STARTED.md) and [`quickstart.md`](quickstart.md). For token economics detail, see `PRSM_Tokenomics.md`. For terminology disambiguation, see [`glossary.md`](glossary.md).

Welcome to PRSM (**Protocol for Research, Storage, and Modeling**), a decentralized peer-to-peer infrastructure protocol where anyone can participate in providing compute, storage, or data for AI workloads and earn rewards.

## What is PRSM?

PRSM is a peer-to-peer infrastructure protocol that connects people who have computing resources with users who need compute, storage, and data access for AI workloads. Instead of relying on a single company's datacenters, PRSM distributes work across thousands of consumer devices and independent operators worldwide — including yours.

**Important framing:** PRSM is **infrastructure**, not a model. Third-party LLMs (Claude, GPT, Gemini) do the reasoning; PRSM provides compute, storage, and data access underneath them via MCP (Model Context Protocol). You contribute to the substrate, not to building a model.

**Why does this matter?**

- **Democratized AI infrastructure**: Compute becomes accessible without depending on a handful of hyperscaler datacenters
- **Fair compensation**: Contributors earn FTNS for providing compute, storage, and data
- **Verifiable provenance**: On-chain royalty distribution means every content access pays the creator, enforced by smart contracts rather than platform policy
- **Transparency**: All transactions are publicly verifiable on-chain

PRSM uses a token called **FTNS** (pronounced "photons") to compensate contributors. FTNS is distributed **only as compensation for services rendered** to the network — it is never sold to investors by the foundation. See `PRSM_Tokenomics.md` §3 for the full distribution model.

## What is FTNS?

FTNS is the native token of the PRSM network. It serves three key purposes:

1. **Payment**: Users spend FTNS to run AI queries and access models
2. **Rewards**: Contributors earn FTNS by providing compute, data, or expertise
3. **Governance**: Token holders can vote on network decisions

### How Much Can I Earn?

Earnings depend on what you contribute and how much:

| Contribution Type | Typical Earnings | Requirements |
|-------------------|------------------|--------------|
| GPU Compute | 0.5-5 FTNS/hour | NVIDIA GPU with 8GB+ VRAM |
| CPU Compute | 0.1-1 FTNS/hour | Modern multi-core CPU |
| Data Sharing | 1-50 FTNS/dataset | Curated, verified datasets |
| Model Development | 10-100+ FTNS/model | Working AI models |

*Note: FTNS is currently in Sepolia testnet bake-in as of April 2026; Phase 1 on-chain royalty contracts are mainnet-imminent. FTNS price is not set by the foundation — it emerges organically from secondary-market activity on third-party venues once the network is live. The foundation does not operate FTNS sales, does not announce prices, and does not guarantee appreciation. Earnings above are illustrative targets that depend on network adoption and are not commitments.*

### Getting Your First FTNS

New users receive a **welcome grant of 100 FTNS** upon registration. This allows you to:
- Run AI queries immediately
- Test the marketplace
- Participate in governance

## How to Participate

### Option A: Contribute Compute Power

Share your computer's processing power to help run AI queries and earn FTNS automatically.

#### Step 1: Install PRSM

PRSM is a Python package. Install with `pip` on any platform (macOS, Linux, Windows with WSL):

```bash
pip install prsm-network
```

Requires Python 3.11+. For the full installation walkthrough including virtualenv setup and first-run configuration, see [`GETTING_STARTED.md`](GETTING_STARTED.md).

#### Step 2: Initialize Your Node

```bash
prsm setup --minimal   # smart defaults, fast path
# OR
prsm setup             # interactive wizard
```

On first run, PRSM generates an Ed25519 identity keypair (stored in `~/.prsm/identity.json`) and credits your node with a welcome grant of 100 FTNS. There is no separate "account" — your node identity *is* your account. Back up `~/.prsm/identity.json` the same way you would back up a crypto wallet: losing it means losing access to your FTNS balance.

A browser/SDK-based wallet UX is planned for Phase 4 (Q4 2026) per the [master roadmap](2026-04-10-audit-gap-roadmap.md). Until then, the CLI is the primary interface.

#### Step 3: Start Contributing

```bash
prsm daemon start              # run in the background
prsm daemon status             # check it's running
prsm daemon logs -f            # follow logs
```

Adjust resource allocation via `prsm config set`:

```bash
prsm config set cpu_pct 50     # share 50% of CPU
prsm config set memory_pct 40  # share 40% of RAM
prsm config set gpu_pct 80     # share 80% of GPU (0 = disabled)
prsm config set storage_gb 50  # pledge 50 GB for storage
```

See `prsm config show` for all options. Your node will automatically process compute jobs from the network and earn FTNS.

#### Step 4: Monitor Your Earnings

```bash
curl http://localhost:8000/balance         # FTNS balance + recent transactions
prsm ftns yield-estimate --hours 24        # projected daily earnings
```

Operator dashboard UI with historical charts and leaderboards is planned for Phase 3 per the [master roadmap](2026-04-10-audit-gap-roadmap.md).

**Tips for Maximum Earnings:**
- Keep your node online during peak hours (typically 9 AM - 9 PM UTC)
- Share GPU resources if available (higher earnings)
- Maintain good internet connectivity (low latency = more queries)
- Keep your system updated for optimal performance

---

### Option B: Share Your Data

Contribute datasets to the PRSM marketplace and earn FTNS when researchers use your data.

#### What Kind of Data is Valuable?

PRSM specializes in scientific and research data:

- **Scientific Papers**: Published research with proper licensing
- **Experimental Results**: Lab data, measurements, observations
- **Code Repositories**: Well-documented research code
- **Datasets**: Curated collections for ML training
- **Model Weights**: Trained neural network parameters

#### Step 1: Prepare Your Data

1. **Organize**: Clean and document your dataset
2. **Format**: Use standard formats (CSV, JSON, Parquet, HDF5)
3. **Document**: Create a README describing:
   - What the data contains
   - How it was collected
   - Any usage restrictions
   - Citation requirements
4. **License**: Choose an appropriate license (MIT, CC-BY, Apache, etc.)

#### Step 2: Upload to PRSM

Using the PRSM interface:
1. Navigate to the "Storage" tab
2. Click "Upload Dataset"
3. Select your files
4. Fill in metadata:
   - Title
   - Description
   - Tags (for discoverability)
   - License type
5. Set pricing (FTNS per access)
6. Click "Publish"

#### Step 3: Earn from Usage

You'll earn FTNS automatically when:
- Users download your dataset
- Users cite your data in their research
- Your data is used in AI model training

**Tips for Data Success:**
- High-quality documentation attracts more users
- Unique datasets command higher prices
- Respond to user questions promptly
- Update your datasets with new data periodically

---

### Option C: Use PRSM via a Third-Party LLM

**PRSM does not host AI models.** The reasoning layer is always a third-party LLM (Claude, GPT, Gemini, local Ollama, etc.). PRSM provides the compute, storage, and data-access substrate *underneath* your LLM via MCP. You spend FTNS when your LLM invokes PRSM tools — not for the reasoning itself.

#### Using PRSM from an MCP-compatible LLM client

When Phase 3's MCP server ships (target Q3 2026, per the [roadmap](2026-04-10-audit-gap-roadmap.md)), any MCP client (Claude Desktop, Cursor, Gemini CLI, etc.) can invoke PRSM tools such as:

- `prsm_retrieve` — retrieve data from PRSM's data layer with creator royalty settlement
- `prsm_compute` — dispatch heavy compute via the PRSM compute mesh
- `prsm_inference` — run private inference via PRSM's zero-trust stack

Your LLM does the reasoning; PRSM tools provide data and compute the LLM can't hold in its own context. Every PRSM tool call debits FTNS from your node's balance.

#### Using PRSM directly via CLI (today)

Before the MCP server ships, you can interact with PRSM's compute pipeline via CLI:

```bash
# Get a cost estimate first (free)
prsm compute quote "analyze EV registration trends in NC" --shards 5 --tier t2

# Run with a budget
prsm compute run --query "analyze EV registration trends in NC" --budget 1.0
```

This routes through the Ring 1-10 pipeline — decompose → plan → quote → execute → trace. Under the hood, a third-party LLM (configured via `OPENROUTER_API_KEY` or equivalent) does the reasoning; PRSM dispatches WASM agents (SPRKs) to data-holding nodes.

#### Budget controls

- `--budget N` caps FTNS spend at N
- `prsm compute quote ...` estimates cost before committing
- Higher tier (T2, T3) = faster compute but higher cost
- More shards = more parallelism but higher aggregate cost

## Frequently Asked Questions

### Do I need an API key?

No, you don't need an API key to use the basic features. When you create an account, you can:
- Use the PRSM application directly
- Run queries from the web interface
- Contribute compute without any keys

API keys are only needed if you're a developer building applications that connect to PRSM programmatically.

### How much can I earn?

Earnings vary based on:
- **Hardware**: Better GPUs earn more
- **Uptime**: More hours = more earnings
- **Network Demand**: Higher demand = higher rates
- **Data Quality**: Better data = more downloads

Typical contributors earn 5-50 FTNS per day. Top contributors with powerful hardware can earn 100+ FTNS daily.

### Is my data safe?

Yes. PRSM implements several protections:

1. **Sandboxing**: Your data is processed in isolated environments
2. **Encryption**: All data transfers are encrypted
3. **Provenance**: Every access is logged and verifiable
4. **Control**: You can delete your data at any time

When you contribute compute, your personal files are **never** accessed. PRSM only processes data specifically submitted to the network.

### What is FTNS worth?

As of April 2026, PRSM is in Sepolia testnet bake-in with mainnet launch imminent. During the testnet phase FTNS has no secondary-market value — it can only be earned and spent within the testnet.

**Post-mainnet (important framing):**

- **The foundation does not set FTNS price and does not operate a sale venue.** FTNS is distributed exclusively as compensation for services rendered (creator royalties, node operator compensation, contributor grants). It is never sold by the foundation to anyone.
- **Secondary-market value emerges organically** via third-party exchanges (Aerodrome on Base, eventual CEX listings) once the network is live. The foundation does not seed AMM pools, announce prices, or guarantee appreciation.
- **The protocol includes value-trajectory mechanisms** (Bitcoin-style halving schedule, 20% burn on every transaction, staking locks, organic demand growth) that collectively produce modest appreciation during bootstrap and steady-state stabilization as adoption matures. Appreciation is a consequence of protocol design, not a foundation promise. See `PRSM_Tokenomics.md` §4 for the full design.
- **Bear case is a real possibility.** If PRSM fails to achieve adoption, FTNS remains low-value regardless of the halving / burn mechanics. Contribute what you can afford to lose the USD-equivalent value of.

### How do I get started without any FTNS?

New users receive a **100 FTNS welcome grant** upon registration. This is enough to:
- Run 50-200 basic AI queries
- Test the marketplace
- Participate in governance votes

If you run low on FTNS, you can:
- Contribute compute to earn more
- Share data for ongoing earnings
- Wait for daily login bonuses (coming soon)

### Can I run PRSM on multiple computers?

Yes! You can run PRSM on as many computers as you like. Each computer:
- Connects to your single account
- Earns FTNS independently
- Shows up in your dashboard

This is a great way to maximize earnings if you have access to multiple machines.

### What happens if my computer goes offline?

Nothing bad happens:
- Your node simply disconnects from the network
- Pending work is redistributed to other nodes
- When you reconnect, you automatically start earning again
- Your FTNS balance and history are preserved

PRSM is designed for intermittent connectivity — many contributors only run their nodes part-time.

## Safety and Privacy

### What PRSM Sees

PRSM collects only essential information:
- Account email (for notifications and recovery)
- FTNS transaction history (for transparency)
- Query metadata (for billing and quality)

### What PRSM Doesn't See

PRSM **cannot** access:
- Your personal files
- Data outside the PRSM storage system
- Your identity (unless you choose to share it)
- Your location or IP address (beyond what's needed for network routing)

### Data Ownership

When you upload data to PRSM:
- **You retain ownership** of your data
- You set the license and usage terms
- You can delete your data at any time
- PRSM cannot claim ownership of your contributions

### Security Best Practices

1. **Use a strong password** (12+ characters, mix of types)
2. **Save your recovery phrase** in a secure location
3. **Keep software updated** for security patches
4. **Review permissions** before running third-party tools
5. **Report suspicious activity** to security@prsm.ai

## Community and Support

### Getting Help

- **Documentation**: [docs.prsm.ai](https://docs.prsm.ai)
- **Community Forum**: [community.prsm.ai](https://community.prsm.ai)
- **GitHub Issues**: [github.com/PRSM-AI/PRSM/issues](https://github.com/PRSM-AI/PRSM/issues)
- **Email Support**: support@prsm.ai

### Reporting Problems

Found a bug or have a suggestion?
1. Check [GitHub Issues](https://github.com/PRSM-AI/PRSM/issues) for known issues
2. Submit a new issue with:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable

### Contributing to Development

PRSM is open source! We welcome contributions:
- **Code**: Submit pull requests on GitHub
- **Documentation**: Help improve guides and API docs
- **Testing**: Report bugs and test new releases
- **Translation**: Help translate PRSM into more languages

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Staying Updated

- **Newsletter**: Subscribe at [prsm.ai/newsletter](https://prsm.ai/newsletter)
- **Blog**: [blog.prsm.ai](https://blog.prsm.ai)
- **Twitter/X**: [@prsm_protocol](https://twitter.com/prsm_protocol)
- **Discord**: [discord.gg/prsm](https://discord.gg/prsm)

---

*Welcome to the future of decentralized AI. We're excited to have you as part of the PRSM community!*
