/**
 * PRSM Dashboard JavaScript
 * Handles API calls, WebSocket connections, and UI interactions
 */

// ── Configuration ────────────────────────────────────────────────────────────────
const CONFIG = {
    API_BASE: '/api',
    WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/status`,
    RECONNECT_DELAY: 5000,
    STATUS_UPDATE_INTERVAL: 30000,
};

// ── State Management ──────────────────────────────────────────────────────────────
const state = {
    token: localStorage.getItem('prsm_token'),
    user: JSON.parse(localStorage.getItem('prsm_user') || 'null'),
    nodeInfo: null,
    peers: [],
    jobs: [],
    transactions: [],
    agents: [],
    balance: {
        available: 0,
        staked: 0,
        total: 0
    },
    ws: null,
    wsConnected: false,
    reconnectAttempts: 0,
};

// ── Utility Functions ─────────────────────────────────────────────────────────────
function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toFixed(2);
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

function formatTimestamp(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
}

function truncateId(id, length = 12) {
    if (!id) return 'N/A';
    return id.length > length ? `${id.substring(0, length)}...` : id;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `alert alert-${type}`;
    toast.style.marginTop = '10px';
    toast.innerHTML = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// ── API Client ───────────────────────────────────────────────────────────────────
const api = {
    async request(endpoint, options = {}) {
        const url = `${CONFIG.API_BASE}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers,
        };
        
        if (state.token) {
            headers['Authorization'] = `Bearer ${state.token}`;
        }
        
        try {
            const response = await fetch(url, {
                ...options,
                headers,
            });
            
            if (response.status === 401) {
                // Token expired or invalid
                state.token = null;
                state.user = null;
                localStorage.removeItem('prsm_token');
                localStorage.removeItem('prsm_user');
                showLoginModal();
                throw new Error('Authentication required');
            }
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', endpoint, error);
            throw error;
        }
    },
    
    // Authentication
    async login(username, password) {
        const data = await this.request('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
        return data;
    },
    
    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } catch (e) {
            // Ignore logout errors
        }
    },
    
    async getCurrentUser() {
        return await this.request('/auth/me');
    },
    
    // Status
    async getStatus() {
        return await this.request('/status');
    },
    
    async getNodeInfo() {
        return await this.request('/node');
    },
    
    // Jobs
    async getJobs(status = null) {
        const params = status ? `?status_filter=${status}` : '';
        return await this.request(`/jobs${params}`);
    },
    
    async submitJob(jobType, budget, payload = {}) {
        return await this.request('/jobs/submit', {
            method: 'POST',
            body: JSON.stringify({
                job_type: jobType,
                ftns_budget: budget,
                payload: payload,
            }),
        });
    },
    
    async getJobStatus(jobId) {
        return await this.request(`/jobs/${jobId}`);
    },
    
    // FTNS
    async getBalance() {
        return await this.request('/ftns/balance');
    },
    
    async getTransactions(limit = 50) {
        return await this.request(`/ftns/history?limit=${limit}`);
    },
    
    async transfer(toWallet, amount, description = null) {
        return await this.request('/ftns/transfer', {
            method: 'POST',
            body: JSON.stringify({
                to_wallet: toWallet,
                amount: amount,
                description: description,
            }),
        });
    },
    
    async stake(amount, durationEpochs = 10) {
        return await this.request('/ftns/stake', {
            method: 'POST',
            body: JSON.stringify({
                amount: amount,
                duration_epochs: durationEpochs,
            }),
        });
    },
    
    // Peers
    async getPeers() {
        return await this.request('/peers');
    },
    
    // Agents
    async getAgents(localOnly = false) {
        return await this.request(`/agents?local_only=${localOnly}`);
    },
    
    async getAgent(agentId) {
        return await this.request(`/agents/${agentId}`);
    },
    
    async setAgentAllowance(agentId, amount, epochHours = 24) {
        return await this.request(`/agents/${agentId}/allowance?amount=${amount}&epoch_hours=${epochHours}`, {
            method: 'POST',
        });
    },
    
    async pauseAgent(agentId) {
        return await this.request(`/agents/${agentId}/pause`, { method: 'POST' });
    },
    
    async resumeAgent(agentId) {
        return await this.request(`/agents/${agentId}/resume`, { method: 'POST' });
    },
    
    // Content
    async searchContent(query, limit = 20) {
        return await this.request(`/content/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    },
};

// ── WebSocket Management ──────────────────────────────────────────────────────────
function connectWebSocket() {
    if (state.ws) {
        state.ws.close();
    }
    
    state.ws = new WebSocket(CONFIG.WS_URL);
    
    state.ws.onopen = () => {
        console.log('WebSocket connected');
        state.wsConnected = true;
        state.reconnectAttempts = 0;
        updateConnectionStatus('online');
        
        // Request initial status
        state.ws.send(JSON.stringify({ type: 'get_status' }));
    };
    
    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('WebSocket message parse error:', e);
        }
    };
    
    state.ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.wsConnected = false;
        updateConnectionStatus('offline');
        
        // Attempt to reconnect
        state.reconnectAttempts++;
        const delay = Math.min(CONFIG.RECONNECT_DELAY * state.reconnectAttempts, 30000);
        setTimeout(connectWebSocket, delay);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('error');
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'status_update':
            updateDashboardStats(data.data);
            break;
        case 'job_submitted':
            showToast(`Job submitted: ${data.data.job_id}`, 'success');
            refreshJobs();
            break;
        case 'transfer':
            showToast(`Transfer completed: ${formatNumber(data.data.amount)} FTNS`, 'success');
            refreshBalance();
            refreshTransactions();
            break;
        case 'pong':
            // Heartbeat response
            break;
        default:
            console.log('Unknown WebSocket message type:', data.type);
    }
}

function updateConnectionStatus(status) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');
    
    if (statusDot && statusText) {
        statusDot.className = 'status-dot ' + status;
        
        switch (status) {
            case 'online':
                statusText.textContent = 'Connected';
                break;
            case 'offline':
                statusText.textContent = 'Disconnected';
                break;
            case 'connecting':
                statusText.textContent = 'Connecting...';
                break;
            default:
                statusText.textContent = 'Error';
        }
    }
}

// ── Dashboard Updates ─────────────────────────────────────────────────────────────
function updateDashboardStats(status) {
    // Update stats
    document.getElementById('stat-peers').textContent = status.connected_peers || 0;
    document.getElementById('stat-balance').textContent = formatNumber(status.ftns_balance || 0);
    document.getElementById('stat-jobs').textContent = status.active_jobs || 0;
    document.getElementById('stat-uptime').textContent = formatDuration(status.uptime_seconds || 0);
    
    // Update badges
    document.getElementById('peer-count-badge').textContent = status.connected_peers || 0;
    document.getElementById('active-jobs-badge').textContent = status.active_jobs || 0;
    
    // Store node info
    state.nodeInfo = status;
}

async function refreshBalance() {
    try {
        const data = await api.getBalance();
        state.balance = {
            available: data.available_balance || 0,
            staked: data.staked_balance || 0,
            total: data.total_balance || 0,
        };
        
        // Update wallet page
        document.getElementById('wallet-available').textContent = formatNumber(state.balance.available);
        document.getElementById('wallet-staked').textContent = formatNumber(state.balance.staked);
        document.getElementById('wallet-total').textContent = formatNumber(state.balance.total);
        document.getElementById('stake-available').textContent = formatNumber(state.balance.available);
        
        // Update dashboard stat
        document.getElementById('stat-balance').textContent = formatNumber(state.balance.available);
    } catch (e) {
        console.error('Failed to refresh balance:', e);
    }
}

async function refreshPeers() {
    try {
        const data = await api.getPeers();
        state.peers = data.connected || [];
        
        const tbody = document.getElementById('peers-table-body');
        if (!tbody) return;
        
        if (state.peers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No peers connected</td></tr>';
            return;
        }
        
        tbody.innerHTML = state.peers.map(peer => `
            <tr>
                <td><code>${truncateId(peer.peer_id)}</code></td>
                <td>${peer.address || 'N/A'}</td>
                <td><span class="badge badge-success">Connected</span></td>
                <td>${formatTimestamp(peer.last_seen)}</td>
            </tr>
        `).join('');
        
        // Update badge
        document.getElementById('peer-count-badge').textContent = state.peers.length;
    } catch (e) {
        console.error('Failed to refresh peers:', e);
    }
}

async function refreshJobs() {
    try {
        const data = await api.getJobs();
        state.jobs = data.jobs || [];
        
        const tbody = document.getElementById('jobs-table-body');
        if (!tbody) return;
        
        if (state.jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No jobs submitted</td></tr>';
            return;
        }
        
        tbody.innerHTML = state.jobs.map(job => {
            const statusClass = {
                'pending': 'badge-warning',
                'running': 'badge-info',
                'completed': 'badge-success',
                'failed': 'badge-error',
            }[job.status] || 'badge-info';
            
            return `
                <tr>
                    <td><code>${truncateId(job.job_id)}</code></td>
                    <td>${job.job_type}</td>
                    <td><span class="badge ${statusClass}">${job.status}</span></td>
                    <td>${formatNumber(job.ftns_budget)} FTNS</td>
                    <td>${formatTimestamp(job.created_at)}</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="viewJob('${job.job_id}')">View</button>
                    </td>
                </tr>
            `;
        }).join('');
        
        // Update badge
        const activeCount = state.jobs.filter(j => j.status === 'pending' || j.status === 'running').length;
        document.getElementById('active-jobs-badge').textContent = activeCount;
    } catch (e) {
        console.error('Failed to refresh jobs:', e);
    }
}

async function refreshTransactions() {
    try {
        const data = await api.getTransactions();
        state.transactions = data.transactions || [];
        
        const tbody = document.getElementById('transactions-table-body');
        if (!tbody) return;
        
        if (state.transactions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No transactions yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = state.transactions.map(tx => {
            const typeClass = {
                'transfer': 'badge-info',
                'reward': 'badge-success',
                'stake': 'badge-warning',
                'unstake': 'badge-warning',
            }[tx.type] || 'badge-info';
            
            return `
                <tr>
                    <td><code>${truncateId(tx.tx_id)}</code></td>
                    <td><span class="badge ${typeClass}">${tx.type}</span></td>
                    <td><code>${truncateId(tx.from)}</code></td>
                    <td><code>${truncateId(tx.to)}</code></td>
                    <td>${formatNumber(tx.amount)} FTNS</td>
                    <td>${formatTimestamp(tx.timestamp)}</td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error('Failed to refresh transactions:', e);
    }
}

async function refreshAgents() {
    try {
        const data = await api.getAgents();
        state.agents = data.agents || [];
        
        const tbody = document.getElementById('agents-table-body');
        if (!tbody) return;
        
        if (state.agents.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No agents registered</td></tr>';
            return;
        }
        
        tbody.innerHTML = state.agents.map(agent => {
            const statusClass = {
                'online': 'badge-success',
                'offline': 'badge-error',
                'paused': 'badge-warning',
            }[agent.status] || 'badge-info';
            
            return `
                <tr>
                    <td><code>${truncateId(agent.agent_id)}</code></td>
                    <td>${agent.agent_name || 'Unnamed'}</td>
                    <td><span class="badge ${statusClass}">${agent.status}</span></td>
                    <td>${formatNumber(agent.allowance || 0)} FTNS</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="viewAgent('${agent.agent_id}')">View</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error('Failed to refresh agents:', e);
    }
}

async function refreshActivity() {
    // Combine recent jobs and transactions for activity feed
    const activityList = document.getElementById('activity-list');
    if (!activityList) return;
    
    const activities = [];
    
    // Add recent jobs
    state.jobs.slice(0, 5).forEach(job => {
        activities.push({
            type: 'job',
            data: job,
            timestamp: new Date(job.created_at || 0),
        });
    });
    
    // Add recent transactions
    state.transactions.slice(0, 5).forEach(tx => {
        activities.push({
            type: 'transaction',
            data: tx,
            timestamp: new Date(tx.timestamp || 0),
        });
    });
    
    // Sort by timestamp
    activities.sort((a, b) => b.timestamp - a.timestamp);
    
    if (activities.length === 0) {
        activityList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                </div>
                <div class="empty-state-title">No recent activity</div>
                <p>Activity will appear here as you use PRSM</p>
            </div>
        `;
        return;
    }
    
    activityList.innerHTML = activities.slice(0, 10).map(activity => {
        if (activity.type === 'job') {
            return `
                <div class="activity-item" style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                    <div class="flex items-center gap-sm">
                        <span class="badge badge-info">Job</span>
                        <span>${activity.data.job_type}</span>
                        <span class="text-muted">${truncateId(activity.data.job_id)}</span>
                    </div>
                    <div class="text-muted" style="font-size: 0.875rem;">${formatTimestamp(activity.data.created_at)}</div>
                </div>
            `;
        } else {
            return `
                <div class="activity-item" style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                    <div class="flex items-center gap-sm">
                        <span class="badge badge-success">Transfer</span>
                        <span>${formatNumber(activity.data.amount)} FTNS</span>
                    </div>
                    <div class="text-muted" style="font-size: 0.875rem;">${formatTimestamp(activity.data.timestamp)}</div>
                </div>
            `;
        }
    }).join('');
}

// ── Page Navigation ───────────────────────────────────────────────────────────────
function showPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
        page.classList.remove('active');
    });
    
    // Show selected page
    const page = document.getElementById(`page-${pageName}`);
    if (page) {
        page.classList.remove('hidden');
        page.classList.add('active');
    }
    
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === pageName) {
            item.classList.add('active');
        }
    });
    
    // Update title
    const titles = {
        'dashboard': 'Dashboard',
        'network': 'Network',
        'jobs': 'Jobs',
        'agents': 'Agents',
        'wallet': 'Wallet',
        'transactions': 'Transactions',
        'staking': 'Staking',
        'content': 'Content',
        'settings': 'Settings',
    };
    document.getElementById('page-title').textContent = titles[pageName] || 'Dashboard';
    
    // Refresh data for the page
    switch (pageName) {
        case 'dashboard':
            refreshAll();
            break;
        case 'network':
            refreshPeers();
            break;
        case 'jobs':
            refreshJobs();
            break;
        case 'agents':
            refreshAgents();
            break;
        case 'wallet':
            refreshBalance();
            break;
        case 'transactions':
            refreshTransactions();
            break;
        case 'staking':
            refreshBalance();
            break;
    }
}

// ── Form Handlers ─────────────────────────────────────────────────────────────────
async function handleJobSubmit(event) {
    event.preventDefault();
    
    const jobType = document.getElementById('job-type').value;
    const budget = parseFloat(document.getElementById('job-budget').value);
    const payloadStr = document.getElementById('job-payload').value;
    
    let payload = {};
    if (payloadStr.trim()) {
        try {
            payload = JSON.parse(payloadStr);
        } catch (e) {
            showToast('Invalid JSON in payload', 'error');
            return;
        }
    }
    
    try {
        const result = await api.submitJob(jobType, budget, payload);
        showToast(`Job submitted: ${result.job_id}`, 'success');
        document.getElementById('job-submit-form').reset();
        refreshJobs();
    } catch (e) {
        showToast(`Failed to submit job: ${e.message}`, 'error');
    }
}

async function handleTransfer(event) {
    event.preventDefault();
    
    const toWallet = document.getElementById('transfer-to').value;
    const amount = parseFloat(document.getElementById('transfer-amount').value);
    const description = document.getElementById('transfer-description').value || null;
    
    try {
        const result = await api.transfer(toWallet, amount, description);
        showToast(`Transfer successful: ${result.tx_id}`, 'success');
        document.getElementById('transfer-form').reset();
        refreshBalance();
        refreshTransactions();
    } catch (e) {
        showToast(`Transfer failed: ${e.message}`, 'error');
    }
}

async function handleStake(event) {
    event.preventDefault();
    
    const amount = parseFloat(document.getElementById('stake-amount').value);
    const duration = parseInt(document.getElementById('stake-duration').value);
    
    try {
        const result = await api.stake(amount, duration);
        showToast(`Staked ${formatNumber(amount)} FTNS`, 'success');
        document.getElementById('stake-form').reset();
        refreshBalance();
    } catch (e) {
        showToast(`Staking failed: ${e.message}`, 'error');
    }
}

async function handleContentSearch(event) {
    event.preventDefault();
    
    const query = document.getElementById('content-search-query').value;
    if (!query.trim()) return;
    
    try {
        const result = await api.searchContent(query);
        const resultsDiv = document.getElementById('content-results');
        
        if (!result.results || result.results.length === 0) {
            resultsDiv.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        </svg>
                    </div>
                    <div class="empty-state-title">No results found</div>
                    <p>Try a different search query</p>
                </div>
            `;
            return;
        }
        
        resultsDiv.innerHTML = `
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>CID</th>
                            <th>Filename</th>
                            <th>Size</th>
                            <th>Creator</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${result.results.map(item => `
                            <tr>
                                <td><code>${truncateId(item.cid, 16)}</code></td>
                                <td>${item.filename || 'N/A'}</td>
                                <td>${formatNumber(item.size_bytes || 0)} bytes</td>
                                <td><code>${truncateId(item.creator_id)}</code></td>
                                <td>${formatTimestamp(item.created_at)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        showToast(`Search failed: ${e.message}`, 'error');
    }
}

// ── Authentication ───────────────────────────────────────────────────────────────
function showLoginModal() {
    document.getElementById('login-modal').classList.add('active');
    document.getElementById('dashboard').classList.add('hidden');
}

function hideLoginModal() {
    document.getElementById('login-modal').classList.remove('active');
    document.getElementById('dashboard').classList.remove('hidden');
}

async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    try {
        const result = await api.login(username, password);
        state.token = result.access_token;
        state.user = {
            id: result.user_id,
            username: result.username,
        };
        
        localStorage.setItem('prsm_token', state.token);
        localStorage.setItem('prsm_user', JSON.stringify(state.user));
        
        hideLoginModal();
        initializeDashboard();
        showToast('Login successful', 'success');
    } catch (e) {
        document.getElementById('login-error').textContent = e.message;
        document.getElementById('login-error').classList.remove('hidden');
    }
}

async function handleLogout() {
    await api.logout();
    state.token = null;
    state.user = null;
    localStorage.removeItem('prsm_token');
    localStorage.removeItem('prsm_user');
    showLoginModal();
}

// ── Theme Management ─────────────────────────────────────────────────────────────
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('prsm_theme', theme);
}

function initTheme() {
    const savedTheme = localStorage.getItem('prsm_theme') || 'dark';
    setTheme(savedTheme);
    
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
        themeSelect.value = savedTheme;
        themeSelect.addEventListener('change', (e) => setTheme(e.target.value));
    }
}

// ── Initialization ────────────────────────────────────────────────────────────────
async function initializeDashboard() {
    // Hide loading screen
    document.getElementById('loading-screen').style.display = 'none';
    
    // Initialize theme
    initTheme();
    
    // Update user info
    if (state.user) {
        document.getElementById('user-name').textContent = state.user.username;
        document.getElementById('user-avatar').textContent = state.user.username.charAt(0).toUpperCase();
    }
    
    // Set node ID in settings
    try {
        const nodeInfo = await api.getNodeInfo();
        document.getElementById('settings-node-id').value = nodeInfo.node_id || 'N/A';
    } catch (e) {
        document.getElementById('settings-node-id').value = 'N/A';
    }
    
    // Connect WebSocket
    connectWebSocket();
    
    // Initial data load
    await refreshAll();
    
    // Setup navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page) showPage(page);
        });
    });
    
    // Setup form handlers
    document.getElementById('job-submit-form')?.addEventListener('submit', handleJobSubmit);
    document.getElementById('transfer-form')?.addEventListener('submit', handleTransfer);
    document.getElementById('stake-form')?.addEventListener('submit', handleStake);
    document.getElementById('content-search-form')?.addEventListener('submit', handleContentSearch);
    document.getElementById('login-form')?.addEventListener('submit', handleLogin);
    
    // Setup periodic refresh
    setInterval(refreshAll, CONFIG.STATUS_UPDATE_INTERVAL);
}

async function refreshAll() {
    await Promise.all([
        refreshBalance(),
        refreshJobs(),
        refreshPeers(),
        refreshTransactions(),
        refreshAgents(),
    ]);
    refreshActivity();
}

// ── Start Application ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Check if already authenticated
    if (state.token) {
        try {
            // Verify token is still valid
            state.user = await api.getCurrentUser();
            localStorage.setItem('prsm_user', JSON.stringify(state.user));
            await initializeDashboard();
        } catch (e) {
            // Token invalid, show login
            showLoginModal();
            document.getElementById('loading-screen').style.display = 'none';
        }
    } else {
        // Demo mode - allow access without login
        await initializeDashboard();
    }
});

// ── Global Functions for HTML onclick handlers ───────────────────────────────────
window.refreshActivity = refreshActivity;
window.refreshPeers = refreshPeers;
window.refreshJobs = refreshJobs;
window.refreshTransactions = refreshTransactions;
window.refreshAgents = refreshAgents;
window.viewJob = (jobId) => showToast(`Job details: ${jobId}`, 'info');
window.viewAgent = (agentId) => showToast(`Agent details: ${agentId}`, 'info');