/**
 * PRSM Production Integration
 * 
 * Enhanced JavaScript client that connects the existing UI to the real
 * PostgreSQL + pgvector backend for investor demonstrations.
 */

class PRSMProductionClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.wsURL = baseURL.replace('http://', 'ws://').replace('https://', 'wss://');
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // UI element references
        this.initializeUIElements();
        
        // Connect to backend
        this.connectWebSocket();
        this.checkSystemHealth();
        
        // Start live metrics updates
        this.startLiveMetricsUpdates();
    }
    
    initializeUIElements() {
        // Create metrics dashboard in the right panel
        this.createMetricsDashboard();
        
        // Add PRSM query interface to conversation area
        this.enhanceConversationInterface();
        
        // Add system status indicators
        this.addSystemStatusIndicators();
    }
    
    createMetricsDashboard() {
        // Find the right panel tabs container
        const rightPanel = document.querySelector('.right-panel .tab-content');
        if (!rightPanel) return;
        
        // Create PRSM Metrics tab
        const metricsTab = document.createElement('div');
        metricsTab.className = 'tab-pane';
        metricsTab.id = 'prsm-metrics';
        metricsTab.innerHTML = `
            <div class="prsm-dashboard">
                <div class="dashboard-header">
                    <h3><i class="fas fa-chart-line"></i> PRSM Live Metrics</h3>
                    <div class="system-status">
                        <span class="status-indicator" id="system-status">
                            <i class="fas fa-circle"></i> Initializing...
                        </span>
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-header">
                            <i class="fas fa-search"></i>
                            <span>Query Performance</span>
                        </div>
                        <div class="metric-value" id="queries-processed">-</div>
                        <div class="metric-label">Queries Processed</div>
                        <div class="metric-sub">
                            <span id="success-rate">-%</span> success rate
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <i class="fas fa-database"></i>
                            <span>Database</span>
                        </div>
                        <div class="metric-value" id="database-vectors">-</div>
                        <div class="metric-label">Vectors Stored</div>
                        <div class="metric-sub">
                            <span id="avg-response-time">-ms</span> avg response
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <i class="fas fa-coins"></i>
                            <span>FTNS Economics</span>
                        </div>
                        <div class="metric-value" id="ftns-volume">-</div>
                        <div class="metric-label">Total Volume</div>
                        <div class="metric-sub">
                            <span id="creators-compensated">-</span> creators paid
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <i class="fas fa-robot"></i>
                            <span>AI Integration</span>
                        </div>
                        <div class="metric-value" id="embedding-calls">-</div>
                        <div class="metric-label">API Calls</div>
                        <div class="metric-sub">
                            Provider: <span id="embedding-provider">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="demo-scenarios">
                    <h4><i class="fas fa-play-circle"></i> Investor Demo Scenarios</h4>
                    <div class="scenario-buttons">
                        <button class="scenario-btn" onclick="prsmClient.runScenario('academic_research')">
                            <i class="fas fa-graduation-cap"></i>
                            Academic Research
                        </button>
                        <button class="scenario-btn" onclick="prsmClient.runScenario('enterprise_knowledge')">
                            <i class="fas fa-building"></i>
                            Enterprise Knowledge
                        </button>
                        <button class="scenario-btn" onclick="prsmClient.runScenario('investor_technical')">
                            <i class="fas fa-cog"></i>
                            Technical Deep Dive
                        </button>
                    </div>
                </div>
                
                <div class="recent-activity">
                    <h4><i class="fas fa-history"></i> Recent Activity</h4>
                    <div class="activity-feed" id="activity-feed">
                        <div class="activity-item">
                            <span class="activity-time">System ready</span>
                            <span class="activity-desc">PRSM production demo initialized</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        rightPanel.appendChild(metricsTab);
        
        // Add tab button to the tab navigation
        const tabNav = document.querySelector('.tab-nav');
        if (tabNav) {
            const metricsTabBtn = document.createElement('button');
            metricsTabBtn.className = 'tab-btn';
            metricsTabBtn.setAttribute('data-tab', 'prsm-metrics');
            metricsTabBtn.innerHTML = '<i class="fas fa-chart-line"></i> PRSM Metrics';
            tabNav.appendChild(metricsTabBtn);
            
            // Add click handler for the new tab
            metricsTabBtn.addEventListener('click', () => {
                // Hide all tabs
                document.querySelectorAll('.tab-pane').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Show metrics tab
                metricsTab.classList.add('active');
                metricsTabBtn.classList.add('active');
            });
        }
    }
    
    enhanceConversationInterface() {
        // Find the message input area
        const messageInput = document.querySelector('.message-input-area');
        if (!messageInput) return;
        
        // Add PRSM query enhancement
        const queryEnhancement = document.createElement('div');
        queryEnhancement.className = 'prsm-query-enhancement';
        queryEnhancement.innerHTML = `
            <div class="query-options">
                <label>
                    <input type="checkbox" id="show-reasoning" checked>
                    Show detailed reasoning
                </label>
                <label>
                    User:
                    <select id="user-select">
                        <option value="demo_investor">Demo Investor</option>
                        <option value="researcher_alice">Researcher Alice</option>
                        <option value="enterprise_org">Enterprise Org</option>
                        <option value="data_scientist_bob">Data Scientist Bob</option>
                    </select>
                </label>
            </div>
        `;
        
        messageInput.insertBefore(queryEnhancement, messageInput.firstChild);
        
        // Override the send button to use PRSM API
        const sendBtn = document.querySelector('.send-btn');
        if (sendBtn) {
            sendBtn.addEventListener('click', this.handlePRSMQuery.bind(this));
        }
        
        // Handle Enter key
        const textInput = document.querySelector('.message-input');
        if (textInput) {
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handlePRSMQuery();
                }
            });
        }
    }
    
    addSystemStatusIndicators() {
        // Add status indicator to the top panel
        const header = document.querySelector('.panel-header h3');
        if (header) {
            const statusIndicator = document.createElement('div');
            statusIndicator.className = 'prsm-status-indicator';
            statusIndicator.innerHTML = `
                <span class="status-dot initializing"></span>
                <span class="status-text">PRSM Initializing...</span>
            `;
            header.appendChild(statusIndicator);
        }
    }
    
    async handlePRSMQuery() {
        const textInput = document.querySelector('.message-input');
        const showReasoning = document.querySelector('#show-reasoning')?.checked || false;
        const selectedUser = document.querySelector('#user-select')?.value || 'demo_investor';
        
        if (!textInput || !textInput.value.trim()) return;
        
        const query = textInput.value.trim();
        textInput.value = '';
        
        // Add user message to conversation
        this.addMessageToConversation(query, 'user');
        
        // Add thinking indicator
        const thinkingId = this.addThinkingIndicator();
        
        try {
            // Send query to PRSM backend
            const response = await fetch(`${this.baseURL}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    user_id: selectedUser,
                    show_reasoning: showReasoning,
                    max_results: 5
                })
            });
            
            const result = await response.json();
            
            // Remove thinking indicator
            this.removeThinkingIndicator(thinkingId);
            
            if (result.success) {
                // Add PRSM response to conversation
                this.addPRSMResponse(result);
                
                // Update activity feed
                this.addActivityItem(`Query processed: "${query.substring(0, 30)}..."`, result.results_found + ' results found');
                
            } else {
                this.addMessageToConversation(`Error: ${result.error}`, 'assistant error');
            }
            
        } catch (error) {
            this.removeThinkingIndicator(thinkingId);
            this.addMessageToConversation(`Connection error: ${error.message}`, 'assistant error');
            console.error('PRSM query error:', error);
        }
    }
    
    addMessageToConversation(content, sender) {
        const conversationArea = document.querySelector('.conversation-messages');
        if (!conversationArea) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${content}</div>
            </div>
        `;
        
        conversationArea.appendChild(messageDiv);
        conversationArea.scrollTop = conversationArea.scrollHeight;
    }
    
    addPRSMResponse(result) {
        const conversationArea = document.querySelector('.conversation-messages');
        if (!conversationArea) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant prsm-response';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="prsm-response-header">
                    <i class="fas fa-brain"></i>
                    <span>PRSM Response</span>
                    <div class="response-metrics">
                        <span class="metric">${result.processing_time.toFixed(2)}s</span>
                        <span class="metric">${result.results_found} results</span>
                        <span class="metric">${result.query_cost.toFixed(4)} FTNS</span>
                    </div>
                </div>
                <div class="message-text">${this.formatPRSMResponse(result.response)}</div>
                <div class="prsm-response-footer">
                    <span class="intent-tag">${result.intent_category}</span>
                    <span class="complexity-tag">Complexity: ${(result.complexity_score * 100).toFixed(0)}%</span>
                    <span class="creators-tag">${result.creators_compensated} creators compensated</span>
                </div>
            </div>
        `;
        
        conversationArea.appendChild(messageDiv);
        conversationArea.scrollTop = conversationArea.scrollHeight;
    }
    
    formatPRSMResponse(response) {
        // Convert markdown-like formatting to HTML
        return response
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/^\* (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }
    
    addThinkingIndicator() {
        const thinkingId = 'thinking-' + Date.now();
        const conversationArea = document.querySelector('.conversation-messages');
        if (!conversationArea) return thinkingId;
        
        const thinkingDiv = document.createElement('div');
        thinkingDiv.id = thinkingId;
        thinkingDiv.className = 'message assistant thinking';
        thinkingDiv.innerHTML = `
            <div class="message-content">
                <div class="thinking-indicator">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span class="thinking-text">PRSM is processing your query...</span>
                </div>
            </div>
        `;
        
        conversationArea.appendChild(thinkingDiv);
        conversationArea.scrollTop = conversationArea.scrollHeight;
        
        return thinkingId;
    }
    
    removeThinkingIndicator(thinkingId) {
        const thinkingDiv = document.getElementById(thinkingId);
        if (thinkingDiv) {
            thinkingDiv.remove();
        }
    }
    
    async connectWebSocket() {
        try {
            this.ws = new WebSocket(`${this.wsURL}/ws`);
            
            this.ws.onopen = () => {
                console.log('✅ Connected to PRSM WebSocket');
                this.reconnectAttempts = 0;
                this.updateSystemStatus('connected', 'Connected to PRSM system');
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            this.ws.onclose = () => {
                console.log('🔌 PRSM WebSocket disconnected');
                this.updateSystemStatus('disconnected', 'Disconnected from PRSM system');
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ PRSM WebSocket error:', error);
                this.updateSystemStatus('error', 'Connection error');
            };
            
        } catch (error) {
            console.error('Failed to connect to PRSM WebSocket:', error);
            this.updateSystemStatus('error', 'Failed to connect');
        }
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'system_status':
                this.updateSystemStatus(message.status, message.message);
                break;
            case 'live_metrics':
                this.updateLiveMetrics(message.data);
                break;
            case 'query_completed':
                this.addActivityItem(`Query by ${message.user_id}`, `${message.results_found} results in ${message.processing_time.toFixed(2)}s`);
                break;
            case 'scenario_completed':
                this.addActivityItem(`Demo scenario: ${message.title}`, `${message.queries_completed} queries completed`);
                break;
            case 'system_error':
                this.updateSystemStatus('error', message.message);
                break;
        }
    }
    
    updateSystemStatus(status, message) {
        const statusIndicator = document.querySelector('#system-status');
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusIndicator) {
            statusIndicator.innerHTML = `<i class="fas fa-circle ${status}"></i> ${message}`;
        }
        
        if (statusDot && statusText) {
            statusDot.className = `status-dot ${status}`;
            statusText.textContent = message;
        }
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            const health = await response.json();
            
            if (health.status === 'healthy') {
                this.updateSystemStatus('ready', 'System ready for queries');
            } else {
                this.updateSystemStatus('warning', health.error || 'System initializing');
            }
        } catch (error) {
            this.updateSystemStatus('error', 'Cannot reach PRSM backend');
        }
    }
    
    startLiveMetricsUpdates() {
        // Update metrics every 5 seconds
        setInterval(async () => {
            try {
                const response = await fetch(`${this.baseURL}/api/metrics/live`);
                const metrics = await response.json();
                this.updateLiveMetrics(metrics);
            } catch (error) {
                console.warn('Failed to update live metrics:', error);
            }
        }, 5000);
    }
    
    updateLiveMetrics(metrics) {
        // Update query performance
        const queriesProcessed = document.querySelector('#queries-processed');
        const successRate = document.querySelector('#success-rate');
        if (queriesProcessed) queriesProcessed.textContent = metrics.queries_processed || 0;
        if (successRate) successRate.textContent = (metrics.success_rate || 0).toFixed(1) + '%';
        
        // Update database metrics
        const databaseVectors = document.querySelector('#database-vectors');
        const avgResponseTime = document.querySelector('#avg-response-time');
        if (databaseVectors) databaseVectors.textContent = metrics.database_operations || 0;
        if (avgResponseTime) avgResponseTime.textContent = ((metrics.average_response_time || 0) * 1000).toFixed(0) + 'ms';
        
        // Update FTNS economics
        const ftnsVolume = document.querySelector('#ftns-volume');
        const creatorsCompensated = document.querySelector('#creators-compensated');
        if (ftnsVolume) ftnsVolume.textContent = (metrics.total_ftns_volume || 0).toFixed(4);
        if (creatorsCompensated) creatorsCompensated.textContent = metrics.total_creators_compensated || 0;
        
        // Update AI integration
        const embeddingCalls = document.querySelector('#embedding-calls');
        const embeddingProvider = document.querySelector('#embedding-provider');
        if (embeddingCalls) embeddingCalls.textContent = metrics.embedding_api_calls || 0;
        if (embeddingProvider) embeddingProvider.textContent = metrics.embedding_provider || 'Unknown';
    }
    
    addActivityItem(title, description) {
        const activityFeed = document.querySelector('#activity-feed');
        if (!activityFeed) return;
        
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <span class="activity-time">${new Date().toLocaleTimeString()}</span>
            <span class="activity-title">${title}</span>
            <span class="activity-desc">${description}</span>
        `;
        
        activityFeed.insertBefore(activityItem, activityFeed.firstChild);
        
        // Keep only last 10 items
        const items = activityFeed.querySelectorAll('.activity-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    async runScenario(scenarioId) {
        try {
            this.addActivityItem('Demo scenario started', `Running ${scenarioId} scenario`);
            
            const response = await fetch(`${this.baseURL}/api/demo/run_scenario/${scenarioId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // Add scenario results to conversation
                this.addMessageToConversation(`Demo Scenario: ${result.scenario.title}`, 'user');
                
                for (const queryResult of result.results) {
                    this.addMessageToConversation(queryResult.query, 'user');
                    
                    if (queryResult.success) {
                        this.addMessageToConversation(
                            `Found ${queryResult.results_found} results in ${queryResult.processing_time.toFixed(2)}s (Cost: ${queryResult.query_cost.toFixed(4)} FTNS)`,
                            'assistant'
                        );
                    } else {
                        this.addMessageToConversation('Query failed', 'assistant error');
                    }
                }
                
                // Add summary
                const summary = result.summary;
                this.addMessageToConversation(
                    `Scenario Summary: ${summary.queries_completed} queries, ${summary.total_results_found} total results, ${summary.total_processing_time.toFixed(2)}s total time, ${summary.total_cost.toFixed(4)} FTNS total cost`,
                    'assistant'
                );
                
            } else {
                this.addActivityItem('Scenario failed', result.detail || 'Unknown error');
            }
            
        } catch (error) {
            console.error('Failed to run scenario:', error);
            this.addActivityItem('Scenario error', error.message);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`Attempting to reconnect in ${delay}ms...`);
        setTimeout(() => {
            this.connectWebSocket();
        }, delay);
    }
}

// Initialize PRSM client when page loads
let prsmClient = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize PRSM production client
    prsmClient = new PRSMProductionClient();
    
    console.log('🚀 PRSM Production Client initialized');
});