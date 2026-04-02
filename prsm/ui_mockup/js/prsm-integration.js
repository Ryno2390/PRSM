/**
 * PRSM Production Integration
 * 
 * Enhanced JavaScript client that connects the existing UI to the real
 * PostgreSQL + pgvector backend for live demonstrations.
 * 
 * Features:
 * - Real-time WebSocket communication
 * - FTNS token balance tracking
 * - Model discovery and semantic search
 * - Streaming AI responses
 * - System health monitoring
 * - Authentication integration
 */

class PRSMProductionClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.wsURL = baseURL.replace('http://', 'ws://').replace('https://', 'wss://');
        this.ws = null;
        this.conversationWs = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.currentUserId = 'demo-user'; // In production, get from auth
        this.currentConversationId = this.generateConversationId();
        
        // State management
        this.state = {
            isConnected: false,
            systemHealth: {},
            ftnsBalance: 0,
            availableModels: [],
            currentConversation: [],
            isTyping: false
        };
        
        // Initialize components
        this.initializeUIElements();
        this.setupEventListeners();
        
        // Connect to backend
        this.connectWebSocket();
        this.checkSystemHealth();
        this.loadUserBalance();
        this.loadAvailableModels();
        
        // Start live updates
        this.startLiveMetricsUpdates();
        
        console.log('🚀 PRSM Production Client initialized');
    }
    
    generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupEventListeners() {
        // Send button handler
        const sendBtn = document.getElementById('send-message-btn');
        const promptInput = document.getElementById('prompt-input');
        
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        if (promptInput) {
            promptInput.addEventListener('keydown', (e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Typing indicators
            promptInput.addEventListener('input', () => this.handleTypingStart());
        }
        
        // Model selection handler
        const modelSelect = document.getElementById('ai-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', () => this.updateSelectedModel());
        }
    }
    
    // === WebSocket Communication ===
    
    connectWebSocket() {
        try {
            // Main WebSocket for general updates
            this.ws = new WebSocket(`${this.wsURL}/ws/${this.currentUserId}`);
            
            this.ws.onopen = () => {
                console.log('✅ Main WebSocket connected');
                this.state.isConnected = true;
                this.updateConnectionStatus('Connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('🔌 Main WebSocket disconnected');
                this.state.isConnected = false;
                this.updateConnectionStatus('Disconnected');
                this.scheduleReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.updateConnectionStatus('Error');
            };
            
            // Conversation WebSocket for streaming
            this.connectConversationWebSocket();
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus('Failed');
            this.scheduleReconnect();
        }
    }
    
    connectConversationWebSocket() {
        try {
            this.conversationWs = new WebSocket(
                `${this.wsURL}/ws/conversation/${this.currentUserId}/${this.currentConversationId}`
            );
            
            this.conversationWs.onopen = () => {
                console.log('✅ Conversation WebSocket connected');
            };
            
            this.conversationWs.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleConversationMessage(data);
                } catch (e) {
                    console.error('Failed to parse conversation message:', e);
                }
            };
            
            this.conversationWs.onclose = () => {
                console.log('🔌 Conversation WebSocket disconnected');
            };
            
        } catch (error) {
            console.error('Failed to connect conversation WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'connection_established':
                console.log('📡 Connection established:', data);
                break;
                
            case 'balance_update':
                this.state.ftnsBalance = data.balance;
                this.updateBalanceDisplay();
                break;
                
            case 'system_status_update':
                this.state.systemHealth = data.health;
                this.updateHealthDisplay();
                break;
                
            case 'notification':
                this.showNotification(data.message, data.notification_type);
                break;
                
            default:
                console.log('Unknown WebSocket message:', data);
        }
    }
    
    handleConversationMessage(data) {
        switch (data.type) {
            case 'ai_response_chunk':
                this.appendToConversation(data.partial_content, 'ai', false);
                break;
                
            case 'ai_response_complete':
                this.appendToConversation(data.final_content, 'ai', true);
                this.updateTokenUsage(data.context_tokens);
                this.hideTypingIndicator();
                break;
                
            case 'ai_typing':
                if (data.typing) {
                    this.showTypingIndicator();
                } else {
                    this.hideTypingIndicator();
                }
                break;
                
            case 'typing_indicator':
                // Handle other users typing (for multi-user conversations)
                break;
                
            default:
                console.log('Unknown conversation message:', data);
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
            console.log(`🔄 Reconnecting in ${delay}ms...`);
            
            setTimeout(() => {
                this.reconnectAttempts++;
                this.connectWebSocket();
            }, delay);
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.updateConnectionStatus('Failed to reconnect');
        }
    }
    
    // === API Communication ===
    
    async apiCall(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer demo-token` // In production, use real auth
            }
        };
        
        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API call to ${endpoint} failed:`, error);
            throw error;
        }
    }
    
    async checkSystemHealth() {
        try {
            const health = await this.apiCall('/health');
            this.state.systemHealth = health;
            this.updateHealthDisplay();
            console.log('💚 System health check completed');
        } catch (error) {
            console.error('❤️‍🩹 Health check failed:', error);
            this.updateConnectionStatus('API Unavailable');
        }
    }
    
    async loadUserBalance() {
        try {
            const balance = await this.apiCall(`/users/${this.currentUserId}/balance`);
            this.state.ftnsBalance = balance.balance;
            this.updateBalanceDisplay();
            console.log('💰 Balance loaded:', balance);
        } catch (error) {
            console.error('Failed to load balance:', error);
        }
    }
    
    async loadAvailableModels() {
        try {
            const models = await this.apiCall('/models');
            this.state.availableModels = models;
            this.updateModelSelector();
            console.log('🤖 Models loaded:', models);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }
    
    async searchModels(query) {
        try {
            const results = await this.apiCall('/models/search/semantic', {
                method: 'POST',
                body: JSON.stringify({ query, top_k: 10 })
            });
            return results;
        } catch (error) {
            console.error('Model search failed:', error);
            return { results: [] };
        }
    }
    
    // === Message Handling ===
    
    sendMessage() {
        const promptInput = document.getElementById('prompt-input');
        if (!promptInput) return;
        
        const content = promptInput.value.trim();
        if (!content) return;
        
        // Add user message to conversation
        this.appendToConversation(content, 'user', true);
        
        // Clear input
        promptInput.value = '';
        
        // Send via WebSocket for streaming response
        if (this.conversationWs && this.conversationWs.readyState === WebSocket.OPEN) {
            this.conversationWs.send(JSON.stringify({
                type: 'send_message',
                content: content,
                timestamp: Date.now()
            }));
        } else {
            // Fallback to REST API
            this.sendMessageViaAPI(content);
        }
    }
    
    async sendMessageViaAPI(content) {
        try {
            this.showTypingIndicator();
            
            const response = await this.apiCall('/ui/conversations/' + this.currentConversationId + '/messages/streaming', {
                method: 'POST',
                body: JSON.stringify({
                    content: content,
                    user_id: this.currentUserId
                })
            });
            
            console.log('Message sent via API:', response);
        } catch (error) {
            console.error('Failed to send message via API:', error);
            this.hideTypingIndicator();
        }
    }
    
    appendToConversation(content, role, isComplete) {
        const responseArea = document.querySelector('.response-area');
        if (!responseArea) return;
        
        // Find or create message container
        let messageContainer = responseArea.querySelector(`.message.${role}:last-child`);
        
        if (!messageContainer || (role === 'user') || isComplete) {
            // Create new message container
            messageContainer = document.createElement('div');
            messageContainer.className = `message ${role}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = content;
            
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date().toLocaleTimeString();
            
            messageContainer.appendChild(messageContent);
            messageContainer.appendChild(messageTime);
            responseArea.appendChild(messageContainer);
        } else {
            // Update existing message (for streaming)
            const messageContent = messageContainer.querySelector('.message-content');
            if (messageContent) {
                messageContent.textContent = content;
            }
        }
        
        // Scroll to bottom
        responseArea.scrollTop = responseArea.scrollHeight;
    }
    
    handleTypingStart() {
        if (this.conversationWs && this.conversationWs.readyState === WebSocket.OPEN) {
            this.conversationWs.send(JSON.stringify({
                type: 'typing_start',
                timestamp: Date.now()
            }));
        }
    }
    
    showTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'block';
        }
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }
    
    // === UI Update Methods ===
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('api-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = 'api-status ' + (
                status === 'Connected' ? 'connected' :
                status === 'Disconnected' ? 'disconnected' :
                'error'
            );
        }
    }
    
    updateBalanceDisplay() {
        // Update main balance display
        const balanceElements = document.querySelectorAll('.ftns-balance');
        balanceElements.forEach(el => {
            el.textContent = `${this.state.ftnsBalance.toFixed(2)} FTNS`;
        });
        
        // Update any specific balance containers
        const balanceContainer = document.getElementById('user-balance');
        if (balanceContainer) {
            balanceContainer.innerHTML = `
                <div class="balance-info">
                    <div class="balance-amount">${this.state.ftnsBalance.toFixed(2)} FTNS</div>
                    <div class="balance-label">Available Balance</div>
                </div>
            `;
        }
    }
    
    updateHealthDisplay() {
        const healthContainer = document.getElementById('system-health');
        if (!healthContainer || !this.state.systemHealth) return;
        
        const health = this.state.systemHealth;
        const overallStatus = health.status || 'unknown';
        
        healthContainer.innerHTML = `
            <div class="health-overview">
                <div class="health-status ${overallStatus}">
                    <i class="fas fa-${overallStatus === 'healthy' ? 'check-circle' : 'exclamation-triangle'}"></i>
                    System: ${overallStatus.toUpperCase()}
                </div>
                <div class="health-components">
                    ${Object.entries(health.components || {}).map(([component, status]) => `
                        <div class="component-status">
                            <span class="component-name">${component}</span>
                            <span class="component-health ${status.status}">${status.status}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    updateModelSelector() {
        const modelSelect = document.getElementById('ai-model-select');
        if (!modelSelect || !this.state.availableModels) return;
        
        // Clear existing options except NWTN
        const options = Array.from(modelSelect.options);
        options.forEach(option => {
            if (option.value !== 'nwtn-v1') {
                option.remove();
            }
        });
        
        // Add available models
        if (this.state.availableModels.teacher_models) {
            this.state.availableModels.teacher_models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id;
                option.textContent = `${model.name} (Teacher)`;
                modelSelect.appendChild(option);
            });
        }
        
        if (this.state.availableModels.specialist_models) {
            this.state.availableModels.specialist_models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id;
                option.textContent = `${model.name} (Specialist)`;
                modelSelect.appendChild(option);
            });
        }
    }
    
    updateTokenUsage(tokens) {
        const progressBar = document.querySelector('.context-progress-bar progress');
        const valueSpan = document.querySelector('.context-progress-bar .value');
        const costSpan = document.querySelector('.api-cost');
        
        if (progressBar) {
            progressBar.value = tokens;
        }
        
        if (valueSpan) {
            valueSpan.textContent = `${tokens} / ${progressBar ? progressBar.max : 4096}`;
        }
        
        if (costSpan) {
            const cost = (tokens * 0.00002).toFixed(4); // Rough cost estimate
            costSpan.textContent = `Cost: $${cost}`;
        }
    }
    
    updateSelectedModel() {
        const modelSelect = document.getElementById('ai-model-select');
        if (modelSelect) {
            console.log('Model selected:', modelSelect.value);
            // In production, this would affect routing
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        // Add to page
        const container = document.querySelector('.app-container');
        if (container) {
            container.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 5000);
        }
    }
    
    // === Dashboard Creation ===
    
    initializeUIElements() {
        // Create metrics dashboard in the right panel
        this.createMetricsDashboard();
        
        // Add PRSM-specific UI elements
        this.enhanceConversationInterface();
        
        // Add system status indicators
        this.addSystemStatusIndicators();
        
        // Add FTNS balance display
        this.addBalanceDisplay();
        
        // Add system status indicators
        this.addSystemStatusIndicators();
    }
    
    createMetricsDashboard() {
        // Find or create the right panel tab container
        const rightPanelNav = document.querySelector('.right-panel-nav');
        const rightPanelContent = document.querySelector('.right-panel-content-area');
        
        if (!rightPanelNav || !rightPanelContent) return;
        
        // Add PRSM Dashboard tab
        const dashboardTab = document.createElement('button');
        dashboardTab.className = 'nav-tab-btn';
        dashboardTab.setAttribute('data-target', 'prsm-dashboard');
        dashboardTab.innerHTML = '<i class="fas fa-chart-line"></i> PRSM';
        rightPanelNav.appendChild(dashboardTab);
        
        // Add dashboard content
        const dashboardContent = document.createElement('div');
        dashboardContent.className = 'content-section';
        dashboardContent.id = 'prsm-dashboard';
        dashboardContent.innerHTML = `
            <div class="prsm-dashboard">
                <div class="dashboard-header">
                    <h3>PRSM Live Dashboard</h3>
                    <div class="connection-indicator" id="dashboard-connection">
                        <span class="status-dot"></span>
                        <span class="status-text">Connecting...</span>
                    </div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-coins"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-label">FTNS Balance</div>
                            <div class="metric-value" id="user-balance">Loading...</div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-heart"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-label">System Health</div>
                            <div class="metric-value" id="system-health">Loading...</div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-label">Available Models</div>
                            <div class="metric-value" id="model-count">Loading...</div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-network-wired"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-label">P2P Network</div>
                            <div class="metric-value" id="network-status">Loading...</div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h4>Model Marketplace</h4>
                    <div class="model-search-container">
                        <input type="text" id="model-search" placeholder="Search models by capability..." />
                        <button id="search-models-btn" class="search-btn">
                            <i class="fas fa-search"></i>
                        </button>
                    </div>
                    <div class="model-results" id="model-search-results">
                        <p class="placeholder-text">Enter a search query to discover models...</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h4>Recent Activity</h4>
                    <div class="activity-feed" id="activity-feed">
                        <div class="activity-item">
                            <i class="fas fa-plug"></i>
                            <span>Connected to PRSM backend</span>
                            <span class="activity-time">${new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        rightPanelContent.appendChild(dashboardContent);
        
        // Add click handler for tab
        dashboardTab.addEventListener('click', () => {
            // Remove active class from all tabs
            document.querySelectorAll('.nav-tab-btn').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
            
            // Add active class to this tab and content
            dashboardTab.classList.add('active');
            dashboardContent.classList.add('active');
        });
        
        // Set up search functionality
        this.setupModelSearch();
    }
    
    setupModelSearch() {
        const searchInput = document.getElementById('model-search');
        const searchBtn = document.getElementById('search-models-btn');
        const resultsContainer = document.getElementById('model-search-results');
        
        const performSearch = async () => {
            const query = searchInput.value.trim();
            if (!query) return;
            
            resultsContainer.innerHTML = '<div class="loading">Searching models...</div>';
            
            try {
                const results = await this.searchModels(query);
                this.displaySearchResults(results, resultsContainer);
            } catch (error) {
                resultsContainer.innerHTML = '<div class="error">Search failed. Please try again.</div>';
            }
        };
        
        if (searchBtn) {
            searchBtn.addEventListener('click', performSearch);
        }
        
        if (searchInput) {
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    performSearch();
                }
            });
        }
    }
    
    displaySearchResults(results, container) {
        if (!results.results || results.results.length === 0) {
            container.innerHTML = '<p class="no-results">No models found matching your query.</p>';
            return;
        }
        
        const resultsHTML = results.results.map(model => `
            <div class="model-result">
                <div class="model-info">
                    <div class="model-name">${model.name || model.model_id}</div>
                    <div class="model-type">${model.model_type}</div>
                    ${model.specialization ? `<div class="model-specialization">${model.specialization}</div>` : ''}
                </div>
                <div class="model-score">
                    <div class="similarity-score">
                        ${(model.similarity_score * 100).toFixed(1)}% match
                    </div>
                    <div class="performance-score">
                        Performance: ${(model.performance_score * 100).toFixed(0)}%
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = resultsHTML;
    }
    
    enhanceConversationInterface() {
        // Add real-time status to the conversation area
        const responseArea = document.querySelector('.response-area');
        if (responseArea && responseArea.children.length <= 1) {
            responseArea.innerHTML = `
                <div class="prsm-welcome">
                    <div class="welcome-header">
                        <i class="fas fa-brain"></i>
                        <h3>PRSM Neural Network</h3>
                    </div>
                    <p>Connected to the Protocol for Recursive Scientific Modeling</p>
                    <p>Ask questions, request analysis, or explore the decentralized AI marketplace.</p>
                    <div class="quick-actions">
                        <button class="quick-action" onclick="prsmClient.quickQuery('Explain FTNS tokenomics')">
                            <i class="fas fa-coins"></i> FTNS Tokenomics
                        </button>
                        <button class="quick-action" onclick="prsmClient.quickQuery('Search for teacher models')">
                            <i class="fas fa-graduation-cap"></i> Teacher Models
                        </button>
                        <button class="quick-action" onclick="prsmClient.quickQuery('Show system health')">
                            <i class="fas fa-heart"></i> System Status
                        </button>
                    </div>
                </div>
                <div id="typing-indicator" style="display: none; padding: 10px; font-style: italic; color: var(--text-secondary);">
                    PRSM is thinking...
                </div>
            `;
        }
    }
    
    addBalanceDisplay() {
        // Add balance indicator to the conversation sub-header
        const subHeader = document.querySelector('.conversation-sub-header');
        if (subHeader) {
            const balanceIndicator = document.createElement('div');
            balanceIndicator.className = 'balance-indicator';
            balanceIndicator.innerHTML = `
                <i class="fas fa-coins"></i>
                <span class="ftns-balance">Loading...</span>
            `;
            subHeader.insertBefore(balanceIndicator, subHeader.firstChild);
        }
    }
    
    addSystemStatusIndicators() {
        // Update the connection status with real backend connection
        this.updateConnectionStatus('Connecting...');
    }
    
    quickQuery(query) {
        const promptInput = document.getElementById('prompt-input');
        if (promptInput) {
            promptInput.value = query;
            this.sendMessage();
        }
    }
    
    // === Live Metrics Updates ===
    
    startLiveMetricsUpdates() {
        // Update metrics every 30 seconds
        setInterval(() => {
            this.updateLiveMetrics();
        }, 30000);
        
        // Initial update
        setTimeout(() => this.updateLiveMetrics(), 2000);
    }
    
    async updateLiveMetrics() {
        try {
            // Update system health
            await this.checkSystemHealth();
            
            // Update model count
            if (this.state.availableModels) {
                const totalModels = (this.state.availableModels.teacher_models?.length || 0) +
                                  (this.state.availableModels.specialist_models?.length || 0) +
                                  (this.state.availableModels.general_models?.length || 0);
                
                const modelCountElement = document.getElementById('model-count');
                if (modelCountElement) {
                    modelCountElement.textContent = `${totalModels} models`;
                }
            }
            
            // Update network status (placeholder)
            const networkStatusElement = document.getElementById('network-status');
            if (networkStatusElement) {
                networkStatusElement.innerHTML = `
                    <div class="network-info">
                        <div>Single Node</div>
                        <div class="network-details">P2P coming v0.3.0</div>
                    </div>
                `;
            }
            
            // Update dashboard connection indicator
            const dashboardConnection = document.getElementById('dashboard-connection');
            if (dashboardConnection) {
                const isConnected = this.state.isConnected;
                dashboardConnection.innerHTML = `
                    <span class="status-dot ${isConnected ? 'connected' : 'disconnected'}"></span>
                    <span class="status-text">${isConnected ? 'Connected' : 'Disconnected'}</span>
                `;
            }
            
            // Add activity item
            this.addActivityItem('Metrics updated', 'fas fa-chart-line');
            
        } catch (error) {
            console.error('Failed to update live metrics:', error);
        }
    }
    
    addActivityItem(message, iconClass = 'fas fa-info-circle') {
        const activityFeed = document.getElementById('activity-feed');
        if (!activityFeed) return;
        
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <i class="${iconClass}"></i>
            <span>${message}</span>
            <span class="activity-time">${new Date().toLocaleTimeString()}</span>
        `;
        
        // Add to top of feed
        activityFeed.insertBefore(activityItem, activityFeed.firstChild);
        
        // Keep only last 10 items
        while (activityFeed.children.length > 10) {
            activityFeed.removeChild(activityFeed.lastChild);
        }
    }
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global PRSM client instance
    window.prsmClient = new PRSMProductionClient();
    console.log('🎉 PRSM UI Integration loaded successfully');
});
    
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