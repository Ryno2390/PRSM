/**
 * PRSM API Client
 * 
 * JavaScript client library for connecting the PRSM UI to the REST API endpoints.
 * Handles all communication with the PRSM backend services.
 */

class PRSMAPIClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.wsBaseURL = baseURL.replace('http://', 'ws://').replace('https://', 'wss://');
        this.currentUserId = 'ui_user_' + Date.now(); // Generate a session user ID
        this.currentConversationId = null;
        
        // WebSocket connections
        this.globalWS = null;
        this.conversationWS = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        
        // Event handlers for WebSocket messages
        this.messageHandlers = {
            'ai_response_chunk': [],
            'ai_response_complete': [],
            'notification': [],
            'typing_indicator': [],
            'tokenomics_update': [],
            'task_update': [],
            'file_update': [],
            'error': []
        };
        
        // Initialize connection
        this.checkConnection();
    }

    // --- Connection Management ---
    
    async checkConnection() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            if (response.ok) {
                console.log('✅ PRSM API connection established');
                this.connected = true;
                this.displayConnectionStatus(true);
                return true;
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.warn('⚠️ PRSM API not available:', error.message);
            this.connected = false;
            this.displayConnectionStatus(false);
            return false;
        }
    }

    displayConnectionStatus(connected) {
        // Display connection status in UI
        const statusElement = document.getElementById('api-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'API Connected' : 'API Offline (Mock Mode)';
            statusElement.className = connected ? 'api-status connected' : 'api-status disconnected';
        }
        
        // Initialize WebSocket connections if API is available
        if (connected) {
            this.initializeWebSockets();
        }
    }

    // --- WebSocket Management ---

    initializeWebSockets() {
        if (!this.connected) return;
        
        // Initialize global WebSocket for notifications
        this.connectGlobalWebSocket();
    }

    connectGlobalWebSocket() {
        if (this.globalWS && this.globalWS.readyState === WebSocket.OPEN) {
            return; // Already connected
        }

        try {
            this.globalWS = new WebSocket(`${this.wsBaseURL}/ws/${this.currentUserId}`);
            
            this.globalWS.onopen = () => {
                console.log('🔌 Global WebSocket connected');
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                
                // Send initial connection message
                this.sendWebSocketMessage(this.globalWS, {
                    type: 'connection',
                    user_id: this.currentUserId,
                    client_type: 'web_ui'
                });
            };

            this.globalWS.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.globalWS.onclose = (event) => {
                console.log('❌ Global WebSocket disconnected:', event.code);
                this.scheduleReconnect('global');
            };

            this.globalWS.onerror = (error) => {
                console.error('🚨 Global WebSocket error:', error);
            };

        } catch (error) {
            console.error('Failed to create global WebSocket:', error);
            this.scheduleReconnect('global');
        }
    }

    connectConversationWebSocket(conversationId) {
        if (!this.connected || !conversationId) return;

        // Close existing conversation WebSocket
        if (this.conversationWS) {
            this.conversationWS.close();
        }

        try {
            this.conversationWS = new WebSocket(
                `${this.wsBaseURL}/ws/conversation/${this.currentUserId}/${conversationId}`
            );

            this.conversationWS.onopen = () => {
                console.log('💬 Conversation WebSocket connected:', conversationId);
                
                // Send initial connection message
                this.sendWebSocketMessage(this.conversationWS, {
                    type: 'join_conversation',
                    conversation_id: conversationId,
                    user_id: this.currentUserId
                });
            };

            this.conversationWS.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.conversationWS.onclose = (event) => {
                console.log('❌ Conversation WebSocket disconnected:', event.code);
            };

            this.conversationWS.onerror = (error) => {
                console.error('🚨 Conversation WebSocket error:', error);
            };

        } catch (error) {
            console.error('Failed to create conversation WebSocket:', error);
        }
    }

    scheduleReconnect(type) {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.warn(`Max reconnection attempts reached for ${type} WebSocket`);
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnecting ${type} WebSocket in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            if (type === 'global') {
                this.connectGlobalWebSocket();
            }
            // Conversation reconnection handled when creating new conversations
        }, this.reconnectDelay);

        // Exponential backoff
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    }

    sendWebSocketMessage(ws, message) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not ready, message not sent:', message);
        }
    }

    handleWebSocketMessage(message) {
        const { type, data } = message;
        
        console.log('📨 WebSocket message received:', type);

        // Trigger registered handlers for this message type
        if (this.messageHandlers[type]) {
            this.messageHandlers[type].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error('Error in WebSocket message handler:', error);
                }
            });
        }

        // Built-in message handling
        switch (type) {
            case 'ai_response_chunk':
                this.handleAIResponseChunk(data);
                break;
            case 'ai_response_complete':
                this.handleAIResponseComplete(data);
                break;
            case 'notification':
                this.handleNotification(data);
                break;
            case 'typing_indicator':
                this.handleTypingIndicator(data);
                break;
            case 'tokenomics_update':
                this.handleTokenomicsUpdate(data);
                break;
            case 'task_update':
                this.handleTaskUpdate(data);
                break;
            case 'file_update':
                this.handleFileUpdate(data);
                break;
            case 'error':
                this.handleWebSocketError(data);
                break;
            default:
                console.log('Unhandled WebSocket message type:', type);
        }
    }

    // WebSocket message handlers
    handleAIResponseChunk(data) {
        const { conversation_id, message_id, chunk, is_complete } = data;
        
        // Update UI with streaming response
        const responseElement = document.querySelector(`[data-message-id="${message_id}"]`);
        if (responseElement) {
            const contentElement = responseElement.querySelector('.message-content');
            if (contentElement) {
                contentElement.textContent += chunk;
                
                // Auto-scroll to latest content
                responseElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        } else {
            // Create new message element for streaming response
            this.createStreamingMessageElement(message_id, chunk, conversation_id);
        }
    }

    handleAIResponseComplete(data) {
        const { conversation_id, message_id, complete_content, metadata } = data;
        
        // Update final message content and metadata
        const responseElement = document.querySelector(`[data-message-id="${message_id}"]`);
        if (responseElement) {
            const contentElement = responseElement.querySelector('.message-content');
            if (contentElement) {
                contentElement.textContent = complete_content;
            }
            
            // Add metadata like model used, tokens, etc.
            const metadataElement = responseElement.querySelector('.message-metadata');
            if (metadataElement && metadata) {
                metadataElement.innerHTML = `
                    <span class="model-tag">${metadata.model_used}</span>
                    <span class="tokens-used">${metadata.tokens_used} tokens</span>
                `;
            }
        }

        // Update conversation context display
        this.updateContextDisplay(metadata);
        
        // Re-enable message input
        this.enableMessageInput();
    }

    updateContextDisplay(metadata) {
        if (!metadata) return;

        // Update token usage progress bar
        const progressBar = document.querySelector('.context-progress-bar progress');
        const progressValue = document.querySelector('.context-progress-bar .value');
        
        if (progressBar && metadata.context_used && metadata.context_limit) {
            progressBar.value = metadata.context_used;
            progressBar.max = metadata.context_limit;
            
            if (progressValue) {
                progressValue.textContent = `${metadata.context_used} / ${metadata.context_limit}`;
            }
        }

        // Update cost display
        const costElement = document.querySelector('.api-cost');
        if (costElement && metadata.total_cost) {
            costElement.textContent = `Cost: $${metadata.total_cost.toFixed(4)}`;
        }
    }

    handleNotification(data) {
        const { title, message, type, priority } = data;
        
        // Show notification in UI
        this.showNotification(title, message, type);
        
        console.log(`📢 ${type.toUpperCase()} notification:`, title, message);
    }

    handleTypingIndicator(data) {
        const { conversation_id, is_typing, user_id } = data;
        
        // Show/hide typing indicator
        const typingElement = document.getElementById('typing-indicator');
        if (typingElement) {
            if (is_typing && user_id !== this.currentUserId) {
                typingElement.style.display = 'block';
                typingElement.textContent = 'PRSM is thinking...';
            } else {
                typingElement.style.display = 'none';
            }
        }
    }

    handleWebSocketError(data) {
        const { error_message, error_code } = data;
        console.error('WebSocket error received:', error_message);
        
        // Show error notification
        this.showNotification('Connection Error', error_message, 'error');
    }

    handleTokenomicsUpdate(data) {
        const { balance, staking, earnings, metadata } = data;
        
        console.log('💰 Live tokenomics update received');
        
        // Update tokenomics display in real-time
        this.updateTokenomicsDisplay({
            success: true,
            tokenomics: { balance, staking, earnings }
        });
        
        // Show notification for significant changes
        if (metadata && metadata.significant_change) {
            this.showNotification(
                'Tokenomics Update',
                `Your FTNS balance has been updated: ${balance.total} FTNS`,
                'info'
            );
        }
    }

    handleTaskUpdate(data) {
        const { task, action, metadata } = data;
        
        console.log('📋 Live task update received:', action);
        
        // Refresh task display
        this.refreshTasksDisplay();
        
        // Show notification for task changes
        const actionMessages = {
            'created': `New task created: ${task.title}`,
            'completed': `Task completed: ${task.title}`,
            'updated': `Task updated: ${task.title}`,
            'deleted': `Task deleted: ${task.title}`
        };
        
        if (actionMessages[action]) {
            this.showNotification('Task Update', actionMessages[action], 'info');
        }
    }

    handleFileUpdate(data) {
        const { file, action, metadata } = data;
        
        console.log('📁 Live file update received:', action);
        
        // Refresh files display
        this.refreshFilesDisplay();
        
        // Show notification for file changes
        const actionMessages = {
            'uploaded': `File uploaded: ${file.filename}`,
            'deleted': `File deleted: ${file.filename}`,
            'processed': `File processing complete: ${file.filename}`
        };
        
        if (actionMessages[action]) {
            this.showNotification('File Update', actionMessages[action], 'info');
        }
    }

    // Helper methods for real-time UI updates
    updateTokenomicsDisplay(data) {
        // Update the tokenomics tab content with new data
        if (data.success && data.tokenomics) {
            const { balance, staking, earnings } = data.tokenomics;
            
            // Update balance display
            const balanceElements = {
                total: document.querySelector('[data-tokenomics="balance-total"]'),
                available: document.querySelector('[data-tokenomics="balance-available"]'),
                locked: document.querySelector('[data-tokenomics="balance-locked"]')
            };
            
            if (balanceElements.total) balanceElements.total.textContent = `${balance.total} FTNS`;
            if (balanceElements.available) balanceElements.available.textContent = `${balance.available} FTNS`;
            if (balanceElements.locked) balanceElements.locked.textContent = `${balance.locked} FTNS`;
            
            // Update staking display
            const stakingElements = {
                amount: document.querySelector('[data-tokenomics="staking-amount"]'),
                apy: document.querySelector('[data-tokenomics="staking-apy"]'),
                rewards: document.querySelector('[data-tokenomics="staking-rewards"]')
            };
            
            if (stakingElements.amount) stakingElements.amount.textContent = `${staking.staked_amount} FTNS`;
            if (stakingElements.apy) stakingElements.apy.textContent = `${staking.apy}%`;
            if (stakingElements.rewards) stakingElements.rewards.textContent = `${staking.rewards_earned} FTNS`;
            
            // Update earnings display
            const earningsElements = {
                total: document.querySelector('[data-tokenomics="earnings-total"]'),
                ipfs: document.querySelector('[data-tokenomics="earnings-ipfs"]'),
                model: document.querySelector('[data-tokenomics="earnings-model"]'),
                compute: document.querySelector('[data-tokenomics="earnings-compute"]')
            };
            
            if (earningsElements.total) earningsElements.total.textContent = `${earnings.total_earned} FTNS`;
            if (earnings.sources) {
                if (earningsElements.ipfs) earningsElements.ipfs.textContent = `${earnings.sources.ipfs_hosting} FTNS`;
                if (earningsElements.model) earningsElements.model.textContent = `${earnings.sources.model_hosting} FTNS`;
                if (earningsElements.compute) earningsElements.compute.textContent = `${earnings.sources.compute_contribution} FTNS`;
            }
        }
    }

    refreshTasksDisplay() {
        // Trigger a refresh of the tasks tab
        if (typeof window.loadTasksData === 'function') {
            window.loadTasksData();
        } else {
            // Fallback: reload tasks data
            this.getTasks().then(response => {
                if (response.success) {
                    console.log('📋 Tasks refreshed via WebSocket update');
                }
            });
        }
    }

    refreshFilesDisplay() {
        // Trigger a refresh of the files tab
        if (typeof window.loadFilesData === 'function') {
            window.loadFilesData();
        } else {
            // Fallback: reload files data
            this.listFiles().then(response => {
                if (response.success) {
                    console.log('📁 Files refreshed via WebSocket update');
                }
            });
        }
    }

    // Event handler registration
    onWebSocketMessage(type, handler) {
        if (this.messageHandlers[type]) {
            this.messageHandlers[type].push(handler);
        } else {
            console.warn('Unknown WebSocket message type:', type);
        }
    }

    // Utility methods for UI integration
    createStreamingMessageElement(messageId, initialChunk, conversationId) {
        const responseArea = document.querySelector('.response-area');
        if (!responseArea) return;

        const messageElement = document.createElement('div');
        messageElement.className = 'message message-assistant';
        messageElement.setAttribute('data-message-id', messageId);
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="message-role">PRSM</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
                <div class="message-metadata"></div>
            </div>
            <div class="message-content">${initialChunk}</div>
        `;

        responseArea.appendChild(messageElement);
        messageElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    showNotification(title, message, type = 'info') {
        // Create or update notification element
        let notificationContainer = document.getElementById('notification-container');
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                pointer-events: none;
            `;
            document.body.appendChild(notificationContainer);
        }

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
            min-width: 300px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            pointer-events: auto;
            opacity: 0;
            transform: translateX(100%);
            transition: opacity 0.3s ease, transform 0.3s ease;
        `;

        notification.innerHTML = `
            <div style="font-weight: 500; margin-bottom: 5px; color: var(--text-primary);">${title}</div>
            <div style="color: var(--text-secondary); font-size: 0.9em;">${message}</div>
        `;

        notificationContainer.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    enableMessageInput() {
        const sendButton = document.getElementById('send-message-btn');
        const messageInput = document.getElementById('prompt-input');
        
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
        }
        
        if (messageInput) {
            messageInput.disabled = false;
        }
    }

    disableMessageInput() {
        const sendButton = document.getElementById('send-message-btn');
        const messageInput = document.getElementById('prompt-input');
        
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';
        }
        
        if (messageInput) {
            messageInput.disabled = true;
        }
    }

    // --- Real-time Message Sending ---

    async sendMessageWithStreaming(content) {
        if (!this.currentConversationId) {
            await this.createConversation();
        }

        // Ensure conversation WebSocket is connected
        if (!this.conversationWS || this.conversationWS.readyState !== WebSocket.OPEN) {
            this.connectConversationWebSocket(this.currentConversationId);
            
            // Wait a moment for connection to establish
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Disable input during processing
        this.disableMessageInput();

        // Send message via WebSocket for real-time response
        if (this.conversationWS && this.conversationWS.readyState === WebSocket.OPEN) {
            this.sendWebSocketMessage(this.conversationWS, {
                type: 'send_message',
                content: content,
                conversation_id: this.currentConversationId,
                streaming: true
            });

            // Add user message to UI immediately
            this.addUserMessageToUI(content);
            
            return { success: true, streaming: true };
        } else {
            // Fallback to REST API if WebSocket not available
            this.enableMessageInput();
            return await this.sendMessage(content);
        }
    }

    addUserMessageToUI(content) {
        const responseArea = document.querySelector('.response-area');
        if (!responseArea) return;

        const messageElement = document.createElement('div');
        messageElement.className = 'message message-user';
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="message-role">You</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${content}</div>
        `;

        responseArea.appendChild(messageElement);
        messageElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                const errorData = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorData}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            
            // Return mock data for offline mode
            if (!this.connected) {
                return this.getMockResponse(endpoint, options.method || 'GET');
            }
            
            throw error;
        }
    }

    // --- Conversation Management ---

    async createConversation(title = 'New Conversation', model = 'nwtn-v1', mode = 'dynamic') {
        const conversationData = {
            user_id: this.currentUserId,
            title: title,
            model: model,
            mode: mode
        };

        const response = await this.makeRequest('/ui/conversations', {
            method: 'POST',
            body: JSON.stringify(conversationData)
        });

        if (response.success) {
            this.currentConversationId = response.conversation_id;
            console.log('📝 New conversation created:', response.conversation_id);
        }

        return response;
    }

    async sendMessage(content) {
        if (!this.currentConversationId) {
            // Auto-create conversation if none exists
            await this.createConversation();
        }

        const messageData = {
            content: content
        };

        const response = await this.makeRequest(
            `/ui/conversations/${this.currentConversationId}/messages`,
            {
                method: 'POST',
                body: JSON.stringify(messageData)
            }
        );

        return response;
    }

    async getConversation(conversationId = null) {
        const id = conversationId || this.currentConversationId;
        if (!id) return null;

        return await this.makeRequest(`/ui/conversations/${id}`);
    }

    async listConversations() {
        return await this.makeRequest(`/ui/conversations?user_id=${this.currentUserId}`);
    }

    // --- File Management ---

    async uploadFile(file, filename = null) {
        const reader = new FileReader();
        
        return new Promise((resolve, reject) => {
            reader.onload = async () => {
                try {
                    const base64Content = reader.result.split(',')[1]; // Remove data URL prefix
                    
                    const fileData = {
                        filename: filename || file.name,
                        content: base64Content,
                        content_type: file.type,
                        encoding: 'base64',
                        user_id: this.currentUserId
                    };

                    const response = await this.makeRequest('/ui/files/upload', {
                        method: 'POST',
                        body: JSON.stringify(fileData)
                    });

                    resolve(response);
                } catch (error) {
                    reject(error);
                }
            };

            reader.onerror = () => reject(new Error('File reading failed'));
            reader.readAsDataURL(file);
        });
    }

    async listFiles() {
        return await this.makeRequest(`/ui/files?user_id=${this.currentUserId}`);
    }

    // --- Tokenomics ---

    async getTokenomics() {
        return await this.makeRequest(`/ui/tokenomics/${this.currentUserId}`);
    }

    // --- Task Management ---

    async getTasks() {
        return await this.makeRequest(`/ui/tasks/${this.currentUserId}`);
    }

    async createTask(taskData) {
        return await this.makeRequest('/ui/tasks', {
            method: 'POST',
            body: JSON.stringify(taskData)
        });
    }

    // --- Settings ---

    async saveSettings(settingsData) {
        const data = {
            user_id: this.currentUserId,
            ...settingsData
        };

        return await this.makeRequest('/ui/settings/save', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async getSettings() {
        return await this.makeRequest(`/ui/settings/${this.currentUserId}`);
    }

    // --- Information Space ---

    async getInformationSpaceData() {
        return await this.makeRequest('/ui/information-space');
    }

    // --- Mock Data for Offline Mode ---

    getMockResponse(endpoint, method) {
        console.log('📱 Using mock data for:', endpoint);
        
        if (endpoint === '/ui/conversations' && method === 'POST') {
            return {
                success: true,
                conversation_id: 'mock_conv_' + Date.now(),
                status: 'created',
                model: 'nwtn-v1',
                mode: 'dynamic'
            };
        }

        if (endpoint.includes('/messages') && method === 'POST') {
            return {
                success: true,
                user_message: {
                    message_id: 'mock_user_' + Date.now(),
                    role: 'user',
                    content: 'Test message',
                    timestamp: new Date().toISOString()
                },
                ai_response: {
                    message_id: 'mock_ai_' + Date.now(),
                    role: 'assistant',
                    content: 'This is a mock response. The PRSM API server is not running, but the UI is working correctly!',
                    timestamp: new Date().toISOString(),
                    model_used: 'nwtn-v1',
                    mode_used: 'dynamic'
                },
                conversation_status: {
                    context_used: 50,
                    context_limit: 4096,
                    message_count: 2
                }
            };
        }

        if (endpoint.includes('/ui/conversations') && method === 'GET') {
            return {
                success: true,
                conversations: [
                    {
                        conversation_id: 'mock_1',
                        title: 'Research on Atomically Precise Manufacturing',
                        last_message_at: '2024-01-15T10:30:00Z',
                        message_count: 12,
                        status: 'active'
                    },
                    {
                        conversation_id: 'mock_2',
                        title: 'Initial Brainstorming Session',
                        last_message_at: '2024-01-14T15:45:00Z',
                        message_count: 8,
                        status: 'archived'
                    }
                ],
                total_count: 2
            };
        }

        if (endpoint.includes('/ui/tokenomics')) {
            return {
                success: true,
                tokenomics: {
                    balance: {
                        total: 1500,
                        available: 1000,
                        locked: 500,
                        currency: 'FTNS'
                    },
                    staking: {
                        staked_amount: 500,
                        apy: 8.5,
                        rewards_earned: 42.5,
                        staking_period: '30_days'
                    },
                    earnings: {
                        total_earned: 156.8,
                        sources: {
                            ipfs_hosting: 89.2,
                            model_hosting: 45.6,
                            compute_contribution: 22.0
                        },
                        current_status: 'active'
                    }
                }
            };
        }

        if (endpoint.includes('/ui/tasks')) {
            return {
                success: true,
                tasks: [
                    {
                        task_id: 'mock_task_1',
                        title: 'Review research paper draft',
                        description: 'Review the quantum computing research paper',
                        status: 'pending',
                        priority: 'high',
                        created_at: '2024-01-15T08:00:00Z'
                    }
                ],
                statistics: {
                    total: 1,
                    pending: 1,
                    in_progress: 0,
                    completed: 0
                }
            };
        }

        if (endpoint.includes('/ui/files')) {
            return {
                success: true,
                files: [
                    {
                        file_id: 'mock_file_1',
                        filename: 'analysis.ipynb',
                        content_type: 'application/x-ipynb+json',
                        size: 245760,
                        privacy: 'private',
                        ai_access: 'core_only'
                    }
                ],
                total_count: 1,
                storage_used: 245760
            };
        }

        // Default mock response
        return {
            success: true,
            message: 'Mock response - API server not available',
            endpoint: endpoint,
            method: method
        };
    }

    // WebSocket cleanup and lifecycle management
    cleanup() {
        console.log('🧹 Cleaning up PRSM API client...');
        
        // Close WebSocket connections
        if (this.globalWS) {
            this.globalWS.close();
            this.globalWS = null;
        }
        
        if (this.conversationWS) {
            this.conversationWS.close();
            this.conversationWS = null;
        }
        
        // Clear reconnect attempts
        this.reconnectAttempts = 0;
        
        console.log('✅ PRSM API client cleanup complete');
    }

    // Check WebSocket health and reconnect if needed
    checkWebSocketHealth() {
        if (!this.connected) return false;

        let needsReconnect = false;

        // Check global WebSocket
        if (!this.globalWS || this.globalWS.readyState !== WebSocket.OPEN) {
            console.log('🔍 Global WebSocket needs reconnection');
            needsReconnect = true;
            this.connectGlobalWebSocket();
        }

        // Check conversation WebSocket if we have an active conversation
        if (this.currentConversationId) {
            if (!this.conversationWS || this.conversationWS.readyState !== WebSocket.OPEN) {
                console.log('🔍 Conversation WebSocket needs reconnection');
                this.connectConversationWebSocket(this.currentConversationId);
            }
        }

        return !needsReconnect;
    }

    // Periodic health check
    startHealthCheck() {
        setInterval(() => {
            this.checkWebSocketHealth();
        }, 30000); // Check every 30 seconds
    }

    // Enhanced connection status reporting
    getConnectionStatus() {
        return {
            api_connected: this.connected,
            global_ws_status: this.globalWS ? this.globalWS.readyState : 'not_connected',
            conversation_ws_status: this.conversationWS ? this.conversationWS.readyState : 'not_connected',
            current_conversation: this.currentConversationId,
            reconnect_attempts: this.reconnectAttempts,
            user_id: this.currentUserId
        };
    }

    // Send heartbeat to keep connections alive
    sendHeartbeat() {
        if (this.globalWS && this.globalWS.readyState === WebSocket.OPEN) {
            this.sendWebSocketMessage(this.globalWS, {
                type: 'heartbeat',
                timestamp: Date.now(),
                user_id: this.currentUserId
            });
        }
    }

    // Start heartbeat interval
    startHeartbeat() {
        setInterval(() => {
            this.sendHeartbeat();
        }, 60000); // Send heartbeat every minute
    }
}

// Create global API client instance
window.prsmAPI = new PRSMAPIClient();

// Global cleanup handler
window.addEventListener('beforeunload', () => {
    if (window.prsmAPI) {
        window.prsmAPI.cleanup();
    }
});

// Start health monitoring
if (window.prsmAPI) {
    // Start health checks after initial connection
    setTimeout(() => {
        window.prsmAPI.startHealthCheck();
        window.prsmAPI.startHeartbeat();
        
        console.log('🔄 WebSocket health monitoring started');
    }, 5000);
}