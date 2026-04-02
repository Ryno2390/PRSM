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
        this.maxReconnectAttempts = 10; // Increased from 5
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 60000; // Max 1 minute
        
        // Connection state management
        this.isReconnecting = false;
        this.lastConnectionAttempt = 0;
        this.connectionId = 'conn_' + Date.now();
        this.wsConnectionState = {
            global: 'disconnected',
            conversation: 'disconnected'
        };
        
        // Message queue for offline scenarios
        this.messageQueue = [];
        this.maxQueueSize = 100;
        
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
            if (connected) {
                if (this.isReconnecting) {
                    statusElement.textContent = 'Reconnecting...';
                    statusElement.className = 'api-status reconnecting';
                } else {
                    statusElement.textContent = 'API Connected';
                    statusElement.className = 'api-status connected';
                }
            } else {
                statusElement.textContent = 'API Offline (Mock Mode)';
                statusElement.className = 'api-status disconnected';
            }
        }
        
        // Update detailed connection status
        this.updateDetailedConnectionStatus();
        
        // Update connection status widget
        this.updateConnectionStatusWidget();
        
        // Initialize WebSocket connections if API is available
        if (connected && !this.isReconnecting) {
            this.initializeWebSockets();
        }
    }
    
    updateDetailedConnectionStatus() {
        // Create or update detailed status indicator
        let detailStatusElement = document.getElementById('connection-details');
        if (!detailStatusElement) {
            detailStatusElement = document.createElement('div');
            detailStatusElement.id = 'connection-details';
            detailStatusElement.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                padding: 12px;
                font-size: 12px;
                color: var(--text-secondary);
                z-index: 1000;
                max-width: 300px;
                display: none;
            `;
            document.body.appendChild(detailStatusElement);
        }
        
        const status = this.getConnectionStatus();
        detailStatusElement.innerHTML = `
            <div><strong>Connection Status</strong></div>
            <div>API: ${status.api_connected ? '✅' : '❌'}</div>
            <div>Global WS: ${this.getWebSocketStatusText(status.global_ws_status)}</div>
            <div>Conv WS: ${this.getWebSocketStatusText(status.conversation_ws_status)}</div>
            <div>Reconnect attempts: ${status.reconnect_attempts}/${this.maxReconnectAttempts}</div>
            <div>Queue: ${this.messageQueue.length} messages</div>
        `;
        
        // Auto-hide after 10 seconds
        if (this.connected || this.isReconnecting) {
            detailStatusElement.style.display = 'block';
            setTimeout(() => {
                if (detailStatusElement) {
                    detailStatusElement.style.display = 'none';
                }
            }, 10000);
        }
    }
    
    getWebSocketStatusText(readyState) {
        const states = {
            0: '🟡 Connecting',
            1: '✅ Open',
            2: '🟠 Closing',
            3: '❌ Closed',
            'not_connected': '⚫ Not Connected'
        };
        return states[readyState] || '❓ Unknown';
    }

    // --- WebSocket Management ---

    initializeWebSockets() {
        if (!this.connected) return;
        
        // Initialize global WebSocket for notifications
        this.connectGlobalWebSocket();
    }

    connectGlobalWebSocket() {
        if (this.globalWS && this.globalWS.readyState === WebSocket.OPEN) {
            console.log('🔌 Global WebSocket already connected');
            return Promise.resolve();
        }

        // Prevent multiple simultaneous connection attempts
        if (this.isReconnecting) {
            console.log('🔄 Already attempting to reconnect global WebSocket');
            return Promise.reject(new Error('Reconnection in progress'));
        }

        return new Promise((resolve, reject) => {
            try {
                this.isReconnecting = true;
                this.lastConnectionAttempt = Date.now();
                this.wsConnectionState.global = 'connecting';
                
                console.log(`🔌 Attempting global WebSocket connection (attempt ${this.reconnectAttempts + 1})`);
                
                this.globalWS = new WebSocket(`${this.wsBaseURL}/ws/${this.currentUserId}?connection_id=${this.connectionId}`);
                
                const connectionTimeout = setTimeout(() => {
                    if (this.globalWS && this.globalWS.readyState === WebSocket.CONNECTING) {
                        console.warn('⏰ Global WebSocket connection timeout');
                        this.globalWS.close();
                        reject(new Error('Connection timeout'));
                    }
                }, 10000); // 10 second timeout
                
                this.globalWS.onopen = () => {
                    clearTimeout(connectionTimeout);
                    console.log('✅ Global WebSocket connected successfully');
                    
                    this.reconnectAttempts = 0;
                    this.reconnectDelay = 1000;
                    this.isReconnecting = false;
                    this.wsConnectionState.global = 'connected';
                    
                    // Send initial connection message
                    this.sendWebSocketMessage(this.globalWS, {
                        type: 'connection',
                        user_id: this.currentUserId,
                        client_type: 'web_ui',
                        connection_id: this.connectionId,
                        timestamp: Date.now()
                    });
                    
                    // Process queued messages
                    this.processMessageQueue();
                    
                    // Update UI
                    this.displayConnectionStatus(true);
                    
                    resolve();
                };

                this.globalWS.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleWebSocketMessage(data);
                    } catch (error) {
                        console.error('❌ Failed to parse WebSocket message:', error);
                    }
                };

                this.globalWS.onclose = (event) => {
                    clearTimeout(connectionTimeout);
                    this.wsConnectionState.global = 'disconnected';
                    this.isReconnecting = false;
                    
                    console.log(`❌ Global WebSocket disconnected (code: ${event.code}, reason: ${event.reason})`);
                    
                    // Don't reconnect if it was a clean close
                    if (event.code !== 1000 && event.code !== 1001) {
                        this.scheduleReconnect('global');
                    }
                    
                    reject(new Error(`WebSocket closed: ${event.code}`));
                };

                this.globalWS.onerror = (error) => {
                    clearTimeout(connectionTimeout);
                    console.error('🚨 Global WebSocket error:', error);
                    
                    this.wsConnectionState.global = 'error';
                    this.isReconnecting = false;
                    
                    // Show user-friendly error notification
                    this.showNotification(
                        'Connection Error',
                        'Lost connection to PRSM server. Attempting to reconnect...',
                        'error'
                    );
                    
                    reject(error);
                };

            } catch (error) {
                console.error('❌ Failed to create global WebSocket:', error);
                this.isReconnecting = false;
                this.scheduleReconnect('global');
                reject(error);
            }
        });
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
            console.warn(`❌ Max reconnection attempts (${this.maxReconnectAttempts}) reached for ${type} WebSocket`);
            
            // Show persistent error to user
            this.showNotification(
                'Connection Failed',
                `Unable to connect to PRSM server after ${this.maxReconnectAttempts} attempts. Please check your connection and refresh the page.`,
                'error'
            );
            
            // Switch to offline mode
            this.connected = false;
            this.displayConnectionStatus(false);
            return;
        }

        // Prevent scheduling if already reconnecting
        if (this.isReconnecting) {
            console.log(`🔄 Reconnection already scheduled for ${type} WebSocket`);
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay, this.maxReconnectDelay);
        
        console.log(`🔄 Scheduling ${type} WebSocket reconnection in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        // Update UI to show reconnecting status
        this.displayConnectionStatus(this.connected);

        setTimeout(() => {
            // Check if we should still try to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                if (type === 'global') {
                    this.connectGlobalWebSocket().catch(error => {
                        console.error('🚨 Global WebSocket reconnection failed:', error);
                        // Schedule another attempt
                        setTimeout(() => this.scheduleReconnect('global'), 2000);
                    });
                } else if (type === 'conversation' && this.currentConversationId) {
                    this.connectConversationWebSocket(this.currentConversationId);
                }
            }
        }, delay);

        // Exponential backoff with jitter to prevent thundering herd
        const jitter = Math.random() * 1000; // 0-1 second jitter
        this.reconnectDelay = Math.min(this.reconnectDelay * 1.5 + jitter, this.maxReconnectDelay);
    }
    
    // Add message queuing for offline scenarios
    queueMessage(message) {
        if (this.messageQueue.length >= this.maxQueueSize) {
            // Remove oldest message to make room
            this.messageQueue.shift();
            console.warn('⚠️ Message queue full, removing oldest message');
        }
        
        this.messageQueue.push({
            message: message,
            timestamp: Date.now(),
            attempts: 0
        });
        
        console.log(`📥 Queued message (queue size: ${this.messageQueue.length})`);
    }
    
    processMessageQueue() {
        if (this.messageQueue.length === 0) return;
        
        console.log(`📤 Processing ${this.messageQueue.length} queued messages`);
        
        const messagesToProcess = [...this.messageQueue];
        this.messageQueue = [];
        
        messagesToProcess.forEach((queuedItem, index) => {
            setTimeout(() => {
                if (this.globalWS && this.globalWS.readyState === WebSocket.OPEN) {
                    this.sendWebSocketMessage(this.globalWS, queuedItem.message);
                    console.log(`📤 Sent queued message ${index + 1}/${messagesToProcess.length}`);
                } else {
                    // Re-queue if connection is lost again
                    this.queueMessage(queuedItem.message);
                }
            }, index * 100); // Stagger sends by 100ms
        });
    }

    sendWebSocketMessage(ws, message) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify(message));
                console.log('📤 WebSocket message sent:', message.type);
                return true;
            } catch (error) {
                console.error('❌ Failed to send WebSocket message:', error);
                return false;
            }
        } else {
            console.warn('⚠️ WebSocket not ready, queueing message:', message.type);
            
            // Queue message for later sending
            this.queueMessage(message);
            
            // Try to reconnect if not already attempting
            if (!this.isReconnecting && ws === this.globalWS) {
                console.log('🔄 Attempting to reconnect global WebSocket due to send failure');
                this.scheduleReconnect('global');
            }
            
            return false;
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
        
        // Clear response timeout
        this.clearResponseTimeout();
        
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
                    <span class="model-tag">${metadata.model_used || 'nwtn-v1'}</span>
                    <span class="tokens-used">${metadata.tokens_used || 0} tokens</span>
                    ${metadata.processing_time ? `<span class="processing-time">${metadata.processing_time}ms</span>` : ''}
                `;
            }
        }

        // Update conversation context display
        this.updateContextDisplay(metadata);
        
        // Re-enable message input
        this.enableMessageInput();
        
        console.log('✅ AI response completed successfully');
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
        try {
            if (!this.currentConversationId) {
                const convResponse = await this.createConversation();
                if (!convResponse.success) {
                    throw new Error('Failed to create conversation');
                }
            }

            // Disable input during processing
            this.disableMessageInput();

            // Try WebSocket first for real-time streaming
            if (this.connected && this.globalWS && this.globalWS.readyState === WebSocket.OPEN) {
                
                const messageSent = this.sendWebSocketMessage(this.globalWS, {
                    type: 'send_message',
                    content: content,
                    conversation_id: this.currentConversationId,
                    streaming: true,
                    timestamp: Date.now()
                });

                if (messageSent) {
                    // Add user message to UI immediately
                    this.addUserMessageToUI(content);
                    
                    // Start timeout for response
                    this.startResponseTimeout();
                    
                    return { success: true, streaming: true, method: 'websocket' };
                }
            }

            // Fallback to REST API if WebSocket not available
            console.log('🔄 WebSocket unavailable, falling back to REST API');
            this.enableMessageInput();
            
            const restResponse = await this.sendMessage(content);
            if (restResponse.success) {
                // Add both user and AI messages to UI
                this.addUserMessageToUI(content);
                if (restResponse.ai_response) {
                    this.addAIMessageToUI(restResponse.ai_response);
                }
            }
            
            return { 
                ...restResponse, 
                method: 'rest_api',
                fallback: true 
            };
            
        } catch (error) {
            console.error('❌ Failed to send message:', error);
            this.enableMessageInput();
            
            this.showNotification(
                'Message Failed',
                'Unable to send message. Please try again.',
                'error'
            );
            
            return { 
                success: false, 
                error: error.message,
                method: 'failed'
            };
        }
    }
    
    startResponseTimeout() {
        // Clear any existing timeout
        if (this.responseTimeout) {
            clearTimeout(this.responseTimeout);
        }
        
        // Set 30 second timeout for AI response
        this.responseTimeout = setTimeout(() => {
            console.warn('⏰ AI response timeout');
            this.enableMessageInput();
            
            this.showNotification(
                'Response Timeout',
                'AI response is taking longer than expected. You can try sending another message.',
                'warning'
            );
        }, 30000);
    }
    
    clearResponseTimeout() {
        if (this.responseTimeout) {
            clearTimeout(this.responseTimeout);
            this.responseTimeout = null;
        }
    }
    
    addAIMessageToUI(aiResponse) {
        const responseArea = document.querySelector('.response-area');
        if (!responseArea) return;

        const messageElement = document.createElement('div');
        messageElement.className = 'message message-assistant';
        messageElement.setAttribute('data-message-id', aiResponse.message_id || 'ai_' + Date.now());
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="message-role">PRSM</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
                <div class="message-metadata">
                    ${aiResponse.model_used ? `<span class="model-tag">${aiResponse.model_used}</span>` : ''}
                </div>
            </div>
            <div class="message-content">${aiResponse.content}</div>
        `;

        responseArea.appendChild(messageElement);
        messageElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
        
        // Re-enable input
        this.enableMessageInput();
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

        const response = await this.makeRequestWithRetry(
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

        // Integration layer mock responses
        if (endpoint.includes('/integrations/connectors/register')) {
            return {
                message: "GitHub connector registered successfully",
                platform: "github",
                user_id: this.currentUserId
            };
        }

        if (endpoint.includes('/integrations/connectors/health')) {
            return {
                github: {
                    status: "healthy",
                    platform: "github",
                    last_check: new Date().toISOString(),
                    error_count: 0,
                    response_time: 120
                },
                huggingface: {
                    status: "healthy",
                    platform: "huggingface",
                    last_check: new Date().toISOString(),
                    error_count: 0,
                    response_time: 200
                },
                ollama: {
                    status: "healthy",
                    platform: "ollama",
                    last_check: new Date().toISOString(),
                    error_count: 0,
                    response_time: 50,
                    ollama_version: "0.1.17",
                    available_models: 3,
                    running_models: 1
                }
            };
        }

        if (endpoint.includes('/integrations/search')) {
            return [
                {
                    platform: "github",
                    external_id: "microsoft/vscode",
                    display_name: "Visual Studio Code",
                    owner_id: "microsoft",
                    url: "https://github.com/microsoft/vscode",
                    description: "Visual Studio Code - Code Editing. Redefined.",
                    metadata: {
                        stars: 140000,
                        forks: 25000,
                        license: "MIT"
                    }
                },
                {
                    platform: "huggingface",
                    external_id: "microsoft/DialoGPT-medium",
                    display_name: "DialoGPT Medium",
                    owner_id: "microsoft",
                    url: "https://huggingface.co/microsoft/DialoGPT-medium",
                    description: "Large-scale pretrained dialogue response generation model",
                    metadata: {
                        downloads: 50000,
                        likes: 250,
                        license: "MIT"
                    }
                },
                {
                    platform: "ollama",
                    external_id: "llama2:7b",
                    display_name: "Llama 2 7B",
                    owner_id: "local",
                    url: "http://localhost:11434/api/show?name=llama2:7b",
                    description: "Local Ollama model - llama2:7b",
                    metadata: {
                        size: 3825819519,
                        size_gb: 3.57,
                        tag: "7b",
                        type: "local_model"
                    }
                }
            ];
        }

        if (endpoint.includes('/integrations/import') && method === 'POST') {
            return {
                request_id: 'mock_import_' + Date.now(),
                status: 'submitted',
                message: 'Import request submitted successfully',
                source: {
                    platform: 'github',
                    external_id: 'test/repo'
                }
            };
        }

        if (endpoint.includes('/integrations/health')) {
            return {
                overall_status: "healthy",
                health_percentage: 95.0,
                connectors: {
                    total: 3,
                    healthy: 3,
                    degraded: 0,
                    unhealthy: 0
                },
                imports: {
                    active: 0,
                    total: 5,
                    successful: 4,
                    failed: 1
                },
                last_health_check: new Date().toISOString(),
                sandbox_status: {
                    status: "idle",
                    active_scans: 0
                }
            };
        }

        if (endpoint.includes('/integrations/imports/history')) {
            return [
                {
                    request_id: 'req_001',
                    result_id: 'res_001',
                    status: 'completed',
                    created_at: new Date(Date.now() - 86400000).toISOString(),
                    completed_at: new Date(Date.now() - 86000000).toISOString(),
                    import_duration: 45.2,
                    success_message: 'Repository imported successfully',
                    error_details: null
                }
            ];
        }

        // Configuration management mock responses
        if (endpoint.includes('/integrations/config/credentials') && method === 'POST') {
            return {
                message: "Credential stored successfully",
                credential_id: 'cred_' + Date.now(),
                platform: 'github'
            };
        }

        if (endpoint.includes('/integrations/config/credentials') && method === 'GET') {
            return [
                {
                    credential_id: 'cred_001',
                    platform: 'github',
                    credential_type: 'oauth_token',
                    created_at: new Date(Date.now() - 86400000).toISOString(),
                    expires_at: new Date(Date.now() + 86400000 * 30).toISOString(),
                    is_active: true,
                    metadata: {}
                },
                {
                    credential_id: 'cred_002',
                    platform: 'huggingface',
                    credential_type: 'api_key',
                    created_at: new Date(Date.now() - 172800000).toISOString(),
                    expires_at: null,
                    is_active: true,
                    metadata: {}
                }
            ];
        }

        if (endpoint.includes('/integrations/config/settings') && method === 'GET') {
            return {
                config_id: 'config_001',
                user_id: this.currentUserId,
                preferences: {
                    auto_connect_on_startup: false,
                    show_notifications: true,
                    notification_level: 'info',
                    auto_scan_security: true,
                    auto_validate_licenses: true,
                    confirm_before_import: true,
                    default_search_limit: 10
                },
                platforms: {
                    github: {
                        platform: 'github',
                        enabled: true,
                        api_base_url: 'https://api.github.com',
                        timeout_seconds: 30,
                        rate_limit: {
                            requests_per_minute: 5000,
                            burst_limit: 100
                        }
                    },
                    huggingface: {
                        platform: 'huggingface',
                        enabled: true,
                        api_base_url: 'https://huggingface.co',
                        timeout_seconds: 45,
                        rate_limit: {
                            requests_per_minute: 1000,
                            burst_limit: 50
                        }
                    },
                    ollama: {
                        platform: 'ollama',
                        enabled: true,
                        api_base_url: 'http://localhost:11434',
                        timeout_seconds: 60,
                        rate_limit: {
                            requests_per_minute: 120,
                            burst_limit: 20
                        }
                    }
                },
                global_security: {
                    security_level: 'standard',
                    require_license_validation: true,
                    require_vulnerability_scan: true,
                    allow_copyleft_licenses: false,
                    max_file_size_mb: 100
                }
            };
        }

        if (endpoint.includes('/integrations/config/settings/validate')) {
            return {
                valid: true,
                issues: [],
                warnings: ['High search limit may impact performance'],
                config_version: '1.0',
                last_updated: new Date().toISOString()
            };
        }

        if (endpoint.includes('/integrations/config/stats')) {
            return {
                configuration: {
                    total_users: 1,
                    active_configs: 1,
                    platform_usage: {
                        github: 1,
                        huggingface: 1,
                        ollama: 1
                    },
                    security_levels: {
                        standard: 1
                    }
                },
                credentials: {
                    total_credentials: 3,
                    active_credentials: 3,
                    expired_credentials: 0,
                    unique_users: 1,
                    platforms: {
                        github: 1,
                        huggingface: 1,
                        ollama: 1
                    }
                }
            };
        }

        if (endpoint.includes('/integrations/config/health')) {
            return {
                status: 'healthy',
                components: {
                    credential_storage: 'healthy',
                    configuration_storage: 'healthy',
                    encryption: 'healthy'
                },
                last_check: new Date().toISOString()
            };
        }

        // Enhanced Security API mock responses
        if (endpoint.includes('/integrations/security/scan') && method === 'POST') {
            return {
                assessment_id: 'sec_' + Date.now(),
                content_id: 'test_content_123',
                platform: 'github',
                timestamp: new Date().toISOString(),
                security_passed: true,
                overall_risk_level: 'low',
                scan_duration: 2.5,
                vulnerability_scan: {
                    completed: true,
                    risk_level: 'low',
                    vulnerabilities_found: 0,
                    scan_method: 'pattern_based'
                },
                license_scan: {
                    completed: true,
                    compliant: true,
                    license_type: 'permissive',
                    issues_found: 0
                },
                threat_scan: {
                    completed: true,
                    threat_level: 'none',
                    threats_found: 0,
                    scan_method: 'comprehensive'
                },
                sandbox_scan: {
                    completed: true,
                    success: true,
                    execution_time: 1.2,
                    security_events: 0,
                    exit_code: 0
                },
                issues: [],
                warnings: ['Content contains network requests'],
                recommendations: ['Content passed security validation', 'All security scans completed successfully'],
                scans_completed: ['vulnerability_scan', 'license_scan', 'threat_scan', 'sandbox_execution'],
                scans_failed: []
            };
        }

        if (endpoint.includes('/integrations/security/events/recent')) {
            return [
                {
                    event_id: 'evt_001',
                    event_type: 'vulnerability_scan',
                    level: 'info',
                    user_id: this.currentUserId,
                    platform: 'github',
                    description: 'Vulnerability scan completed for content test_repo',
                    timestamp: new Date(Date.now() - 3600000).toISOString(),
                    metadata: {
                        content_id: 'test_repo',
                        vulnerabilities_found: 0,
                        risk_level: 'low'
                    }
                },
                {
                    event_id: 'evt_002',
                    event_type: 'threat_detection',
                    level: 'warning',
                    user_id: this.currentUserId,
                    platform: 'huggingface',
                    description: 'Suspicious pattern detected in model weights',
                    timestamp: new Date(Date.now() - 7200000).toISOString(),
                    metadata: {
                        content_id: 'suspicious_model',
                        threat_level: 'medium'
                    }
                },
                {
                    event_id: 'evt_003',
                    event_type: 'sandbox_execution',
                    level: 'info',
                    user_id: this.currentUserId,
                    platform: 'ollama',
                    description: 'Sandbox execution for content local_model',
                    timestamp: new Date(Date.now() - 10800000).toISOString(),
                    metadata: {
                        content_id: 'local_model',
                        success: true,
                        execution_time: 0.8
                    }
                }
            ];
        }

        if (endpoint.includes('/integrations/security/stats')) {
            return {
                total_assessments: 150,
                assessments_passed: 142,
                assessments_failed: 8,
                security_events: {
                    info: 120,
                    warning: 25,
                    error: 4,
                    critical: 1
                },
                threat_levels: {
                    none: 90,
                    low: 40,
                    medium: 15,
                    high: 4,
                    critical: 1
                },
                risk_levels: {
                    none: 75,
                    low: 50,
                    medium: 20,
                    high: 4,
                    critical: 1
                },
                last_updated: new Date().toISOString()
            };
        }

        if (endpoint.includes('/integrations/security/health')) {
            return {
                status: 'healthy',
                components: {
                    vulnerability_scanner: 'healthy',
                    license_scanner: 'healthy',
                    threat_detector: 'healthy',
                    enhanced_sandbox: 'healthy',
                    audit_logger: 'healthy',
                    security_orchestrator: 'healthy'
                },
                last_check: new Date().toISOString(),
                security_policies_active: true
            };
        }

        if (endpoint.includes('/integrations/security/policies')) {
            if (method === 'POST') {
                return {
                    message: 'Updated 3 security policies',
                    updated_policies: {
                        require_license_compliance: true,
                        block_high_risk_vulnerabilities: true,
                        block_medium_risk_threats: false
                    }
                };
            } else {
                return {
                    policies: {
                        require_license_compliance: true,
                        block_high_risk_vulnerabilities: true,
                        block_medium_risk_threats: true,
                        require_sandbox_validation: true,
                        auto_quarantine_threats: true
                    },
                    last_updated: new Date().toISOString()
                };
            }
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

    // === Integration Layer API Methods ===

    async registerConnector(platform, credentials) {
        const requestData = {
            platform: platform,
            oauth_credentials: credentials.oauth_credentials || null,
            api_key: credentials.api_key || null,
            custom_settings: credentials.custom_settings || {}
        };

        return await this.makeRequest('/integrations/connectors/register', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });
    }

    async getConnectorsHealth() {
        return await this.makeRequest('/integrations/connectors/health');
    }

    async getConnectorStatus(platform) {
        return await this.makeRequest(`/integrations/connectors/${platform}/status`);
    }

    async searchContent(query, platforms = null, contentType = 'model', limit = 10, filters = null) {
        const searchData = {
            query: query,
            platforms: platforms,
            content_type: contentType,
            limit: limit,
            filters: filters
        };

        return await this.makeRequest('/integrations/search', {
            method: 'POST',
            body: JSON.stringify(searchData)
        });
    }

    async getContentMetadata(platform, externalId) {
        const encodedId = encodeURIComponent(externalId);
        return await this.makeRequest(`/integrations/content/${platform}/${encodedId}/metadata`);
    }

    async validateContentLicense(platform, externalId) {
        const encodedId = encodeURIComponent(externalId);
        return await this.makeRequest(`/integrations/content/${platform}/${encodedId}/license`);
    }

    async submitImportRequest(importData) {
        return await this.makeRequest('/integrations/import', {
            method: 'POST',
            body: JSON.stringify(importData)
        });
    }

    async getImportStatus(requestId) {
        return await this.makeRequest(`/integrations/import/${requestId}/status`);
    }

    async getImportResult(requestId) {
        return await this.makeRequest(`/integrations/import/${requestId}/result`);
    }

    async cancelImportRequest(requestId) {
        return await this.makeRequest(`/integrations/import/${requestId}`, {
            method: 'DELETE'
        });
    }

    async getIntegrationHealth() {
        return await this.makeRequest('/integrations/health');
    }

    async getIntegrationStats() {
        return await this.makeRequest('/integrations/stats');
    }

    async getActiveImports() {
        return await this.makeRequest('/integrations/imports/active');
    }

    async getImportHistory(limit = 20, offset = 0) {
        return await this.makeRequest(`/integrations/imports/history?limit=${limit}&offset=${offset}`);
    }

    // === Configuration Management API Methods ===

    async storeCredential(platform, credentialData) {
        const requestData = {
            platform: platform,
            ...credentialData
        };

        return await this.makeRequest('/integrations/config/credentials', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });
    }

    async listCredentials(platform = null, includeExpired = false) {
        const params = new URLSearchParams();
        if (platform) params.append('platform', platform);
        if (includeExpired) params.append('include_expired', 'true');

        return await this.makeRequest(`/integrations/config/credentials?${params}`);
    }

    async validateCredential(credentialId) {
        return await this.makeRequest(`/integrations/config/credentials/${credentialId}/validate`);
    }

    async deleteCredential(credentialId) {
        return await this.makeRequest(`/integrations/config/credentials/${credentialId}`, {
            method: 'DELETE'
        });
    }

    async cleanupExpiredCredentials() {
        return await this.makeRequest('/integrations/config/credentials/cleanup', {
            method: 'POST'
        });
    }

    async getUserConfiguration() {
        return await this.makeRequest('/integrations/config/settings');
    }

    async updateUserConfiguration(configData) {
        return await this.makeRequest('/integrations/config/settings', {
            method: 'PUT',
            body: JSON.stringify(configData)
        });
    }

    async getPlatformConfiguration(platform) {
        return await this.makeRequest(`/integrations/config/settings/platforms/${platform}`);
    }

    async updatePlatformConfiguration(platform, configData) {
        return await this.makeRequest(`/integrations/config/settings/platforms/${platform}`, {
            method: 'PUT',
            body: JSON.stringify(configData)
        });
    }

    async validateConfiguration(platform = null) {
        const params = platform ? `?platform=${platform}` : '';
        return await this.makeRequest(`/integrations/config/settings/validate${params}`);
    }

    async exportConfiguration() {
        return await this.makeRequest('/integrations/config/settings/export');
    }

    async importConfiguration(configData, merge = true) {
        return await this.makeRequest(`/integrations/config/settings/import?merge=${merge}`, {
            method: 'POST',
            body: JSON.stringify(configData)
        });
    }

    async resetConfiguration() {
        return await this.makeRequest('/integrations/config/settings/reset', {
            method: 'POST'
        });
    }

    async getConfigurationStats() {
        return await this.makeRequest('/integrations/config/stats');
    }

    async getConfigurationHealth() {
        return await this.makeRequest('/integrations/config/health');
    }

    // === Enhanced Security API Methods ===

    async performSecurityScan(contentPath, contentId, platform, metadata, enableSandbox = true) {
        const requestData = {
            content_path: contentPath,
            content_id: contentId,
            platform: platform,
            metadata: metadata,
            enable_sandbox: enableSandbox
        };

        return await this.makeRequest('/integrations/security/scan', {
            method: 'POST',
            body: JSON.stringify(requestData)
        });
    }

    async getSecurityAssessment(assessmentId) {
        return await this.makeRequest(`/integrations/security/assessment/${assessmentId}`);
    }

    async getRecentSecurityEvents(limit = 50, level = null) {
        const params = new URLSearchParams();
        params.append('limit', limit.toString());
        if (level) params.append('level', level);

        return await this.makeRequest(`/integrations/security/events/recent?${params}`);
    }

    async getSecurityStatistics() {
        return await this.makeRequest('/integrations/security/stats');
    }

    async getSecurityHealth() {
        return await this.makeRequest('/integrations/security/health');
    }

    async getSecurityPolicies() {
        return await this.makeRequest('/integrations/security/policies');
    }

    async updateSecurityPolicies(policies) {
        return await this.makeRequest('/integrations/security/policies/update', {
            method: 'POST',
            body: JSON.stringify(policies)
        });
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