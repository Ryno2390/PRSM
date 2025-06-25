/**
 * PRSM SDK WebSocket Manager
 * Handles real-time communication, session updates, and streaming responses
 */

import { EventEmitter } from 'eventemitter3';
import {
  WebSocketMessage,
  WebSocketSubscription,
  StreamChunk,
  SessionInfo,
  SafetyLevel,
  Callback,
  ErrorCallback,
} from './types';
import {
  WebSocketError,
  WebSocketConnectionError,
  InvalidWebSocketMessageError,
  AuthenticationError,
  NetworkError,
  toPRSMError,
} from './errors';

// ============================================================================
// WEBSOCKET MANAGER CONFIGURATION
// ============================================================================

export interface WebSocketManagerConfig {
  /** WebSocket URL */
  websocketUrl: string;
  /** Authentication token or API key */
  authToken?: string;
  /** Authentication headers function */
  getAuthHeaders?: () => Promise<Record<string, string>>;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Reconnect interval in milliseconds */
  reconnectInterval?: number;
  /** Maximum reconnect attempts */
  maxReconnectAttempts?: number;
  /** Connection timeout in milliseconds */
  connectionTimeout?: number;
  /** Heartbeat interval in milliseconds */
  heartbeatInterval?: number;
  /** Debug logging */
  debug?: boolean;
  /** Custom protocols */
  protocols?: string[];
}

// ============================================================================
// CONNECTION STATES
// ============================================================================

export enum WebSocketState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTING = 'disconnecting',
  DISCONNECTED = 'disconnected',
  RECONNECTING = 'reconnecting',
  FAILED = 'failed'
}

// ============================================================================
// MESSAGE TYPES
// ============================================================================

export enum MessageType {
  // Connection management
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  PING = 'ping',
  PONG = 'pong',
  AUTH = 'auth',
  AUTH_SUCCESS = 'auth_success',
  AUTH_FAILED = 'auth_failed',

  // Subscriptions
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  SUBSCRIPTION_SUCCESS = 'subscription_success',
  SUBSCRIPTION_ERROR = 'subscription_error',

  // Session updates
  SESSION_PROGRESS = 'session_progress',
  SESSION_COMPLETED = 'session_completed',
  SESSION_FAILED = 'session_failed',
  SESSION_CANCELLED = 'session_cancelled',

  // Streaming
  STREAM_START = 'stream_start',
  STREAM_CHUNK = 'stream_chunk',
  STREAM_END = 'stream_end',
  STREAM_ERROR = 'stream_error',

  // Safety and monitoring
  SAFETY_ALERT = 'safety_alert',
  SECURITY_NOTICE = 'security_notice',
  SYSTEM_NOTICE = 'system_notice',

  // Tool execution
  TOOL_EXECUTION_UPDATE = 'tool_execution_update',
  TOOL_EXECUTION_COMPLETED = 'tool_execution_completed',
  TOOL_EXECUTION_FAILED = 'tool_execution_failed',

  // Marketplace
  MODEL_STATUS_UPDATE = 'model_status_update',
  RENTAL_EXPIRING = 'rental_expiring',
  RENTAL_EXPIRED = 'rental_expired',

  // FTNS and payments
  BALANCE_UPDATE = 'balance_update',
  TRANSACTION_UPDATE = 'transaction_update',
  PAYMENT_STATUS = 'payment_status',

  // Errors
  ERROR = 'error',
  RATE_LIMIT = 'rate_limit',
  QUOTA_EXCEEDED = 'quota_exceeded'
}

// ============================================================================
// EVENT INTERFACES
// ============================================================================

export interface ConnectionEvent {
  state: WebSocketState;
  timestamp: string;
  reconnectAttempt?: number;
  error?: string;
}

export interface SessionProgressEvent {
  sessionId: string;
  progress: number;
  status: string;
  currentStep?: string;
  estimatedCompletion?: string;
  metadata?: Record<string, any>;
}

export interface StreamEvent {
  streamId: string;
  chunk: StreamChunk;
  metadata?: Record<string, any>;
}

export interface SafetyAlertEvent {
  level: SafetyLevel;
  type: string;
  message: string;
  sessionId?: string;
  timestamp: string;
  actions?: string[];
}

// ============================================================================
// SUBSCRIPTION MANAGER
// ============================================================================

class SubscriptionManager {
  private subscriptions: Map<string, WebSocketSubscription> = new Map();
  private callbacks: Map<string, Set<Callback<any>>> = new Map();

  subscribe(channel: string, params: Record<string, any> = {}, callback?: Callback<any>): string {
    const subscriptionId = `${channel}:${JSON.stringify(params)}`;
    
    this.subscriptions.set(subscriptionId, { channel, params });
    
    if (callback) {
      if (!this.callbacks.has(subscriptionId)) {
        this.callbacks.set(subscriptionId, new Set());
      }
      this.callbacks.get(subscriptionId)!.add(callback);
    }
    
    return subscriptionId;
  }

  unsubscribe(subscriptionId: string, callback?: Callback<any>): boolean {
    if (callback && this.callbacks.has(subscriptionId)) {
      const callbacks = this.callbacks.get(subscriptionId)!;
      callbacks.delete(callback);
      
      if (callbacks.size === 0) {
        this.callbacks.delete(subscriptionId);
        this.subscriptions.delete(subscriptionId);
        return true;
      }
      return false;
    } else {
      this.callbacks.delete(subscriptionId);
      return this.subscriptions.delete(subscriptionId);
    }
  }

  getSubscription(subscriptionId: string): WebSocketSubscription | undefined {
    return this.subscriptions.get(subscriptionId);
  }

  getAllSubscriptions(): WebSocketSubscription[] {
    return Array.from(this.subscriptions.values());
  }

  notifyCallbacks(subscriptionId: string, data: any): void {
    const callbacks = this.callbacks.get(subscriptionId);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Subscription callback error:', error);
        }
      });
    }
  }

  clear(): void {
    this.subscriptions.clear();
    this.callbacks.clear();
  }
}

// ============================================================================
// WEBSOCKET MANAGER CLASS
// ============================================================================

/**
 * Manages WebSocket connections and real-time communication with PRSM
 */
export class WebSocketManager extends EventEmitter {
  private readonly websocketUrl: string;
  private readonly config: WebSocketManagerConfig;
  private readonly subscriptions: SubscriptionManager;

  private ws: WebSocket | null = null;
  private state: WebSocketState = WebSocketState.DISCONNECTED;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private lastPongTime: number = 0;

  constructor(config: WebSocketManagerConfig) {
    super();
    
    this.websocketUrl = config.websocketUrl;
    this.config = {
      autoReconnect: true,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      connectionTimeout: 30000,
      heartbeatInterval: 30000,
      debug: false,
      protocols: [],
      ...config,
    };
    
    this.subscriptions = new SubscriptionManager();
  }

  // ============================================================================
  // CONNECTION MANAGEMENT
  // ============================================================================

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.state === WebSocketState.CONNECTED || 
        this.state === WebSocketState.CONNECTING) {
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        this.setState(WebSocketState.CONNECTING);
        
        if (this.config.debug) {
          console.log('[PRSM WebSocket] Connecting to:', this.websocketUrl);
        }

        // Build connection URL with auth
        const url = this.buildConnectionUrl();
        
        this.ws = new WebSocket(url, this.config.protocols);
        
        // Set connection timeout
        this.connectionTimeout = setTimeout(() => {
          this.handleConnectionTimeout();
          reject(new WebSocketConnectionError('Connection timeout'));
        }, this.config.connectionTimeout);

        this.ws.onopen = () => {
          this.handleOpen();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.handleClose(event);
        };

        this.ws.onerror = (event) => {
          this.handleError(event);
          reject(new WebSocketConnectionError('Connection failed'));
        };

      } catch (error) {
        this.setState(WebSocketState.FAILED);
        reject(new WebSocketConnectionError('Failed to create WebSocket connection'));
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.state === WebSocketState.DISCONNECTED || 
        this.state === WebSocketState.DISCONNECTING) {
      return;
    }

    this.setState(WebSocketState.DISCONNECTING);
    
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Disconnecting');
    }

    this.clearTimers();
    this.config.autoReconnect = false; // Prevent auto-reconnect
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.state === WebSocketState.CONNECTED && 
           this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get current connection state
   */
  getState(): WebSocketState {
    return this.state;
  }

  // ============================================================================
  // MESSAGING
  // ============================================================================

  /**
   * Send message to server
   */
  send(message: WebSocketMessage): void {
    if (!this.isConnected()) {
      if (this.config.autoReconnect) {
        this.messageQueue.push(message);
        this.reconnect();
        return;
      } else {
        throw new WebSocketError('WebSocket not connected');
      }
    }

    try {
      const messageStr = JSON.stringify(message);
      this.ws!.send(messageStr);
      
      if (this.config.debug) {
        console.log('[PRSM WebSocket] Sent:', message.type, message.data);
      }
    } catch (error) {
      throw new WebSocketError('Failed to send message', message.type);
    }
  }

  /**
   * Send authentication message
   */
  private async sendAuth(): Promise<void> {
    try {
      let authData: Record<string, any> = {};

      if (this.config.getAuthHeaders) {
        const headers = await this.config.getAuthHeaders();
        authData.headers = headers;
      } else if (this.config.authToken) {
        authData.token = this.config.authToken;
      }

      this.send({
        type: MessageType.AUTH,
        data: authData,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      throw new AuthenticationError('Failed to send authentication');
    }
  }

  // ============================================================================
  // SUBSCRIPTIONS
  // ============================================================================

  /**
   * Subscribe to a channel
   */
  subscribe(
    channel: string,
    params: Record<string, any> = {},
    callback?: Callback<any>
  ): string {
    const subscriptionId = this.subscriptions.subscribe(channel, params, callback);
    
    if (this.isConnected()) {
      this.send({
        type: MessageType.SUBSCRIBE,
        data: { channel, params, subscriptionId },
        timestamp: new Date().toISOString(),
      });
    }
    
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Subscribed to:', channel, params);
    }
    
    return subscriptionId;
  }

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(subscriptionId: string, callback?: Callback<any>): void {
    const wasRemoved = this.subscriptions.unsubscribe(subscriptionId, callback);
    
    if (wasRemoved && this.isConnected()) {
      this.send({
        type: MessageType.UNSUBSCRIBE,
        data: { subscriptionId },
        timestamp: new Date().toISOString(),
      });
    }
    
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Unsubscribed from:', subscriptionId);
    }
  }

  // ============================================================================
  // CONVENIENCE METHODS
  // ============================================================================

  /**
   * Subscribe to session updates
   */
  subscribeToSession(
    sessionId: string,
    callback: (event: SessionProgressEvent) => void
  ): string {
    return this.subscribe('session_updates', { sessionId }, callback);
  }

  /**
   * Subscribe to streaming responses
   */
  subscribeToStream(
    streamId: string,
    callback: (event: StreamEvent) => void
  ): string {
    return this.subscribe('stream', { streamId }, callback);
  }

  /**
   * Subscribe to safety alerts
   */
  subscribeToSafetyAlerts(
    callback: (event: SafetyAlertEvent) => void
  ): string {
    return this.subscribe('safety_alerts', {}, callback);
  }

  /**
   * Subscribe to balance updates
   */
  subscribeToBalanceUpdates(
    callback: (balance: any) => void
  ): string {
    return this.subscribe('balance_updates', {}, callback);
  }

  /**
   * Subscribe to tool execution updates
   */
  subscribeToToolExecution(
    executionId: string,
    callback: (update: any) => void
  ): string {
    return this.subscribe('tool_execution', { executionId }, callback);
  }

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  private handleOpen(): void {
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Connected');
    }

    this.clearTimers();
    this.setState(WebSocketState.CONNECTED);
    this.reconnectAttempts = 0;
    
    // Send authentication
    this.sendAuth().catch(error => {
      this.emit('error', new AuthenticationError('WebSocket authentication failed'));
    });
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Send queued messages
    this.processMessageQueue();
    
    // Re-establish subscriptions
    this.reestablishSubscriptions();
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      if (this.config.debug) {
        console.log('[PRSM WebSocket] Received:', message.type, message.data);
      }

      this.processMessage(message);
    } catch (error) {
      this.emit('error', new InvalidWebSocketMessageError('Invalid message format'));
    }
  }

  private handleClose(event: CloseEvent): void {
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Disconnected:', event.code, event.reason);
    }

    this.clearTimers();
    this.setState(WebSocketState.DISCONNECTED);
    
    this.emit('disconnected', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
    });

    // Auto-reconnect if enabled and not a clean disconnect
    if (this.config.autoReconnect && !event.wasClean && event.code !== 1000) {
      this.reconnect();
    }
  }

  private handleError(event: Event): void {
    if (this.config.debug) {
      console.error('[PRSM WebSocket] Error:', event);
    }

    this.emit('error', new WebSocketError('WebSocket error occurred'));
  }

  private handleConnectionTimeout(): void {
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Connection timeout');
    }

    this.setState(WebSocketState.FAILED);
    
    if (this.ws) {
      this.ws.close();
    }
  }

  // ============================================================================
  // MESSAGE PROCESSING
  // ============================================================================

  private processMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case MessageType.AUTH_SUCCESS:
        this.emit('authenticated', message.data);
        break;

      case MessageType.AUTH_FAILED:
        this.emit('error', new AuthenticationError('WebSocket authentication failed'));
        break;

      case MessageType.PONG:
        this.lastPongTime = Date.now();
        break;

      case MessageType.SUBSCRIPTION_SUCCESS:
        this.emit('subscribed', message.data);
        break;

      case MessageType.SUBSCRIPTION_ERROR:
        this.emit('subscription_error', message.data);
        break;

      case MessageType.SESSION_PROGRESS:
        this.handleSessionProgress(message);
        break;

      case MessageType.SESSION_COMPLETED:
      case MessageType.SESSION_FAILED:
      case MessageType.SESSION_CANCELLED:
        this.handleSessionUpdate(message);
        break;

      case MessageType.STREAM_CHUNK:
      case MessageType.STREAM_END:
      case MessageType.STREAM_ERROR:
        this.handleStreamEvent(message);
        break;

      case MessageType.SAFETY_ALERT:
        this.handleSafetyAlert(message);
        break;

      case MessageType.BALANCE_UPDATE:
        this.handleBalanceUpdate(message);
        break;

      case MessageType.TOOL_EXECUTION_UPDATE:
      case MessageType.TOOL_EXECUTION_COMPLETED:
      case MessageType.TOOL_EXECUTION_FAILED:
        this.handleToolExecutionUpdate(message);
        break;

      case MessageType.ERROR:
        this.emit('server_error', message.data);
        break;

      case MessageType.RATE_LIMIT:
        this.emit('rate_limit', message.data);
        break;

      case MessageType.QUOTA_EXCEEDED:
        this.emit('quota_exceeded', message.data);
        break;

      default:
        this.emit('message', message);
        break;
    }
  }

  private handleSessionProgress(message: WebSocketMessage): void {
    const event: SessionProgressEvent = {
      sessionId: message.data.sessionId,
      progress: message.data.progress,
      status: message.data.status,
      currentStep: message.data.currentStep,
      estimatedCompletion: message.data.estimatedCompletion,
      metadata: message.data.metadata,
    };

    this.emit('session_progress', event);
    this.notifySubscriptions('session_updates', event);
  }

  private handleSessionUpdate(message: WebSocketMessage): void {
    this.emit('session_update', message.data);
    this.notifySubscriptions('session_updates', message.data);
  }

  private handleStreamEvent(message: WebSocketMessage): void {
    const event: StreamEvent = {
      streamId: message.data.streamId,
      chunk: message.data.chunk,
      metadata: message.data.metadata,
    };

    this.emit('stream_event', event);
    this.notifySubscriptions('stream', event);
  }

  private handleSafetyAlert(message: WebSocketMessage): void {
    const event: SafetyAlertEvent = {
      level: message.data.level,
      type: message.data.type,
      message: message.data.message,
      sessionId: message.data.sessionId,
      timestamp: message.data.timestamp,
      actions: message.data.actions,
    };

    this.emit('safety_alert', event);
    this.notifySubscriptions('safety_alerts', event);
  }

  private handleBalanceUpdate(message: WebSocketMessage): void {
    this.emit('balance_update', message.data);
    this.notifySubscriptions('balance_updates', message.data);
  }

  private handleToolExecutionUpdate(message: WebSocketMessage): void {
    this.emit('tool_execution_update', message.data);
    this.notifySubscriptions('tool_execution', message.data);
  }

  // ============================================================================
  // SUBSCRIPTION MANAGEMENT
  // ============================================================================

  private notifySubscriptions(channel: string, data: any): void {
    // Find matching subscriptions and notify callbacks
    this.subscriptions.getAllSubscriptions().forEach(subscription => {
      if (subscription.channel === channel) {
        const subscriptionId = `${channel}:${JSON.stringify(subscription.params)}`;
        this.subscriptions.notifyCallbacks(subscriptionId, data);
      }
    });
  }

  private reestablishSubscriptions(): void {
    const subscriptions = this.subscriptions.getAllSubscriptions();
    
    subscriptions.forEach(subscription => {
      this.send({
        type: MessageType.SUBSCRIBE,
        data: {
          channel: subscription.channel,
          params: subscription.params,
        },
        timestamp: new Date().toISOString(),
      });
    });

    if (this.config.debug && subscriptions.length > 0) {
      console.log('[PRSM WebSocket] Re-established', subscriptions.length, 'subscriptions');
    }
  }

  // ============================================================================
  // RECONNECTION LOGIC
  // ============================================================================

  private reconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts!) {
      this.setState(WebSocketState.FAILED);
      this.emit('error', new WebSocketConnectionError('Max reconnect attempts exceeded'));
      return;
    }

    this.setState(WebSocketState.RECONNECTING);
    this.reconnectAttempts++;

    if (this.config.debug) {
      console.log(`[PRSM WebSocket] Reconnecting... (attempt ${this.reconnectAttempts})`);
    }

    this.emit('reconnecting', {
      attempt: this.reconnectAttempts,
      maxAttempts: this.config.maxReconnectAttempts!,
    });

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        if (this.config.debug) {
          console.error('[PRSM WebSocket] Reconnect failed:', error);
        }
        this.reconnect(); // Try again
      });
    }, this.config.reconnectInterval);
  }

  // ============================================================================
  // HEARTBEAT
  // ============================================================================

  private startHeartbeat(): void {
    if (!this.config.heartbeatInterval) return;

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({
          type: MessageType.PING,
          data: { timestamp: Date.now() },
          timestamp: new Date().toISOString(),
        });

        // Check if we received pong recently
        if (this.lastPongTime > 0 && 
            Date.now() - this.lastPongTime > this.config.heartbeatInterval! * 2) {
          if (this.config.debug) {
            console.log('[PRSM WebSocket] Heartbeat timeout, reconnecting');
          }
          this.reconnect();
        }
      }
    }, this.config.heartbeatInterval);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private setState(newState: WebSocketState): void {
    if (this.state !== newState) {
      const oldState = this.state;
      this.state = newState;
      
      this.emit('state_change', {
        oldState,
        newState,
        timestamp: new Date().toISOString(),
      });
    }
  }

  private buildConnectionUrl(): string {
    const url = new URL(this.websocketUrl);
    
    if (this.config.authToken) {
      url.searchParams.append('token', this.config.authToken);
    }
    
    return url.toString();
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift()!;
      this.send(message);
    }
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }
  }

  /**
   * Clean up resources and disconnect
   */
  destroy(): void {
    this.config.autoReconnect = false;
    this.disconnect();
    this.clearTimers();
    this.subscriptions.clear();
    this.messageQueue.length = 0;
    this.removeAllListeners();
    
    if (this.config.debug) {
      console.log('[PRSM WebSocket] Manager destroyed');
    }
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    state: WebSocketState;
    reconnectAttempts: number;
    subscriptions: number;
    queuedMessages: number;
    lastPongTime: number;
  } {
    return {
      state: this.state,
      reconnectAttempts: this.reconnectAttempts,
      subscriptions: this.subscriptions.getAllSubscriptions().length,
      queuedMessages: this.messageQueue.length,
      lastPongTime: this.lastPongTime,
    };
  }
}