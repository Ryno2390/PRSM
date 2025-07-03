// Package websocket provides WebSocket functionality for real-time PRSM communication
package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pkg/errors"
)

// Manager handles WebSocket connections and communication
type Manager struct {
	url              string
	getAuthHeaders   func() (map[string]string, error)
	autoReconnect    bool
	reconnectInterval time.Duration
	maxReconnectAttempts int
	connectionTimeout time.Duration
	heartbeatInterval time.Duration
	debug            bool

	conn       *websocket.Conn
	connected  bool
	mu         sync.RWMutex
	handlers   map[string][]MessageHandler
	stopCh     chan struct{}
	doneCh     chan struct{}
}

// Config represents WebSocket manager configuration
type Config struct {
	URL                  string
	GetAuthHeaders       func() (map[string]string, error)
	AutoReconnect        bool
	ReconnectInterval    time.Duration
	MaxReconnectAttempts int
	ConnectionTimeout    time.Duration
	HeartbeatInterval    time.Duration
	Debug                bool
}

// MessageHandler represents a handler for WebSocket messages
type MessageHandler func(*Message)

// Message represents a WebSocket message
type Message struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
	RequestID string                 `json:"request_id,omitempty"`
}

// StreamQueryRequest represents a streaming query request
type StreamQueryRequest struct {
	Query       string                 `json:"query"`
	ModelID     *string                `json:"model_id,omitempty"`
	MaxTokens   int                    `json:"max_tokens"`
	Temperature float64                `json:"temperature"`
	Context     map[string]interface{} `json:"context,omitempty"`
}

// New creates a new WebSocket manager
func New(config *Config) *Manager {
	if config.ReconnectInterval == 0 {
		config.ReconnectInterval = 5 * time.Second
	}
	if config.MaxReconnectAttempts == 0 {
		config.MaxReconnectAttempts = 10
	}
	if config.ConnectionTimeout == 0 {
		config.ConnectionTimeout = 30 * time.Second
	}
	if config.HeartbeatInterval == 0 {
		config.HeartbeatInterval = 30 * time.Second
	}

	return &Manager{
		url:                  config.URL,
		getAuthHeaders:       config.GetAuthHeaders,
		autoReconnect:        config.AutoReconnect,
		reconnectInterval:    config.ReconnectInterval,
		maxReconnectAttempts: config.MaxReconnectAttempts,
		connectionTimeout:    config.ConnectionTimeout,
		heartbeatInterval:    config.HeartbeatInterval,
		debug:                config.Debug,
		handlers:             make(map[string][]MessageHandler),
		stopCh:               make(chan struct{}),
		doneCh:               make(chan struct{}),
	}
}

// Connect establishes WebSocket connection
func (m *Manager) Connect(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.connected {
		return nil
	}

	// Get authentication headers
	headers := http.Header{}
	if m.getAuthHeaders != nil {
		authHeaders, err := m.getAuthHeaders()
		if err != nil {
			return errors.Wrap(err, "failed to get auth headers")
		}
		for key, value := range authHeaders {
			headers.Set(key, value)
		}
	}

	// Set up dialer with timeout
	dialer := websocket.Dialer{
		HandshakeTimeout: m.connectionTimeout,
	}

	// Connect to WebSocket
	conn, _, err := dialer.DialContext(ctx, m.url, headers)
	if err != nil {
		return errors.Wrap(err, "failed to connect to WebSocket")
	}

	m.conn = conn
	m.connected = true

	// Start message handling goroutines
	go m.readMessages()
	go m.heartbeat()

	if m.debug {
		fmt.Printf("[WebSocket] Connected to %s\n", m.url)
	}

	return nil
}

// Disconnect closes WebSocket connection
func (m *Manager) Disconnect() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if !m.connected {
		return nil
	}

	close(m.stopCh)
	<-m.doneCh

	if m.conn != nil {
		err := m.conn.Close()
		m.conn = nil
		m.connected = false
		
		if m.debug {
			fmt.Println("[WebSocket] Disconnected")
		}
		
		return err
	}

	return nil
}

// IsConnected returns whether the WebSocket is connected
func (m *Manager) IsConnected() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.connected
}

// OnMessage registers a message handler for a specific message type
func (m *Manager) OnMessage(messageType string, handler MessageHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.handlers[messageType] = append(m.handlers[messageType], handler)
}

// SendMessage sends a message over WebSocket
func (m *Manager) SendMessage(message *Message) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.connected || m.conn == nil {
		return errors.New("WebSocket not connected")
	}

	return m.conn.WriteJSON(message)
}

// StreamQuery initiates a streaming query
func (m *Manager) StreamQuery(ctx context.Context, req *StreamQueryRequest) (<-chan *Message, <-chan error) {
	messageCh := make(chan *Message, 100)
	errorCh := make(chan error, 1)

	go func() {
		defer close(messageCh)
		defer close(errorCh)

		// Send query request
		queryMessage := &Message{
			Type: "stream_query",
			Data: map[string]interface{}{
				"query":       req.Query,
				"model_id":    req.ModelID,
				"max_tokens":  req.MaxTokens,
				"temperature": req.Temperature,
				"context":     req.Context,
			},
			Timestamp: time.Now(),
		}

		if err := m.SendMessage(queryMessage); err != nil {
			errorCh <- errors.Wrap(err, "failed to send query message")
			return
		}

		// Listen for streaming responses
		responseCh := make(chan *Message)
		m.OnMessage("stream_response", func(msg *Message) {
			select {
			case responseCh <- msg:
			case <-ctx.Done():
			}
		})

		for {
			select {
			case <-ctx.Done():
				return
			case msg := <-responseCh:
				if msg.Type == "stream_complete" {
					return
				}
				messageCh <- msg
			}
		}
	}()

	return messageCh, errorCh
}

// readMessages handles incoming WebSocket messages
func (m *Manager) readMessages() {
	defer func() {
		m.doneCh <- struct{}{}
	}()

	for {
		select {
		case <-m.stopCh:
			return
		default:
			var message Message
			err := m.conn.ReadJSON(&message)
			if err != nil {
				if m.debug {
					fmt.Printf("[WebSocket] Read error: %v\n", err)
				}
				
				if m.autoReconnect {
					go m.reconnect()
				}
				return
			}

			// Dispatch message to handlers
			m.mu.RLock()
			handlers := m.handlers[message.Type]
			m.mu.RUnlock()

			for _, handler := range handlers {
				go handler(&message)
			}
		}
	}
}

// heartbeat sends periodic ping messages
func (m *Manager) heartbeat() {
	ticker := time.NewTicker(m.heartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			if m.connected && m.conn != nil {
				if err := m.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					if m.debug {
						fmt.Printf("[WebSocket] Ping failed: %v\n", err)
					}
				}
			}
		}
	}
}

// reconnect attempts to reconnect to WebSocket
func (m *Manager) reconnect() {
	for attempt := 0; attempt < m.maxReconnectAttempts; attempt++ {
		if m.debug {
			fmt.Printf("[WebSocket] Reconnection attempt %d/%d\n", attempt+1, m.maxReconnectAttempts)
		}

		time.Sleep(m.reconnectInterval)

		ctx, cancel := context.WithTimeout(context.Background(), m.connectionTimeout)
		err := m.Connect(ctx)
		cancel()

		if err == nil {
			if m.debug {
				fmt.Println("[WebSocket] Reconnected successfully")
			}
			return
		}

		if m.debug {
			fmt.Printf("[WebSocket] Reconnection attempt %d failed: %v\n", attempt+1, err)
		}
	}

	if m.debug {
		fmt.Printf("[WebSocket] Failed to reconnect after %d attempts\n", m.maxReconnectAttempts)
	}
}

// GetStats returns WebSocket connection statistics
func (m *Manager) GetStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"connected":             m.connected,
		"url":                   m.url,
		"auto_reconnect":        m.autoReconnect,
		"reconnect_interval":    m.reconnectInterval.String(),
		"max_reconnect_attempts": m.maxReconnectAttempts,
		"connection_timeout":    m.connectionTimeout.String(),
		"heartbeat_interval":    m.heartbeatInterval.String(),
	}
}