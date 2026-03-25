// Package auth provides authentication functionality for the PRSM Go SDK
package auth

import (
	"crypto/rand"
	"encoding/hex"
)

// Manager handles authentication for PRSM API requests
type Manager struct {
	apiKey string
	token  string
}

// New creates a new authentication manager with the given API key
func New(apiKey string) *Manager {
	return &Manager{
		apiKey: apiKey,
	}
}

// GetHeaders returns the authentication headers for API requests
func (m *Manager) GetHeaders() (map[string]string, error) {
	headers := map[string]string{
		"Content-Type": "application/json",
		"User-Agent":   "prsm-go-sdk/1.0",
	}

	if m.apiKey != "" {
		headers["Authorization"] = "Bearer " + m.apiKey
	}

	if m.token != "" {
		headers["X-Session-Token"] = m.token
	}

	return headers, nil
}

// IsAuthenticated returns whether the manager has valid credentials
func (m *Manager) IsAuthenticated() bool {
	return m.apiKey != "" || m.token != ""
}

// SetToken sets a session token for authentication
func (m *Manager) SetToken(token string) {
	m.token = token
}

// GetAPIKey returns the current API key
func (m *Manager) GetAPIKey() string {
	return m.apiKey
}

// GenerateSecureID generates a secure random ID for request tracing
func GenerateSecureID() (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes), nil
}
