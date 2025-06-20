// Package client provides the main PRSM Go SDK client
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/PRSM-AI/prsm-go-sdk/auth"
	"github.com/PRSM-AI/prsm-go-sdk/ftns"
	"github.com/PRSM-AI/prsm-go-sdk/marketplace"
	"github.com/PRSM-AI/prsm-go-sdk/tools"
	"github.com/PRSM-AI/prsm-go-sdk/types"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

// Client is the main PRSM client for AI queries, token management, and marketplace access
type Client struct {
	baseURL      string
	websocketURL string
	timeout      time.Duration
	maxRetries   int
	httpClient   *http.Client
	rateLimiter  *rate.Limiter

	// Managers
	Auth        *auth.Manager
	FTNS        *ftns.Manager
	Marketplace *marketplace.Manager
	Tools       *tools.Executor
}

// Config holds configuration for the PRSM client
type Config struct {
	APIKey       string
	BaseURL      string
	WebSocketURL string
	Timeout      time.Duration
	MaxRetries   int
	RateLimit    rate.Limit // requests per second
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		BaseURL:      "https://api.prsm.ai/v1",
		WebSocketURL: "wss://ws.prsm.ai/v1",
		Timeout:      60 * time.Second,
		MaxRetries:   3,
		RateLimit:    rate.Limit(10), // 10 requests per second
	}
}

// New creates a new PRSM client with the given API key
func New(apiKey string) *Client {
	config := DefaultConfig()
	config.APIKey = apiKey
	return NewWithConfig(config)
}

// NewWithConfig creates a new PRSM client with custom configuration
func NewWithConfig(config *Config) *Client {
	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	client := &Client{
		baseURL:      config.BaseURL,
		websocketURL: config.WebSocketURL,
		timeout:      config.Timeout,
		maxRetries:   config.MaxRetries,
		httpClient:   httpClient,
		rateLimiter:  rate.NewLimiter(config.RateLimit, 1),
	}

	// Initialize managers
	client.Auth = auth.New(config.APIKey)
	client.FTNS = ftns.New(client)
	client.Marketplace = marketplace.New(client)
	client.Tools = tools.New(client)

	return client
}

// Query executes an AI query with PRSM
func (c *Client) Query(ctx context.Context, req *types.QueryRequest) (*types.PRSMResponse, error) {
	if req.Prompt == "" {
		return nil, errors.New("prompt cannot be empty")
	}

	// Set defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = 1000
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}
	if req.Context == nil {
		req.Context = make(map[string]interface{})
	}

	var response types.PRSMResponse
	err := c.makeRequest(ctx, "POST", "/nwtn/query", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to execute query")
	}

	return &response, nil
}

// EstimateCost estimates the FTNS cost for a query without executing it
func (c *Client) EstimateCost(ctx context.Context, prompt string, modelID *string) (float64, error) {
	reqData := map[string]interface{}{
		"prompt": prompt,
	}
	if modelID != nil {
		reqData["model_id"] = *modelID
	}

	var response struct {
		EstimatedCost float64 `json:"estimated_cost"`
	}

	err := c.makeRequest(ctx, "POST", "/nwtn/estimate-cost", reqData, &response)
	if err != nil {
		return 0, errors.Wrap(err, "failed to estimate cost")
	}

	return response.EstimatedCost, nil
}

// GetSafetyStatus retrieves the current safety monitoring status
func (c *Client) GetSafetyStatus(ctx context.Context) (*types.SafetyStatus, error) {
	var status types.SafetyStatus
	err := c.makeRequest(ctx, "GET", "/safety/status", nil, &status)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get safety status")
	}
	return &status, nil
}

// ListAvailableModels lists all available models in the network
func (c *Client) ListAvailableModels(ctx context.Context) ([]*types.ModelInfo, error) {
	var response struct {
		Models []*types.ModelInfo `json:"models"`
	}

	err := c.makeRequest(ctx, "GET", "/models", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list models")
	}

	return response.Models, nil
}

// HealthCheck checks API health and connectivity
func (c *Client) HealthCheck(ctx context.Context) (map[string]interface{}, error) {
	var health map[string]interface{}
	err := c.makeRequest(ctx, "GET", "/health", nil, &health)
	if err != nil {
		return nil, errors.Wrap(err, "health check failed")
	}
	return health, nil
}

// makeRequest is a helper method for making HTTP requests with retry logic
func (c *Client) makeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error {
	// Rate limiting
	if err := c.rateLimiter.Wait(ctx); err != nil {
		return errors.Wrap(err, "rate limit error")
	}

	var bodyReader *bytes.Reader
	if reqBody != nil {
		jsonBody, err := json.Marshal(reqBody)
		if err != nil {
			return errors.Wrap(err, "failed to marshal request body")
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	reqURL, err := url.JoinPath(c.baseURL, endpoint)
	if err != nil {
		return errors.Wrap(err, "failed to construct request URL")
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if bodyReader != nil {
			bodyReader.Seek(0, 0) // Reset reader for retries
		}

		req, err := http.NewRequestWithContext(ctx, method, reqURL, bodyReader)
		if err != nil {
			return errors.Wrap(err, "failed to create request")
		}

		// Add headers
		headers, err := c.Auth.GetHeaders()
		if err != nil {
			return errors.Wrap(err, "failed to get auth headers")
		}

		for key, value := range headers {
			req.Header.Set(key, value)
		}

		if reqBody != nil {
			req.Header.Set("Content-Type", "application/json")
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = errors.Wrap(err, "request failed")
			if attempt < c.maxRetries {
				// Exponential backoff
				time.Sleep(time.Duration(1<<attempt) * time.Second)
				continue
			}
			return lastErr
		}
		defer resp.Body.Close()

		// Handle HTTP errors
		if resp.StatusCode >= 400 {
			var errorResp map[string]interface{}
			json.NewDecoder(resp.Body).Decode(&errorResp)
			
			message := "unknown error"
			if msg, ok := errorResp["message"].(string); ok {
				message = msg
			}

			switch resp.StatusCode {
			case 401:
				return &types.AuthenticationError{Message: message}
			case 402:
				return &types.InsufficientFundsError{Message: message}
			case 429:
				return &types.RateLimitError{Message: message}
			default:
				return &types.PRSMError{
					Message:   fmt.Sprintf("API error %d: %s", resp.StatusCode, message),
					ErrorCode: "API_ERROR",
				}
			}
		}

		// Parse successful response
		if respBody != nil {
			if err := json.NewDecoder(resp.Body).Decode(respBody); err != nil {
				return errors.Wrap(err, "failed to decode response")
			}
		}

		return nil
	}

	return lastErr
}