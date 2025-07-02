package client

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/PRSM-AI/prsm-go-sdk/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	client := New("test-api-key")
	
	assert.NotNil(t, client)
	assert.Equal(t, "test-api-key", client.Auth.GetAPIKey())
	assert.Equal(t, "https://api.prsm.ai/v1", client.baseURL)
	assert.Equal(t, 60*time.Second, client.timeout)
	assert.Equal(t, 3, client.maxRetries)
}

func TestNewWithConfig(t *testing.T) {
	config := &Config{
		APIKey:       "test-key",
		BaseURL:      "https://custom.api.com",
		WebSocketURL: "wss://custom.ws.com",
		Timeout:      30 * time.Second,
		MaxRetries:   5,
		RateLimit:    20,
	}
	
	client := NewWithConfig(config)
	
	assert.NotNil(t, client)
	assert.Equal(t, "https://custom.api.com", client.baseURL)
	assert.Equal(t, "wss://custom.ws.com", client.websocketURL)
	assert.Equal(t, 30*time.Second, client.timeout)
	assert.Equal(t, 5, client.maxRetries)
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()
	
	assert.Equal(t, "https://api.prsm.ai/v1", config.BaseURL)
	assert.Equal(t, "wss://ws.prsm.ai/v1", config.WebSocketURL)
	assert.Equal(t, 60*time.Second, config.Timeout)
	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 10.0, float64(config.RateLimit))
}

func TestQuery_Success(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/nwtn/query", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		
		// Mock response
		response := types.PRSMResponse{
			Content:       "Test response content",
			ModelID:       "test-model-123",
			Provider:      types.ModelProviderOpenAI,
			ExecutionTime: 1.23,
			FTNSCost:      0.05,
			SafetyStatus:  types.SafetyLevelNone,
			RequestID:     "req-123",
			Timestamp:     time.Now(),
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	// Create client with test server
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	// Execute query
	req := &types.QueryRequest{
		Prompt:      "Test prompt",
		MaxTokens:   100,
		Temperature: 0.7,
		SafetyLevel: types.SafetyLevelModerate,
	}
	
	response, err := client.Query(context.Background(), req)
	
	require.NoError(t, err)
	assert.Equal(t, "Test response content", response.Content)
	assert.Equal(t, "test-model-123", response.ModelID)
	assert.Equal(t, types.ModelProviderOpenAI, response.Provider)
	assert.Equal(t, 1.23, response.ExecutionTime)
	assert.Equal(t, 0.05, response.FTNSCost)
}

func TestQuery_EmptyPrompt(t *testing.T) {
	client := New("test-key")
	
	req := &types.QueryRequest{
		Prompt: "",
	}
	
	_, err := client.Query(context.Background(), req)
	
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "prompt cannot be empty")
}

func TestQuery_AuthenticationError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Invalid API key",
		})
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "invalid-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	req := &types.QueryRequest{
		Prompt: "Test prompt",
	}
	
	_, err := client.Query(context.Background(), req)
	
	require.Error(t, err)
	assert.IsType(t, &types.AuthenticationError{}, err)
}

func TestQuery_InsufficientFunds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Insufficient FTNS balance",
		})
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	req := &types.QueryRequest{
		Prompt: "Test prompt",
	}
	
	_, err := client.Query(context.Background(), req)
	
	require.Error(t, err)
	assert.IsType(t, &types.InsufficientFundsError{}, err)
}

func TestQuery_RateLimitError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Rate limit exceeded",
		})
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	req := &types.QueryRequest{
		Prompt: "Test prompt",
	}
	
	_, err := client.Query(context.Background(), req)
	
	require.Error(t, err)
	assert.IsType(t, &types.RateLimitError{}, err)
}

func TestEstimateCost_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/nwtn/estimate-cost", r.URL.Path)
		
		response := map[string]float64{
			"estimated_cost": 0.123,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	cost, err := client.EstimateCost(context.Background(), "Test prompt", nil)
	
	require.NoError(t, err)
	assert.Equal(t, 0.123, cost)
}

func TestEstimateCost_WithModelID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)
		
		assert.Equal(t, "Test prompt", reqBody["prompt"])
		assert.Equal(t, "specific-model", reqBody["model_id"])
		
		response := map[string]float64{
			"estimated_cost": 0.456,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	modelID := "specific-model"
	cost, err := client.EstimateCost(context.Background(), "Test prompt", &modelID)
	
	require.NoError(t, err)
	assert.Equal(t, 0.456, cost)
}

func TestGetSafetyStatus_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/safety/status", r.URL.Path)
		
		status := types.SafetyStatus{
			OverallStatus:            types.SafetyLevelLow,
			ActiveMonitors:           5,
			ThreatsDetected:          2,
			CircuitBreakersTriggered: 0,
			LastAssessment:           time.Now(),
			NetworkHealth:            0.95,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	status, err := client.GetSafetyStatus(context.Background())
	
	require.NoError(t, err)
	assert.Equal(t, types.SafetyLevelLow, status.OverallStatus)
	assert.Equal(t, 5, status.ActiveMonitors)
	assert.Equal(t, 2, status.ThreatsDetected)
	assert.Equal(t, 0, status.CircuitBreakersTriggered)
	assert.Equal(t, 0.95, status.NetworkHealth)
}

func TestListAvailableModels_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/models", r.URL.Path)
		
		models := []*types.ModelInfo{
			{
				ID:                "model-1",
				Name:              "GPT-4",
				Provider:          types.ModelProviderOpenAI,
				Description:       "Advanced language model",
				Capabilities:      []string{"text-generation", "reasoning"},
				CostPerToken:      0.00003,
				MaxTokens:         4096,
				ContextWindow:     8192,
				IsAvailable:       true,
				PerformanceRating: 4.8,
				SafetyRating:      4.5,
				CreatedAt:         time.Now(),
			},
		}
		
		response := map[string][]*types.ModelInfo{
			"models": models,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	models, err := client.ListAvailableModels(context.Background())
	
	require.NoError(t, err)
	require.Len(t, models, 1)
	assert.Equal(t, "model-1", models[0].ID)
	assert.Equal(t, "GPT-4", models[0].Name)
	assert.Equal(t, types.ModelProviderOpenAI, models[0].Provider)
	assert.True(t, models[0].IsAvailable)
}

func TestHealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/health", r.URL.Path)
		
		health := map[string]interface{}{
			"status":        "healthy",
			"response_time": 45,
			"active_models": 12,
			"system_load":   0.65,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(health)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	client := NewWithConfig(config)
	
	health, err := client.HealthCheck(context.Background())
	
	require.NoError(t, err)
	assert.Equal(t, "healthy", health["status"])
	assert.Equal(t, float64(45), health["response_time"])
	assert.Equal(t, float64(12), health["active_models"])
	assert.Equal(t, 0.65, health["system_load"])
}

func TestMakeRequest_RetryLogic(t *testing.T) {
	attemptCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount < 3 {
			// Simulate temporary failure
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// Success on third attempt
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "success"})
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	config.MaxRetries = 3
	client := NewWithConfig(config)
	
	var response map[string]string
	err := client.makeRequest(context.Background(), "GET", "/test", nil, &response)
	
	require.NoError(t, err)
	assert.Equal(t, "success", response["status"])
	assert.Equal(t, 3, attemptCount)
}

func TestMakeRequest_ExceedsMaxRetries(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()
	
	config := DefaultConfig()
	config.APIKey = "test-key"
	config.BaseURL = server.URL
	config.MaxRetries = 2
	client := NewWithConfig(config)
	
	var response map[string]string
	err := client.makeRequest(context.Background(), "GET", "/test", nil, &response)
	
	require.Error(t, err)
	assert.Contains(t, err.Error(), "request failed")
}