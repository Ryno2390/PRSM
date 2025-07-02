package types

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelProvider_Constants(t *testing.T) {
	assert.Equal(t, ModelProvider("openai"), ModelProviderOpenAI)
	assert.Equal(t, ModelProvider("anthropic"), ModelProviderAnthropic)
	assert.Equal(t, ModelProvider("huggingface"), ModelProviderHuggingFace)
	assert.Equal(t, ModelProvider("local"), ModelProviderLocal)
	assert.Equal(t, ModelProvider("prsm_distilled"), ModelProviderPRSMDistilled)
}

func TestSafetyLevel_Constants(t *testing.T) {
	assert.Equal(t, SafetyLevel("none"), SafetyLevelNone)
	assert.Equal(t, SafetyLevel("low"), SafetyLevelLow)
	assert.Equal(t, SafetyLevel("moderate"), SafetyLevelModerate)
	assert.Equal(t, SafetyLevel("high"), SafetyLevelHigh)
	assert.Equal(t, SafetyLevel("critical"), SafetyLevelCritical)
	assert.Equal(t, SafetyLevel("emergency"), SafetyLevelEmergency)
}

func TestQueryRequest_JSONSerialization(t *testing.T) {
	modelID := "test-model-123"
	systemPrompt := "You are a helpful assistant"
	
	req := QueryRequest{
		Prompt:       "Test prompt",
		ModelID:      &modelID,
		MaxTokens:    500,
		Temperature:  0.7,
		SystemPrompt: &systemPrompt,
		Context: map[string]interface{}{
			"user_id": "user123",
			"session": "session456",
		},
		Tools:       []string{"web_search", "calculator"},
		SafetyLevel: SafetyLevelModerate,
	}
	
	// Test serialization
	jsonData, err := json.Marshal(req)
	require.NoError(t, err)
	
	// Test deserialization
	var decoded QueryRequest
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, req.Prompt, decoded.Prompt)
	assert.Equal(t, req.ModelID, decoded.ModelID)
	assert.Equal(t, req.MaxTokens, decoded.MaxTokens)
	assert.Equal(t, req.Temperature, decoded.Temperature)
	assert.Equal(t, req.SystemPrompt, decoded.SystemPrompt)
	assert.Equal(t, req.Context, decoded.Context)
	assert.Equal(t, req.Tools, decoded.Tools)
	assert.Equal(t, req.SafetyLevel, decoded.SafetyLevel)
}

func TestQueryRequest_OptionalFields(t *testing.T) {
	req := QueryRequest{
		Prompt:      "Simple prompt",
		MaxTokens:   100,
		Temperature: 0.5,
		Context:     make(map[string]interface{}),
		SafetyLevel: SafetyLevelLow,
	}
	
	jsonData, err := json.Marshal(req)
	require.NoError(t, err)
	
	var decoded QueryRequest
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Nil(t, decoded.ModelID)
	assert.Nil(t, decoded.SystemPrompt)
	assert.Empty(t, decoded.Tools)
	assert.NotNil(t, decoded.Context)
}

func TestPRSMResponse_JSONSerialization(t *testing.T) {
	now := time.Now()
	
	response := PRSMResponse{
		Content:       "This is a test response",
		ModelID:       "gpt-4",
		Provider:      ModelProviderOpenAI,
		ExecutionTime: 2.5,
		TokenUsage: map[string]int{
			"prompt_tokens":     50,
			"completion_tokens": 100,
			"total_tokens":      150,
		},
		FTNSCost:       0.075,
		ReasoningTrace: []string{"Step 1", "Step 2", "Step 3"},
		SafetyStatus:   SafetyLevelNone,
		Metadata: map[string]interface{}{
			"temperature": 0.7,
			"model_version": "2024-01",
		},
		RequestID: "req-123456",
		Timestamp: now,
	}
	
	// Test serialization
	jsonData, err := json.Marshal(response)
	require.NoError(t, err)
	
	// Test deserialization
	var decoded PRSMResponse
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, response.Content, decoded.Content)
	assert.Equal(t, response.ModelID, decoded.ModelID)
	assert.Equal(t, response.Provider, decoded.Provider)
	assert.Equal(t, response.ExecutionTime, decoded.ExecutionTime)
	assert.Equal(t, response.TokenUsage, decoded.TokenUsage)
	assert.Equal(t, response.FTNSCost, decoded.FTNSCost)
	assert.Equal(t, response.ReasoningTrace, decoded.ReasoningTrace)
	assert.Equal(t, response.SafetyStatus, decoded.SafetyStatus)
	assert.Equal(t, response.Metadata, decoded.Metadata)
	assert.Equal(t, response.RequestID, decoded.RequestID)
	
	// Time comparison with tolerance for JSON serialization/deserialization
	assert.WithinDuration(t, response.Timestamp, decoded.Timestamp, time.Second)
}

func TestFTNSBalance_JSONSerialization(t *testing.T) {
	now := time.Now()
	
	balance := FTNSBalance{
		TotalBalance:     1000.50,
		AvailableBalance: 850.25,
		ReservedBalance:  150.25,
		EarnedToday:      25.75,
		SpentToday:       15.50,
		LastUpdated:      now,
	}
	
	jsonData, err := json.Marshal(balance)
	require.NoError(t, err)
	
	var decoded FTNSBalance
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, balance.TotalBalance, decoded.TotalBalance)
	assert.Equal(t, balance.AvailableBalance, decoded.AvailableBalance)
	assert.Equal(t, balance.ReservedBalance, decoded.ReservedBalance)
	assert.Equal(t, balance.EarnedToday, decoded.EarnedToday)
	assert.Equal(t, balance.SpentToday, decoded.SpentToday)
	assert.WithinDuration(t, balance.LastUpdated, decoded.LastUpdated, time.Second)
}

func TestModelInfo_JSONSerialization(t *testing.T) {
	now := time.Now()
	
	model := ModelInfo{
		ID:          "model-123",
		Name:        "Advanced Language Model",
		Provider:    ModelProviderAnthropic,
		Description: "A powerful language model for complex tasks",
		Capabilities: []string{
			"text-generation",
			"reasoning",
			"code-generation",
		},
		CostPerToken:      0.00005,
		MaxTokens:         4096,
		ContextWindow:     100000,
		IsAvailable:       true,
		PerformanceRating: 4.8,
		SafetyRating:      4.9,
		CreatedAt:         now,
	}
	
	jsonData, err := json.Marshal(model)
	require.NoError(t, err)
	
	var decoded ModelInfo
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, model.ID, decoded.ID)
	assert.Equal(t, model.Name, decoded.Name)
	assert.Equal(t, model.Provider, decoded.Provider)
	assert.Equal(t, model.Description, decoded.Description)
	assert.Equal(t, model.Capabilities, decoded.Capabilities)
	assert.Equal(t, model.CostPerToken, decoded.CostPerToken)
	assert.Equal(t, model.MaxTokens, decoded.MaxTokens)
	assert.Equal(t, model.ContextWindow, decoded.ContextWindow)
	assert.Equal(t, model.IsAvailable, decoded.IsAvailable)
	assert.Equal(t, model.PerformanceRating, decoded.PerformanceRating)
	assert.Equal(t, model.SafetyRating, decoded.SafetyRating)
	assert.WithinDuration(t, model.CreatedAt, decoded.CreatedAt, time.Second)
}

func TestToolSpec_JSONSerialization(t *testing.T) {
	tool := ToolSpec{
		Name:        "web_search",
		Description: "Search the web for information",
		Parameters: map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query",
				"required":    true,
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of results",
				"default":     10,
			},
		},
		CostPerExecution: 0.01,
		SafetyLevel:      SafetyLevelModerate,
		Provider:         "search_provider",
		Version:          "1.2.0",
	}
	
	jsonData, err := json.Marshal(tool)
	require.NoError(t, err)
	
	var decoded ToolSpec
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, tool.Name, decoded.Name)
	assert.Equal(t, tool.Description, decoded.Description)
	assert.Equal(t, tool.Parameters, decoded.Parameters)
	assert.Equal(t, tool.CostPerExecution, decoded.CostPerExecution)
	assert.Equal(t, tool.SafetyLevel, decoded.SafetyLevel)
	assert.Equal(t, tool.Provider, decoded.Provider)
	assert.Equal(t, tool.Version, decoded.Version)
}

func TestSafetyStatus_JSONSerialization(t *testing.T) {
	now := time.Now()
	
	status := SafetyStatus{
		OverallStatus:            SafetyLevelHigh,
		ActiveMonitors:           8,
		ThreatsDetected:          3,
		CircuitBreakersTriggered: 1,
		LastAssessment:           now,
		NetworkHealth:            0.85,
	}
	
	jsonData, err := json.Marshal(status)
	require.NoError(t, err)
	
	var decoded SafetyStatus
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, status.OverallStatus, decoded.OverallStatus)
	assert.Equal(t, status.ActiveMonitors, decoded.ActiveMonitors)
	assert.Equal(t, status.ThreatsDetected, decoded.ThreatsDetected)
	assert.Equal(t, status.CircuitBreakersTriggered, decoded.CircuitBreakersTriggered)
	assert.WithinDuration(t, status.LastAssessment, decoded.LastAssessment, time.Second)
	assert.Equal(t, status.NetworkHealth, decoded.NetworkHealth)
}

func TestMarketplaceQuery_JSONSerialization(t *testing.T) {
	provider := ModelProviderOpenAI
	maxCost := 0.001
	minPerformance := 4.0
	
	query := MarketplaceQuery{
		Query:          "language model",
		Provider:       &provider,
		MaxCost:        &maxCost,
		MinPerformance: &minPerformance,
		Capabilities:   []string{"text-generation", "reasoning"},
		Limit:          10,
	}
	
	jsonData, err := json.Marshal(query)
	require.NoError(t, err)
	
	var decoded MarketplaceQuery
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, query.Query, decoded.Query)
	assert.Equal(t, query.Provider, decoded.Provider)
	assert.Equal(t, query.MaxCost, decoded.MaxCost)
	assert.Equal(t, query.MinPerformance, decoded.MinPerformance)
	assert.Equal(t, query.Capabilities, decoded.Capabilities)
	assert.Equal(t, query.Limit, decoded.Limit)
}

func TestToolExecutionRequest_JSONSerialization(t *testing.T) {
	req := ToolExecutionRequest{
		ToolName: "calculator",
		Parameters: map[string]interface{}{
			"expression": "2 + 2 * 3",
			"precision":  10,
		},
		Context: map[string]interface{}{
			"user_id": "user123",
		},
		SafetyLevel: SafetyLevelLow,
	}
	
	jsonData, err := json.Marshal(req)
	require.NoError(t, err)
	
	var decoded ToolExecutionRequest
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, req.ToolName, decoded.ToolName)
	assert.Equal(t, req.Parameters, decoded.Parameters)
	assert.Equal(t, req.Context, decoded.Context)
	assert.Equal(t, req.SafetyLevel, decoded.SafetyLevel)
}

func TestToolExecutionResponse_JSONSerialization(t *testing.T) {
	errorMsg := "Division by zero"
	
	response := ToolExecutionResponse{
		Result:        8.0,
		ExecutionTime: 0.15,
		FTNSCost:      0.001,
		SafetyStatus:  SafetyLevelNone,
		Success:       true,
		Error:         &errorMsg,
		Metadata: map[string]interface{}{
			"calculation_steps": []string{"2 + 2", "4 * 3", "8"},
		},
	}
	
	jsonData, err := json.Marshal(response)
	require.NoError(t, err)
	
	var decoded ToolExecutionResponse
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Equal(t, response.Result, decoded.Result)
	assert.Equal(t, response.ExecutionTime, decoded.ExecutionTime)
	assert.Equal(t, response.FTNSCost, decoded.FTNSCost)
	assert.Equal(t, response.SafetyStatus, decoded.SafetyStatus)
	assert.Equal(t, response.Success, decoded.Success)
	assert.Equal(t, response.Error, decoded.Error)
	assert.Equal(t, response.Metadata, decoded.Metadata)
}

func TestToolExecutionResponse_NoError(t *testing.T) {
	response := ToolExecutionResponse{
		Result:        "Success",
		ExecutionTime: 1.0,
		FTNSCost:      0.005,
		SafetyStatus:  SafetyLevelNone,
		Success:       true,
		Error:         nil,
		Metadata:      make(map[string]interface{}),
	}
	
	jsonData, err := json.Marshal(response)
	require.NoError(t, err)
	
	var decoded ToolExecutionResponse
	err = json.Unmarshal(jsonData, &decoded)
	require.NoError(t, err)
	
	assert.Nil(t, decoded.Error)
	assert.True(t, decoded.Success)
}