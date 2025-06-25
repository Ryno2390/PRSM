// Package types defines the data structures used by the PRSM Go SDK
package types

import "time"

// ModelProvider represents the AI model providers supported by PRSM
type ModelProvider string

const (
	ModelProviderOpenAI        ModelProvider = "openai"
	ModelProviderAnthropic     ModelProvider = "anthropic"
	ModelProviderHuggingFace   ModelProvider = "huggingface"
	ModelProviderLocal         ModelProvider = "local"
	ModelProviderPRSMDistilled ModelProvider = "prsm_distilled"
)

// SafetyLevel represents the safety monitoring levels
type SafetyLevel string

const (
	SafetyLevelNone      SafetyLevel = "none"
	SafetyLevelLow       SafetyLevel = "low"
	SafetyLevelModerate  SafetyLevel = "moderate"
	SafetyLevelHigh      SafetyLevel = "high"
	SafetyLevelCritical  SafetyLevel = "critical"
	SafetyLevelEmergency SafetyLevel = "emergency"
)

// QueryRequest represents a request for AI query execution
type QueryRequest struct {
	Prompt       string                 `json:"prompt"`
	ModelID      *string                `json:"model_id,omitempty"`
	MaxTokens    int                    `json:"max_tokens"`
	Temperature  float64                `json:"temperature"`
	SystemPrompt *string                `json:"system_prompt,omitempty"`
	Context      map[string]interface{} `json:"context"`
	Tools        []string               `json:"tools,omitempty"`
	SafetyLevel  SafetyLevel            `json:"safety_level"`
}

// PRSMResponse represents a response from PRSM AI query
type PRSMResponse struct {
	Content        string                 `json:"content"`
	ModelID        string                 `json:"model_id"`
	Provider       ModelProvider          `json:"provider"`
	ExecutionTime  float64                `json:"execution_time"`
	TokenUsage     map[string]int         `json:"token_usage"`
	FTNSCost       float64                `json:"ftns_cost"`
	ReasoningTrace []string               `json:"reasoning_trace,omitempty"`
	SafetyStatus   SafetyLevel            `json:"safety_status"`
	Metadata       map[string]interface{} `json:"metadata"`
	RequestID      string                 `json:"request_id"`
	Timestamp      time.Time              `json:"timestamp"`
}

// FTNSBalance represents FTNS token balance information
type FTNSBalance struct {
	TotalBalance     float64   `json:"total_balance"`
	AvailableBalance float64   `json:"available_balance"`
	ReservedBalance  float64   `json:"reserved_balance"`
	EarnedToday      float64   `json:"earned_today"`
	SpentToday       float64   `json:"spent_today"`
	LastUpdated      time.Time `json:"last_updated"`
}

// ModelInfo represents information about available AI models
type ModelInfo struct {
	ID                string        `json:"id"`
	Name              string        `json:"name"`
	Provider          ModelProvider `json:"provider"`
	Description       string        `json:"description"`
	Capabilities      []string      `json:"capabilities"`
	CostPerToken      float64       `json:"cost_per_token"`
	MaxTokens         int           `json:"max_tokens"`
	ContextWindow     int           `json:"context_window"`
	IsAvailable       bool          `json:"is_available"`
	PerformanceRating float64       `json:"performance_rating"`
	SafetyRating      float64       `json:"safety_rating"`
	CreatedAt         time.Time     `json:"created_at"`
}

// ToolSpec represents an MCP tool specification
type ToolSpec struct {
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	Parameters        map[string]interface{} `json:"parameters"`
	CostPerExecution  float64                `json:"cost_per_execution"`
	SafetyLevel       SafetyLevel            `json:"safety_level"`
	Provider          string                 `json:"provider"`
	Version           string                 `json:"version"`
}

// SafetyStatus represents current safety monitoring status
type SafetyStatus struct {
	OverallStatus            SafetyLevel `json:"overall_status"`
	ActiveMonitors           int         `json:"active_monitors"`
	ThreatsDetected          int         `json:"threats_detected"`
	CircuitBreakersTriggered int         `json:"circuit_breakers_triggered"`
	LastAssessment           time.Time   `json:"last_assessment"`
	NetworkHealth            float64     `json:"network_health"`
}

// MarketplaceQuery represents a query for marketplace model search
type MarketplaceQuery struct {
	Query          string         `json:"query"`
	Provider       *ModelProvider `json:"provider,omitempty"`
	MaxCost        *float64       `json:"max_cost,omitempty"`
	MinPerformance *float64       `json:"min_performance,omitempty"`
	Capabilities   []string       `json:"capabilities,omitempty"`
	Limit          int            `json:"limit"`
}

// ToolExecutionRequest represents a request for MCP tool execution
type ToolExecutionRequest struct {
	ToolName    string                 `json:"tool_name"`
	Parameters  map[string]interface{} `json:"parameters"`
	Context     map[string]interface{} `json:"context,omitempty"`
	SafetyLevel SafetyLevel            `json:"safety_level"`
}

// ToolExecutionResponse represents a response from MCP tool execution
type ToolExecutionResponse struct {
	Result        interface{}            `json:"result"`
	ExecutionTime float64                `json:"execution_time"`
	FTNSCost      float64                `json:"ftns_cost"`
	SafetyStatus  SafetyLevel            `json:"safety_status"`
	Success       bool                   `json:"success"`
	Error         *string                `json:"error,omitempty"`
	Metadata      map[string]interface{} `json:"metadata"`
}