// Package tools provides MCP tool execution functionality for the PRSM Go SDK
package tools

import (
	"context"
	"time"

	"github.com/Ryno2390/PRSM/sdks/go/types"
	"github.com/pkg/errors"
)

// Executor handles MCP tool execution
type Executor struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new tools executor
func New(client HTTPClient) *Executor {
	return &Executor{
		client: client,
	}
}

// ToolSpec represents an MCP tool specification
type ToolSpec struct {
	Name             string                 `json:"name"`
	Description      string                 `json:"description"`
	Parameters       map[string]interface{} `json:"parameters"`
	CostPerExecution float64                `json:"cost_per_execution"`
	SafetyLevel      types.SafetyLevel      `json:"safety_level"`
	Provider         string                 `json:"provider"`
	Version          string                 `json:"version"`
}

// ExecutionRequest represents a tool execution request
type ExecutionRequest struct {
	ToolName    string                 `json:"tool_name"`
	Parameters  map[string]interface{} `json:"parameters"`
	Context     map[string]interface{} `json:"context,omitempty"`
	SafetyLevel types.SafetyLevel      `json:"safety_level"`
}

// ExecutionResponse represents a tool execution response
type ExecutionResponse struct {
	Result        interface{}            `json:"result"`
	ExecutionTime float64                `json:"execution_time"`
	FTNSCost      float64                `json:"ftns_cost"`
	SafetyStatus  types.SafetyLevel      `json:"safety_status"`
	Success       bool                   `json:"success"`
	Error         *string                `json:"error,omitempty"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// ToolInfo represents detailed tool information
type ToolInfo struct {
	Name             string                 `json:"name"`
	Description      string                 `json:"description"`
	Parameters       map[string]interface{} `json:"parameters"`
	CostPerExecution float64                `json:"cost_per_execution"`
	SafetyLevel      types.SafetyLevel      `json:"safety_level"`
	Provider         string                 `json:"provider"`
	Version          string                 `json:"version"`
	TotalExecutions  int                    `json:"total_executions"`
	SuccessRate      float64                `json:"success_rate"`
	AvgExecutionTime float64                `json:"avg_execution_time"`
	LastUpdated      time.Time              `json:"last_updated"`
}

// ListAvailable lists all available MCP tools
func (e *Executor) ListAvailable(ctx context.Context) ([]ToolSpec, error) {
	var response struct {
		Tools []ToolSpec `json:"tools"`
	}
	err := e.client.MakeRequest(ctx, "GET", "/api/v1/tools", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list tools")
	}
	return response.Tools, nil
}

// Execute executes an MCP tool with the given parameters
func (e *Executor) Execute(ctx context.Context, req *ExecutionRequest) (*ExecutionResponse, error) {
	if req == nil {
		return nil, errors.New("execution request cannot be nil")
	}
	if req.ToolName == "" {
		return nil, errors.New("tool name is required")
	}
	if req.Parameters == nil {
		req.Parameters = make(map[string]interface{})
	}
	if req.SafetyLevel == "" {
		req.SafetyLevel = types.SafetyLevelModerate
	}

	var response ExecutionResponse
	err := e.client.MakeRequest(ctx, "POST", "/api/v1/tools/execute", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to execute tool")
	}
	return &response, nil
}

// GetToolInfo retrieves detailed information about a specific tool
func (e *Executor) GetToolInfo(ctx context.Context, toolName string) (*ToolInfo, error) {
	if toolName == "" {
		return nil, errors.New("tool name cannot be empty")
	}

	var response ToolInfo
	err := e.client.MakeRequest(ctx, "GET", "/api/v1/tools/"+toolName, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get tool info")
	}
	return &response, nil
}

// EstimateCost estimates the FTNS cost for executing a tool
func (e *Executor) EstimateCost(ctx context.Context, toolName string, params map[string]interface{}) (float64, error) {
	if toolName == "" {
		return 0, errors.New("tool name cannot be empty")
	}

	req := map[string]interface{}{
		"tool_name":  toolName,
		"parameters": params,
	}

	var response struct {
		EstimatedCost float64 `json:"estimated_cost"`
	}
	err := e.client.MakeRequest(ctx, "POST", "/api/v1/tools/estimate-cost", req, &response)
	if err != nil {
		return 0, errors.Wrap(err, "failed to estimate tool cost")
	}
	return response.EstimatedCost, nil
}

// SearchTools searches for tools matching a query
func (e *Executor) SearchTools(ctx context.Context, query string, limit int) ([]ToolSpec, error) {
	if query == "" {
		return nil, errors.New("query cannot be empty")
	}
	if limit <= 0 {
		limit = 20
	}

	req := map[string]interface{}{
		"query": query,
		"limit": limit,
	}

	var response struct {
		Tools []ToolSpec `json:"tools"`
	}
	err := e.client.MakeRequest(ctx, "POST", "/api/v1/tools/search", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to search tools")
	}
	return response.Tools, nil
}

// GetExecutionHistory retrieves tool execution history
func (e *Executor) GetExecutionHistory(ctx context.Context, limit int, offset int) ([]ExecutionResponse, error) {
	if limit <= 0 {
		limit = 50
	}

	req := map[string]interface{}{
		"limit":  limit,
		"offset": offset,
	}

	var response struct {
		Executions []ExecutionResponse `json:"executions"`
	}
	err := e.client.MakeRequest(ctx, "GET", "/api/v1/tools/history", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get execution history")
	}
	return response.Executions, nil
}
