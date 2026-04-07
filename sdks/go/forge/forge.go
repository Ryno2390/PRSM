// Package forge provides client methods for the Agent Architect (AgentForge) system
// Handles task decomposition, planning, execution, and WASM agent generation
package forge

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// HTTPClient is the interface for making HTTP requests
type HTTPClient interface {
	DoRequest(ctx context.Context, method, path string, body map[string]interface{}) ([]byte, error)
}

// TaskType represents the type of task to decompose
type TaskType string

const (
	TaskTypeAnalysis    TaskType = "analysis"
	TaskTypeGeneration   TaskType = "generation"
	TaskTypeReasoning    TaskType = "reasoning"
	TaskTypeSynthesis    TaskType = "synthesis"
	TaskTypeVerification TaskType = "verification"
)

// TaskStatus represents the status of a forge task
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "pending"
	TaskStatusDecomposed TaskStatus = "decomposed"
	TaskStatusPlanning    TaskStatus = "planning"
	TaskStatusExecuting   TaskStatus = "executing"
	TaskStatusCompleted   TaskStatus = "completed"
	TaskStatusFailed      TaskStatus = "failed"
)

// ExecutionRoute represents the execution route for a subtask
type ExecutionRoute struct {
	ModelID        string   `json:"model_id,omitempty"`
	ToolsRequired  []string `json:"tools_required,omitempty"`
	WASMModule      string   `json:"wasm_module,omitempty"`
	TargetRing     int      `json:"target_ring,omitempty"`
	EstimatedFTNS  float64  `json:"estimated_ftns,omitempty"`
}

// SubTask represents a decomposed subtask
type SubTask struct {
	ID               string          `json:"id"`
	Type             TaskType        `json:"type"`
	Description      string          `json:"description"`
	Priority         int             `json:"priority"`
	Dependencies     []string        `json:"dependencies,omitempty"`
	ExecutionRoute   ExecutionRoute  `json:"execution_route"`
	Status           TaskStatus      `json:"status"`
	CreatedAt        time.Time       `json:"created_at"`
	CompletedAt      *time.Time      `json:"completed_at,omitempty"`
	Result           string          `json:"result,omitempty"`
}

// ForgeTask represents a task submitted to the forge
type ForgeTask struct {
	ID              string     `json:"id"`
	OriginalPrompt  string     `json:"original_prompt"`
	TaskType        TaskType   `json:"task_type"`
	Status          TaskStatus `json:"status"`
	SubTasks        []SubTask  `json:"subtasks,omitempty"`
	TotalCost       float64    `json:"total_cost,omitempty"`
	ConfidenceScore float64    `json:"confidence_score,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
	FinalResult     string     `json:"final_result,omitempty"`
}

// DecomposeRequest represents a request to decompose a task
type DecomposeRequest struct {
	Prompt      string            `json:"prompt"`
	TaskType    TaskType          `json:"task_type,omitempty"`
	MaxSubTasks int               `json:"max_subtasks,omitempty"`
	Budget      float64           `json:"budget,omitempty"`
	Context     map[string]string `json:"context,omitempty"`
}

// DecomposeResponse represents the response from decomposing a task
type DecomposeResponse struct {
	TaskID        string     `json:"task_id"`
	SubTasks      []SubTask  `json:"subtasks"`
	EstimatedCost float64    `json:"estimated_cost"`
	DecomposedAt  time.Time  `json:"decomposed_at"`
}

// PlanRequest represents a request to create an execution plan
type PlanRequest struct {
	TaskID       string `json:"task_id"`
	Strategy    string `json:"strategy,omitempty"`
	Parallelize bool   `json:"parallelize,omitempty"`
}

// PlanResponse represents the response from planning
type PlanResponse struct {
	TaskID    string            `json:"task_id"`
	Plan      []ExecutionRoute `json:"plan"`
	Status    TaskStatus       `json:"status"`
	PlannedAt time.Time        `json:"planned_at"`
}

// ExecuteRequest represents a request to execute a plan
type ExecuteRequest struct {
	TaskID string `json:"task_id"`
	Async  bool   `json:"async,omitempty"`
}

// ExecuteResponse represents the response from execution
type ExecuteResponse struct {
	TaskID    string     `json:"task_id"`
	Status    TaskStatus `json:"status"`
	Result    string     `json:"result,omitempty"`
	FTNSCharged float64  `json:"ftns_charged,omitempty"`
}

// QuoteRequest represents a request for a cost quote
type QuoteRequest struct {
	Prompt   string   `json:"prompt"`
	TaskType TaskType `json:"task_type,omitempty"`
}

// QuoteResponse represents the cost quote
type QuoteResponse struct {
	EstimatedCost float64 `json:"estimated_cost"`
	EstimatedTime int     `json:"estimated_time_seconds"`
	RingsUsed     []int   `json:"rings_used"`
}

// WASMManifest represents a WASM agent manifest
type WASMManifest struct {
	ModuleID   string            `json:"module_id"`
	Version    string            `json:"version"`
	Entrypoint string            `json:"entrypoint"`
	Memory     int               `json:"memory_mb"`
	Capabilities []string        `json:"capabilities"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Manager provides forge management methods
type Manager struct {
	client HTTPClient
}

// NewManager creates a new forge manager
func NewManager(client HTTPClient) *Manager {
	return &Manager{client: client}
}

// Decompose decomposes a complex task into subtasks
func (m *Manager) Decompose(ctx context.Context, req DecomposeRequest) (*DecomposeResponse, error) {
	body := map[string]interface{}{
		"prompt": req.Prompt,
	}
	if req.TaskType != "" {
		body["task_type"] = req.TaskType
	} else {
		body["task_type"] = TaskTypeAnalysis
	}
	if req.MaxSubTasks > 0 {
		body["max_subtasks"] = req.MaxSubTasks
	} else {
		body["max_subtasks"] = 5
	}
	if req.Budget > 0 {
		body["budget"] = req.Budget
	}
	if req.Context != nil {
		body["context"] = req.Context
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/forge/decompose", body)
	if err != nil {
		return nil, fmt.Errorf("failed to decompose task: %w", err)
	}

	var result DecomposeResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse decompose response: %w", err)
	}

	return &result, nil
}

// Plan creates an execution plan for decomposed tasks
func (m *Manager) Plan(ctx context.Context, req PlanRequest) (*PlanResponse, error) {
	body := map[string]interface{}{
		"task_id": req.TaskID,
	}
	if req.Strategy != "" {
		body["strategy"] = req.Strategy
	} else {
		body["strategy"] = "cost_optimized"
	}
	body["parallelize"] = req.Parallelize

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/forge/plan", body)
	if err != nil {
		return nil, fmt.Errorf("failed to create plan: %w", err)
	}

	var result PlanResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse plan response: %w", err)
	}

	return &result, nil
}

// Execute executes a planned task
func (m *Manager) Execute(ctx context.Context, req ExecuteRequest) (*ExecuteResponse, error) {
	body := map[string]interface{}{
		"task_id": req.TaskID,
	}
	body["async"] = req.Async

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/forge/execute", body)
	if err != nil {
		return nil, fmt.Errorf("failed to execute task: %w", err)
	}

	var result ExecuteResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse execute response: %w", err)
	}

	return &result, nil
}

// GetQuote gets a cost estimate for a task
func (m *Manager) GetQuote(ctx context.Context, req QuoteRequest) (*QuoteResponse, error) {
	body := map[string]interface{}{
		"prompt": req.Prompt,
	}
	if req.TaskType != "" {
		body["task_type"] = req.TaskType
	} else {
		body["task_type"] = TaskTypeAnalysis
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/forge/quote", body)
	if err != nil {
		return nil, fmt.Errorf("failed to get quote: %w", err)
	}

	var result QuoteResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse quote response: %w", err)
	}

	return &result, nil
}

// GetTask retrieves a forge task by ID
func (m *Manager) GetTask(ctx context.Context, taskID string) (*ForgeTask, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/forge/tasks/"+taskID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get task: %w", err)
	}

	var result ForgeTask
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse task response: %w", err)
	}

	return &result, nil
}

// GenerateWASM generates a WASM agent manifest for a task
func (m *Manager) GenerateWASM(ctx context.Context, taskID string) (*WASMManifest, error) {
	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/forge/tasks/"+taskID+"/wasm", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate WASM: %w", err)
	}

	var result WASMManifest
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse WASM manifest: %w", err)
	}

	return &result, nil
}

// ListTasks lists recent forge tasks
func (m *Manager) ListTasks(ctx context.Context, limit, offset int) ([]ForgeTask, error) {
	path := fmt.Sprintf("/api/v1/forge/tasks?limit=%d&offset=%d", limit, offset)
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list tasks: %w", err)
	}

	var result struct {
		Tasks []ForgeTask `json:"tasks"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse tasks response: %w", err)
	}

	return result.Tasks, nil
}
