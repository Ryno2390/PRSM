// Package compute provides client methods for submitting and managing compute jobs
// on the PRSM network (Ring 1-10)
package compute

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

// JobStatus represents the current state of a compute job
type JobStatus string

const (
	JobStatusPending   JobStatus = "pending"
	JobStatusQueued    JobStatus = "queued"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed    JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
	JobStatusTimeout   JobStatus = "timeout"
)

// JobPriority represents the priority level of a compute job
type JobPriority string

const (
	JobPriorityLow    JobPriority = "low"
	JobPriorityNormal JobPriority = "normal"
	JobPriorityHigh   JobPriority = "high"
	JobPriorityUrgent JobPriority = "urgent"
)

// JobRequest represents a request to submit a compute job
type JobRequest struct {
	Prompt       string                 `json:"prompt"`
	Model        string                 `json:"model,omitempty"`
	MaxTokens    int                    `json:"max_tokens,omitempty"`
	Temperature  float64                `json:"temperature,omitempty"`
	Budget       float64                `json:"budget,omitempty"`
	Priority     JobPriority            `json:"priority,omitempty"`
	Timeout      int                    `json:"timeout,omitempty"`
	Context      map[string]interface{} `json:"context,omitempty"`
	Tools        []string               `json:"tools,omitempty"`
	Stream       bool                   `json:"stream,omitempty"`
}

// JobResponse represents the response from submitting a job
type JobResponse struct {
	JobID             string    `json:"job_id"`
	Status            JobStatus `json:"status"`
	CreatedAt         time.Time `json:"created_at"`
	EstimatedCost     float64   `json:"estimated_cost"`
	EstimatedDuration int       `json:"estimated_duration"`
	QueuePosition     int       `json:"queue_position,omitempty"`
}

// JobResult represents the result of a completed job
type JobResult struct {
	JobID           string                 `json:"job_id"`
	Status          JobStatus              `json:"status"`
	Content         string                 `json:"content"`
	Model           string                 `json:"model"`
	Provider        string                 `json:"provider"`
	ExecutionTime   int                    `json:"execution_time"`
	TokenUsage      map[string]int         `json:"token_usage"`
	FTNSCost        float64                `json:"ftns_cost"`
	ReasoningTrace  []string               `json:"reasoning_trace,omitempty"`
	Citations       []Citation             `json:"citations,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	CompletedAt     time.Time              `json:"completed_at"`
}

// Citation represents a citation in a job result
type Citation struct {
	Source string `json:"source"`
	URL    string `json:"url,omitempty"`
}

// JobInfo represents detailed information about a job
type JobInfo struct {
	JobID       string      `json:"job_id"`
	Status      JobStatus   `json:"status"`
	Request     JobRequest  `json:"request"`
	Result      *JobResult  `json:"result,omitempty"`
	Progress    int         `json:"progress"`
	CreatedAt   time.Time   `json:"created_at"`
	StartedAt   *time.Time  `json:"started_at,omitempty"`
	CompletedAt *time.Time  `json:"completed_at,omitempty"`
	Error       string      `json:"error,omitempty"`
	NodeID      string      `json:"node_id,omitempty"`
}

// JobListResponse represents a paginated list of jobs
type JobListResponse struct {
	Jobs   []JobInfo `json:"jobs"`
	Total  int       `json:"total"`
	Offset int       `json:"offset"`
	Limit  int       `json:"limit"`
}

// Manager provides compute job management methods
type Manager struct {
	client HTTPClient
}

// NewManager creates a new compute manager
func NewManager(client HTTPClient) *Manager {
	return &Manager{client: client}
}

// SubmitJob submits a compute job to the PRSM network
func (m *Manager) SubmitJob(ctx context.Context, req JobRequest) (*JobResponse, error) {
	body := map[string]interface{}{
		"prompt": req.Prompt,
	}
	if req.Model != "" {
		body["model"] = req.Model
	} else {
		body["model"] = "nwtn"
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	} else {
		body["max_tokens"] = 1000
	}
	if req.Temperature > 0 {
		body["temperature"] = req.Temperature
	} else {
		body["temperature"] = 0.7
	}
	if req.Budget > 0 {
		body["budget"] = req.Budget
	}
	if req.Priority != "" {
		body["priority"] = req.Priority
	} else {
		body["priority"] = JobPriorityNormal
	}
	if req.Timeout > 0 {
		body["timeout"] = req.Timeout
	}
	if req.Context != nil {
		body["context"] = req.Context
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}
	body["stream"] = req.Stream

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/compute/jobs", body)
	if err != nil {
		return nil, fmt.Errorf("failed to submit compute job: %w", err)
	}

	var result JobResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse job response: %w", err)
	}

	return &result, nil
}

// GetJob retrieves detailed information about a job
func (m *Manager) GetJob(ctx context.Context, jobID string) (*JobInfo, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/compute/jobs/"+jobID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get job: %w", err)
	}

	var result JobInfo
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse job info: %w", err)
	}

	return &result, nil
}

// GetResult retrieves the result of a completed job
func (m *Manager) GetResult(ctx context.Context, jobID string) (*JobResult, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/compute/jobs/"+jobID+"/result", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get job result: %w", err)
	}

	var result JobResult
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse job result: %w", err)
	}

	return &result, nil
}

// CancelJob cancels a running job
func (m *Manager) CancelJob(ctx context.Context, jobID string) (bool, error) {
	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/compute/jobs/"+jobID+"/cancel", nil)
	if err != nil {
		return false, fmt.Errorf("failed to cancel job: %w", err)
	}

	var result struct {
		Cancelled bool `json:"cancelled"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return false, fmt.Errorf("failed to parse cancel response: %w", err)
	}

	return result.Cancelled, nil
}

// ListJobs lists recent jobs
func (m *Manager) ListJobs(ctx context.Context, status JobStatus, limit, offset int) (*JobListResponse, error) {
	path := fmt.Sprintf("/api/v1/compute/jobs?limit=%d&offset=%d", limit, offset)
	if status != "" {
		path += "&status=" + string(status)
	}

	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list jobs: %w", err)
	}

	var result JobListResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse job list: %w", err)
	}

	return &result, nil
}

// WaitForCompletion waits for a job to complete
func (m *Manager) WaitForCompletion(ctx context.Context, jobID string, timeout time.Duration) (*JobResult, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("timeout waiting for job %s", jobID)
		case <-ticker.C:
			job, err := m.GetJob(ctx, jobID)
			if err != nil {
				return nil, err
			}

			switch job.Status {
			case JobStatusCompleted:
				return m.GetResult(ctx, jobID)
			case JobStatusFailed, JobStatusCancelled, JobStatusTimeout:
				return nil, fmt.Errorf("job %s ended with status: %s", jobID, job.Status)
			}
		}
	}
}

// EstimateCost estimates the FTNS cost for a job
func (m *Manager) EstimateCost(ctx context.Context, prompt, model string, maxTokens int) (float64, error) {
	body := map[string]interface{}{
		"prompt": prompt,
	}
	if model != "" {
		body["model"] = model
	} else {
		body["model"] = "nwtn"
	}
	if maxTokens > 0 {
		body["max_tokens"] = maxTokens
	} else {
		body["max_tokens"] = 1000
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/compute/estimate", body)
	if err != nil {
		return 0, fmt.Errorf("failed to estimate cost: %w", err)
	}

	var result struct {
		EstimatedCost float64 `json:"estimated_cost"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("failed to parse estimate response: %w", err)
	}

	return result.EstimatedCost, nil
}

// GetQueueStatus gets the current queue status
func (m *Manager) GetQueueStatus(ctx context.Context) (map[string]interface{}, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/compute/queue/status", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get queue status: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse queue status: %w", err)
	}

	return result, nil
}

// GetAvailableModels gets available compute models
func (m *Manager) GetAvailableModels(ctx context.Context) ([]map[string]interface{}, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/compute/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get available models: %w", err)
	}

	var result struct {
		Models []map[string]interface{} `json:"models"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse models response: %w", err)
	}

	return result.Models, nil
}
