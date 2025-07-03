// Package nwtn provides NWTN (Neural Web of Thought Networks) API functionality
package nwtn

import (
	"context"
	"time"

	"github.com/PRSM-AI/prsm-go-sdk/types"
	"github.com/pkg/errors"
)

// Manager handles NWTN operations
type Manager struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new NWTN manager
func New(client HTTPClient) *Manager {
	return &Manager{
		client: client,
	}
}

// SessionInfo represents information about an NWTN session
type SessionInfo struct {
	SessionID       string                 `json:"session_id"`
	Status          SessionStatus          `json:"status"`
	Query           string                 `json:"query"`
	ModelID         *string                `json:"model_id,omitempty"`
	Domain          *string                `json:"domain,omitempty"`
	Methodology     *string                `json:"methodology,omitempty"`
	MaxIterations   int                    `json:"max_iterations"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	CompletedAt     *time.Time             `json:"completed_at,omitempty"`
	Results         *SessionResults        `json:"results,omitempty"`
	CostEstimate    *CostEstimate          `json:"cost_estimate,omitempty"`
	CostActual      *CostActual            `json:"cost_actual,omitempty"`
	Progress        float64                `json:"progress"`
	EstimatedETA    *time.Time             `json:"estimated_eta,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// SessionStatus represents the status of an NWTN session
type SessionStatus string

const (
	SessionStatusPending    SessionStatus = "pending"
	SessionStatusProcessing SessionStatus = "processing"
	SessionStatusCompleted  SessionStatus = "completed"
	SessionStatusFailed     SessionStatus = "failed"
	SessionStatusCancelled  SessionStatus = "cancelled"
)

// SessionResults represents the results of a completed NWTN session
type SessionResults struct {
	Summary     string                 `json:"summary"`
	Citations   []Citation             `json:"citations,omitempty"`
	Artifacts   map[string]interface{} `json:"artifacts,omitempty"`
	Confidence  float64                `json:"confidence"`
	Methodology string                 `json:"methodology"`
}

// Citation represents a citation in the results
type Citation struct {
	Source      string `json:"source"`
	URL         string `json:"url,omitempty"`
	Title       string `json:"title"`
	Relevance   float64 `json:"relevance"`
	Excerpt     string `json:"excerpt"`
}

// CostEstimate represents estimated costs for a session
type CostEstimate struct {
	FTNSTokens      float64 `json:"ftns_tokens"`
	ComputeTime     float64 `json:"compute_time"`
	NetworkRequests int     `json:"network_requests"`
}

// CostActual represents actual costs incurred by a session
type CostActual struct {
	FTNSTokens      float64 `json:"ftns_tokens"`
	ComputeTime     float64 `json:"compute_time"`
	NetworkRequests int     `json:"network_requests"`
}

// QueryRequest represents a request to submit a query to NWTN
type QueryRequest struct {
	Query             string                 `json:"query"`
	ModelID           *string                `json:"model_id,omitempty"`
	Domain            *string                `json:"domain,omitempty"`
	Methodology       *string                `json:"methodology,omitempty"`
	MaxIterations     int                    `json:"max_iterations"`
	MaxTokens         int                    `json:"max_tokens"`
	Temperature       float64                `json:"temperature"`
	SystemPrompt      *string                `json:"system_prompt,omitempty"`
	Context           map[string]interface{} `json:"context,omitempty"`
	Tools             []string               `json:"tools,omitempty"`
	SafetyLevel       types.SafetyLevel      `json:"safety_level"`
	IncludeCitations  bool                   `json:"include_citations"`
	SEALEnhancement   *SEALConfig            `json:"seal_enhancement,omitempty"`
	Stream            bool                   `json:"stream"`
}

// SEALConfig represents SEAL enhancement configuration
type SEALConfig struct {
	Enabled              bool    `json:"enabled"`
	AutonomousImprovement bool    `json:"autonomous_improvement"`
	TargetLearningGain   float64 `json:"target_learning_gain"`
	RestemMethodology    bool    `json:"restem_methodology"`
}

// ListSessionsOptions represents options for listing sessions
type ListSessionsOptions struct {
	Limit  int            `json:"limit,omitempty"`
	Offset int            `json:"offset,omitempty"`
	Status *SessionStatus `json:"status,omitempty"`
}

// PaginatedResponse represents a paginated response
type PaginatedResponse[T any] struct {
	Data       []T `json:"data"`
	TotalCount int `json:"total_count"`
	Limit      int `json:"limit"`
	Offset     int `json:"offset"`
	HasMore    bool `json:"has_more"`
}

// SubmitQuery submits a research query to NWTN
func (m *Manager) SubmitQuery(ctx context.Context, req *QueryRequest) (*SessionInfo, error) {
	if req.Query == "" {
		return nil, errors.New("query cannot be empty")
	}

	// Set defaults
	if req.MaxIterations == 0 {
		req.MaxIterations = 3
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = 1000
	}
	if req.Temperature == 0 {
		req.Temperature = 0.7
	}
	if req.SafetyLevel == "" {
		req.SafetyLevel = types.SafetyLevelModerate
	}

	var response SessionInfo
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/nwtn/query", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to submit NWTN query")
	}

	return &response, nil
}

// GetSession retrieves session status and details
func (m *Manager) GetSession(ctx context.Context, sessionID string) (*SessionInfo, error) {
	if sessionID == "" {
		return nil, errors.New("session ID cannot be empty")
	}

	var response SessionInfo
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/nwtn/sessions/"+sessionID, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get session details")
	}

	return &response, nil
}

// ListSessions lists user's sessions with filtering
func (m *Manager) ListSessions(ctx context.Context, options *ListSessionsOptions) (*PaginatedResponse[SessionInfo], error) {
	endpoint := "/api/v1/nwtn/sessions"
	
	var response PaginatedResponse[SessionInfo]
	err := m.client.MakeRequest(ctx, "GET", endpoint, options, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list sessions")
	}

	return &response, nil
}

// CancelSession cancels a running session
func (m *Manager) CancelSession(ctx context.Context, sessionID string) error {
	if sessionID == "" {
		return errors.New("session ID cannot be empty")
	}

	err := m.client.MakeRequest(ctx, "POST", "/api/v1/nwtn/sessions/"+sessionID+"/cancel", nil, nil)
	if err != nil {
		return errors.Wrap(err, "failed to cancel session")
	}

	return nil
}

// WaitForCompletionOptions represents options for waiting for session completion
type WaitForCompletionOptions struct {
	TimeoutDuration  time.Duration
	PollInterval     time.Duration
	OnProgress       func(*SessionInfo)
}

// WaitForCompletion waits for session completion with polling
func (m *Manager) WaitForCompletion(ctx context.Context, sessionID string, options *WaitForCompletionOptions) (*SessionInfo, error) {
	if options == nil {
		options = &WaitForCompletionOptions{
			TimeoutDuration: 10 * time.Minute,
			PollInterval:    5 * time.Second,
		}
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, options.TimeoutDuration)
	defer cancel()

	ticker := time.NewTicker(options.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-timeoutCtx.Done():
			return nil, errors.New("session completion timeout")
		case <-ticker.C:
			session, err := m.GetSession(ctx, sessionID)
			if err != nil {
				return nil, errors.Wrap(err, "failed to get session status")
			}

			if options.OnProgress != nil {
				options.OnProgress(session)
			}

			switch session.Status {
			case SessionStatusCompleted:
				return session, nil
			case SessionStatusFailed:
				return nil, errors.New("session failed")
			case SessionStatusCancelled:
				return nil, errors.New("session was cancelled")
			}
		}
	}
}