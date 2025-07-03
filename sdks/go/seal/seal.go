// Package seal provides SEAL (Self-Adapting Language model) Technology API functionality
package seal

import (
	"context"

	"github.com/pkg/errors"
)

// Manager handles SEAL operations
type Manager struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new SEAL manager
func New(client HTTPClient) *Manager {
	return &Manager{
		client: client,
	}
}

// Metrics represents SEAL performance metrics
type Metrics struct {
	SEALSystemStatus   string             `json:"seal_system_status"`
	ProductionMetrics  ProductionMetrics  `json:"production_metrics"`
	RealTimePerformance RealTimePerformance `json:"real_time_performance"`
}

// ProductionMetrics represents production-level SEAL metrics
type ProductionMetrics struct {
	KnowledgeIncorporationBaseline         float64 `json:"knowledge_incorporation_baseline"`
	KnowledgeIncorporationCurrent          float64 `json:"knowledge_incorporation_current"`
	ImprovementPercentage                  float64 `json:"improvement_percentage"`
	FewShotLearningSuccessRate             float64 `json:"few_shot_learning_success_rate"`
	SelfEditGenerationRate                 float64 `json:"self_edit_generation_rate"`
	AutonomousImprovementCyclesCompleted   int     `json:"autonomous_improvement_cycles_completed"`
}

// RealTimePerformance represents real-time SEAL performance metrics
type RealTimePerformance struct {
	RestemPolicyUpdatesPerSecond     float64 `json:"restem_policy_updates_per_second"`
	SEALRewardCalculationsPerSecond  float64 `json:"seal_reward_calculations_per_second"`
	AutonomousImprovementRate        float64 `json:"autonomous_improvement_rate"`
}

// ImprovementConfig represents configuration for triggering SEAL improvement
type ImprovementConfig struct {
	Domain               string  `json:"domain"`
	TargetImprovement    float64 `json:"target_improvement"`
	ImprovementStrategy  string  `json:"improvement_strategy"`
	MaxIterations        int     `json:"max_iterations"`
}

// ImprovementResponse represents the response from triggering SEAL improvement
type ImprovementResponse struct {
	ImprovementID string `json:"improvement_id"`
	Status        string `json:"status"`
}

// SessionStatus represents SEAL enhancement status for a session
type SessionStatus struct {
	EnhancementEnabled           bool    `json:"enhancement_enabled"`
	AutonomousImprovementActive  bool    `json:"autonomous_improvement_active"`
	EstimatedLearningGain        float64 `json:"estimated_learning_gain"`
	SelfEditGenerationRate       float64 `json:"self_edit_generation_rate"`
}

// GetMetrics retrieves SEAL performance metrics
func (m *Manager) GetMetrics(ctx context.Context) (*Metrics, error) {
	var response Metrics
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/seal/metrics", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get SEAL metrics")
	}

	return &response, nil
}

// TriggerImprovement triggers SEAL autonomous improvement
func (m *Manager) TriggerImprovement(ctx context.Context, config *ImprovementConfig) (*ImprovementResponse, error) {
	if config == nil {
		return nil, errors.New("improvement config cannot be nil")
	}

	if config.Domain == "" {
		return nil, errors.New("domain cannot be empty")
	}

	if config.MaxIterations == 0 {
		config.MaxIterations = 5
	}

	var response ImprovementResponse
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/seal/improve", config, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to trigger SEAL improvement")
	}

	return &response, nil
}

// GetSessionStatus retrieves SEAL enhancement status for a session
func (m *Manager) GetSessionStatus(ctx context.Context, sessionID string) (*SessionStatus, error) {
	if sessionID == "" {
		return nil, errors.New("session ID cannot be empty")
	}

	var response SessionStatus
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/seal/sessions/"+sessionID+"/status", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get SEAL session status")
	}

	return &response, nil
}