// Package marketplace provides model marketplace functionality for the PRSM Go SDK
package marketplace

import (
	"context"
	"time"

	"github.com/Ryno2390/PRSM/sdks/go/types"
	"github.com/pkg/errors"
)

// Manager handles marketplace operations
type Manager struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new marketplace manager
func New(client HTTPClient) *Manager {
	return &Manager{
		client: client,
	}
}

// ModelCategory represents model categories in the marketplace
type ModelCategory string

const (
	ModelCategoryLanguage   ModelCategory = "language"
	ModelCategoryVision     ModelCategory = "vision"
	ModelCategoryAudio      ModelCategory = "audio"
	ModelCategoryMultimodal ModelCategory = "multimodal"
	ModelCategoryScientific ModelCategory = "scientific"
	ModelCategoryReasoning  ModelCategory = "reasoning"
	ModelCategoryCode       ModelCategory = "code"
)

// ModelInfo represents information about an AI model in the marketplace
type ModelInfo struct {
	ID                string          `json:"id"`
	Name              string          `json:"name"`
	Provider          types.ModelProvider `json:"provider"`
	Description       string          `json:"description"`
	Category          ModelCategory   `json:"category"`
	Capabilities      []string        `json:"capabilities"`
	CostPerToken      float64         `json:"cost_per_token"`
	MaxTokens         int             `json:"max_tokens"`
	ContextWindow     int             `json:"context_window"`
	IsAvailable       bool            `json:"is_available"`
	PerformanceRating float64         `json:"performance_rating"`
	SafetyRating      float64         `json:"safety_rating"`
	Popularity        int             `json:"popularity"`
	CreatedAt         time.Time       `json:"created_at"`
	UpdatedAt         time.Time       `json:"updated_at"`
}

// SearchRequest represents a search request for models
type SearchRequest struct {
	Query          string          `json:"query,omitempty"`
	Provider       *types.ModelProvider `json:"provider,omitempty"`
	Category       *ModelCategory  `json:"category,omitempty"`
	MaxCost        *float64        `json:"max_cost,omitempty"`
	MinPerformance *float64        `json:"min_performance,omitempty"`
	MinSafety      *float64        `json:"min_safety,omitempty"`
	Capabilities   []string        `json:"capabilities,omitempty"`
	Limit          int             `json:"limit"`
	Offset         int             `json:"offset"`
}

// SearchResult represents search results containing models
type SearchResult struct {
	Models []ModelInfo `json:"models"`
	Total  int         `json:"total"`
	Offset int         `json:"offset"`
	Limit  int         `json:"limit"`
}

// ModelRental represents model rental information
type ModelRental struct {
	ModelID       string    `json:"model_id"`
	RentalID      string    `json:"rental_id"`
	StartTime     time.Time `json:"start_time"`
	EndTime       time.Time `json:"end_time"`
	Cost          float64   `json:"cost"`
	RequestsUsed  int       `json:"requests_used"`
	RequestLimit  *int      `json:"request_limit,omitempty"`
}

// RentalRequest represents a model rental request
type RentalRequest struct {
	ModelID       string `json:"model_id"`
	DurationHours int    `json:"duration_hours"`
	MaxRequests   *int   `json:"max_requests,omitempty"`
}

// ModelStats represents model usage statistics
type ModelStats struct {
	ModelID      string    `json:"model_id"`
	TotalRequests int      `json:"total_requests"`
	TotalTokens  int       `json:"total_tokens"`
	TotalCost    float64   `json:"total_cost"`
	AvgLatency   float64   `json:"avg_latency"`
	SuccessRate  float64   `json:"success_rate"`
	LastUsed     time.Time `json:"last_used"`
}

// ListModelsRequest represents parameters for listing models
type ListModelsRequest struct {
	Category *ModelCategory     `json:"category,omitempty"`
	Provider *types.ModelProvider `json:"provider,omitempty"`
	Limit    int                `json:"limit"`
}

// SearchModels searches for models in the marketplace
func (m *Manager) SearchModels(ctx context.Context, req *SearchRequest) (*SearchResult, error) {
	if req == nil {
		req = &SearchRequest{Limit: 20}
	}
	if req.Limit <= 0 {
		req.Limit = 20
	}

	var response SearchResult
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/marketplace/search", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to search models")
	}
	return &response, nil
}

// GetModel retrieves detailed information about a specific model
func (m *Manager) GetModel(ctx context.Context, modelID string) (*ModelInfo, error) {
	if modelID == "" {
		return nil, errors.New("model ID cannot be empty")
	}

	var response ModelInfo
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/models/"+modelID, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get model info")
	}
	return &response, nil
}

// ListModels lists available models with optional filtering
func (m *Manager) ListModels(ctx context.Context, req *ListModelsRequest) ([]ModelInfo, error) {
	if req == nil {
		req = &ListModelsRequest{Limit: 20}
	}
	if req.Limit <= 0 {
		req.Limit = 20
	}

	var response struct {
		Models []ModelInfo `json:"models"`
	}
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/models", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list models")
	}
	return response.Models, nil
}

// RentModel rents a model for use
func (m *Manager) RentModel(ctx context.Context, req *RentalRequest) (*ModelRental, error) {
	if req == nil {
		return nil, errors.New("rental request cannot be nil")
	}
	if req.ModelID == "" {
		return nil, errors.New("model ID is required")
	}
	if req.DurationHours <= 0 {
		req.DurationHours = 1 // Default to 1 hour
	}

	var response ModelRental
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/marketplace/models/"+req.ModelID+"/rent", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to rent model")
	}
	return &response, nil
}

// GetRental retrieves rental information
func (m *Manager) GetRental(ctx context.Context, rentalID string) (*ModelRental, error) {
	if rentalID == "" {
		return nil, errors.New("rental ID cannot be empty")
	}

	var response ModelRental
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/rentals/"+rentalID, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get rental info")
	}
	return &response, nil
}

// ListRentals lists user's model rentals
func (m *Manager) ListRentals(ctx context.Context, activeOnly bool) ([]ModelRental, error) {
	req := map[string]interface{}{"active_only": activeOnly}

	var response struct {
		Rentals []ModelRental `json:"rentals"`
	}
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/rentals", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list rentals")
	}
	return response.Rentals, nil
}

// GetModelStats retrieves usage statistics for a model
func (m *Manager) GetModelStats(ctx context.Context, modelID string) (*ModelStats, error) {
	if modelID == "" {
		return nil, errors.New("model ID cannot be empty")
	}

	var response ModelStats
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/models/"+modelID+"/stats", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get model stats")
	}
	return &response, nil
}

// GetFeaturedModels retrieves featured/popular models
func (m *Manager) GetFeaturedModels(ctx context.Context, limit int) ([]ModelInfo, error) {
	if limit <= 0 {
		limit = 10
	}

	req := map[string]interface{}{"limit": limit}

	var response struct {
		Models []ModelInfo `json:"models"`
	}
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/marketplace/featured", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get featured models")
	}
	return response.Models, nil
}

// GetRecommendedModels retrieves recommended models for a specific task
func (m *Manager) GetRecommendedModels(ctx context.Context, task string, limit int) ([]ModelInfo, error) {
	if task == "" {
		return nil, errors.New("task cannot be empty")
	}
	if limit <= 0 {
		limit = 5
	}

	req := map[string]interface{}{"task": task, "limit": limit}

	var response struct {
		Models []ModelInfo `json:"models"`
	}
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/marketplace/recommend", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get recommended models")
	}
	return response.Models, nil
}
