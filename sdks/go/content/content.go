// Package content provides client methods for the Content Economy module
// Handles dataset storage, search, and marketplace access
package content

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

// ContentStatus represents the status of content
type ContentStatus string

const (
	ContentStatusPending   ContentStatus = "pending"
	ContentStatusUploaded  ContentStatus = "uploaded"
	ContentStatusVerified  ContentStatus = "verified"
	ContentStatusPublished ContentStatus = "published"
	ContentStatusRetired   ContentStatus = "retired"
)

// ContentType represents the type of content
type ContentType string

const (
	ContentTypeDataset   ContentType = "dataset"
	ContentTypeModel     ContentType = "model"
	ContentTypeTool      ContentType = "tool"
	ContentTypeWASM      ContentType = "wasm"
	ContentTypeDocument  ContentType = "document"
)

// Dataset represents a stored dataset
type Dataset struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Type        ContentType   `json:"type"`
	Status      ContentStatus `json:"status"`
	Size        int64         `json:"size_bytes"`
	Hash        string        `json:"content_hash"`
	Owner       string        `json:"owner_address"`
	AccessCount int           `json:"access_count"`
	Price       float64       `json:"price_ftns"`
	Tags        []string      `json:"tags,omitempty"`
	CreatedAt   time.Time     `json:"created_at"`
	UpdatedAt   time.Time     `json:"updated_at"`
}

// UploadRequest represents a request to upload content
type UploadRequest struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Type        ContentType `json:"type"`
	ContentHash string      `json:"content_hash"`
	Size        int64       `json:"size_bytes"`
	Price       float64     `json:"price_ftns,omitempty"`
	Tags        []string    `json:"tags,omitempty"`
}

// UploadResponse represents the response from uploading content
type UploadResponse struct {
	ContentID   string        `json:"content_id"`
	Name        string        `json:"name"`
	Status      ContentStatus `json:"status"`
	UploadURL   string        `json:"upload_url,omitempty"`
	CreatedAt   time.Time     `json:"created_at"`
}

// SearchRequest represents a content search request
type SearchRequest struct {
	Query    string      `json:"query"`
	Type     ContentType `json:"type,omitempty"`
	Tags     []string    `json:"tags,omitempty"`
	MaxPrice float64     `json:"max_price,omitempty"`
	MinSize  int64       `json:"min_size,omitempty"`
	MaxSize  int64       `json:"max_size,omitempty"`
	Limit    int         `json:"limit,omitempty"`
	Offset   int         `json:"offset,omitempty"`
}

// SearchResult represents a search result
type SearchResult struct {
	Datasets   []Dataset `json:"datasets"`
	Total      int       `json:"total"`
	Query      string    `json:"query"`
	SearchedAt time.Time `json:"searched_at"`
}

// AccessRequest represents a request to access content
type AccessRequest struct {
	ContentID string `json:"content_id"`
	Duration  int    `json:"duration_hours,omitempty"`
}

// AccessResponse represents the response from accessing content
type AccessResponse struct {
	ContentID   string    `json:"content_id"`
	AccessURL   string    `json:"access_url"`
	ExpiresAt   time.Time `json:"expires_at"`
	FTNSCharged float64   `json:"ftns_charged"`
}

// MarketplaceListResponse represents a paginated list of marketplace items
type MarketplaceListResponse struct {
	Items    []Dataset `json:"items"`
	Total    int       `json:"total"`
	Offset   int       `json:"offset"`
	Limit    int       `json:"limit"`
}

// Manager provides content economy management methods
type Manager struct {
	client HTTPClient
}

// NewManager creates a new content manager
func NewManager(client HTTPClient) *Manager {
	return &Manager{client: client}
}

// Upload uploads new content to the network
func (m *Manager) Upload(ctx context.Context, req UploadRequest) (*UploadResponse, error) {
	body := map[string]interface{}{
		"name":         req.Name,
		"description":  req.Description,
		"type":         req.Type,
		"content_hash": req.ContentHash,
		"size_bytes":   req.Size,
	}
	if req.Price > 0 {
		body["price_ftns"] = req.Price
	}
	if len(req.Tags) > 0 {
		body["tags"] = req.Tags
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/content-economy/upload", body)
	if err != nil {
		return nil, fmt.Errorf("failed to upload content: %w", err)
	}

	var result UploadResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse upload response: %w", err)
	}

	return &result, nil
}

// GetDataset retrieves a dataset by ID
func (m *Manager) GetDataset(ctx context.Context, contentID string) (*Dataset, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/content-economy/datasets/"+contentID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get dataset: %w", err)
	}

	var result Dataset
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse dataset: %w", err)
	}

	return &result, nil
}

// Search searches for content in the marketplace
func (m *Manager) Search(ctx context.Context, req SearchRequest) (*SearchResult, error) {
	body := map[string]interface{}{
		"query": req.Query,
	}
	if req.Type != "" {
		body["type"] = req.Type
	}
	if len(req.Tags) > 0 {
		body["tags"] = req.Tags
	}
	if req.MaxPrice > 0 {
		body["max_price"] = req.MaxPrice
	}
	if req.MinSize > 0 {
		body["min_size"] = req.MinSize
	}
	if req.MaxSize > 0 {
		body["max_size"] = req.MaxSize
	}
	if req.Limit > 0 {
		body["limit"] = req.Limit
	} else {
		body["limit"] = 20
	}
	if req.Offset > 0 {
		body["offset"] = req.Offset
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/content-economy/search", body)
	if err != nil {
		return nil, fmt.Errorf("failed to search content: %w", err)
	}

	var result SearchResult
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse search results: %w", err)
	}

	return &result, nil
}

// Access requests access to content
func (m *Manager) Access(ctx context.Context, req AccessRequest) (*AccessResponse, error) {
	body := map[string]interface{}{
		"content_id": req.ContentID,
	}
	if req.Duration > 0 {
		body["duration_hours"] = req.Duration
	} else {
		body["duration_hours"] = 24
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/content-economy/access", body)
	if err != nil {
		return nil, fmt.Errorf("failed to access content: %w", err)
	}

	var result AccessResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse access response: %w", err)
	}

	return &result, nil
}

// ListDatasets lists datasets with pagination
func (m *Manager) ListDatasets(ctx context.Context, limit, offset int) (*MarketplaceListResponse, error) {
	path := fmt.Sprintf("/api/v1/content-economy/datasets?limit=%d&offset=%d", limit, offset)
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list datasets: %w", err)
	}

	var result MarketplaceListResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse dataset list: %w", err)
	}

	return &result, nil
}

// ListByOwner lists content owned by a specific address
func (m *Manager) ListByOwner(ctx context.Context, ownerAddress string, limit, offset int) ([]Dataset, error) {
	path := fmt.Sprintf("/api/v1/content-economy/owner/%s?limit=%d&offset=%d", ownerAddress, limit, offset)
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list owner content: %w", err)
	}

	var result struct {
		Items []Dataset `json:"items"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse owner content: %w", err)
	}

	return result.Items, nil
}

// GetAccessHistory gets access history for content
func (m *Manager) GetAccessHistory(ctx context.Context, contentID string, limit int) ([]map[string]interface{}, error) {
	path := fmt.Sprintf("/api/v1/content-economy/datasets/%s/access-history?limit=%d", contentID, limit)
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get access history: %w", err)
	}

	var result struct {
		History []map[string]interface{} `json:"history"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse access history: %w", err)
	}

	return result.History, nil
}

// Retire retires content from the marketplace
func (m *Manager) Retire(ctx context.Context, contentID string) (bool, error) {
	path := "/api/v1/content-economy/datasets/" + contentID + "/retire"
	resp, err := m.client.DoRequest(ctx, "POST", path, nil)
	if err != nil {
		return false, fmt.Errorf("failed to retire content: %w", err)
	}

	var result struct {
		Retired bool `json:"retired"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return false, fmt.Errorf("failed to parse retire response: %w", err)
	}

	return result.Retired, nil
}
