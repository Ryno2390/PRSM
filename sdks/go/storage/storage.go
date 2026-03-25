// Package storage provides IPFS storage functionality for the PRSM Go SDK
package storage

import (
	"context"
	"time"

	"github.com/pkg/errors"
)

// Client handles IPFS storage operations
type Client struct {
	httpClient HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new storage client
func New(client HTTPClient) *Client {
	return &Client{
		httpClient: client,
	}
}

// StorageStatus represents the status of stored content
type StorageStatus string

const (
	StorageStatusUploading  StorageStatus = "uploading"
	StorageStatusAvailable  StorageStatus = "available"
	StorageStatusPinned     StorageStatus = "pinned"
	StorageStatusUnavailable StorageStatus = "unavailable"
	StorageStatusExpired    StorageStatus = "expired"
)

// ContentType represents types of content in storage
type ContentType string

const (
	ContentTypeFile     ContentType = "file"
	ContentTypeDataset  ContentType = "dataset"
	ContentTypeModel    ContentType = "model"
	ContentTypeDocument ContentType = "document"
	ContentTypeCode     ContentType = "code"
	ContentTypeOther    ContentType = "other"
)

// UploadRequest represents a request to upload content
type UploadRequest struct {
	ContentType  ContentType `json:"content_type"`
	Filename     string      `json:"filename,omitempty"`
	Description  string      `json:"description,omitempty"`
	Tags         []string    `json:"tags"`
	IsPublic     bool        `json:"is_public"`
	Pin          bool        `json:"pin"`
	Replication  int         `json:"replication"`
}

// UploadResult represents the result of content upload
type UploadResult struct {
	CID         string        `json:"cid"`
	Size        int64         `json:"size"`
	ContentType ContentType   `json:"content_type"`
	Filename    string        `json:"filename,omitempty"`
	UploadTime  time.Time     `json:"upload_time"`
	FTNSCost    float64       `json:"ftns_cost"`
	GatewayURL  string        `json:"gateway_url"`
	IsPinned    bool          `json:"is_pinned"`
}

// StorageInfo represents information about stored content
type StorageInfo struct {
	CID         string        `json:"cid"`
	ContentType ContentType   `json:"content_type"`
	Size        int64         `json:"size"`
	Filename    string        `json:"filename,omitempty"`
	Description string        `json:"description,omitempty"`
	Tags        []string      `json:"tags"`
	Status      StorageStatus `json:"status"`
	IsPublic    bool          `json:"is_public"`
	IsPinned    bool          `json:"is_pinned"`
	Replication int           `json:"replication"`
	CreatedAt   time.Time     `json:"created_at"`
	ExpiresAt   *time.Time    `json:"expires_at,omitempty"`
	Owner       string        `json:"owner"`
	AccessCount int           `json:"access_count"`
}

// PinInfo represents information about pinned content
type PinInfo struct {
	CID          string    `json:"cid"`
	PinnedAt     time.Time `json:"pinned_at"`
	Size         int64     `json:"size"`
	Replication  int       `json:"replication"`
	MonthlyCost  float64   `json:"monthly_cost"`
}

// SearchRequest represents a search request for stored content
type SearchRequest struct {
	Query       string      `json:"query,omitempty"`
	ContentType *ContentType `json:"content_type,omitempty"`
	Tags        []string    `json:"tags,omitempty"`
	Owner       string      `json:"owner,omitempty"`
	IsPublic    *bool       `json:"is_public,omitempty"`
	MinSize     *int64      `json:"min_size,omitempty"`
	MaxSize     *int64      `json:"max_size,omitempty"`
	Limit       int         `json:"limit"`
	Offset      int         `json:"offset"`
}

// SearchResult represents search results for stored content
type SearchResult struct {
	Items  []StorageInfo `json:"items"`
	Total  int           `json:"total"`
	Offset int           `json:"offset"`
	Limit  int           `json:"limit"`
}

// Upload uploads content to IPFS storage
func (c *Client) Upload(ctx context.Context, req *UploadRequest, data []byte) (*UploadResult, error) {
	if req == nil {
		req = &UploadRequest{}
	}
	if req.ContentType == "" {
		req.ContentType = ContentTypeFile
	}
	if req.Replication <= 0 {
		req.Replication = 3
	}
	if len(data) == 0 {
		return nil, errors.New("data cannot be empty")
	}

	// For binary data, we wrap it in a request that includes both metadata and data
	fullReq := map[string]interface{}{
		"content_type": req.ContentType,
		"filename":     req.Filename,
		"description":  req.Description,
		"tags":         req.Tags,
		"is_public":    req.IsPublic,
		"pin":          req.Pin,
		"replication":  req.Replication,
		"data_size":    len(data),
	}

	var response UploadResult
	err := c.httpClient.MakeRequest(ctx, "POST", "/api/v1/storage/upload", fullReq, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to upload content")
	}
	return &response, nil
}

// Download downloads content from IPFS by CID
func (c *Client) Download(ctx context.Context, cid string) ([]byte, error) {
	if cid == "" {
		return nil, errors.New("CID cannot be empty")
	}

	var response struct {
		Data []byte `json:"data"`
	}
	err := c.httpClient.MakeRequest(ctx, "GET", "/api/v1/storage/"+cid+"/download", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to download content")
	}
	return response.Data, nil
}

// GetInfo retrieves information about stored content
func (c *Client) GetInfo(ctx context.Context, cid string) (*StorageInfo, error) {
	if cid == "" {
		return nil, errors.New("CID cannot be empty")
	}

	var response StorageInfo
	err := c.httpClient.MakeRequest(ctx, "GET", "/api/v1/storage/"+cid, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get storage info")
	}
	return &response, nil
}

// Pin pins content for persistent storage
func (c *Client) Pin(ctx context.Context, cid string, replication int) (*PinInfo, error) {
	if cid == "" {
		return nil, errors.New("CID cannot be empty")
	}
	if replication <= 0 {
		replication = 3
	}

	req := map[string]interface{}{"replication": replication}

	var response PinInfo
	err := c.httpClient.MakeRequest(ctx, "POST", "/api/v1/storage/"+cid+"/pin", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to pin content")
	}
	return &response, nil
}

// Unpin unpins content from storage
func (c *Client) Unpin(ctx context.Context, cid string) (bool, error) {
	if cid == "" {
		return false, errors.New("CID cannot be empty")
	}

	var response struct {
		Unpinned bool `json:"unpinned"`
	}
	err := c.httpClient.MakeRequest(ctx, "POST", "/api/v1/storage/"+cid+"/unpin", nil, &response)
	if err != nil {
		return false, errors.Wrap(err, "failed to unpin content")
	}
	return response.Unpinned, nil
}

// ListPins lists all pinned content
func (c *Client) ListPins(ctx context.Context, limit int) ([]PinInfo, error) {
	if limit <= 0 {
		limit = 50
	}

	req := map[string]interface{}{"limit": limit}

	var response struct {
		Pins []PinInfo `json:"pins"`
	}
	err := c.httpClient.MakeRequest(ctx, "GET", "/api/v1/storage/pins", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list pins")
	}
	return response.Pins, nil
}

// Search searches for stored content
func (c *Client) Search(ctx context.Context, req *SearchRequest) (*SearchResult, error) {
	if req == nil {
		req = &SearchRequest{Limit: 20}
	}
	if req.Limit <= 0 {
		req.Limit = 20
	}

	var response SearchResult
	err := c.httpClient.MakeRequest(ctx, "POST", "/api/v1/storage/search", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to search storage")
	}
	return &response, nil
}

// Delete deletes content from storage
func (c *Client) Delete(ctx context.Context, cid string) (bool, error) {
	if cid == "" {
		return false, errors.New("CID cannot be empty")
	}

	var response struct {
		Deleted bool `json:"deleted"`
	}
	err := c.httpClient.MakeRequest(ctx, "DELETE", "/api/v1/storage/"+cid, nil, &response)
	if err != nil {
		return false, errors.Wrap(err, "failed to delete content")
	}
	return response.Deleted, nil
}

// UpdateMetadata updates content metadata
func (c *Client) UpdateMetadata(ctx context.Context, cid string, description string, tags []string, isPublic *bool) (*StorageInfo, error) {
	if cid == "" {
		return nil, errors.New("CID cannot be empty")
	}

	data := map[string]interface{}{}
	if description != "" {
		data["description"] = description
	}
	if tags != nil {
		data["tags"] = tags
	}
	if isPublic != nil {
		data["is_public"] = *isPublic
	}

	var response StorageInfo
	err := c.httpClient.MakeRequest(ctx, "PATCH", "/api/v1/storage/"+cid, data, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to update metadata")
	}
	return &response, nil
}

// EstimateUploadCost estimates the FTNS cost for uploading content
func (c *Client) EstimateUploadCost(ctx context.Context, sizeBytes int64, replication int, durationDays int) (float64, error) {
	if sizeBytes <= 0 {
		return 0, errors.New("size must be positive")
	}
	if replication <= 0 {
		replication = 3
	}
	if durationDays <= 0 {
		durationDays = 30
	}

	req := map[string]interface{}{
		"size":           sizeBytes,
		"replication":    replication,
		"duration_days":  durationDays,
	}

	var response struct {
		EstimatedCost float64 `json:"estimated_cost"`
	}
	err := c.httpClient.MakeRequest(ctx, "POST", "/api/v1/storage/estimate-cost", req, &response)
	if err != nil {
		return 0, errors.Wrap(err, "failed to estimate upload cost")
	}
	return response.EstimatedCost, nil
}

// GetStorageStats retrieves storage usage statistics
func (c *Client) GetStorageStats(ctx context.Context) (map[string]interface{}, error) {
	var response map[string]interface{}
	err := c.httpClient.MakeRequest(ctx, "GET", "/api/v1/storage/stats", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get storage stats")
	}
	return response, nil
}

// GetGatewayURL returns the HTTP gateway URL for content
func (c *Client) GetGatewayURL(cid string) string {
	return "https://ipfs.io/ipfs/" + cid
}
