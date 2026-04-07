// Package settler provides client methods for the Settler module
// Handles staking, slashing, and FTNS token operations on Base mainnet
package settler

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

// SettlerStatus represents the status of a settler
type SettlerStatus string

const (
	SettlerStatusActive    SettlerStatus = "active"
	SettlerStatusInactive  SettlerStatus = "inactive"
	SettlerStatusJailed    SettlerStatus = "jailed"
	SettlerStatusSlashed   SettlerStatus = "slashed"
)

// StakeInfo represents staking information
type StakeInfo struct {
	NodeID        string       `json:"node_id"`
	StakerAddress string       `json:"staker_address"`
	Amount        float64      `json:"amount"`
	Status        SettlerStatus `json:"status"`
	StakedAt      time.Time    `json:"staked_at"`
	LockEnd       *time.Time   `json:"lock_end,omitempty"`
	RewardsEarned float64      `json:"rewards_earned"`
}

// StakeRequest represents a request to stake tokens
type StakeRequest struct {
	NodeID    string  `json:"node_id"`
	Amount    float64 `json:"amount"`
	LockDays  int     `json:"lock_days,omitempty"`
}

// StakeResponse represents the response from staking
type StakeResponse struct {
	StakeID     string    `json:"stake_id"`
	NodeID      string    `json:"node_id"`
	Amount      float64   `json:"amount"`
	Status      SettlerStatus `json:"status"`
	TxHash      string    `json:"tx_hash"`
	StakedAt    time.Time `json:"staked_at"`
	LockEnd     *time.Time `json:"lock_end,omitempty"`
}

// UnstakeRequest represents a request to unstake tokens
type UnstakeRequest struct {
	StakeID string `json:"stake_id"`
}

// UnstakeResponse represents the response from unstaking
type UnstakeResponse struct {
	StakeID   string    `json:"stake_id"`
	Amount    float64   `json:"amount"`
	Status    string    `json:"status"`
	TxHash    string    `json:"tx_hash"`
	Completed time.Time `json:"completed"`
}

// BatchSignRequest represents a request for batch signing
type BatchSignRequest struct {
	Operations []SignOperation `json:"operations"`
}

// SignOperation represents a single signing operation
type SignOperation struct {
	Type     string      `json:"type"`
	TargetID string      `json:"target_id"`
	Data     interface{} `json:"data"`
}

// BatchSignResponse represents the response from batch signing
type BatchSignResponse struct {
	BatchID    string        `json:"batch_id"`
	Operations []SignResult  `json:"operations"`
	TxHash     string        `json:"tx_hash"`
	Status     string        `json:"status"`
	FTNSUsed   float64       `json:"ftns_used"`
}

// SignResult represents the result of a signing operation
type SignResult struct {
	OperationID string `json:"operation_id"`
	Success     bool   `json:"success"`
	Error       string `json:"error,omitempty"`
}

// SlashingEvent represents a slashing event
type SlashingEvent struct {
	EventID     string       `json:"event_id"`
	NodeID      string       `json:"node_id"`
	Reason      string       `json:"reason"`
	Amount      float64      `json:"amount_slashed"`
	Status      SettlerStatus `json:"status"`
	BlockNumber int64        `json:"block_number"`
	Timestamp   time.Time    `json:"timestamp"`
}

// SlashingHistory represents slashing history for a node
type SlashingHistory struct {
	NodeID       string          `json:"node_id"`
	TotalSlashed float64         `json:"total_slashed"`
	Events       []SlashingEvent `json:"events"`
}

// Manager provides settler management methods
type Manager struct {
	client HTTPClient
}

// NewManager creates a new settler manager
func NewManager(client HTTPClient) *Manager {
	return &Manager{client: client}
}

// Stake stakes tokens on a node
func (m *Manager) Stake(ctx context.Context, req StakeRequest) (*StakeResponse, error) {
	body := map[string]interface{}{
		"node_id": req.NodeID,
		"amount":  req.Amount,
	}
	if req.LockDays > 0 {
		body["lock_days"] = req.LockDays
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/settler/stake", body)
	if err != nil {
		return nil, fmt.Errorf("failed to stake: %w", err)
	}

	var result StakeResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse stake response: %w", err)
	}

	return &result, nil
}

// Unstake unstakes tokens from a node
func (m *Manager) Unstake(ctx context.Context, req UnstakeRequest) (*UnstakeResponse, error) {
	body := map[string]interface{}{
		"stake_id": req.StakeID,
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/settler/unstake", body)
	if err != nil {
		return nil, fmt.Errorf("failed to unstake: %w", err)
	}

	var result UnstakeResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse unstake response: %w", err)
	}

	return &result, nil
}

// GetStakeInfo gets staking information for a node
func (m *Manager) GetStakeInfo(ctx context.Context, stakeID string) (*StakeInfo, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/settler/stakes/"+stakeID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get stake info: %w", err)
	}

	var result StakeInfo
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse stake info: %w", err)
	}

	return &result, nil
}

// ListStakes lists stakes for a staker
func (m *Manager) ListStakes(ctx context.Context, stakerAddress string, limit, offset int) ([]StakeInfo, error) {
	path := fmt.Sprintf("/api/v1/settler/stakes?staker=%s&limit=%d&offset=%d", stakerAddress, limit, offset)
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list stakes: %w", err)
	}

	var result struct {
		Stakes []StakeInfo `json:"stakes"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse stakes: %w", err)
	}

	return result.Stakes, nil
}

// BatchSign signs multiple operations in a batch
func (m *Manager) BatchSign(ctx context.Context, req BatchSignRequest) (*BatchSignResponse, error) {
	body := map[string]interface{}{
		"operations": req.Operations,
	}

	resp, err := m.client.DoRequest(ctx, "POST", "/api/v1/settler/batch-sign", body)
	if err != nil {
		return nil, fmt.Errorf("failed to batch sign: %w", err)
	}

	var result BatchSignResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse batch sign response: %w", err)
	}

	return &result, nil
}

// GetSlashingHistory gets slashing history for a node
func (m *Manager) GetSlashingHistory(ctx context.Context, nodeID string) (*SlashingHistory, error) {
	resp, err := m.client.DoRequest(ctx, "GET", "/api/v1/settler/slashing/"+nodeID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get slashing history: %w", err)
	}

	var result SlashingHistory
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("failed to parse slashing history: %w", err)
	}

	return &result, nil
}

// GetRewards gets unclaimed rewards for a staker
func (m *Manager) GetRewards(ctx context.Context, stakerAddress string) (float64, error) {
	path := "/api/v1/settler/rewards/" + stakerAddress
	resp, err := m.client.DoRequest(ctx, "GET", path, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to get rewards: %w", err)
	}

	var result struct {
		Rewards float64 `json:"rewards"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("failed to parse rewards: %w", err)
	}

	return result.Rewards, nil
}

// ClaimRewards claims staking rewards
func (m *Manager) ClaimRewards(ctx context.Context, stakerAddress string) (float64, string, error) {
	path := "/api/v1/settler/rewards/" + stakerAddress + "/claim"
	resp, err := m.client.DoRequest(ctx, "POST", path, nil)
	if err != nil {
		return 0, "", fmt.Errorf("failed to claim rewards: %w", err)
	}

	var result struct {
		Amount float64 `json:"amount"`
		TxHash string  `json:"tx_hash"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, "", fmt.Errorf("failed to parse claim response: %w", err)
	}

	return result.Amount, result.TxHash, nil
}
