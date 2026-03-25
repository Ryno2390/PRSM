// Package ftns provides FTNS token management for the PRSM Go SDK
package ftns

import (
	"context"
	"time"

	"github.com/pkg/errors"
)

// Manager handles FTNS token operations
type Manager struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new FTNS manager
func New(client HTTPClient) *Manager {
	return &Manager{
		client: client,
	}
}

// FTNSBalance represents FTNS token balance information
type FTNSBalance struct {
	TotalBalance     float64   `json:"total_balance"`
	AvailableBalance float64   `json:"available_balance"`
	ReservedBalance  float64   `json:"reserved_balance"`
	StakedBalance    float64   `json:"staked_balance"`
	EarnedToday      float64   `json:"earned_today"`
	SpentToday       float64   `json:"spent_today"`
	LastUpdated      time.Time `json:"last_updated"`
}

// Transaction represents an FTNS transaction record
type Transaction struct {
	TransactionID   string     `json:"transaction_id"`
	TransactionType string     `json:"transaction_type"` // transfer, stake, unstake, reward
	Amount          float64    `json:"amount"`
	FromAddress     string     `json:"from_address,omitempty"`
	ToAddress       string     `json:"to_address,omitempty"`
	Status          string     `json:"status"` // pending, completed, failed
	Timestamp       time.Time  `json:"timestamp"`
	Memo            string     `json:"memo,omitempty"`
	BlockNumber     *int64     `json:"block_number,omitempty"`
}

// StakeInfo represents staking information
type StakeInfo struct {
	StakedAmount  float64    `json:"staked_amount"`
	RewardsEarned float64    `json:"rewards_earned"`
	LockPeriod    int        `json:"lock_period"` // days
	APY           float64    `json:"apy"`
	UnlockDate    *time.Time `json:"unlock_date,omitempty"`
}

// TransferRequest represents a transfer request
type TransferRequest struct {
	ToAddress string  `json:"to_address"`
	Amount    float64 `json:"amount"`
	Memo      string  `json:"memo,omitempty"`
}

// TransferResponse represents a transfer response
type TransferResponse struct {
	TransactionID string    `json:"transaction_id"`
	Status        string    `json:"status"`
	Amount        float64   `json:"amount"`
	Fee           float64   `json:"fee"`
	Timestamp     time.Time `json:"timestamp"`
}

// TransactionHistoryRequest represents parameters for transaction history
type TransactionHistoryRequest struct {
	Limit           int    `json:"limit,omitempty"`
	Offset          int    `json:"offset,omitempty"`
	TransactionType string `json:"transaction_type,omitempty"`
}

// TransactionHistoryResponse represents transaction history response
type TransactionHistoryResponse struct {
	Transactions []Transaction `json:"transactions"`
	Total        int           `json:"total"`
	Offset       int           `json:"offset"`
	Limit        int           `json:"limit"`
}

// StakeRequest represents a staking request
type StakeRequest struct {
	Amount     float64 `json:"amount"`
	LockPeriod int     `json:"lock_period"` // days
}

// GetBalance retrieves the current FTNS balance
func (m *Manager) GetBalance(ctx context.Context) (*FTNSBalance, error) {
	var balance FTNSBalance
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/ftns/balance", nil, &balance)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get FTNS balance")
	}
	return &balance, nil
}

// Transfer transfers FTNS tokens to another address
func (m *Manager) Transfer(ctx context.Context, req *TransferRequest) (*TransferResponse, error) {
	if req == nil {
		return nil, errors.New("transfer request cannot be nil")
	}
	if req.ToAddress == "" {
		return nil, errors.New("to_address is required")
	}
	if req.Amount <= 0 {
		return nil, errors.New("amount must be positive")
	}

	var response TransferResponse
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/ftns/transfer", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to transfer FTNS")
	}
	return &response, nil
}

// GetTransactionHistory retrieves transaction history
func (m *Manager) GetTransactionHistory(ctx context.Context, req *TransactionHistoryRequest) (*TransactionHistoryResponse, error) {
	if req == nil {
		req = &TransactionHistoryRequest{Limit: 50}
	}
	if req.Limit <= 0 {
		req.Limit = 50
	}

	var response TransactionHistoryResponse
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/ftns/history", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get transaction history")
	}
	return &response, nil
}

// Stake stakes FTNS tokens for network participation
func (m *Manager) Stake(ctx context.Context, req *StakeRequest) (*StakeInfo, error) {
	if req == nil {
		return nil, errors.New("stake request cannot be nil")
	}
	if req.Amount <= 0 {
		return nil, errors.New("amount must be positive")
	}
	if req.LockPeriod <= 0 {
		req.LockPeriod = 30 // Default to 30 days
	}

	var response StakeInfo
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/ftns/stake", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to stake FTNS")
	}
	return &response, nil
}

// Unstake unstakes FTNS tokens
func (m *Manager) Unstake(ctx context.Context, amount *float64) (*StakeInfo, error) {
	req := map[string]interface{}{}
	if amount != nil {
		req["amount"] = *amount
	}

	var response StakeInfo
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/ftns/unstake", req, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to unstake FTNS")
	}
	return &response, nil
}

// GetStakeInfo retrieves current staking information
func (m *Manager) GetStakeInfo(ctx context.Context) (*StakeInfo, error) {
	var response StakeInfo
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/ftns/stake/info", nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get stake info")
	}
	return &response, nil
}

// EstimateTransferFee estimates the fee for a transfer
func (m *Manager) EstimateTransferFee(ctx context.Context, amount float64) (float64, error) {
	req := map[string]interface{}{"amount": amount}

	var response struct {
		Fee float64 `json:"fee"`
	}
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/ftns/estimate-fee", req, &response)
	if err != nil {
		return 0, errors.Wrap(err, "failed to estimate transfer fee")
	}
	return response.Fee, nil
}
