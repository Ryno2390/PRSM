// Package governance provides governance API functionality for PRSM DAO operations
package governance

import (
	"context"

	"github.com/pkg/errors"
)

// Manager handles governance operations
type Manager struct {
	client HTTPClient
}

// HTTPClient interface for making HTTP requests
type HTTPClient interface {
	MakeRequest(ctx context.Context, method, endpoint string, reqBody interface{}, respBody interface{}) error
}

// New creates a new governance manager
func New(client HTTPClient) *Manager {
	return &Manager{
		client: client,
	}
}

// Proposal represents a governance proposal
type Proposal struct {
	ID                 string  `json:"id"`
	Title              string  `json:"title"`
	Description        string  `json:"description"`
	Category           string  `json:"category"`
	Status             string  `json:"status"`
	ImplementationPlan string  `json:"implementation_plan"`
	BudgetRequired     float64 `json:"budget_required"`
	VotesFor           int     `json:"votes_for"`
	VotesAgainst       int     `json:"votes_against"`
	VotesAbstain       int     `json:"votes_abstain"`
	CreatedAt          string  `json:"created_at"`
	VotingEndsAt       string  `json:"voting_ends_at"`
	Proposer           string  `json:"proposer"`
}

// ProposalRequest represents a request to create a new proposal
type ProposalRequest struct {
	Title              string  `json:"title"`
	Description        string  `json:"description"`
	Category           string  `json:"category"`
	ImplementationPlan string  `json:"implementation_plan"`
	BudgetRequired     float64 `json:"budget_required"`
}

// VoteRequest represents a vote on a proposal
type VoteRequest struct {
	Vote        string  `json:"vote"` // "yes", "no", "abstain"
	VotingPower float64 `json:"voting_power"`
	Comment     *string `json:"comment,omitempty"`
}

// ProposalResponse represents the response when creating a proposal
type ProposalResponse struct {
	ProposalID string `json:"proposal_id"`
}

// ListProposalsOptions represents options for listing proposals
type ListProposalsOptions struct {
	Status *string `json:"status,omitempty"`
	Limit  int     `json:"limit,omitempty"`
	Offset int     `json:"offset,omitempty"`
}

// PaginatedResponse represents a paginated response
type PaginatedResponse[T any] struct {
	Data       []T `json:"data"`
	TotalCount int `json:"total_count"`
	Limit      int `json:"limit"`
	Offset     int `json:"offset"`
	HasMore    bool `json:"has_more"`
}

// ListProposals lists active proposals with filtering
func (m *Manager) ListProposals(ctx context.Context, options *ListProposalsOptions) (*PaginatedResponse[Proposal], error) {
	endpoint := "/api/v1/governance/proposals"
	
	var response PaginatedResponse[Proposal]
	err := m.client.MakeRequest(ctx, "GET", endpoint, options, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list proposals")
	}

	return &response, nil
}

// SubmitProposal submits a new governance proposal
func (m *Manager) SubmitProposal(ctx context.Context, proposal *ProposalRequest) (*ProposalResponse, error) {
	if proposal == nil {
		return nil, errors.New("proposal cannot be nil")
	}

	if proposal.Title == "" {
		return nil, errors.New("proposal title cannot be empty")
	}

	if proposal.Description == "" {
		return nil, errors.New("proposal description cannot be empty")
	}

	if proposal.Category == "" {
		return nil, errors.New("proposal category cannot be empty")
	}

	var response ProposalResponse
	err := m.client.MakeRequest(ctx, "POST", "/api/v1/governance/proposals", proposal, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to submit proposal")
	}

	return &response, nil
}

// Vote votes on a governance proposal
func (m *Manager) Vote(ctx context.Context, proposalID string, vote *VoteRequest) error {
	if proposalID == "" {
		return errors.New("proposal ID cannot be empty")
	}

	if vote == nil {
		return errors.New("vote cannot be nil")
	}

	if vote.Vote != "yes" && vote.Vote != "no" && vote.Vote != "abstain" {
		return errors.New("vote must be 'yes', 'no', or 'abstain'")
	}

	if vote.VotingPower <= 0 {
		return errors.New("voting power must be positive")
	}

	endpoint := "/api/v1/governance/proposals/" + proposalID + "/vote"
	err := m.client.MakeRequest(ctx, "POST", endpoint, vote, nil)
	if err != nil {
		return errors.Wrap(err, "failed to vote on proposal")
	}

	return nil
}

// GetProposal retrieves details of a specific proposal
func (m *Manager) GetProposal(ctx context.Context, proposalID string) (*Proposal, error) {
	if proposalID == "" {
		return nil, errors.New("proposal ID cannot be empty")
	}

	var response Proposal
	err := m.client.MakeRequest(ctx, "GET", "/api/v1/governance/proposals/"+proposalID, nil, &response)
	if err != nil {
		return nil, errors.Wrap(err, "failed to get proposal details")
	}

	return &response, nil
}