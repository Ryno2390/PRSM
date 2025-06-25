// Package types defines error types for the PRSM Go SDK
package types

import "fmt"

// PRSMError is the base error type for all PRSM SDK errors
type PRSMError struct {
	Message   string                 `json:"message"`
	ErrorCode string                 `json:"error_code,omitempty"`
	Details   map[string]interface{} `json:"details,omitempty"`
}

func (e *PRSMError) Error() string {
	if e.ErrorCode != "" {
		return fmt.Sprintf("[%s] %s", e.ErrorCode, e.Message)
	}
	return e.Message
}

// AuthenticationError is raised when authentication fails
type AuthenticationError struct {
	Message string `json:"message"`
}

func (e *AuthenticationError) Error() string {
	return fmt.Sprintf("Authentication failed: %s", e.Message)
}

// InsufficientFundsError is raised when FTNS balance is insufficient
type InsufficientFundsError struct {
	Message   string  `json:"message"`
	Required  float64 `json:"required,omitempty"`
	Available float64 `json:"available,omitempty"`
}

func (e *InsufficientFundsError) Error() string {
	if e.Required > 0 && e.Available > 0 {
		return fmt.Sprintf("Insufficient FTNS balance. Required: %.2f, Available: %.2f", e.Required, e.Available)
	}
	return fmt.Sprintf("Insufficient funds: %s", e.Message)
}

// SafetyViolationError is raised when content violates safety policies
type SafetyViolationError struct {
	Message     string      `json:"message"`
	SafetyLevel SafetyLevel `json:"safety_level,omitempty"`
}

func (e *SafetyViolationError) Error() string {
	if e.SafetyLevel != "" {
		return fmt.Sprintf("Safety violation [%s]: %s", e.SafetyLevel, e.Message)
	}
	return fmt.Sprintf("Safety violation: %s", e.Message)
}

// NetworkError is raised when network requests fail
type NetworkError struct {
	Message string `json:"message"`
}

func (e *NetworkError) Error() string {
	return fmt.Sprintf("Network error: %s", e.Message)
}

// ModelNotFoundError is raised when requested model is not available
type ModelNotFoundError struct {
	ModelID string `json:"model_id"`
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("Model not found: %s", e.ModelID)
}

// ToolExecutionError is raised when MCP tool execution fails
type ToolExecutionError struct {
	ToolName     string `json:"tool_name"`
	ErrorMessage string `json:"error_message"`
}

func (e *ToolExecutionError) Error() string {
	return fmt.Sprintf("Tool execution failed: %s - %s", e.ToolName, e.ErrorMessage)
}

// RateLimitError is raised when API rate limits are exceeded
type RateLimitError struct {
	Message    string `json:"message"`
	RetryAfter *int   `json:"retry_after,omitempty"`
}

func (e *RateLimitError) Error() string {
	if e.RetryAfter != nil {
		return fmt.Sprintf("Rate limit exceeded. Retry after %d seconds", *e.RetryAfter)
	}
	return fmt.Sprintf("Rate limit exceeded: %s", e.Message)
}

// ValidationError is raised when request validation fails
type ValidationError struct {
	Field             string `json:"field"`
	ValidationMessage string `json:"validation_message"`
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("Validation error for field '%s': %s", e.Field, e.ValidationMessage)
}