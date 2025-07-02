package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPRSMError_Error(t *testing.T) {
	// Test with error code
	err := &PRSMError{
		Message:   "Something went wrong",
		ErrorCode: "API_ERROR",
		Details: map[string]interface{}{
			"request_id": "req-123",
		},
	}
	
	expected := "[API_ERROR] Something went wrong"
	assert.Equal(t, expected, err.Error())
	
	// Test without error code
	errNoCode := &PRSMError{
		Message: "Basic error message",
	}
	
	assert.Equal(t, "Basic error message", errNoCode.Error())
}

func TestAuthenticationError_Error(t *testing.T) {
	err := &AuthenticationError{
		Message: "Invalid API key provided",
	}
	
	expected := "Authentication failed: Invalid API key provided"
	assert.Equal(t, expected, err.Error())
}

func TestInsufficientFundsError_Error(t *testing.T) {
	// Test with balance details
	err := &InsufficientFundsError{
		Message:   "Not enough FTNS tokens",
		Required:  10.5,
		Available: 7.25,
	}
	
	expected := "Insufficient FTNS balance. Required: 10.50, Available: 7.25"
	assert.Equal(t, expected, err.Error())
	
	// Test without balance details
	errNoDetails := &InsufficientFundsError{
		Message: "Insufficient funds for operation",
	}
	
	expected2 := "Insufficient funds: Insufficient funds for operation"
	assert.Equal(t, expected2, errNoDetails.Error())
	
	// Test with zero values
	errZeroValues := &InsufficientFundsError{
		Message:   "Zero balance",
		Required:  0,
		Available: 0,
	}
	
	expected3 := "Insufficient funds: Zero balance"
	assert.Equal(t, expected3, errZeroValues.Error())
}

func TestSafetyViolationError_Error(t *testing.T) {
	// Test with safety level
	err := &SafetyViolationError{
		Message:     "Content contains harmful material",
		SafetyLevel: SafetyLevelCritical,
	}
	
	expected := "Safety violation [critical]: Content contains harmful material"
	assert.Equal(t, expected, err.Error())
	
	// Test without safety level
	errNoLevel := &SafetyViolationError{
		Message: "General safety violation",
	}
	
	expected2 := "Safety violation: General safety violation"
	assert.Equal(t, expected2, errNoLevel.Error())
}

func TestNetworkError_Error(t *testing.T) {
	err := &NetworkError{
		Message: "Connection timeout",
	}
	
	expected := "Network error: Connection timeout"
	assert.Equal(t, expected, err.Error())
}

func TestModelNotFoundError_Error(t *testing.T) {
	err := &ModelNotFoundError{
		ModelID: "gpt-5-turbo",
	}
	
	expected := "Model not found: gpt-5-turbo"
	assert.Equal(t, expected, err.Error())
}

func TestToolExecutionError_Error(t *testing.T) {
	err := &ToolExecutionError{
		ToolName:     "web_search",
		ErrorMessage: "Search service unavailable",
	}
	
	expected := "Tool execution failed: web_search - Search service unavailable"
	assert.Equal(t, expected, err.Error())
}

func TestRateLimitError_Error(t *testing.T) {
	// Test with retry after
	retryAfter := 60
	err := &RateLimitError{
		Message:    "Too many requests",
		RetryAfter: &retryAfter,
	}
	
	expected := "Rate limit exceeded. Retry after 60 seconds"
	assert.Equal(t, expected, err.Error())
	
	// Test without retry after
	errNoRetry := &RateLimitError{
		Message: "Rate limit exceeded",
	}
	
	expected2 := "Rate limit exceeded: Rate limit exceeded"
	assert.Equal(t, expected2, errNoRetry.Error())
}

func TestValidationError_Error(t *testing.T) {
	err := &ValidationError{
		Field:             "temperature",
		ValidationMessage: "must be between 0.0 and 2.0",
	}
	
	expected := "Validation error for field 'temperature': must be between 0.0 and 2.0"
	assert.Equal(t, expected, err.Error())
}

// Test error type assertions
func TestErrorTypeAssertions(t *testing.T) {
	var err error
	
	// Test PRSMError
	err = &PRSMError{Message: "test"}
	_, ok := err.(*PRSMError)
	assert.True(t, ok)
	
	// Test AuthenticationError
	err = &AuthenticationError{Message: "test"}
	_, ok = err.(*AuthenticationError)
	assert.True(t, ok)
	
	// Test InsufficientFundsError
	err = &InsufficientFundsError{Message: "test"}
	_, ok = err.(*InsufficientFundsError)
	assert.True(t, ok)
	
	// Test SafetyViolationError
	err = &SafetyViolationError{Message: "test"}
	_, ok = err.(*SafetyViolationError)
	assert.True(t, ok)
	
	// Test NetworkError
	err = &NetworkError{Message: "test"}
	_, ok = err.(*NetworkError)
	assert.True(t, ok)
	
	// Test ModelNotFoundError
	err = &ModelNotFoundError{ModelID: "test"}
	_, ok = err.(*ModelNotFoundError)
	assert.True(t, ok)
	
	// Test ToolExecutionError
	err = &ToolExecutionError{ToolName: "test", ErrorMessage: "test"}
	_, ok = err.(*ToolExecutionError)
	assert.True(t, ok)
	
	// Test RateLimitError
	err = &RateLimitError{Message: "test"}
	_, ok = err.(*RateLimitError)
	assert.True(t, ok)
	
	// Test ValidationError
	err = &ValidationError{Field: "test", ValidationMessage: "test"}
	_, ok = err.(*ValidationError)
	assert.True(t, ok)
}

// Test complex error scenarios
func TestComplexErrorScenarios(t *testing.T) {
	// Insufficient funds with precise balance tracking
	err := &InsufficientFundsError{
		Message:   "Cannot execute query",
		Required:  15.789,
		Available: 12.345,
	}
	
	expected := "Insufficient FTNS balance. Required: 15.79, Available: 12.35"
	assert.Equal(t, expected, err.Error())
	
	// Rate limit with large retry after value
	retryAfter := 3600 // 1 hour
	rateLimitErr := &RateLimitError{
		Message:    "Daily limit exceeded",
		RetryAfter: &retryAfter,
	}
	
	expected2 := "Rate limit exceeded. Retry after 3600 seconds"
	assert.Equal(t, expected2, rateLimitErr.Error())
	
	// Safety violation with emergency level
	safetyErr := &SafetyViolationError{
		Message:     "Content triggers emergency protocols",
		SafetyLevel: SafetyLevelEmergency,
	}
	
	expected3 := "Safety violation [emergency]: Content triggers emergency protocols"
	assert.Equal(t, expected3, safetyErr.Error())
}

// Test empty and nil values
func TestErrorsWithEmptyValues(t *testing.T) {
	// Empty message in PRSMError
	err := &PRSMError{
		Message:   "",
		ErrorCode: "EMPTY_MSG",
	}
	
	assert.Equal(t, "[EMPTY_MSG] ", err.Error())
	
	// Empty model ID
	modelErr := &ModelNotFoundError{
		ModelID: "",
	}
	
	assert.Equal(t, "Model not found: ", modelErr.Error())
	
	// Empty field in validation error
	validationErr := &ValidationError{
		Field:             "",
		ValidationMessage: "Some validation failed",
	}
	
	assert.Equal(t, "Validation error for field '': Some validation failed", validationErr.Error())
}