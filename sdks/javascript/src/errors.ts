/**
 * PRSM SDK Error Classes
 * Comprehensive error handling for the Protocol for Recursive Scientific Modeling
 */

import { APIError, SafetyLevel } from './types';

// ============================================================================
// BASE ERROR CLASS
// ============================================================================

/**
 * Base error class for all PRSM SDK errors
 */
export class PRSMError extends Error {
  public readonly code: string;
  public readonly requestId?: string;
  public readonly timestamp: string;
  public readonly details?: Record<string, any>;
  public readonly statusCode?: number;

  constructor(
    message: string,
    code: string = 'PRSM_ERROR',
    requestId?: string,
    details?: Record<string, any>,
    statusCode?: number
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.requestId = requestId;
    this.timestamp = new Date().toISOString();
    this.details = details;
    this.statusCode = statusCode;

    // Maintain proper stack trace for where our error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert error to JSON representation
   */
  toJSON(): APIError {
    return {
      code: this.code,
      message: this.message,
      details: this.details,
      requestId: this.requestId || 'unknown',
      timestamp: this.timestamp,
    };
  }

  /**
   * Create PRSMError from API error response
   */
  static fromAPIError(apiError: APIError, statusCode?: number): PRSMError {
    return new PRSMError(
      apiError.message,
      apiError.code,
      apiError.requestId,
      apiError.details,
      statusCode
    );
  }
}

// ============================================================================
// AUTHENTICATION ERRORS
// ============================================================================

/**
 * Error thrown when authentication fails
 */
export class AuthenticationError extends PRSMError {
  constructor(
    message: string = 'Authentication failed',
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'AUTHENTICATION_FAILED', requestId, details, 401);
  }
}

/**
 * Error thrown when API key is invalid or missing
 */
export class InvalidAPIKeyError extends AuthenticationError {
  constructor(requestId?: string) {
    super('Invalid or missing API key', requestId, {
      hint: 'Check your API key and ensure it is properly configured',
    });
    this.code = 'INVALID_API_KEY';
  }
}

/**
 * Error thrown when token has expired
 */
export class TokenExpiredError extends AuthenticationError {
  constructor(requestId?: string) {
    super('Authentication token has expired', requestId, {
      hint: 'Refresh your token or re-authenticate',
    });
    this.code = 'TOKEN_EXPIRED';
  }
}

/**
 * Error thrown when insufficient permissions
 */
export class InsufficientPermissionsError extends PRSMError {
  constructor(
    permission: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Insufficient permissions: ${permission}`,
      'INSUFFICIENT_PERMISSIONS',
      requestId,
      { requiredPermission: permission, ...details },
      403
    );
  }
}

// ============================================================================
// FINANCIAL ERRORS
// ============================================================================

/**
 * Error thrown when insufficient FTNS balance
 */
export class InsufficientFundsError extends PRSMError {
  public readonly required: number;
  public readonly available: number;

  constructor(
    required: number,
    available: number,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Insufficient FTNS balance. Required: ${required}, Available: ${available}`,
      'INSUFFICIENT_FUNDS',
      requestId,
      { required, available, ...details },
      402
    );
    this.required = required;
    this.available = available;
  }
}

/**
 * Error thrown when payment processing fails
 */
export class PaymentError extends PRSMError {
  public readonly paymentMethod?: string;
  public readonly transactionId?: string;

  constructor(
    message: string,
    paymentMethod?: string,
    transactionId?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'PAYMENT_ERROR', requestId, {
      paymentMethod,
      transactionId,
      ...details,
    }, 402);
    this.paymentMethod = paymentMethod;
    this.transactionId = transactionId;
  }
}

/**
 * Error thrown when Web3 transaction fails
 */
export class Web3TransactionError extends PRSMError {
  public readonly txHash?: string;
  public readonly gasUsed?: number;

  constructor(
    message: string,
    txHash?: string,
    gasUsed?: number,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'WEB3_TRANSACTION_ERROR', requestId, {
      txHash,
      gasUsed,
      ...details,
    }, 400);
    this.txHash = txHash;
    this.gasUsed = gasUsed;
  }
}

// ============================================================================
// SAFETY & SECURITY ERRORS
// ============================================================================

/**
 * Error thrown when safety violation is detected
 */
export class SafetyViolationError extends PRSMError {
  public readonly safetyLevel: SafetyLevel;
  public readonly violationType: string;

  constructor(
    violationType: string,
    safetyLevel: SafetyLevel,
    message?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      message || `Safety violation detected: ${violationType}`,
      'SAFETY_VIOLATION',
      requestId,
      { violationType, safetyLevel, ...details },
      400
    );
    this.safetyLevel = safetyLevel;
    this.violationType = violationType;
  }
}

/**
 * Error thrown when content is blocked by safety filters
 */
export class ContentBlockedError extends SafetyViolationError {
  constructor(
    reason: string,
    safetyLevel: SafetyLevel,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      'content_blocked',
      safetyLevel,
      `Content blocked: ${reason}`,
      requestId,
      details
    );
    this.code = 'CONTENT_BLOCKED';
  }
}

/**
 * Error thrown when circuit breaker is triggered
 */
export class CircuitBreakerError extends PRSMError {
  public readonly circuitName: string;
  public readonly resetTime: Date;

  constructor(
    circuitName: string,
    resetTime: Date,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Circuit breaker '${circuitName}' is open. Reset at: ${resetTime.toISOString()}`,
      'CIRCUIT_BREAKER_OPEN',
      requestId,
      { circuitName, resetTime: resetTime.toISOString(), ...details },
      503
    );
    this.circuitName = circuitName;
    this.resetTime = resetTime;
  }
}

// ============================================================================
// NETWORK & API ERRORS
// ============================================================================

/**
 * Error thrown for network-related issues
 */
export class NetworkError extends PRSMError {
  public readonly originalError?: Error;

  constructor(
    message: string,
    originalError?: Error,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'NETWORK_ERROR', requestId, {
      originalError: originalError?.message,
      ...details,
    }, 500);
    this.originalError = originalError;
  }
}

/**
 * Error thrown when request times out
 */
export class TimeoutError extends NetworkError {
  public readonly timeoutMs: number;

  constructor(
    timeoutMs: number,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Request timed out after ${timeoutMs}ms`,
      undefined,
      requestId,
      { timeoutMs, ...details }
    );
    this.code = 'TIMEOUT_ERROR';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Error thrown when rate limit is exceeded
 */
export class RateLimitError extends PRSMError {
  public readonly limit: number;
  public readonly remaining: number;
  public readonly resetTime: Date;

  constructor(
    limit: number,
    remaining: number,
    resetTime: Date,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Rate limit exceeded. Limit: ${limit}, Remaining: ${remaining}, Reset: ${resetTime.toISOString()}`,
      'RATE_LIMIT_EXCEEDED',
      requestId,
      { limit, remaining, resetTime: resetTime.toISOString(), ...details },
      429
    );
    this.limit = limit;
    this.remaining = remaining;
    this.resetTime = resetTime;
  }
}

// ============================================================================
// RESOURCE ERRORS
// ============================================================================

/**
 * Error thrown when requested resource is not found
 */
export class ResourceNotFoundError extends PRSMError {
  public readonly resourceType: string;
  public readonly resourceId: string;

  constructor(
    resourceType: string,
    resourceId: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `${resourceType} not found: ${resourceId}`,
      'RESOURCE_NOT_FOUND',
      requestId,
      { resourceType, resourceId, ...details },
      404
    );
    this.resourceType = resourceType;
    this.resourceId = resourceId;
  }
}

/**
 * Error thrown when model is not found or unavailable
 */
export class ModelNotFoundError extends ResourceNotFoundError {
  constructor(
    modelId: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super('Model', modelId, requestId, details);
    this.code = 'MODEL_NOT_FOUND';
  }
}

/**
 * Error thrown when session is not found
 */
export class SessionNotFoundError extends ResourceNotFoundError {
  constructor(
    sessionId: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super('Session', sessionId, requestId, details);
    this.code = 'SESSION_NOT_FOUND';
  }
}

/**
 * Error thrown when tool is not found or unavailable
 */
export class ToolNotFoundError extends ResourceNotFoundError {
  constructor(
    toolName: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super('Tool', toolName, requestId, details);
    this.code = 'TOOL_NOT_FOUND';
  }
}

// ============================================================================
// EXECUTION ERRORS
// ============================================================================

/**
 * Error thrown when tool execution fails
 */
export class ToolExecutionError extends PRSMError {
  public readonly toolName: string;
  public readonly executionId?: string;

  constructor(
    toolName: string,
    message: string,
    executionId?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Tool execution failed for '${toolName}': ${message}`,
      'TOOL_EXECUTION_ERROR',
      requestId,
      { toolName, executionId, ...details },
      400
    );
    this.toolName = toolName;
    this.executionId = executionId;
  }
}

/**
 * Error thrown when model execution fails
 */
export class ModelExecutionError extends PRSMError {
  public readonly modelId: string;
  public readonly provider: string;

  constructor(
    modelId: string,
    provider: string,
    message: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Model execution failed for '${modelId}' (${provider}): ${message}`,
      'MODEL_EXECUTION_ERROR',
      requestId,
      { modelId, provider, ...details },
      400
    );
    this.modelId = modelId;
    this.provider = provider;
  }
}

/**
 * Error thrown when session execution fails
 */
export class SessionExecutionError extends PRSMError {
  public readonly sessionId: string;

  constructor(
    sessionId: string,
    message: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Session execution failed for '${sessionId}': ${message}`,
      'SESSION_EXECUTION_ERROR',
      requestId,
      { sessionId, ...details },
      400
    );
    this.sessionId = sessionId;
  }
}

// ============================================================================
// VALIDATION ERRORS
// ============================================================================

/**
 * Error thrown when input validation fails
 */
export class ValidationError extends PRSMError {
  public readonly field?: string;
  public readonly value?: any;

  constructor(
    message: string,
    field?: string,
    value?: any,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'VALIDATION_ERROR', requestId, {
      field,
      value,
      ...details,
    }, 422);
    this.field = field;
    this.value = value;
  }
}

/**
 * Error thrown when required parameter is missing
 */
export class MissingParameterError extends ValidationError {
  constructor(
    parameterName: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Missing required parameter: ${parameterName}`,
      parameterName,
      undefined,
      requestId,
      details
    );
    this.code = 'MISSING_PARAMETER';
  }
}

/**
 * Error thrown when parameter value is invalid
 */
export class InvalidParameterError extends ValidationError {
  constructor(
    parameterName: string,
    value: any,
    expectedType?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Invalid parameter '${parameterName}': ${value}${
        expectedType ? ` (expected ${expectedType})` : ''
      }`,
      parameterName,
      value,
      requestId,
      { expectedType, ...details }
    );
    this.code = 'INVALID_PARAMETER';
  }
}

// ============================================================================
// WEBSOCKET ERRORS
// ============================================================================

/**
 * Error thrown for WebSocket-related issues
 */
export class WebSocketError extends PRSMError {
  public readonly eventType?: string;

  constructor(
    message: string,
    eventType?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'WEBSOCKET_ERROR', requestId, {
      eventType,
      ...details,
    });
    this.eventType = eventType;
  }
}

/**
 * Error thrown when WebSocket connection fails
 */
export class WebSocketConnectionError extends WebSocketError {
  constructor(
    reason: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `WebSocket connection failed: ${reason}`,
      'connection_failed',
      requestId,
      details
    );
    this.code = 'WEBSOCKET_CONNECTION_ERROR';
  }
}

/**
 * Error thrown when WebSocket message is invalid
 */
export class InvalidWebSocketMessageError extends WebSocketError {
  constructor(
    message: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Invalid WebSocket message: ${message}`,
      'invalid_message',
      requestId,
      details
    );
    this.code = 'INVALID_WEBSOCKET_MESSAGE';
  }
}

// ============================================================================
// SERVICE ERRORS
// ============================================================================

/**
 * Error thrown when external service is unavailable
 */
export class ServiceUnavailableError extends PRSMError {
  public readonly serviceName: string;
  public readonly retryAfter?: number;

  constructor(
    serviceName: string,
    retryAfter?: number,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(
      `Service unavailable: ${serviceName}${
        retryAfter ? ` (retry after ${retryAfter}s)` : ''
      }`,
      'SERVICE_UNAVAILABLE',
      requestId,
      { serviceName, retryAfter, ...details },
      503
    );
    this.serviceName = serviceName;
    this.retryAfter = retryAfter;
  }
}

/**
 * Error thrown when configuration is invalid
 */
export class ConfigurationError extends PRSMError {
  public readonly configKey?: string;

  constructor(
    message: string,
    configKey?: string,
    requestId?: string,
    details?: Record<string, any>
  ) {
    super(message, 'CONFIGURATION_ERROR', requestId, {
      configKey,
      ...details,
    });
    this.configKey = configKey;
  }
}

// ============================================================================
// ERROR UTILITIES
// ============================================================================

/**
 * Type guard to check if error is a PRSM error
 */
export function isPRSMError(error: any): error is PRSMError {
  return error instanceof PRSMError;
}

/**
 * Get error code from any error type
 */
export function getErrorCode(error: any): string {
  if (isPRSMError(error)) {
    return error.code;
  }
  if (error instanceof Error) {
    return error.name || 'UNKNOWN_ERROR';
  }
  return 'UNKNOWN_ERROR';
}

/**
 * Convert any error to PRSM error
 */
export function toPRSMError(
  error: any,
  defaultMessage: string = 'An unknown error occurred',
  requestId?: string
): PRSMError {
  if (isPRSMError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new PRSMError(
      error.message || defaultMessage,
      'WRAPPED_ERROR',
      requestId,
      { originalError: error.name }
    );
  }

  return new PRSMError(
    defaultMessage,
    'UNKNOWN_ERROR',
    requestId,
    { originalError: String(error) }
  );
}

/**
 * Format error for logging
 */
export function formatErrorForLogging(error: any): Record<string, any> {
  if (isPRSMError(error)) {
    return {
      name: error.name,
      code: error.code,
      message: error.message,
      requestId: error.requestId,
      timestamp: error.timestamp,
      statusCode: error.statusCode,
      details: error.details,
      stack: error.stack,
    };
  }

  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }

  return {
    error: String(error),
  };
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: any): boolean {
  if (!isPRSMError(error)) {
    return false;
  }

  const retryableCodes = [
    'NETWORK_ERROR',
    'TIMEOUT_ERROR',
    'SERVICE_UNAVAILABLE',
    'CIRCUIT_BREAKER_OPEN',
  ];

  return retryableCodes.includes(error.code);
}

/**
 * Get retry delay for retryable errors
 */
export function getRetryDelay(error: any, attempt: number): number {
  if (error instanceof RateLimitError) {
    return Math.max(0, error.resetTime.getTime() - Date.now());
  }

  if (error instanceof CircuitBreakerError) {
    return Math.max(0, error.resetTime.getTime() - Date.now());
  }

  if (error instanceof ServiceUnavailableError && error.retryAfter) {
    return error.retryAfter * 1000;
  }

  // Exponential backoff: 2^attempt * 1000ms, max 30 seconds
  return Math.min(Math.pow(2, attempt) * 1000, 30000);
}