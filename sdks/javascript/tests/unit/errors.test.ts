/**
 * Unit tests for PRSM Error classes
 */

import {
  PRSMError,
  AuthenticationError,
  InsufficientFundsError,
  SafetyViolationError,
  NetworkError,
  ValidationError,
  WebSocketError,
  isPRSMError,
  getErrorCode,
  toPRSMError,
  isRetryableError,
  getRetryDelay,
} from '../../src/errors';
import { SafetyLevel } from '../../src/types';

describe('PRSM Error Classes', () => {
  describe('PRSMError', () => {
    it('should create base PRSM error', () => {
      const error = new PRSMError('Test error', 'TEST_CODE', 'req_123');
      
      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_CODE');
      expect(error.requestId).toBe('req_123');
      expect(error.timestamp).toBeDefined();
    });

    it('should convert to JSON', () => {
      const error = new PRSMError('Test error', 'TEST_CODE', 'req_123', { key: 'value' });
      const json = error.toJSON();
      
      expect(json).toEqual({
        code: 'TEST_CODE',
        message: 'Test error',
        details: { key: 'value' },
        requestId: 'req_123',
        timestamp: error.timestamp,
      });
    });

    it('should create from API error', () => {
      const apiError = {
        code: 'API_ERROR',
        message: 'API error message',
        details: { field: 'value' },
        requestId: 'req_456',
        timestamp: '2024-01-01T00:00:00Z',
      };
      
      const error = PRSMError.fromAPIError(apiError, 400);
      
      expect(error.code).toBe('API_ERROR');
      expect(error.message).toBe('API error message');
      expect(error.statusCode).toBe(400);
      expect(error.requestId).toBe('req_456');
    });
  });

  describe('AuthenticationError', () => {
    it('should create authentication error', () => {
      const error = new AuthenticationError('Invalid credentials');
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('AUTHENTICATION_FAILED');
      expect(error.statusCode).toBe(401);
      expect(error.message).toBe('Invalid credentials');
    });
  });

  describe('InsufficientFundsError', () => {
    it('should create insufficient funds error', () => {
      const error = new InsufficientFundsError(100, 50, 'req_789');
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('INSUFFICIENT_FUNDS');
      expect(error.statusCode).toBe(402);
      expect(error.required).toBe(100);
      expect(error.available).toBe(50);
      expect(error.message).toContain('Required: 100');
      expect(error.message).toContain('Available: 50');
    });
  });

  describe('SafetyViolationError', () => {
    it('should create safety violation error', () => {
      const error = new SafetyViolationError(
        'content_blocked',
        SafetyLevel.HIGH,
        'Content contains inappropriate material'
      );
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('SAFETY_VIOLATION');
      expect(error.violationType).toBe('content_blocked');
      expect(error.safetyLevel).toBe(SafetyLevel.HIGH);
      expect(error.message).toBe('Content contains inappropriate material');
    });
  });

  describe('NetworkError', () => {
    it('should create network error', () => {
      const originalError = new Error('Connection failed');
      const error = new NetworkError('Network request failed', originalError);
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('NETWORK_ERROR');
      expect(error.statusCode).toBe(500);
      expect(error.originalError).toBe(originalError);
    });
  });

  describe('ValidationError', () => {
    it('should create validation error', () => {
      const error = new ValidationError('Invalid input', 'username', 'test@');
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('VALIDATION_ERROR');
      expect(error.statusCode).toBe(422);
      expect(error.field).toBe('username');
      expect(error.value).toBe('test@');
    });
  });

  describe('WebSocketError', () => {
    it('should create websocket error', () => {
      const error = new WebSocketError('Connection lost', 'connection_failed');
      
      expect(error).toBeInstanceOf(PRSMError);
      expect(error.code).toBe('WEBSOCKET_ERROR');
      expect(error.eventType).toBe('connection_failed');
    });
  });
});

describe('Error Utilities', () => {
  describe('isPRSMError', () => {
    it('should identify PRSM errors', () => {
      const prsmError = new PRSMError('Test');
      const regularError = new Error('Regular error');
      
      expect(isPRSMError(prsmError)).toBe(true);
      expect(isPRSMError(regularError)).toBe(false);
      expect(isPRSMError('string')).toBe(false);
      expect(isPRSMError(null)).toBe(false);
    });
  });

  describe('getErrorCode', () => {
    it('should get error code from PRSM error', () => {
      const error = new PRSMError('Test', 'TEST_CODE');
      expect(getErrorCode(error)).toBe('TEST_CODE');
    });

    it('should get error name from regular error', () => {
      const error = new TypeError('Type error');
      expect(getErrorCode(error)).toBe('TypeError');
    });

    it('should return unknown for non-error values', () => {
      expect(getErrorCode('string')).toBe('UNKNOWN_ERROR');
      expect(getErrorCode(null)).toBe('UNKNOWN_ERROR');
    });
  });

  describe('toPRSMError', () => {
    it('should return PRSM error as-is', () => {
      const error = new PRSMError('Test');
      expect(toPRSMError(error)).toBe(error);
    });

    it('should wrap regular error', () => {
      const error = new Error('Regular error');
      const wrapped = toPRSMError(error);
      
      expect(wrapped).toBeInstanceOf(PRSMError);
      expect(wrapped.message).toBe('Regular error');
      expect(wrapped.code).toBe('WRAPPED_ERROR');
    });

    it('should handle non-error values', () => {
      const wrapped = toPRSMError('string error');
      
      expect(wrapped).toBeInstanceOf(PRSMError);
      expect(wrapped.code).toBe('UNKNOWN_ERROR');
    });
  });

  describe('isRetryableError', () => {
    it('should identify retryable errors', () => {
      const networkError = new NetworkError('Connection failed');
      const authError = new AuthenticationError('Invalid token');
      
      expect(isRetryableError(networkError)).toBe(true);
      expect(isRetryableError(authError)).toBe(false);
    });
  });

  describe('getRetryDelay', () => {
    it('should calculate exponential backoff', () => {
      const error = new NetworkError('Connection failed');
      
      const delay1 = getRetryDelay(error, 1);
      const delay2 = getRetryDelay(error, 2);
      const delay3 = getRetryDelay(error, 3);
      
      expect(delay1).toBe(2000); // 2^1 * 1000
      expect(delay2).toBe(4000); // 2^2 * 1000
      expect(delay3).toBe(8000); // 2^3 * 1000
    });

    it('should cap maximum delay', () => {
      const error = new NetworkError('Connection failed');
      const delay = getRetryDelay(error, 10);
      
      expect(delay).toBe(30000); // Capped at 30 seconds
    });
  });
});