/**
 * Jest setup file for PRSM SDK tests
 */

import { TextEncoder, TextDecoder } from 'util';

// Polyfill for TextEncoder/TextDecoder in Node.js
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Mock WebSocket for Node.js environment
global.WebSocket = class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen?: (event: Event) => void;
  onclose?: (event: CloseEvent) => void;
  onmessage?: (event: MessageEvent) => void;
  onerror?: (event: Event) => void;

  constructor(url: string) {
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.(new Event('open'));
    }, 10);
  }

  send(data: string): void {
    // Mock send
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent('close', { code, reason }));
  }
};

// Mock fetch for Node.js environment
global.fetch = jest.fn();

// Default test configuration
const TEST_CONFIG = {
  API_KEY: 'test_api_key_12345',
  BASE_URL: 'https://api.test.prsm.org',
  WEBSOCKET_URL: 'wss://api.test.prsm.org/ws',
  TIMEOUT: 5000,
};

// Make test config available globally
(global as any).TEST_CONFIG = TEST_CONFIG;

// Increase test timeout for integration tests
jest.setTimeout(30000);

// Mock console methods for cleaner test output
const originalConsole = console;
beforeAll(() => {
  console.log = jest.fn();
  console.info = jest.fn();
  console.warn = jest.fn();
  // Keep console.error for debugging
});

afterAll(() => {
  console.log = originalConsole.log;
  console.info = originalConsole.info;
  console.warn = originalConsole.warn;
});