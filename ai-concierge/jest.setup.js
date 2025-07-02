/**
 * Jest Setup for PRSM AI Concierge
 * 
 * Global test configuration and mocks for testing environment
 */

// Import jest-dom for additional matchers
import '@testing-library/jest-dom'

// Mock Next.js router
jest.mock('next/router', () => ({
  useRouter() {
    return {
      route: '/',
      pathname: '/',
      query: {},
      asPath: '/',
      push: jest.fn(),
      pop: jest.fn(),
      reload: jest.fn(),
      back: jest.fn(),
      prefetch: jest.fn(),
      beforePopState: jest.fn(),
      events: {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
      }
    }
  }
}))

// Mock environment variables
process.env.NODE_ENV = 'test'
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:3000'

// Mock console methods to reduce noise in tests
const originalConsoleError = console.error
console.error = (...args) => {
  // Suppress specific React warnings in tests
  if (
    typeof args[0] === 'string' &&
    args[0].includes('Warning: ReactDOM.render is no longer supported')
  ) {
    return
  }
  originalConsoleError.call(console, ...args)
}

// Global test utilities
global.testUtils = {
  // Helper for testing async components
  waitFor: (callback, options = {}) => 
    new Promise((resolve) => {
      const timeout = options.timeout || 1000
      const interval = options.interval || 10
      const start = Date.now()
      
      const check = () => {
        try {
          const result = callback()
          if (result) {
            resolve(result)
            return
          }
        } catch (error) {
          // Continue checking
        }
        
        if (Date.now() - start > timeout) {
          throw new Error(`Timeout after ${timeout}ms`)
        }
        
        setTimeout(check, interval)
      }
      
      check()
    }),
    
  // Helper for creating mock API responses
  createMockResponse: (data, status = 200) => ({
    ok: status < 400,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data))
  })
}

// Mock fetch for API testing
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
  })
)

// Reset all mocks before each test
beforeEach(() => {
  jest.clearAllMocks()
  fetch.mockClear()
})

// Clean up after each test
afterEach(() => {
  jest.restoreAllMocks()
})