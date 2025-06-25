module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  
  // E2E test patterns
  testMatch: [
    '**/tests/e2e/**/*.test.ts'
  ],
  
  // Setup files for E2E
  setupFilesAfterEnv: ['<rootDir>/tests/e2e/setup.ts'],
  
  // Longer timeout for E2E tests
  testTimeout: 120000,
  
  // Run tests serially for E2E
  maxWorkers: 1,
  
  // Module path mapping
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  
  // Transform configuration
  transform: {
    '^.+\\.ts$': 'ts-jest'
  },
  
  // Extensions to consider
  moduleFileExtensions: ['ts', 'js', 'json'],
  
  // Verbose output
  verbose: true,
  
  // Clear mocks automatically
  clearMocks: true,
  
  // Global setup/teardown for E2E
  globalSetup: '<rootDir>/tests/e2e/globalSetup.ts',
  globalTeardown: '<rootDir>/tests/e2e/globalTeardown.ts',
  
  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/coverage/',
    '/tests/unit/'
  ]
};