/**
 * Global setup for Jest tests
 */

export default async function globalSetup(): Promise<void> {
  console.log('🚀 Starting PRSM SDK tests...\n');
  
  // Set environment variables for tests
  process.env.NODE_ENV = 'test';
  process.env.PRSM_API_KEY = 'test_api_key_12345';
  process.env.PRSM_BASE_URL = 'https://api.test.prsm.org';
  
  // Disable network requests during tests
  process.env.PRSM_DISABLE_NETWORK = 'true';
}