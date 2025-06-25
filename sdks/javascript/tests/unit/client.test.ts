/**
 * Unit tests for PRSM Client
 */

import { PRSMClient } from '../../src/client';
import { PRSMClientConfig } from '../../src/types';
import { ConfigurationError } from '../../src/errors';

describe('PRSMClient', () => {
  const mockConfig: PRSMClientConfig = {
    apiKey: (global as any).TEST_CONFIG.API_KEY,
    baseUrl: (global as any).TEST_CONFIG.BASE_URL,
    websocketUrl: (global as any).TEST_CONFIG.WEBSOCKET_URL,
    timeout: 5000,
    debug: false,
  };

  beforeEach(() => {
    // Reset fetch mock
    (global.fetch as jest.Mock).mockReset();
  });

  describe('Constructor', () => {
    it('should create client with valid configuration', () => {
      const client = new PRSMClient(mockConfig);
      
      expect(client).toBeInstanceOf(PRSMClient);
      expect(client.baseUrl).toBe(mockConfig.baseUrl);
      expect(client.timeout).toBe(mockConfig.timeout);
      expect(client.debug).toBe(mockConfig.debug);
    });

    it('should use default configuration when not provided', () => {
      const client = new PRSMClient({ apiKey: 'test_key' });
      
      expect(client.baseUrl).toBe('https://api.prsm.org');
      expect(client.timeout).toBe(30000);
      expect(client.debug).toBe(false);
    });

    it('should throw error for invalid timeout', () => {
      expect(() => {
        new PRSMClient({
          ...mockConfig,
          timeout: 500, // Too short
        });
      }).toThrow(ConfigurationError);
    });

    it('should throw error for invalid maxRetries', () => {
      expect(() => {
        new PRSMClient({
          ...mockConfig,
          maxRetries: 15, // Too many
        });
      }).toThrow(ConfigurationError);
    });
  });

  describe('Authentication', () => {
    it('should check authentication status', () => {
      const client = new PRSMClient(mockConfig);
      
      // Initially not authenticated (no token)
      expect(client.isAuthenticated()).toBe(false);
    });

    it('should get auth headers with API key', async () => {
      const client = new PRSMClient(mockConfig);
      const headers = await client.getAuthHeaders();
      
      expect(headers['X-API-Key']).toBe(mockConfig.apiKey);
    });
  });

  describe('Quick Query Method', () => {
    it('should perform quick query with default options', async () => {
      // Mock successful session creation and completion
      const mockSession = {
        sessionId: 'sess_123',
        status: 'queued',
        query: 'Test query',
        createdAt: new Date().toISOString(),
      };

      const mockCompletedSession = {
        ...mockSession,
        status: 'completed',
        results: {
          summary: 'Test response',
          keyFindings: ['Finding 1', 'Finding 2'],
          citations: [
            {
              title: 'Test Paper',
              authors: ['Author 1'],
              year: 2024,
              confidence: 0.9,
            },
          ],
          confidenceScore: 0.95,
        },
        costActual: {
          ftnsTokens: 5.5,
          usdEquivalent: 0.55,
        },
      };

      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: mockSession }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: mockCompletedSession }),
        });

      const client = new PRSMClient(mockConfig);
      const result = await client.query('Test query');

      expect(result.content).toBe('Test response');
      expect(result.citations).toHaveLength(1);
      expect(result.ftnsCost).toBe(5.5);
    });

    it('should handle query with custom options', async () => {
      const client = new PRSMClient(mockConfig);
      
      const mockSession = {
        sessionId: 'sess_456',
        status: 'queued',
        query: 'Custom query',
        createdAt: new Date().toISOString(),
      };

      const mockCompletedSession = {
        ...mockSession,
        status: 'completed',
        results: {
          summary: 'Custom response',
          keyFindings: [],
          citations: [],
          confidenceScore: 0.85,
        },
      };

      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: mockSession }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: mockCompletedSession }),
        });

      const result = await client.query('Custom query', {
        domain: 'computer_science',
        maxIterations: 5,
        sealEnhancement: {
          enabled: true,
          autonomousImprovement: false,
          targetLearningGain: 0.1,
          restemMethodology: true,
        },
      });

      expect(result.content).toBe('Custom response');
    });
  });

  describe('Status and Health', () => {
    it('should get client status', () => {
      const client = new PRSMClient(mockConfig);
      const status = client.getStatus();

      expect(status).toHaveProperty('isAuthenticated');
      expect(status).toHaveProperty('isWebSocketConnected');
      expect(status).toHaveProperty('baseUrl');
      expect(status).toHaveProperty('websocketUrl');
      expect(status.baseUrl).toBe(mockConfig.baseUrl);
    });
  });

  describe('Lifecycle Management', () => {
    it('should initialize client successfully', async () => {
      const client = new PRSMClient(mockConfig);

      // Mock health check
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'healthy',
          version: '1.0.0',
          components: {},
        }),
      });

      await expect(client.initialize()).resolves.toBeUndefined();
    });

    it('should destroy client cleanly', async () => {
      const client = new PRSMClient(mockConfig);

      await expect(client.destroy()).resolves.toBeUndefined();
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors gracefully', async () => {
      const client = new PRSMClient(mockConfig);

      (global.fetch as jest.Mock).mockRejectedValueOnce(
        new Error('Network error')
      );

      await expect(
        client.query('Test query')
      ).rejects.toThrow();
    });

    it('should retry failed requests', async () => {
      const client = new PRSMClient({
        ...mockConfig,
        maxRetries: 2,
      });

      const mockSession = {
        sessionId: 'sess_retry',
        status: 'queued',
        query: 'Retry query',
        createdAt: new Date().toISOString(),
      };

      // First call fails, second succeeds
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Temporary error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: mockSession }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            data: {
              ...mockSession,
              status: 'completed',
              results: {
                summary: 'Retry success',
                keyFindings: [],
                citations: [],
                confidenceScore: 0.9,
              },
            },
          }),
        });

      const result = await client.query('Retry query');
      expect(result.content).toBe('Retry success');
    });
  });
});