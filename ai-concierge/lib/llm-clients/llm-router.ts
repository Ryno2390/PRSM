import { ClaudeClient, LLMResponse as ClaudeResponse, ChatMessage } from './claude-client';
import { GeminiClient, LLMResponse as GeminiResponse } from './gemini-client';

export type { ChatMessage } from './claude-client';

export type LLMProvider = 'claude' | 'gemini' | 'openai';

export interface LLMResponse {
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
  model: string;
  provider: LLMProvider;
  responseTime: number;
}

export interface LLMConfig {
  provider: LLMProvider;
  model?: string;
  fallback?: LLMProvider[];
}

export class LLMRouter {
  private claudeClient?: ClaudeClient;
  private geminiClient?: GeminiClient;
  private config: LLMConfig;

  constructor(config: LLMConfig, apiKeys: { claude?: string; gemini?: string; openai?: string }) {
    this.config = config;

    // Initialize available clients
    if (apiKeys.claude) {
      this.claudeClient = new ClaudeClient(apiKeys.claude);
    }
    if (apiKeys.gemini) {
      this.geminiClient = new GeminiClient(apiKeys.gemini);
    }
  }

  async generateResponse(
    messages: ChatMessage[],
    systemPrompt: string,
    options?: {
      maxRetries?: number;
      timeoutMs?: number;
    }
  ): Promise<LLMResponse> {
    const startTime = Date.now();
    const providers = [this.config.provider, ...(this.config.fallback || [])];
    
    for (let i = 0; i < providers.length; i++) {
      const provider = providers[i];
      
      try {
        const response = await this.callProvider(provider, messages, systemPrompt);
        return {
          ...response,
          responseTime: Date.now() - startTime
        };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error(`Provider ${provider} failed:`, errorMessage);
        
        // If this is the last provider, throw the error
        if (i === providers.length - 1) {
          throw new Error(`All LLM providers failed. Last error: ${errorMessage}`);
        }
        
        // Otherwise, continue to next provider
        console.log(`Falling back to next provider...`);
      }
    }

    throw new Error('No LLM providers available');
  }

  private async callProvider(
    provider: LLMProvider,
    messages: ChatMessage[],
    systemPrompt: string
  ): Promise<LLMResponse> {
    switch (provider) {
      case 'claude':
        if (!this.claudeClient) {
          throw new Error('Claude client not initialized');
        }
        const claudeResponse = await this.claudeClient.generateResponse(
          messages,
          systemPrompt,
          this.config.model || 'claude-3-5-sonnet-20241022'
        );
        return claudeResponse as LLMResponse;

      case 'gemini':
        if (!this.geminiClient) {
          throw new Error('Gemini client not initialized');
        }
        const geminiResponse = await this.geminiClient.generateResponse(
          messages,
          systemPrompt,
          this.config.model || 'gemini-2.5-pro'
        );
        return geminiResponse as LLMResponse;

      case 'openai':
        throw new Error('OpenAI client not yet implemented');

      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  async testConnections(): Promise<{ [key in LLMProvider]?: boolean }> {
    const results: { [key in LLMProvider]?: boolean } = {};

    if (this.claudeClient) {
      results.claude = await this.claudeClient.testConnection();
    }
    if (this.geminiClient) {
      results.gemini = await this.geminiClient.testConnection();
    }

    return results;
  }

  getAvailableProviders(): LLMProvider[] {
    const providers: LLMProvider[] = [];
    if (this.claudeClient) providers.push('claude');
    if (this.geminiClient) providers.push('gemini');
    return providers;
  }
}