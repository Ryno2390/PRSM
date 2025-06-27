import Anthropic from '@anthropic-ai/sdk';

export interface LLMResponse {
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
  model: string;
  provider: 'claude';
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export class ClaudeClient {
  private client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({
      apiKey: apiKey,
    });
  }

  async generateResponse(
    messages: ChatMessage[],
    systemPrompt: string,
    model: string = 'claude-3-5-sonnet-20241022'
  ): Promise<LLMResponse> {
    try {
      // Convert messages to Claude format (system prompt separate)
      const userMessages = messages.filter(m => m.role !== 'system');
      
      const response = await this.client.messages.create({
        model: model,
        max_tokens: 4000,
        temperature: 0.1, // Low temperature for consistent, factual responses
        system: systemPrompt,
        messages: userMessages.map(msg => ({
          role: msg.role as 'user' | 'assistant',
          content: msg.content
        }))
      });

      const content = response.content[0];
      if (content.type !== 'text') {
        throw new Error('Unexpected response type from Claude');
      }

      return {
        content: content.text,
        usage: {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens
        },
        model: response.model,
        provider: 'claude'
      };
    } catch (error) {
      console.error('Claude API Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Claude API request failed: ${errorMessage}`);
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.generateResponse(
        [{ role: 'user', content: 'Test connection' }],
        'You are a test assistant. Respond with "Connection successful".'
      );
      return true;
    } catch (error) {
      console.error('Claude connection test failed:', error);
      return false;
    }
  }
}