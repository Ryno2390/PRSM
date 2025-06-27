import { GoogleGenerativeAI } from '@google/generative-ai';

export interface LLMResponse {
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
  };
  model: string;
  provider: 'gemini';
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export class GeminiClient {
  private client: GoogleGenerativeAI;

  constructor(apiKey: string) {
    this.client = new GoogleGenerativeAI(apiKey);
  }

  async generateResponse(
    messages: ChatMessage[],
    systemPrompt: string,
    model: string = 'gemini-2.5-pro'
  ): Promise<LLMResponse> {
    try {
      const genModel = this.client.getGenerativeModel({ 
        model,
        generationConfig: {
          temperature: 0.1, // Low temperature for consistent, factual responses
          topP: 0.8,
          topK: 40,
          maxOutputTokens: 4000,
        },
        systemInstruction: systemPrompt
      });

      // Convert messages to Gemini format
      const history = messages.slice(0, -1).map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }]
      }));

      const lastMessage = messages[messages.length - 1];
      
      const chat = genModel.startChat({ history });
      const result = await chat.sendMessage(lastMessage.content);
      const response = await result.response;

      return {
        content: response.text(),
        usage: {
          inputTokens: response.usageMetadata?.promptTokenCount || 0,
          outputTokens: response.usageMetadata?.candidatesTokenCount || 0
        },
        model: model,
        provider: 'gemini'
      };
    } catch (error) {
      console.error('Gemini API Error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Gemini API request failed: ${errorMessage}`);
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
      console.error('Gemini connection test failed:', error);
      return false;
    }
  }
}