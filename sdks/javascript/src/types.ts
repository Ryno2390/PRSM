/**
 * TypeScript type definitions for PRSM SDK
 */

export enum ModelProvider {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  HUGGINGFACE = 'huggingface',
  LOCAL = 'local',
  PRSM_DISTILLED = 'prsm_distilled'
}

export enum SafetyLevel {
  NONE = 'none',
  LOW = 'low',
  MODERATE = 'moderate',
  HIGH = 'high',
  CRITICAL = 'critical',
  EMERGENCY = 'emergency'
}

export interface PRSMClientConfig {
  apiKey?: string;
  baseUrl?: string;
  websocketUrl?: string;
  timeout?: number;
  maxRetries?: number;
  headers?: Record<string, string>;
}

export interface QueryRequest {
  prompt: string;
  modelId?: string;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  context?: Record<string, any>;
  tools?: string[];
  safetyLevel?: SafetyLevel;
}

export interface PRSMResponse {
  content: string;
  modelId: string;
  provider: ModelProvider;
  executionTime: number;
  tokenUsage: Record<string, number>;
  ftnsCost: number;
  reasoningTrace?: string[];
  safetyStatus: SafetyLevel;
  metadata: Record<string, any>;
  requestId: string;
  timestamp: string;
}

export interface FTNSBalance {
  totalBalance: number;
  availableBalance: number;
  reservedBalance: number;
  earnedToday: number;
  spentToday: number;
  lastUpdated: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: ModelProvider;
  description: string;
  capabilities: string[];
  costPerToken: number;
  maxTokens: number;
  contextWindow: number;
  isAvailable: boolean;
  performanceRating: number;
  safetyRating: number;
  createdAt: string;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, any>;
  costPerExecution: number;
  safetyLevel: SafetyLevel;
  provider: string;
  version: string;
}

export interface SafetyStatus {
  overallStatus: SafetyLevel;
  activeMonitors: number;
  threatsDetected: number;
  circuitBreakersTriggered: number;
  lastAssessment: string;
  networkHealth: number;
}

export interface StreamChunk {
  content: string;
  isComplete: boolean;
  metadata?: Record<string, any>;
}

export interface MarketplaceQuery {
  query: string;
  provider?: ModelProvider;
  maxCost?: number;
  minPerformance?: number;
  capabilities?: string[];
  limit?: number;
}

export interface ToolExecutionRequest {
  toolName: string;
  parameters: Record<string, any>;
  context?: Record<string, any>;
  safetyLevel?: SafetyLevel;
}

export interface ToolExecutionResponse {
  result: any;
  executionTime: number;
  ftnsCost: number;
  safetyStatus: SafetyLevel;
  success: boolean;
  error?: string;
  metadata: Record<string, any>;
}

export interface WebSocketMessage {
  type: string;
  data: Record<string, any>;
  requestId?: string;
  timestamp: string;
}