/**
 * PRSM JavaScript/TypeScript SDK
 * Official client library for the Protocol for Recursive Scientific Modeling
 * 
 * @example
 * ```typescript
 * import { PRSMClient } from '@prsm/sdk';
 * 
 * const client = new PRSMClient({ apiKey: 'your_api_key' });
 * const response = await client.query('Explain quantum computing');
 * console.log(response.content);
 * ```
 */

export { PRSMClient } from './client';
export type {
  PRSMResponse,
  QueryRequest,
  FTNSBalance,
  ModelInfo,
  ToolSpec,
  SafetyStatus,
  PRSMClientConfig,
  StreamChunk
} from './types';
export {
  ModelProvider,
  SafetyLevel
} from './types';
export {
  PRSMError,
  AuthenticationError,
  InsufficientFundsError,
  SafetyViolationError,
  NetworkError,
  ModelNotFoundError,
  ToolExecutionError,
  RateLimitError,
  ValidationError
} from './errors';
export { AuthManager } from './auth';
export { FTNSManager } from './ftns';
export { ModelMarketplace } from './marketplace';
export { ToolExecutor } from './tools';

// Version information
export const VERSION = '0.1.0';