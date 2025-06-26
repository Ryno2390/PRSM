#!/usr/bin/env node

/**
 * PRSM AI Concierge - Prompt Testing Suite
 * 
 * Tests the concierge against the comprehensive FAQ dataset to validate
 * response accuracy, tone, and escalation handling.
 */

// Load environment variables
import { config } from 'dotenv';
config();

import { LLMRouter } from '../lib/llm-clients/llm-router';
import { ConciergeEngine } from '../lib/prompt-engine/concierge-engine';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface TestCase {
  id: string;
  category: string;
  question: string;
  expectedKeywords?: string[];
  expectedSources?: string[];
  shouldEscalate?: boolean;
}

interface TestScore {
  accuracy: number;
  completeness: number;
  tone: number;
  sources: number;
  escalation: number;
  total: number;
}

interface TestResult {
  testCase: TestCase;
  response?: any;
  error?: string;
  passed: boolean;
  score: TestScore;
  responseTime?: number;
}

// Test questions from our FAQ dataset
const TEST_QUESTIONS: TestCase[] = [
  {
    id: 'funding_status',
    category: 'funding',
    question: "What's PRSM's current funding status and needs?",
    expectedKeywords: ['99/100', 'external validation', '$2-4M', 'seed round', '$10-15M', 'Series A'],
    expectedSources: ['INVESTMENT_READINESS_REPORT.md']
  },
  {
    id: 'security_posture',
    category: 'security',
    question: "What's PRSM's security posture?",
    expectedKeywords: ['100% security compliance', 'zero vulnerabilities', '31 → 0', 'enterprise-grade'],
    expectedSources: ['INVESTMENT_READINESS_REPORT.md', 'SECURITY_ARCHITECTURE.md']
  },
  {
    id: 'technical_differentiation',
    category: 'technical',
    question: "What makes PRSM technically differentiated?",
    expectedKeywords: ['RLT', '100% component success', 'P2P federation', 'post-quantum'],
    expectedSources: ['TECHNICAL_ADVANTAGES.md']
  },
  {
    id: 'business_model',
    category: 'business',
    question: "What's the business model and revenue strategy?",
    expectedKeywords: ['FTNS token', 'transaction fees', 'enterprise licensing', 'marketplace revenue'],
    expectedSources: ['BUSINESS_CASE.md', 'tokenomics.md']
  },
  {
    id: 'team_execution',
    category: 'team',
    question: "What evidence exists of technical execution?",
    expectedKeywords: ['Phase 1', 'Phase 2', 'Phase 3', '100% RLT', 'remarkable progress'],
    expectedSources: ['PHASE_1_TASK_1_EVIDENCE_REPORT.md']
  },
  {
    id: 'escalation_test',
    category: 'escalation',
    question: "What are PRSM's legal exposures and regulatory risks?",
    expectedKeywords: ['legal team', 'consultation', 'compliance framework'],
    shouldEscalate: true
  }
];

class PromptTester {
  private results: {
    totalTests: number;
    passed: number;
    failed: number;
    scores: TestScore[];
    details: TestResult[];
  };

  constructor() {
    this.results = {
      totalTests: 0,
      passed: 0,
      failed: 0,
      scores: [],
      details: []
    };
  }

  async runTests(): Promise<void> {
    console.log('🧪 Starting PRSM Concierge Prompt Testing...\n');

    // Initialize concierge engine
    const engine = await this.initializeEngine();
    
    for (const testCase of TEST_QUESTIONS) {
      console.log(`Testing: ${testCase.question}`);
      await this.runSingleTest(engine, testCase);
      console.log(''); // Empty line for readability
    }

    this.printSummary();
    await this.saveResults();
  }

  private async initializeEngine(): Promise<ConciergeEngine | MockEngine> {
    // Configure with environment variables or fallback to test mode
    const apiKeys = {
      claude: process.env.ANTHROPIC_API_KEY,
      gemini: process.env.GOOGLE_API_KEY,
      openai: process.env.OPENAI_API_KEY
    };

    // Check if we have any API keys
    const hasApiKeys = Object.values(apiKeys).some(key => !!key);
    
    if (!hasApiKeys) {
      console.log('⚠️  No API keys found. Running in simulation mode...\n');
      return this.createMockEngine();
    }

    const llmConfig = {
      provider: 'claude' as const,
      fallback: ['gemini' as const]
    };

    const llmRouter = new LLMRouter(llmConfig, apiKeys);
    const engine = new ConciergeEngine(llmRouter);

    // Load knowledge base
    const knowledgeBasePath = path.resolve(__dirname, '../knowledge-base/compiled-knowledge.json');
    await engine.loadKnowledgeBase(knowledgeBasePath);

    return engine;
  }

  private createMockEngine(): MockEngine {
    // Create a mock engine for testing without API keys
    return new MockEngine();
  }

  private async runSingleTest(engine: ConciergeEngine | MockEngine, testCase: TestCase): Promise<void> {
    this.results.totalTests++;
    
    try {
      const startTime = Date.now();
      const response = await engine.processInvestorQuery(testCase.question);
      const endTime = Date.now();

      const score = this.scoreResponse(response, testCase);
      this.results.scores.push(score);

      const passed = score.total >= 20; // 4.0 average minimum
      if (passed) {
        this.results.passed++;
        console.log(`  ✅ PASSED (${score.total}/25)`);
      } else {
        this.results.failed++;
        console.log(`  ❌ FAILED (${score.total}/25)`);
      }

      const result: TestResult = {
        testCase,
        response,
        score,
        passed,
        responseTime: endTime - startTime
      };

      this.results.details.push(result);
      this.printTestDetails(result);

    } catch (error: any) {
      this.results.failed++;
      console.log(`  ❌ ERROR: ${error.message}`);
      
      this.results.details.push({
        testCase,
        error: error.message,
        passed: false,
        score: { accuracy: 0, completeness: 0, tone: 0, sources: 0, escalation: 0, total: 0 }
      });
    }
  }

  private scoreResponse(response: any, testCase: TestCase): TestScore {
    const score: TestScore = {
      accuracy: 0,
      completeness: 0,
      tone: 0,
      sources: 0,
      escalation: 0,
      total: 0
    };

    // Accuracy: Check for expected keywords
    if (testCase.expectedKeywords) {
      const foundKeywords = testCase.expectedKeywords.filter(keyword =>
        response.content.toLowerCase().includes(keyword.toLowerCase())
      );
      score.accuracy = Math.min(5, (foundKeywords.length / testCase.expectedKeywords.length) * 5);
    } else {
      score.accuracy = 4; // Default for non-keyword tests
    }

    // Completeness: Response length and detail
    const wordCount = response.content.split(/\s+/).length;
    if (wordCount >= 100) score.completeness = 5;
    else if (wordCount >= 50) score.completeness = 4;
    else if (wordCount >= 25) score.completeness = 3;
    else score.completeness = 2;

    // Tone: Professional and authoritative
    const content = response.content.toLowerCase();
    if (content.includes('source:') || content.includes('based on')) score.tone += 2;
    if (!content.includes('i think') && !content.includes('maybe')) score.tone += 2;
    if (content.includes('prsm') && !content.includes('uncertain')) score.tone += 1;

    // Sources: Proper attribution
    if (response.sourceReferences && response.sourceReferences.length > 0) {
      score.sources = Math.min(5, response.sourceReferences.length * 2);
    }

    // Escalation: Appropriate escalation handling
    if (testCase.shouldEscalate) {
      score.escalation = response.escalationSuggested ? 5 : 1;
    } else {
      score.escalation = !response.escalationSuggested ? 5 : 3;
    }

    score.total = score.accuracy + score.completeness + score.tone + score.sources + score.escalation;
    return score;
  }

  private printTestDetails(result: TestResult): void {
    if (result.error) {
      console.log(`    Error: ${result.error}`);
      return;
    }

    const { response, score } = result;
    
    console.log(`    Accuracy: ${score.accuracy}/5 | Completeness: ${score.completeness}/5 | Tone: ${score.tone}/5`);
    console.log(`    Sources: ${score.sources}/5 | Escalation: ${score.escalation}/5`);
    console.log(`    Confidence: ${response.confidence} | Response Time: ${result.responseTime}ms`);
    
    if (response.sourceReferences && response.sourceReferences.length > 0) {
      console.log(`    Sources: ${response.sourceReferences.join(', ')}`);
    }
    
    if (response.escalationSuggested) {
      console.log(`    🔄 Escalation suggested`);
    }
  }

  private printSummary(): void {
    console.log('📊 Test Results Summary');
    console.log('========================');
    console.log(`Total Tests: ${this.results.totalTests}`);
    console.log(`Passed: ${this.results.passed} ✅`);
    console.log(`Failed: ${this.results.failed} ❌`);
    console.log(`Success Rate: ${((this.results.passed / this.results.totalTests) * 100).toFixed(1)}%`);
    
    if (this.results.scores.length > 0) {
      const avgScore = this.results.scores.reduce((sum, score) => sum + score.total, 0) / this.results.scores.length;
      console.log(`Average Score: ${avgScore.toFixed(1)}/25 (${(avgScore/25*100).toFixed(1)}%)`);
      
      const targetMet = avgScore >= 20; // 4.0 average
      console.log(`Production Ready: ${targetMet ? '✅ YES' : '❌ NO'} (target: 20/25)`);
    }
  }

  private async saveResults(): Promise<void> {
    const resultsDir = path.resolve(__dirname, '../test-results');
    await fs.ensureDir(resultsDir);
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `prompt-test-results-${timestamp}.json`;
    const filepath = path.resolve(resultsDir, filename);
    
    await fs.writeJson(filepath, {
      timestamp: new Date().toISOString(),
      summary: {
        totalTests: this.results.totalTests,
        passed: this.results.passed,
        failed: this.results.failed,
        successRate: (this.results.passed / this.results.totalTests) * 100,
        averageScore: this.results.scores.reduce((sum, score) => sum + score.total, 0) / this.results.scores.length
      },
      details: this.results.details
    }, { spaces: 2 });
    
    console.log(`\n💾 Results saved to: ${filename}`);
  }
}

// Mock engine for testing without API keys
class MockEngine {
  async processInvestorQuery(query: string): Promise<any> {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      content: `This is a simulated response for: "${query}". In a real implementation, this would contain comprehensive information from PRSM's knowledge base with specific metrics, achievements, and source references.`,
      sourceReferences: ['INVESTMENT_READINESS_REPORT.md'],
      confidence: 'medium',
      escalationSuggested: query.toLowerCase().includes('legal'),
      responseMetadata: {
        provider: 'mock',
        model: 'test-model',
        responseTime: 500,
        tokensUsed: 100
      }
    };
  }
}

// Run tests if called directly
async function main() {
  try {
    const tester = new PromptTester();
    await tester.runTests();
  } catch (error: any) {
    console.error('❌ Testing failed:', error);
    process.exit(1);
  }
}

// Check if this is the main module (ES module equivalent)
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}