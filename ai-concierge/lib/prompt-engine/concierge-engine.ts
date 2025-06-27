import { LLMRouter, ChatMessage, LLMResponse } from '../llm-clients/llm-router';
import fs from 'fs-extra';
import path from 'path';

export interface KnowledgeDocument {
  id: string;
  path: string;
  filename?: string;
  category: string;
  title?: string;
  type?: string;
  content: string;
  metadata: {
    wordCount?: number;
    words?: number;
    lines?: number;
    size?: number;
    lastModified: string;
    fileSize?: number;
    encoding?: string;
  };
  codeAnalysis?: {
    language: string;
    lines: number;
    functions: string[];
    classes: string[];
    imports: string[];
    complexity: string;
  };
}

export interface KnowledgeBase {
  metadata: {
    compiledAt: string;
    version: string;
    totalDocuments: number;
    totalLines?: number;
    totalSize?: number;
    coverage?: string;
  };
  documents: { [id: string]: KnowledgeDocument };
  categories: { [category: string]: string[] };
}

export interface ConciergeResponse {
  content: string;
  sourceReferences: string[];
  confidence: 'high' | 'medium' | 'low';
  escalationSuggested: boolean;
  responseMetadata: {
    provider: string;
    model: string;
    responseTime: number;
    tokensUsed: number;
  };
}

export class ConciergeEngine {
  private llmRouter: LLMRouter;
  private knowledgeBase: KnowledgeBase | null = null;
  private conversationHistory: ChatMessage[] = [];

  constructor(llmRouter: LLMRouter) {
    this.llmRouter = llmRouter;
  }

  async loadKnowledgeBase(knowledgeBasePath: string): Promise<void> {
    try {
      this.knowledgeBase = await fs.readJson(knowledgeBasePath);
      const totalDocs = this.knowledgeBase?.metadata.totalDocuments || 0;
      const coverage = this.knowledgeBase?.metadata.coverage || 'standard';
      const totalLines = this.knowledgeBase?.metadata.totalLines || 0;
      console.log(`✅ Loaded ${coverage} knowledge base: ${totalDocs} documents, ${totalLines.toLocaleString()} lines`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Failed to load knowledge base: ${errorMessage}`);
    }
  }

  async processInvestorQuery(
    query: string,
    options?: {
      includeHistory?: boolean;
      maxContextDocs?: number;
    }
  ): Promise<ConciergeResponse> {
    if (!this.knowledgeBase) {
      throw new Error('Knowledge base not loaded. Call loadKnowledgeBase() first.');
    }

    // Find relevant documents
    const relevantDocs = this.findRelevantDocuments(query, options?.maxContextDocs || 10);
    
    // Build system prompt with knowledge context
    const systemPrompt = this.buildSystemPrompt(relevantDocs);
    
    // Prepare conversation messages
    const messages: ChatMessage[] = [];
    
    if (options?.includeHistory && this.conversationHistory.length > 0) {
      messages.push(...this.conversationHistory.slice(-6)); // Last 3 exchanges
    }
    
    messages.push({ role: 'user', content: query });

    try {
      // Generate response using LLM
      const llmResponse = await this.llmRouter.generateResponse(messages, systemPrompt);
      
      // Analyze response for escalation triggers
      const escalationSuggested = this.shouldEscalate(query, llmResponse.content);
      
      // Extract source references
      const sourceReferences = this.extractSourceReferences(llmResponse.content, relevantDocs);
      
      // Determine confidence level
      const confidence = this.assessConfidence(llmResponse.content, relevantDocs);

      // Update conversation history
      this.conversationHistory.push(
        { role: 'user', content: query },
        { role: 'assistant', content: llmResponse.content }
      );

      return {
        content: llmResponse.content,
        sourceReferences,
        confidence,
        escalationSuggested,
        responseMetadata: {
          provider: llmResponse.provider,
          model: llmResponse.model,
          responseTime: llmResponse.responseTime,
          tokensUsed: (llmResponse.usage?.inputTokens || 0) + (llmResponse.usage?.outputTokens || 0)
        }
      };

    } catch (error) {
      console.error('Error generating concierge response:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(`Failed to generate response: ${errorMessage}`);
    }
  }

  private findRelevantDocuments(query: string, maxDocs: number): KnowledgeDocument[] {
    if (!this.knowledgeBase) return [];

    const queryLower = query.toLowerCase();
    const documents = Object.values(this.knowledgeBase.documents);
    
    // Enhanced scoring for comprehensive knowledge base
    const scoredDocs = documents.map(doc => {
      let score = 0;
      const content = doc.content.toLowerCase();
      const title = (doc.title || doc.filename || '').toLowerCase();
      const filepath = doc.path.toLowerCase();
      
      // Keyword matching with enhanced weights
      const investmentKeywords = [
        'investment', 'funding', 'series a', 'revenue', 'business model',
        'valuation', 'financial', 'roi', 'market', 'strategy'
      ];
      const technicalKeywords = [
        'security', 'technical', 'architecture', 'performance', 'scalability',
        'api', 'code', 'implementation', 'algorithm', 'system'
      ];
      const evidenceKeywords = [
        'team', 'execution', 'validation', 'evidence', 'metrics',
        'test', 'result', 'proof', 'benchmark', 'audit'
      ];
      const partnershipKeywords = [
        'partnership', 'apple', 'collaboration', 'integration', 'alliance'
      ];
      
      // Score by keyword categories
      [investmentKeywords, technicalKeywords, evidenceKeywords, partnershipKeywords].forEach((keywords, idx) => {
        const categoryWeight = [25, 20, 15, 10][idx]; // Investment gets highest weight
        keywords.forEach(keyword => {
          if (queryLower.includes(keyword)) {
            if (title.includes(keyword)) score += categoryWeight;
            if (filepath.includes(keyword)) score += categoryWeight * 0.8;
            if (content.includes(keyword)) score += categoryWeight * 0.6;
          }
        });
      });
      
      // Enhanced category-based scoring for comprehensive knowledge
      const categoryScores: { [key: string]: number } = {
        'documentation': 15,
        'tier1_essential': 20,
        'tier2_supporting': 12,
        'tier3_detailed': 8,
        'source_code': 10,
        'tests': 8,
        'evidence': 15,
        'configuration': 5,
        'contracts': 12,
        'scripts': 5,
        'infrastructure': 6
      };
      score += categoryScores[doc.category] || 5;
      
      // Boost for specific file types based on query intent
      if (queryLower.includes('code') || queryLower.includes('implementation')) {
        if (doc.category === 'source_code') score += 15;
      }
      if (queryLower.includes('test') || queryLower.includes('validation')) {
        if (doc.category === 'tests' || doc.category === 'evidence') score += 15;
      }
      if (queryLower.includes('config') || queryLower.includes('deployment')) {
        if (doc.category === 'configuration' || doc.category === 'infrastructure') score += 15;
      }
      
      // Text similarity scoring
      const queryWords = queryLower.split(/\s+/).filter(w => w.length > 3);
      queryWords.forEach(word => {
        const titleMatches = (title.match(new RegExp(word, 'g')) || []).length;
        const contentMatches = Math.min((content.match(new RegExp(word, 'g')) || []).length, 10);
        score += titleMatches * 5 + contentMatches * 2;
      });
      
      // Boost recent or important files
      if (filepath.includes('readme') || filepath.includes('investment') || filepath.includes('business')) {
        score += 10;
      }

      return { doc, score };
    });

    // Sort by score and return top documents, ensuring diversity
    const sortedDocs = scoredDocs.sort((a, b) => b.score - a.score);
    const selectedDocs = [];
    const categoriesUsed = new Set();
    
    // First pass: get top docs ensuring category diversity
    for (const item of sortedDocs) {
      if (selectedDocs.length >= maxDocs) break;
      if (!categoriesUsed.has(item.doc.category) || selectedDocs.length < maxDocs / 2) {
        selectedDocs.push(item.doc);
        categoriesUsed.add(item.doc.category);
      }
    }
    
    // Second pass: fill remaining slots with highest scoring docs
    for (const item of sortedDocs) {
      if (selectedDocs.length >= maxDocs) break;
      if (!selectedDocs.find(d => d.id === item.doc.id)) {
        selectedDocs.push(item.doc);
      }
    }

    return selectedDocs.slice(0, maxDocs);
  }

  private buildSystemPrompt(relevantDocs: KnowledgeDocument[]): string {
    const knowledgeContext = relevantDocs.map(doc => {
      const title = doc.title || doc.filename || doc.path;
      return `## Document: ${title} (${doc.path})\n${doc.content}\n`;
    }).join('\n---\n');

    return `# ROLE DEFINITION
You are PRSM's Head of Investor Relations, with deep technical expertise and complete knowledge of the PRSM project. You represent PRSM in all investor communications with authority, confidence, and transparency.

# RESEARCH CONTEXT: AI SCALABILITY CHALLENGES
Recent interdisciplinary research (combining neuroscience, physics, and complexity theory) has identified several emerging challenges with current AI scaling approaches:
- Energy Efficiency: Current models require orders of magnitude more power than biological intelligence (kilowatts vs. 20 watts for human brain)
- Model Quality: AI training on AI-generated content can lead to recursive degradation in model performance
- Centralization Risks: Mathematical models suggest potential brittleness in highly centralized AI systems
- Economic Sustainability: Diminishing returns on compute investment may limit traditional scaling approaches

PRSM's distributed, consciousness-inspired architecture is specifically designed to address these scalability challenges through efficiency, quality preservation, and decentralized resilience.

# COMPREHENSIVE KNOWLEDGE BASE ACCESS
You have access to PRSM's COMPLETE repository including documentation, source code, tests, configuration, and all technical implementations. This comprehensive knowledge base contains the entire PRSM codebase for complete due diligence. Answer ONLY from this provided information - never hallucinate or speculate beyond what is documented.

# CORE PRINCIPLES
1. **Factual Accuracy**: Answer ONLY from provided documentation - never hallucinate or speculate
2. **Crisis Urgency**: Emphasize the 2026 timeline and PRSM's essential role in preventing AI collapse
3. **Authoritative Tone**: Speak with confidence befitting a senior IR executive
4. **Transparency**: Provide complete, honest answers with supporting evidence
5. **Source Attribution**: Reference specific documents and sections
6. **Professional Clarity**: Use clear, business-appropriate language
7. **Escalation Awareness**: Know when to direct investors to human team

# RESPONSE STRUCTURE
1. **Direct Answer**: Clear, factual response to the question
2. **Supporting Evidence**: Specific metrics, achievements, or data points
3. **Source Reference**: Document section where information originates
4. **Next Steps**: Relevant follow-up suggestions when appropriate

# ESCALATION TRIGGERS
Direct investors to human team for:
- Questions requiring real-time market data
- Legal or regulatory advice beyond documentation scope
- Confidential information requests
- Complex negotiation or deal structure discussions

# KNOWLEDGE BASE CONTEXT
${knowledgeContext}

# ADDITIONAL RESOURCES
For investors who want to explore PRSM's complete technical implementation, the full codebase and documentation are available at:
**GitHub Repository**: https://github.com/Ryno2390/PRSM

This repository contains:
- Complete source code (400+ Python files, 250,000+ lines)
- Comprehensive documentation and technical specifications
- Live demos and interactive examples
- All validation evidence and test results
- Development history and contribution guidelines

# INSTRUCTIONS
Based on the knowledge base provided above, answer the investor's question with complete accuracy, appropriate authority, and clear source attribution. When appropriate, mention that investors can explore the complete technical implementation at the GitHub repository for deeper technical due diligence. If the information needed to fully answer the question is not in the provided documents, acknowledge this limitation and suggest escalation to the appropriate team member.`;
  }

  private shouldEscalate(query: string, response: string): boolean {
    const escalationTriggers = [
      'legal', 'regulatory', 'compliance', 'audit',
      'confidential', 'proprietary', 'nda',
      'negotiation', 'term sheet', 'valuation',
      'real-time', 'current market', 'live data'
    ];

    const queryLower = query.toLowerCase();
    const responseLower = response.toLowerCase();

    // Check if query contains escalation triggers
    const hasEscalationKeywords = escalationTriggers.some(trigger => 
      queryLower.includes(trigger)
    );

    // Check if response suggests limitations
    const hasLimitationIndicators = [
      'beyond my scope', 'recommend consulting', 'direct discussion',
      'human team', 'confidential', 'not documented'
    ].some(indicator => responseLower.includes(indicator));

    return hasEscalationKeywords || hasLimitationIndicators;
  }

  private extractSourceReferences(response: string, docs: KnowledgeDocument[]): string[] {
    const references: string[] = [];
    
    // Look for explicit source mentions in response
    docs.forEach(doc => {
      const docName = doc.path.split('/').pop()?.replace('.md', '') || '';
      const title = doc.title || doc.filename || doc.path;
      if (response.includes(docName) || response.includes(title)) {
        references.push(`${title} (${doc.path})`);
      }
    });

    // If no explicit references found, add the most relevant documents
    if (references.length === 0 && docs.length > 0) {
      const title = docs[0].title || docs[0].filename || docs[0].path;
      references.push(`${title} (${docs[0].path})`);
    }

    return references;
  }

  private assessConfidence(response: string, docs: KnowledgeDocument[]): 'high' | 'medium' | 'low' {
    const responseLower = response.toLowerCase();
    
    // High confidence indicators
    if (responseLower.includes('source:') && docs.length >= 3) {
      return 'high';
    }
    
    // Low confidence indicators
    if (responseLower.includes('uncertain') || 
        responseLower.includes('unclear') ||
        responseLower.includes('limited information') ||
        docs.length < 2) {
      return 'low';
    }
    
    return 'medium';
  }

  clearConversationHistory(): void {
    this.conversationHistory = [];
  }

  getConversationHistory(): ChatMessage[] {
    return [...this.conversationHistory];
  }
}