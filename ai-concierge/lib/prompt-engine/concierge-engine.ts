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

    console.log(`Processing investor query: "${query}"`);
    const startTime = Date.now();

    try {
      // Find relevant documents (reduced default for performance)
      const maxDocs = Math.min(options?.maxContextDocs || 5, 8); // Cap at 8 docs max
      const relevantDocs = this.findRelevantDocuments(query, maxDocs);
      
      // Build system prompt with knowledge context
      const systemPrompt = this.buildSystemPrompt(relevantDocs);
      console.log(`Built system prompt with ${relevantDocs.length} documents`);
    
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

      const totalTime = Date.now() - startTime;
      console.log(`Query processed successfully in ${totalTime}ms`);

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

    console.log(`Finding relevant documents for query: "${query}" (max: ${maxDocs})`);
    const startTime = Date.now();
    
    const queryLower = query.toLowerCase();
    const documents = Object.values(this.knowledgeBase.documents);
    
    console.log(`Processing ${documents.length} documents...`);
    
    // Simplified and optimized scoring
    const scoredDocs = documents.map(doc => {
      let score = 0;
      const title = (doc.title || doc.filename || '').toLowerCase();
      const filepath = doc.path.toLowerCase();
      
      // Quick category-based base scoring
      const categoryScores: { [key: string]: number } = {
        'tier1_essential': 20,
        'tier2_supporting': 12,
        'tier3_detailed': 8
      };
      score += categoryScores[doc.category] || 5;
      
      // Fast keyword matching - only check key terms
      const keyTerms = ['prsm', 'investment', 'funding', 'technical', 'business', 'model'];
      keyTerms.forEach(term => {
        if (queryLower.includes(term)) {
          if (title.includes(term)) score += 15;
          if (filepath.includes(term)) score += 10;
          // Simple content check - just indexOf, no regex
          if (doc.content.toLowerCase().indexOf(term) !== -1) score += 5;
        }
      });
      
      // General PRSM queries get investment docs
      if (queryLower.includes('prsm') || queryLower.includes('tell me about')) {
        if (filepath.includes('investment') || title.includes('investment')) score += 25;
        if (filepath.includes('readme') || title.includes('overview')) score += 20;
      }
      
      // Boost important files
      if (filepath.includes('investment_materials') || 
          filepath.includes('investment_readiness') ||
          filepath.includes('business_case')) {
        score += 15;
      }

      return { doc, score };
    });

    // Simple sort and take top results
    const sortedDocs = scoredDocs
      .filter(item => item.score > 5) // Only include docs with some relevance
      .sort((a, b) => b.score - a.score)
      .slice(0, maxDocs)
      .map(item => item.doc);

    const elapsed = Date.now() - startTime;
    console.log(`Found ${sortedDocs.length} relevant documents in ${elapsed}ms`);
    
    return sortedDocs;
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