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

    console.log(`[NETLIFY-OPT] Processing query: "${query.substring(0, 50)}"`);
    const startTime = Date.now();

    try {
      // NETLIFY OPTIMIZATION: Aggressive performance tuning
      const maxDocs = Math.min(options?.maxContextDocs || 2, 3); // Cap at 3 docs for Netlify
      console.log(`[NETLIFY-OPT] Finding max ${maxDocs} documents...`);
      
      const relevantDocs = this.findRelevantDocuments(query, maxDocs);
      console.log(`[NETLIFY-OPT] Found ${relevantDocs.length} documents`);
      
      // Build system prompt with knowledge context
      const systemPrompt = this.buildSystemPrompt(relevantDocs);
      console.log(`[NETLIFY-OPT] Built prompt for ${relevantDocs.length} docs`);
      
      // Prepare conversation messages
      const messages: ChatMessage[] = [];
      
      if (options?.includeHistory && this.conversationHistory.length > 0) {
        messages.push(...this.conversationHistory.slice(-6)); // Last 3 exchanges
      }
      
      messages.push({ role: 'user', content: query });

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

    console.log(`[NETLIFY-OPT] Finding docs for: "${query.substring(0, 50)}" (max: ${maxDocs})`);
    const startTime = Date.now();
    
    try {
      const documents = Object.values(this.knowledgeBase.documents);
      
      // NETLIFY OPTIMIZATION: Aggressive early returns for common queries
      const queryLower = query.toLowerCase();
      
      // Fast path for PRSM queries - pre-selected key documents
      if (queryLower.includes('prsm') || queryLower.includes('tell me about') || queryLower.includes('what is')) {
        // Return only the 3 most essential documents for speed
        const fastDocs = documents.filter(doc => 
          doc.id === 'INVESTOR_MATERIALS' ||
          doc.id === 'INVESTMENT_READINESS_REPORT' ||
          doc.id === 'docs_AI_CRISIS_INVESTOR_BRIEF'
        ).slice(0, Math.min(maxDocs, 3)); // Cap at 3 for Netlify
        
        const elapsed = Date.now() - startTime;
        console.log(`[NETLIFY-OPT] Fast path: ${fastDocs.length} docs in ${elapsed}ms`);
        return fastDocs;
      }
      
      // For investment queries - specific documents
      if (queryLower.includes('invest') || queryLower.includes('fund') || queryLower.includes('series')) {
        const investDocs = documents.filter(doc => 
          doc.id === 'INVESTOR_MATERIALS' ||
          doc.id === 'INVESTMENT_READINESS_REPORT'
        ).slice(0, 2);
        
        const elapsed = Date.now() - startTime;
        console.log(`[NETLIFY-OPT] Investment path: ${investDocs.length} docs in ${elapsed}ms`);
        return investDocs;
      }
      
      // Technical queries
      if (queryLower.includes('technical') || queryLower.includes('architecture') || queryLower.includes('code')) {
        const techDocs = documents.filter(doc => 
          doc.category === 'tier1_essential' || 
          doc.path.includes('technical')
        ).slice(0, 2);
        
        const elapsed = Date.now() - startTime;
        console.log(`[NETLIFY-OPT] Technical path: ${techDocs.length} docs in ${elapsed}ms`);
        return techDocs;
      }
      
      // Default fallback - just tier1 essentials for speed
      const fallbackDocs = documents
        .filter(doc => doc.category === 'tier1_essential')
        .slice(0, 2); // Limit to 2 for performance
      
      const elapsed = Date.now() - startTime;
      console.log(`[NETLIFY-OPT] Fallback: ${fallbackDocs.length} docs in ${elapsed}ms`);
      return fallbackDocs;
      
    } catch (error) {
      console.error('[NETLIFY-OPT] Error in findRelevantDocuments:', error);
      // Emergency fallback: return minimal essential doc
      const documents = Object.values(this.knowledgeBase.documents);
      const emergencyDoc = documents.find(doc => doc.id === 'INVESTOR_MATERIALS');
      return emergencyDoc ? [emergencyDoc] : [];
    }
  }

  private buildSystemPrompt(relevantDocs: KnowledgeDocument[]): string {
    console.log(`[NETLIFY-OPT] Building system prompt with ${relevantDocs.length} documents`);
    
    // NETLIFY OPTIMIZATION: Aggressive truncation for faster processing
    const knowledgeContext = relevantDocs.map(doc => {
      const title = doc.title || doc.filename || doc.path;
      // Limit each document to 800 characters for Netlify performance
      const truncatedContent = doc.content.length > 800 
        ? doc.content.substring(0, 800) + '...\n[Content truncated for Netlify optimization]'
        : doc.content;
      return `## ${title}\n${truncatedContent}\n`;
    }).join('\n---\n');
    
    console.log(`[NETLIFY-OPT] Context length: ${knowledgeContext.length} chars`);

    // NETLIFY OPTIMIZATION: Simplified system prompt for faster processing
    return `You are PRSM's Head of Investor Relations. Provide accurate, professional responses about PRSM based strictly on the provided documentation.

PRSM is a non-profit AI infrastructure protocol addressing critical AI challenges through distributed architecture. Current Series A: $18M at $72M pre-money valuation.

Key Facts:
- Addresses $847B AI market + $2.3T workflow automation
- 99/100 external validation score
- 84 completed infrastructure components  
- Targets 2026 AI crisis with sustainable solution

# KNOWLEDGE BASE CONTEXT
${knowledgeContext}

Respond professionally with specific references to the provided documents. For complex negotiations or confidential matters, direct to human team.`;
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