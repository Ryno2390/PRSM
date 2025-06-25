#!/usr/bin/env node

/**
 * PRSM AI Concierge - Knowledge Base Compiler
 * 
 * This script compiles all relevant PRSM documentation into a structured
 * knowledge base for the AI Concierge system.
 */

const fs = require('fs-extra');
const path = require('path');
const matter = require('gray-matter');

// Configuration
const PRSM_ROOT = path.resolve(__dirname, '../../');
const OUTPUT_DIR = path.resolve(__dirname, '../knowledge-base');
const COMPILED_FILE = path.resolve(OUTPUT_DIR, 'compiled-knowledge.json');

// File categories from our analysis
const KNOWLEDGE_CATEGORIES = {
  tier1_essential: [
    'INVESTOR_MATERIALS.md',
    'INVESTMENT_READINESS_REPORT.md',
    'docs/BUSINESS_CASE.md',
    'docs/GAME_THEORETIC_INVESTOR_THESIS.md',
    'docs/FUNDING_MILESTONES.md',
    'README.md',
    'docs/TECHNICAL_ADVANTAGES.md'
  ],
  tier2_supporting: [
    'docs/architecture.md',
    'docs/SECURITY_ARCHITECTURE.md',
    'docs/TEAM_CAPABILITY.md',
    'docs/PROTOTYPE_CAPABILITIES.md',
    'PHASE_1_TASK_1_EVIDENCE_REPORT.md',
    'docs/tokenomics.md',
    'docs/business/INVESTOR_MATERIALS.md',
    'docs/business/INVESTOR_QUICKSTART.md'
  ],
  tier3_detailed: [
    'docs/API_REFERENCE.md',
    'docs/SECURITY_HARDENING.md',
    'docs/COMPLIANCE_FRAMEWORK.md',
    'docs/PRODUCTION_OPERATIONS_MANUAL.md',
    'docs/PERFORMANCE_CLAIMS_AUDIT.md',
    'docs/TRANSPARENT_VALIDATION_EVIDENCE.md',
    'docs/PRSM_x_Apple/APPLE_PARTNERSHIP_PROPOSAL.md',
    'docs/PRSM_x_Apple/APPLE_EXECUTIVE_ONE_PAGER.md',
    'legal/crypto_legal_strategy.md'
  ]
};

class KnowledgeBaseCompiler {
  constructor() {
    this.knowledgeBase = {
      metadata: {
        compiledAt: new Date().toISOString(),
        version: '1.0.0',
        totalDocuments: 0,
        categories: Object.keys(KNOWLEDGE_CATEGORIES)
      },
      documents: {},
      categories: {}
    };
  }

  async compile() {
    console.log('🚀 Starting PRSM Knowledge Base compilation...');
    
    // Ensure output directory exists
    await fs.ensureDir(OUTPUT_DIR);
    
    // Compile each category
    for (const [category, files] of Object.entries(KNOWLEDGE_CATEGORIES)) {
      console.log(`📚 Processing category: ${category}`);
      this.knowledgeBase.categories[category] = [];
      
      for (const filePath of files) {
        await this.processFile(filePath, category);
      }
    }
    
    // Add additional context files
    await this.addContextFiles();
    
    // Write compiled knowledge base
    await this.writeCompiledKnowledge();
    
    console.log(`✅ Knowledge base compilation complete!`);
    console.log(`📊 Total documents: ${this.knowledgeBase.metadata.totalDocuments}`);
    console.log(`📁 Output: ${COMPILED_FILE}`);
  }

  async processFile(relativePath, category) {
    const fullPath = path.resolve(PRSM_ROOT, relativePath);
    
    try {
      if (await fs.pathExists(fullPath)) {
        const content = await fs.readFile(fullPath, 'utf-8');
        const parsed = matter(content);
        
        const document = {
          id: this.generateDocumentId(relativePath),
          path: relativePath,
          category: category,
          title: this.extractTitle(parsed.content, relativePath),
          content: parsed.content,
          metadata: {
            ...parsed.data,
            fileSize: content.length,
            wordCount: this.countWords(parsed.content),
            lastModified: (await fs.stat(fullPath)).mtime.toISOString()
          }
        };
        
        this.knowledgeBase.documents[document.id] = document;
        this.knowledgeBase.categories[category].push(document.id);
        this.knowledgeBase.metadata.totalDocuments++;
        
        console.log(`  ✓ ${relativePath} (${document.metadata.wordCount} words)`);
      } else {
        console.log(`  ⚠️  File not found: ${relativePath}`);
      }
    } catch (error) {
      console.error(`  ❌ Error processing ${relativePath}:`, error.message);
    }
  }

  async addContextFiles() {
    // Initialize additional categories
    this.knowledgeBase.categories.evidence = [];
    this.knowledgeBase.categories.structure = [];
    
    // Add key evidence files
    const evidenceDir = path.resolve(PRSM_ROOT, 'evidence/latest');
    if (await fs.pathExists(evidenceDir)) {
      const evidenceFiles = await fs.readdir(evidenceDir);
      for (const file of evidenceFiles) {
        if (file.endsWith('.md') || file.endsWith('.json')) {
          await this.processFile(`evidence/latest/${file}`, 'evidence');
        }
      }
    }

    // Add repository structure information
    const repoMapPath = path.resolve(PRSM_ROOT, 'REPOSITORY_MAP.md');
    if (await fs.pathExists(repoMapPath)) {
      await this.processFile('REPOSITORY_MAP.md', 'structure');
    }
  }

  generateDocumentId(filePath) {
    return filePath.replace(/[\/\\]/g, '_').replace(/\.(md|json)$/, '');
  }

  extractTitle(content, filePath) {
    // Try to extract title from first H1 heading
    const h1Match = content.match(/^# (.+)$/m);
    if (h1Match) {
      return h1Match[1];
    }
    
    // Fallback to filename
    return path.basename(filePath, path.extname(filePath))
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  }

  countWords(content) {
    return content.split(/\s+/).filter(word => word.length > 0).length;
  }

  async writeCompiledKnowledge() {
    await fs.writeJson(COMPILED_FILE, this.knowledgeBase, { spaces: 2 });
    
    // Also create a summary file
    const summary = {
      compiledAt: this.knowledgeBase.metadata.compiledAt,
      totalDocuments: this.knowledgeBase.metadata.totalDocuments,
      categories: Object.entries(this.knowledgeBase.categories).map(([name, docs]) => ({
        name,
        documentCount: docs.length
      })),
      documentSummary: Object.values(this.knowledgeBase.documents).map(doc => ({
        id: doc.id,
        title: doc.title,
        category: doc.category,
        wordCount: doc.metadata.wordCount
      }))
    };
    
    await fs.writeJson(path.resolve(OUTPUT_DIR, 'knowledge-summary.json'), summary, { spaces: 2 });
  }
}

// Run compilation if called directly
if (require.main === module) {
  const compiler = new KnowledgeBaseCompiler();
  compiler.compile().catch(error => {
    console.error('❌ Compilation failed:', error);
    process.exit(1);
  });
}

module.exports = KnowledgeBaseCompiler;