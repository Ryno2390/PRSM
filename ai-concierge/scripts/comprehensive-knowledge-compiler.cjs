#!/usr/bin/env node

/**
 * PRSM AI Concierge - Comprehensive Knowledge Base Compiler
 * 
 * This script compiles ALL PRSM repository information into a comprehensive
 * knowledge base for complete due diligence capabilities.
 */

const fs = require('fs-extra');
const path = require('path');
const matter = require('gray-matter');

// Configuration
const PRSM_ROOT = path.resolve(__dirname, '../../');
const OUTPUT_DIR = path.resolve(__dirname, '../knowledge-base');
const COMPILED_FILE = path.resolve(OUTPUT_DIR, 'comprehensive-knowledge.json');

// Size limits for Netlify deployment (avoid JSON.stringify limits)
const MAX_TOTAL_SIZE = 100 * 1024 * 1024; // 100MB total content
const MAX_FILE_SIZE = 1024 * 1024; // 1MB per file

// File extensions to include
const INCLUDE_EXTENSIONS = [
  '.md', '.py', '.js', '.ts', '.tsx', '.json', '.yml', '.yaml', 
  '.toml', '.txt', '.sh', '.sql', '.env.example', '.gitignore'
];

// Directories to exclude (too large or irrelevant)
const EXCLUDE_DIRS = [
  'node_modules', '.git', '__pycache__', '.next', 'out', 
  '.vscode', '.idea', 'dist', 'build', '.cache', 'logs',
  '.claude', 'ai-concierge/node_modules', 'ai-concierge/.next',
  '.venv', 'venv', '.env', 'env', '.virtualenv', 'virtualenv',
  '__pycache__', '.pytest_cache', '.coverage', 'htmlcov',
  '.tox', '.nox'
];

// File patterns to exclude
const EXCLUDE_PATTERNS = [
  '.DS_Store', '*.log', '*.pyc', '*.pyo', '*.pyd', 
  '*.so', '*.egg', '*.egg-info', '*.whl', '*.tar.gz',
  'package-lock.json', '*.lock', '*.tsbuildinfo'
];

class ComprehensiveKnowledgeCompiler {
  constructor() {
    this.knowledgeBase = {
      metadata: {
        compiledAt: new Date().toISOString(),
        version: '2.0.0-comprehensive',
        totalDocuments: 0,
        totalLines: 0,
        totalSize: 0,
        coverage: 'complete_repository'
      },
      documents: {},
      categories: {
        documentation: [],
        source_code: [],
        configuration: [],
        tests: [],
        evidence: [],
        infrastructure: [],
        contracts: [],
        scripts: []
      },
      structure: {},
      codeAnalysis: {}
    };
  }

  async compile() {
    console.log('🚀 Starting Comprehensive PRSM Knowledge Base compilation...');
    console.log('📂 Scanning entire repository for maximum due diligence coverage...');
    
    await fs.ensureDir(OUTPUT_DIR);
    
    // Scan and process all files
    await this.scanRepository(PRSM_ROOT, '');
    
    // Generate code analysis
    await this.generateCodeAnalysis();
    
    // Generate repository structure
    await this.generateRepositoryStructure();
    
    // Write comprehensive knowledge base
    await this.writeCompiledKnowledge();
    
    console.log(`✅ Comprehensive knowledge base compilation complete!`);
    console.log(`📊 Total documents: ${this.knowledgeBase.metadata.totalDocuments}`);
    console.log(`📏 Total lines: ${this.knowledgeBase.metadata.totalLines.toLocaleString()}`);
    console.log(`💾 Total size: ${(this.knowledgeBase.metadata.totalSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`📁 Output: ${COMPILED_FILE}`);
  }

  async scanRepository(currentPath, relativePath) {
    try {
      const items = await fs.readdir(currentPath);
      
      for (const item of items) {
        const fullPath = path.join(currentPath, item);
        const itemRelativePath = path.join(relativePath, item);
        
        // Skip excluded directories
        if (this.shouldExcludeDirectory(item, itemRelativePath)) {
          continue;
        }
        
        try {
          const stat = await fs.stat(fullPath);
          
          if (stat.isDirectory()) {
            await this.scanRepository(fullPath, itemRelativePath);
          } else if (stat.isFile()) {
            await this.processFile(fullPath, itemRelativePath, stat);
          }
        } catch (statError) {
          // Skip broken symlinks or inaccessible files
          console.log(`  ⚠ Skipped ${itemRelativePath}: ${statError.message}`);
          continue;
        }
      }
    } catch (readdirError) {
      console.log(`  ⚠ Cannot read directory ${relativePath}: ${readdirError.message}`);
    }
  }

  shouldExcludeDirectory(dirname, relativePath) {
    return EXCLUDE_DIRS.some(excluded => 
      dirname === excluded || relativePath.includes(excluded)
    );
  }

  shouldIncludeFile(filename, relativePath) {
    // Check file extension
    const ext = path.extname(filename);
    const hasValidExtension = INCLUDE_EXTENSIONS.includes(ext) || ext === '';
    
    // Check exclude patterns
    const isExcluded = EXCLUDE_PATTERNS.some(pattern => {
      if (pattern.startsWith('*.')) {
        return filename.endsWith(pattern.slice(1));
      }
      return filename === pattern;
    });
    
    // Include important files without extensions
    const importantFiles = [
      'Dockerfile', 'Makefile', 'LICENSE', 'README', 'CHANGELOG',
      'requirements.txt', 'pyproject.toml', 'setup.py'
    ];
    const isImportant = importantFiles.some(important => 
      filename.startsWith(important)
    );
    
    return (hasValidExtension || isImportant) && !isExcluded;
  }

  async processFile(fullPath, relativePath, stat) {
    const filename = path.basename(fullPath);
    
    if (!this.shouldIncludeFile(filename, relativePath)) {
      return;
    }
    
    // Skip files that are too large
    if (stat.size > MAX_FILE_SIZE) {
      console.log(`  ⚠ Skipped ${relativePath}: File too large (${Math.round(stat.size / 1024)}KB)`);
      return;
    }
    
    // Check if we're approaching total size limit
    if (this.knowledgeBase.metadata.totalSize > MAX_TOTAL_SIZE) {
      console.log(`  ⚠ Stopping compilation: Reached size limit (${Math.round(MAX_TOTAL_SIZE / 1024 / 1024)}MB)`);
      return;
    }
    
    try {
      const content = await fs.readFile(fullPath, 'utf-8');
      const category = this.categorizeFile(relativePath, content);
      
      const document = {
        id: this.generateId(relativePath),
        path: relativePath,
        filename: filename,
        category: category,
        type: this.getFileType(relativePath),
        content: content,
        metadata: {
          size: stat.size,
          lines: content.split('\n').length,
          words: content.split(/\s+/).filter(w => w.length > 0).length,
          lastModified: stat.mtime.toISOString(),
          encoding: 'utf-8'
        }
      };
      
      // Extract title for documentation files
      if (category === 'documentation') {
        document.title = this.extractTitle(content, filename);
      }
      
      // Add code analysis for source files
      if (category === 'source_code') {
        document.codeAnalysis = this.analyzeCode(content, relativePath);
      }
      
      this.knowledgeBase.documents[document.id] = document;
      this.knowledgeBase.categories[category].push(document.id);
      this.knowledgeBase.metadata.totalDocuments++;
      this.knowledgeBase.metadata.totalLines += document.metadata.lines;
      this.knowledgeBase.metadata.totalSize += stat.size;
      
      console.log(`  ✓ ${relativePath} (${document.metadata.words} words, ${document.metadata.lines} lines)`);
      
    } catch (error) {
      if (error.code !== 'EISDIR') {
        console.log(`  ⚠ Skipped ${relativePath}: ${error.message}`);
      }
    }
  }

  categorizeFile(relativePath, content) {
    const path_lower = relativePath.toLowerCase();
    
    // Documentation
    if (path_lower.endsWith('.md') || path_lower.includes('readme') || path_lower.includes('doc')) {
      return 'documentation';
    }
    
    // Source code
    if (path_lower.endsWith('.py') || path_lower.endsWith('.js') || 
        path_lower.endsWith('.ts') || path_lower.endsWith('.tsx')) {
      return 'source_code';
    }
    
    // Tests
    if (path_lower.includes('test') || path_lower.includes('spec')) {
      return 'tests';
    }
    
    // Configuration
    if (path_lower.endsWith('.json') || path_lower.endsWith('.yml') || 
        path_lower.endsWith('.yaml') || path_lower.endsWith('.toml') ||
        path_lower.includes('config') || path_lower.includes('docker')) {
      return 'configuration';
    }
    
    // Smart contracts
    if (path_lower.includes('contract') || path_lower.endsWith('.sol')) {
      return 'contracts';
    }
    
    // Scripts
    if (path_lower.endsWith('.sh') || path_lower.includes('script')) {
      return 'scripts';
    }
    
    // Evidence and reports
    if (path_lower.includes('evidence') || path_lower.includes('result') || 
        path_lower.includes('report')) {
      return 'evidence';
    }
    
    // Infrastructure
    if (path_lower.includes('deploy') || path_lower.includes('k8s') || 
        path_lower.includes('terraform') || path_lower.includes('infra')) {
      return 'infrastructure';
    }
    
    return 'documentation'; // Default fallback
  }

  getFileType(relativePath) {
    const ext = path.extname(relativePath).toLowerCase();
    const typeMap = {
      '.py': 'python',
      '.js': 'javascript', 
      '.ts': 'typescript',
      '.tsx': 'typescript_react',
      '.md': 'markdown',
      '.json': 'json',
      '.yml': 'yaml',
      '.yaml': 'yaml',
      '.toml': 'toml',
      '.sh': 'shell',
      '.sql': 'sql',
      '.sol': 'solidity'
    };
    return typeMap[ext] || 'text';
  }

  generateId(relativePath) {
    return relativePath.replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_');
  }

  extractTitle(content, filename) {
    // Try to find title in markdown headers
    const lines = content.split('\n');
    for (const line of lines.slice(0, 10)) {
      const match = line.match(/^#+\s+(.+)/);
      if (match) {
        return match[1].trim();
      }
    }
    
    // Fallback to filename
    return filename.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ');
  }

  analyzeCode(content, relativePath) {
    const analysis = {
      language: this.getFileType(relativePath),
      lines: content.split('\n').length,
      functions: [],
      classes: [],
      imports: [],
      complexity: 'unknown'
    };
    
    // Basic Python analysis
    if (relativePath.endsWith('.py')) {
      analysis.functions = this.extractPythonFunctions(content);
      analysis.classes = this.extractPythonClasses(content);
      analysis.imports = this.extractPythonImports(content);
    }
    
    // Basic TypeScript/JavaScript analysis
    if (relativePath.endsWith('.ts') || relativePath.endsWith('.js') || relativePath.endsWith('.tsx')) {
      analysis.functions = this.extractJSFunctions(content);
      analysis.classes = this.extractJSClasses(content);
      analysis.imports = this.extractJSImports(content);
    }
    
    return analysis;
  }

  extractPythonFunctions(content) {
    const functions = [];
    const functionRegex = /^(?:\s*)def\s+(\w+)\s*\([^)]*\):/gm;
    let match;
    while ((match = functionRegex.exec(content)) !== null) {
      functions.push(match[1]);
    }
    return functions;
  }

  extractPythonClasses(content) {
    const classes = [];
    const classRegex = /^(?:\s*)class\s+(\w+)(?:\([^)]*\))?:/gm;
    let match;
    while ((match = classRegex.exec(content)) !== null) {
      classes.push(match[1]);
    }
    return classes;
  }

  extractPythonImports(content) {
    const imports = [];
    const importRegex = /^(?:\s*)(?:from\s+[\w.]+\s+)?import\s+(.+)/gm;
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      imports.push(match[1].trim());
    }
    return imports.slice(0, 10); // Limit to first 10
  }

  extractJSFunctions(content) {
    const functions = [];
    const functionRegexes = [
      /function\s+(\w+)\s*\(/g,
      /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>/g,
      /(\w+)\s*:\s*(?:async\s+)?(?:function\s*)?\([^)]*\)\s*=>/g
    ];
    
    for (const regex of functionRegexes) {
      let match;
      while ((match = regex.exec(content)) !== null) {
        functions.push(match[1]);
      }
    }
    return functions.slice(0, 20); // Limit to first 20
  }

  extractJSClasses(content) {
    const classes = [];
    const classRegex = /class\s+(\w+)(?:\s+extends\s+\w+)?/g;
    let match;
    while ((match = classRegex.exec(content)) !== null) {
      classes.push(match[1]);
    }
    return classes;
  }

  extractJSImports(content) {
    const imports = [];
    const importRegex = /import.*?from\s+['"]([^'"]+)['"]/g;
    let match;
    while ((match = importRegex.exec(content)) !== null) {
      imports.push(match[1]);
    }
    return imports.slice(0, 10); // Limit to first 10
  }

  async generateCodeAnalysis() {
    console.log('🔍 Generating code analysis summary...');
    
    const sourceFiles = Object.values(this.knowledgeBase.documents)
      .filter(doc => doc.category === 'source_code');
    
    this.knowledgeBase.codeAnalysis = {
      totalFiles: sourceFiles.length,
      languages: {},
      totalFunctions: 0,
      totalClasses: 0,
      keyComponents: []
    };
    
    sourceFiles.forEach(file => {
      const lang = file.codeAnalysis?.language || 'unknown';
      if (!this.knowledgeBase.codeAnalysis.languages[lang]) {
        this.knowledgeBase.codeAnalysis.languages[lang] = 0;
      }
      this.knowledgeBase.codeAnalysis.languages[lang]++;
      
      if (file.codeAnalysis) {
        this.knowledgeBase.codeAnalysis.totalFunctions += file.codeAnalysis.functions.length;
        this.knowledgeBase.codeAnalysis.totalClasses += file.codeAnalysis.classes.length;
      }
    });
  }

  async generateRepositoryStructure() {
    console.log('🏗️ Generating repository structure map...');
    
    this.knowledgeBase.structure = {
      totalFiles: this.knowledgeBase.metadata.totalDocuments,
      directories: {},
      fileTypes: {}
    };
    
    Object.values(this.knowledgeBase.documents).forEach(doc => {
      const dir = path.dirname(doc.path);
      const ext = path.extname(doc.path);
      
      if (!this.knowledgeBase.structure.directories[dir]) {
        this.knowledgeBase.structure.directories[dir] = 0;
      }
      this.knowledgeBase.structure.directories[dir]++;
      
      if (!this.knowledgeBase.structure.fileTypes[ext || 'no_extension']) {
        this.knowledgeBase.structure.fileTypes[ext || 'no_extension'] = 0;
      }
      this.knowledgeBase.structure.fileTypes[ext || 'no_extension']++;
    });
  }

  async writeCompiledKnowledge() {
    await fs.writeJson(COMPILED_FILE, this.knowledgeBase, { spaces: 2 });
    
    // Create summary file
    const summaryFile = path.resolve(OUTPUT_DIR, 'comprehensive-summary.json');
    const summary = {
      compiledAt: this.knowledgeBase.metadata.compiledAt,
      version: this.knowledgeBase.metadata.version,
      coverage: this.knowledgeBase.metadata.coverage,
      totalDocuments: this.knowledgeBase.metadata.totalDocuments,
      totalLines: this.knowledgeBase.metadata.totalLines,
      totalSize: this.knowledgeBase.metadata.totalSize,
      categories: Object.fromEntries(
        Object.entries(this.knowledgeBase.categories).map(([key, docs]) => [key, docs.length])
      ),
      codeAnalysis: this.knowledgeBase.codeAnalysis,
      structure: this.knowledgeBase.structure
    };
    
    await fs.writeJson(summaryFile, summary, { spaces: 2 });
  }
}

// Run if called directly
if (require.main === module) {
  const compiler = new ComprehensiveKnowledgeCompiler();
  compiler.compile().catch(error => {
    console.error('❌ Compilation failed:', error);
    process.exit(1);
  });
}

module.exports = ComprehensiveKnowledgeCompiler;