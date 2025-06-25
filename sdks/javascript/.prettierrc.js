module.exports = {
  // Basic formatting
  semi: true,
  trailingComma: 'es5',
  singleQuote: true,
  quoteProps: 'as-needed',
  jsxSingleQuote: true,
  
  // Indentation
  tabWidth: 2,
  useTabs: false,
  
  // Line wrapping
  printWidth: 100,
  proseWrap: 'preserve',
  
  // Spacing
  bracketSpacing: true,
  bracketSameLine: false,
  arrowParens: 'always',
  
  // File handling
  endOfLine: 'lf',
  insertPragma: false,
  requirePragma: false,
  
  // Language specific
  overrides: [
    {
      files: '*.json',
      options: {
        printWidth: 120,
        trailingComma: 'none'
      }
    },
    {
      files: '*.md',
      options: {
        printWidth: 80,
        proseWrap: 'always'
      }
    },
    {
      files: '*.yaml',
      options: {
        tabWidth: 2,
        singleQuote: false
      }
    }
  ]
};