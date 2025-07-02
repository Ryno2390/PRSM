# PRSM Documentation Style Guide

## Overview

This guide establishes consistent documentation standards for the PRSM project to improve readability, maintainability, and user experience across all documentation types.

## Documentation Types & Organization

### 1. **Primary Documentation**
- `README.md` - Project overview, investment status, technical highlights
- `docs/quickstart.md` - Getting started guide with step-by-step instructions
- `docs/API_REFERENCE.md` - Complete API documentation with examples
- `CONTRIBUTING.md` - Comprehensive contribution guidelines

### 2. **Development Documentation**
- `docs/development/CODING_STANDARDS.md` - Code style and technical standards
- `docs/development/DOCUMENTATION_STYLE_GUIDE.md` - This document
- Architecture and technical deep-dive documents

### 3. **Specialized Documentation**
- API references with endpoint specifications
- Tutorial and workflow guides
- Security and compliance documentation

## Writing Style Guidelines

### Tone and Voice
- **Professional yet accessible** - Technical accuracy without unnecessary jargon
- **Action-oriented** - Use active voice and imperative mood for instructions
- **Confident and authoritative** - Present information with certainty
- **User-focused** - Write from the reader's perspective

### Language Conventions
- **Use present tense** for descriptions and current functionality
- **Use future tense** only for planned features explicitly marked as such
- **Prefer active voice** - "The system processes queries" vs "Queries are processed"
- **Use specific terminology** consistently throughout documentation

### Writing Standards
- **Sentence length**: Keep sentences under 25 words when possible
- **Paragraph length**: 2-4 sentences for technical content, 1-3 for introductory material
- **Lists**: Use parallel structure and consistent formatting
- **Acronyms**: Define on first use, then use consistently

## Markdown Formatting Standards

### Headers
```markdown
# H1: Document Title (only one per document)
## H2: Major Sections
### H3: Subsections
#### H4: Sub-subsections (use sparingly)
```

**Header Guidelines:**
- Use title case for H1 and H2
- Use sentence case for H3 and below
- Include emoji icons for major sections where appropriate (🚀, 📋, 🔧, etc.)
- Maintain consistent numbering when appropriate

### Code Formatting

#### Inline Code
- Use `backticks` for file names, commands, variables, and short code snippets
- Use **bold** for UI elements and important terms
- Use *italic* for emphasis and parameter names

#### Code Blocks
```python
# Always specify language for syntax highlighting
async def example_function():
    """Include docstrings in examples."""
    return "Use realistic, working code examples"
```

**Code Block Standards:**
- Always specify language (`python`, `bash`, `json`, `http`, etc.)
- Include comments explaining complex logic
- Use realistic, functional examples
- Keep examples under 20 lines when possible

#### API Examples
```http
POST /api/v1/endpoint
Authorization: Bearer <token>
Content-Type: application/json

{
  "parameter": "value"
}
```

**Response Examples:**
```json
{
  "status": "success",
  "data": {
    "key": "value"
  }
}
```

### Lists and Bullets

#### Unordered Lists
- Use hyphens (`-`) for consistency
- Maintain parallel structure
- Use sub-bullets sparingly (max 2 levels)

#### Ordered Lists
1. **Primary steps** - Use numbers for sequential processes
2. **Sub-steps** - Use letters (a, b, c) when needed
3. **Formatting** - Bold the action, explain the details

#### Checklist Format
- [ ] Incomplete items
- [x] Completed items
- Use for setup guides and validation steps

### Links and References

#### Internal Links
- Use relative paths: `[Quick Start](../quickstart.md)`
- Use descriptive link text: `[API Authentication Guide](API_AUTHENTICATION.md)`
- Avoid "click here" or generic phrases

#### External Links
- Use full URLs with descriptive text
- Open external links in new tabs when appropriate for web documentation
- Include link context when necessary

#### Cross-References
- Reference specific sections: `See [Authentication](#authentication) section`
- Use consistent section anchors
- Maintain link accuracy during updates

### Tables

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| **Bold headers** | Clear data | Aligned content |
| Use pipes consistently | Right-align numbers | Left-align text |

**Table Guidelines:**
- Bold header row
- Use consistent alignment
- Keep tables under 5 columns when possible
- Include table descriptions when complex

### Badges and Status Indicators

Use badges consistently for:
```markdown
[![Status](https://img.shields.io/badge/status-Production%20Ready-green.svg)](#section)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](link)
```

**Badge Standards:**
- Green for positive status (ready, complete, active)
- Blue for informational (version, type)
- Orange for warnings or in-progress
- Red for critical issues or deprecated

## Content Structure Patterns

### Document Template Structure
```markdown
# Document Title

Brief overview paragraph explaining the document's purpose.

## 🎯 Quick Summary (for longer docs)
- Key point 1
- Key point 2
- Key point 3

## 📋 Prerequisites (when applicable)
- Requirement 1
- Requirement 2

## Main Content Sections
### Implementation Details
### Examples
### Best Practices

## 🚀 Next Steps (when applicable)
Links to related documentation

---
*Footer information, last updated, contact info*
```

### README Structure Pattern
1. **Status badges** - Current state and key metrics
2. **Production readiness** - Enterprise features and validation
3. **Investment/business context** - Funding status and opportunity
4. **Technical overview** - What the project does
5. **Problem statement** - Why it exists
6. **Solution approach** - How it works
7. **Key features** - What it provides
8. **Getting started** - Quick setup
9. **Documentation links** - Where to learn more

### API Documentation Pattern
1. **Overview** - Purpose and base information
2. **Authentication** - How to authenticate
3. **Endpoints** - Grouped by functionality
4. **Request/Response examples** - Complete working examples
5. **Error handling** - Common errors and solutions
6. **Rate limiting** - Usage constraints
7. **SDK examples** - Code implementation examples

## Code Documentation Standards

### Inline Comments
```python
# Use comments to explain WHY, not WHAT
async def process_query(user_input: str) -> PRSMResponse:
    """Process user query with SEAL enhancement.
    
    Args:
        user_input: The user's research question
        
    Returns:
        Enhanced response with autonomous improvements
    """
    # Apply SEAL methodology for autonomous improvement
    return enhanced_response
```

### Example Code Requirements
- **Working examples** - All code should be functional
- **Complete context** - Include necessary imports and setup
- **Error handling** - Show proper exception management
- **Comments** - Explain complex logic and business rules
- **Type hints** - Use Python type annotations consistently

### Configuration Examples
```python
# Good: Complete, realistic configuration
PRSM_CONFIG = {
    "api_key": "your_api_key_here",
    "base_url": "https://api.prsm.org",
    "timeout": 30,
    "retry_attempts": 3
}
```

## Visual Elements and Formatting

### Emoji Usage Guidelines
Use emojis consistently for visual navigation:
- 🚀 **Getting Started, Quick Start, Production**
- 📋 **Prerequisites, Requirements, Checklists**
- 🔧 **Configuration, Setup, Tools**
- 📊 **Metrics, Performance, Analytics**
- 🛡️ **Security, Safety, Compliance**
- 💰 **Economics, FTNS, Marketplace**
- 🔬 **Research, Science, SEAL Technology**
- 🤝 **Community, Contributing, Governance**
- ⚡ **Performance, Speed, Optimization**
- 🎯 **Goals, Objectives, Key Points**

### Callout Boxes and Alerts
```markdown
> **Note**: Important information that users should be aware of

> **Warning**: Critical information that could cause issues

> **Pro Tip**: Advanced usage suggestions and optimizations
```

### Status Indicators
- ✅ **Completed, Working, Available**
- ❌ **Failed, Broken, Unavailable**
- 🔄 **In Progress, Processing**
- ⏸️ **Paused, Pending**
- 🚧 **Under Construction, Beta**

## Version Control and Maintenance

### Documentation Updates
- **Update dates** - Include "Last Updated" information
- **Version alignment** - Keep docs in sync with code versions
- **Change tracking** - Document significant updates in commit messages
- **Review process** - All documentation changes should be reviewed

### File Naming Conventions
- Use `UPPERCASE.md` for primary documentation (README.md, CONTRIBUTING.md)
- Use `lowercase_with_underscores.md` for technical guides
- Use descriptive names that clearly indicate content
- Group related files in appropriate subdirectories

### Cross-Reference Maintenance
- **Regular link checking** - Verify internal and external links
- **Section anchor updates** - Update references when sections change
- **Consistent naming** - Use the same terms throughout all documentation

## Quality Assurance Checklist

### Before Publishing Documentation
- [ ] **Accuracy** - All technical information is correct and current
- [ ] **Completeness** - All necessary information is included
- [ ] **Clarity** - Instructions are easy to follow
- [ ] **Examples** - Code examples work as written
- [ ] **Links** - All links are functional and point to correct locations
- [ ] **Formatting** - Consistent markdown formatting throughout
- [ ] **Grammar** - Professional writing with correct grammar and spelling
- [ ] **Audience** - Content is appropriate for the intended audience

### Accessibility Guidelines
- **Alt text** - Include descriptive alt text for images
- **Heading hierarchy** - Use proper heading structure for navigation
- **Link descriptions** - Use descriptive link text
- **Color independence** - Don't rely solely on color to convey information

## Content-Specific Guidelines

### Technical Tutorials
1. **Clear objectives** - State what the user will learn
2. **Prerequisites** - List required knowledge and setup
3. **Step-by-step instructions** - Sequential, actionable steps
4. **Verification steps** - How to confirm success
5. **Troubleshooting** - Common issues and solutions
6. **Next steps** - What to do after completion

### API Documentation
1. **Complete examples** - Full request/response cycles
2. **Error scenarios** - Document error conditions and responses
3. **Rate limits** - Include usage constraints
4. **Authentication** - Clear auth requirements
5. **Parameter validation** - Explain required vs optional parameters

### Architecture Documentation
1. **System overview** - High-level component relationships
2. **Data flow** - How information moves through the system
3. **Design decisions** - Why specific approaches were chosen
4. **Performance characteristics** - Expected behavior under load
5. **Security considerations** - Built-in security measures

## Compliance and Legal

### Copyright and Attribution
- Include appropriate copyright notices
- Attribute external content properly
- Use consistent license information
- Credit contributors and sources

### Confidentiality Guidelines
- **No secrets** - Never include API keys, passwords, or sensitive data
- **Generic examples** - Use placeholder values in examples
- **Public information only** - Only document publicly available features
- **Security review** - Review security-sensitive documentation

---

## Document Maintenance

**Version**: 1.0  
**Last Updated**: July 2025  
**Next Review**: January 2026  
**Maintainer**: PRSM Documentation Team  

**Related Documentation:**
- [Coding Standards](CODING_STANDARDS.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [API Reference](../API_REFERENCE.md)

---

*This style guide is a living document. Updates should be discussed with the documentation team and approved through the standard review process.*