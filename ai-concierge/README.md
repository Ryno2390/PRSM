# PRSM AI Investor Concierge

**24/7 Intelligent Investor Relations Assistant powered by AI**

## Overview

The PRSM AI Investor Concierge is an innovative AI-powered system that provides authoritative, always-available responses to investor inquiries about PRSM. It demonstrates both practical utility and technical sophistication while showcasing PRSM's meta-application capabilities.

## 🎯 Key Benefits

- **24/7 Global Coverage**: Instant responses for international investors across time zones
- **Consistency Guarantee**: Zero risk of off-message communication
- **Unlimited Scalability**: Handle 1 or 1,000 inquiries without additional overhead
- **Technical Demonstration**: Live showcase of PRSM's AI coordination capabilities
- **Competitive Differentiation**: First-of-its-kind AI concierge in startup investor relations

## 🏗️ Architecture

### Phase 1: MVP Concierge (Current)
```
AI Concierge MVP
├── Knowledge Base (Comprehensive PRSM documentation)
├── LLM Integration (Claude 3.5 Sonnet / Gemini 2.5 Pro)
├── Web Interface (React-based chat UI)
└── Deployment (Vercel hosting)
```

### Future: Meta-PRSM Integration
- Concierge running as teacher model within PRSM runtime
- Integration with governance portal
- Version-controlled, auditable responses
- Community-driven improvements via FTNS governance

## 📚 Knowledge Base

The concierge has comprehensive access to:

### Tier 1: Essential Investor Materials
- Investment Readiness Report
- Business Case and Revenue Strategy
- Game-Theoretic Investor Thesis
- Funding Milestones and Strategy
- Technical Advantages

### Tier 2: Supporting Documentation
- Architecture and Security Documentation
- Team Capabilities and Evidence Reports
- Tokenomics and Market Analysis
- Prototype Capabilities

### Tier 3: Detailed Technical & Legal
- API and Integration Documentation
- Compliance and Legal Framework
- Performance and Validation Reports
- Strategic Partnership Plans

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- NPM or Yarn
- API keys for LLM providers (Claude/Gemini/OpenAI)

### Installation
```bash
cd ai-concierge
npm install
```

### Configuration
```bash
cp .env.example .env
# Add your API keys to .env
```

### Knowledge Base Compilation
```bash
npm run knowledge-compile
```

### Development
```bash
npm run dev
# Navigate to http://localhost:3000
```

### Production Build
```bash
npm run build
npm start
```

## 🛠️ Development Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run knowledge-compile` | Compile PRSM docs into knowledge base |
| `npm run prompt-test` | Test prompt engineering with FAQ dataset |
| `npm run lint` | Run ESLint |
| `npm run type-check` | TypeScript type checking |

## 📁 Project Structure

```
ai-concierge/
├── components/           # React components
│   ├── chat/            # Chat interface components
│   ├── ui/              # Reusable UI components
│   └── layout/          # Layout components
├── pages/               # Next.js pages
├── lib/                 # Utility libraries
│   ├── llm-clients/     # LLM provider integrations
│   ├── knowledge-base/  # Knowledge base utilities
│   └── prompt-engine/   # Prompt engineering system
├── knowledge-base/      # Compiled documentation
├── prompt-engineering/  # Prompt templates and frameworks
├── testing/            # Test datasets and validation
└── scripts/            # Build and utility scripts
```

## 🎛️ Configuration

### Environment Variables
```env
# LLM Provider Configuration
ANTHROPIC_API_KEY=your_claude_api_key
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Application Configuration
NEXT_PUBLIC_APP_URL=http://localhost:3000
DEFAULT_LLM_PROVIDER=claude

# Analytics (optional)
VERCEL_ANALYTICS_ID=your_analytics_id
```

### LLM Provider Priority
1. **Claude 3.5 Sonnet** (Primary)
2. **Gemini 2.5 Pro** (Secondary)
3. **GPT-4o** (Fallback)

## 🧪 Testing

### Prompt Testing
```bash
npm run prompt-test
```

This runs the concierge against the comprehensive FAQ dataset to validate:
- Response accuracy (source validation)
- Professional tone and authority
- Proper escalation handling
- Source attribution quality

### Quality Metrics
- **Accuracy**: 95%+ factual accuracy from source materials
- **Completeness**: 90%+ comprehensive response coverage
- **Professional Tone**: Consistent IR executive authority
- **Response Time**: <3 seconds average

## 📊 Monitoring & Analytics

### Response Quality Tracking
- Real-time accuracy monitoring
- Source attribution validation
- Escalation pattern analysis
- User satisfaction scoring

### Usage Analytics
- Question category distribution
- Response time metrics
- User engagement patterns
- Knowledge gap identification

## 🔧 Customization

### Adding New Knowledge Sources
1. Update `KNOWLEDGE_CATEGORIES` in `scripts/compile-knowledge-base.js`
2. Run `npm run knowledge-compile`
3. Test with relevant questions

### Prompt Engineering
1. Modify templates in `prompt-engineering/`
2. Test with `npm run prompt-test`
3. Validate against FAQ dataset

### UI Customization
1. Update components in `components/`
2. Modify styling in Tailwind CSS
3. Test responsive design

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Manual Deployment
```bash
npm run build
# Deploy 'out' directory to your hosting provider
```

### Environment Setup
Ensure all environment variables are configured in your deployment platform.

## 📈 Roadmap

### Phase 1: MVP (Current)
- [x] Knowledge base compilation system
- [x] Prompt engineering framework
- [x] Basic chat interface
- [ ] LLM integration and testing
- [ ] Production deployment

### Phase 2: Enhanced Experience (Weeks 5-12)
- [ ] Advanced conversational UI
- [ ] Multi-modal responses (charts, documents)
- [ ] Analytics dashboard
- [ ] Integration with existing investor materials

### Phase 3: Meta-PRSM Integration (Months 4-12)
- [ ] Integration with PRSM runtime
- [ ] Governance portal integration
- [ ] Version-controlled responses
- [ ] FTNS token-weighted improvements

## 🤝 Contributing

This project follows PRSM's contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## 📄 License

MIT License - see the main PRSM repository for details.

## 🆘 Support

For technical support or questions:
- Create an issue in the main PRSM repository
- Tag with `ai-concierge` label
- Include relevant logs and configuration details

---

**The PRSM AI Investor Concierge represents the future of investor relations - intelligent, scalable, and always available.**