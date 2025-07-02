# Convert Legacy Test Scripts to Pytest Format

**Labels**: testing, good first issue, intermediate, help wanted, priority: low

---
**Type**: Testing
**Difficulty**: ⭐⭐ (2/4 stars)
**Estimated Time**: 3-5 hours
**Skills Needed**: Python, pytest, Testing, Code Migration
**Impact**: 🔥 Medium
**Mentorship Available**: Yes ✅

## 📝 Description

Several test files in the PRSM codebase use legacy testing formats and need conversion to proper pytest structure. This will improve test consistency, maintainability, and integration with our CI/CD pipeline.

**Files to Convert**:
- `tests/test_dashboard.py` - Currently uses print statements instead of assertions
- `tests/standalone_pq_test.py` - Needs conversion to pytest structure
- Other legacy test files identified in the codebase

**Why This Matters**: Consistent testing infrastructure is crucial for code quality. Legacy test formats make it harder to run tests, get proper reporting, and integrate with modern tooling.

## 🚀 Getting Started

### Prerequisites
- [ ] Read the [Contributing Guide](CONTRIBUTING.md)
- [ ] Set up development environment following [Development Setup](docs/DEVELOPMENT_SETUP.md)
- [ ] Join our [Discord community](https://discord.gg/prsm-ai) for support

### Files to Modify
- `tests/test_dashboard.py`
- `tests/standalone_pq_test.py`
- `tests/README.md`

## ✅ Acceptance Criteria

- [ ] All identified legacy tests converted to proper pytest format
- [ ] Print statements replaced with proper assertions and test reporting
- [ ] Tests integrated into main test suite and CI pipeline
- [ ] All converted tests pass consistently
- [ ] Follow pytest naming conventions and best practices
- [ ] Update test documentation and runner instructions

## 📚 Learning Opportunities

- pytest framework features and conventions
- Test migration strategies and best practices
- CI/CD integration for automated testing
- Code quality improvement and technical debt reduction
- Testing infrastructure design and maintenance

## 🤝 Ready to Start?

**Comment below to claim this issue!** We'll assign it to you and provide guidance.

### New to Open Source?
This issue is perfect for newcomers! Check out:
- [How to contribute to open source](https://opensource.guide/how-to-contribute/)
- [PRSM Contributor Onboarding Guide](docs/CONTRIBUTOR_ONBOARDING.md)
- [Good First Issues Guide](docs/CURATED_GOOD_FIRST_ISSUES.md)

### Need Help?
- 💬 Ask questions in [GitHub Discussions](https://github.com/PRSM-AI/prsm/discussions)
- 🆘 Join our [Discord #help channel](https://discord.gg/prsm-ai)
- 📧 Email: contributors@prsm.ai

---

*This issue was created from our [Curated Good First Issues](docs/CURATED_GOOD_FIRST_ISSUES.md) guide.*
