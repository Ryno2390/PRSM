# PyPI Publishing Guide for PRSM

This document provides comprehensive instructions for publishing `prsm-network` to TestPyPI (for testing) and PyPI (for production).

## Package Information

| Property | Value |
|----------|-------|
| **PyPI Distribution Name** | `prsm-network` |
| **Import Name** | `prsm` |
| **CLI Command** | `prsm` |
| **Current Version** | `0.2.0` |

> **Note:** The package name `prsm` is already taken on PyPI by another project. We use `prsm-network` as the distribution name, but the import name remains `prsm` (i.e., `import prsm` still works).

---

## 1. Prerequisites

### 1.1 System Requirements

- **Python 3.10+** (required for modern packaging features)
- **pip** (latest version recommended)
- **git** (for version control)

### 1.2 Install Required Tools

```bash
# Install build and twine for package building and uploading
pip install build twine

# Verify installations
python -m build --version
twine --version
```

### 1.3 Create Accounts

You need accounts on both TestPyPI (for testing) and PyPI (for production):

| Service | Purpose | Registration URL |
|---------|---------|------------------|
| **TestPyPI** | Testing package uploads | https://test.pypi.org/account/register/ |
| **PyPI** | Production package hosting | https://pypi.org/account/register/ |

### 1.4 Generate API Tokens

API tokens are the recommended authentication method for PyPI/TestPyPI:

#### TestPyPI Token
1. Log in to https://test.pypi.org/
2. Navigate to **Account settings** → **API tokens**
3. Click **Add API token**
4. Set **Token name**: `prsm-network-publishing`
5. Set **Scope**: "Entire account" (for first upload) or select specific project
6. Click **Create token**
7. **Copy the token immediately** - it starts with `pypi-` and won't be shown again

#### PyPI Token
1. Log in to https://pypi.org/
2. Follow the same steps as above
3. Save the token securely

> **Security Note:** Store API tokens securely. Consider using a password manager or secure environment variables. Never commit tokens to version control.

---

## 2. Building the Package

### 2.1 Clean Old Artifacts

Before building, remove any old distribution files to ensure a clean build:

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info

# Also clean any __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

### 2.2 Build New Package

```bash
# Build the package (creates both wheel and source distribution)
python -m build
```

This command creates:
- A wheel distribution (`.whl` file) - pre-built binary format
- A source distribution (`.tar.gz` file) - source code archive

### 2.3 Verify Build Artifacts

```bash
# List the built artifacts
ls -la dist/

# Expected output:
# prsm_network-0.2.0-py3-none-any.whl
# prsm_network-0.2.0.tar.gz
```

### 2.4 Validate Package Metadata

```bash
# Check the package metadata and contents
twine check dist/*

# This verifies:
# - Package metadata is valid
# - README renders correctly
# - No common packaging issues
```

---

## 3. TestPyPI Publication (Testing)

TestPyPI is a separate instance of PyPI for testing package uploads before publishing to production.

### 3.1 Configure Credentials

#### Option A: Using `~/.pypirc` (Recommended)

Create or edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
username = __token__
password = <your-testpypi-token>
repository = https://test.pypi.org/legacy/
```

Set restrictive permissions:
```bash
chmod 600 ~/.pypirc
```

#### Option B: Using Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-token>
```

#### Option C: Interactive Prompt

Twine will prompt for credentials if not configured:
```bash
# Username: __token__
# Password: <your-testpypi-token>
```

### 3.2 Upload to TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Expected output:
# Uploading prsm_network-0.2.0-py3-none-any.whl
# Uploading prsm_network-0.2.0.tar.gz
# View at: https://test.pypi.org/project/prsm-network/0.2.0/
```

### 3.3 Verify Installation from TestPyPI

```bash
# Create a fresh virtual environment for testing
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI (with fallback to PyPI for dependencies)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    prsm-network

# Verify the installation
pip show prsm-network

# Test the CLI
prsm --version
# Expected output: prsm, version 0.2.0

# Test the import
python -c "import prsm; print(prsm.__version__)"
# Expected output: 0.2.0

# Clean up
deactivate
rm -rf test_env
```

---

## 4. PyPI Publication (Production)

Once testing is successful on TestPyPI, publish to production PyPI.

### 4.1 Final Pre-Publication Checklist

- [ ] Version number is correct in `pyproject.toml`
- [ ] CHANGELOG.md is updated with release notes
- [ ] All tests pass locally
- [ ] TestPyPI installation verified successfully
- [ ] README.md renders correctly on TestPyPI

### 4.2 Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Expected output:
# Uploading prsm_network-0.2.0-py3-none-any.whl
# Uploading prsm_network-0.2.0.tar.gz
# View at: https://pypi.org/project/prsm-network/0.2.0/
```

### 4.3 Verify Installation from PyPI

```bash
# Create a fresh virtual environment for testing
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from production PyPI
pip install prsm-network

# Verify the installation
pip show prsm-network

# Test the CLI
prsm --version
# Expected output: prsm, version 0.2.0

# Test the import
python -c "import prsm; print(prsm.__version__)"
# Expected output: 0.2.0

# Clean up
deactivate
rm -rf test_env
```

---

## 5. GitHub Actions Release (Automated)

For automated publishing via GitHub Actions, you can trigger publication by creating a GitHub release.

### 5.1 Trusted Publishing (Recommended)

Trusted publishing uses OpenID Connect (OIDC) for secure, credential-free authentication.

#### Setup

1. Go to PyPI → **Publishing settings** for `prsm-network`
2. Add a trusted publisher:
   - **PyPI Project**: `prsm-network`
   - **Owner**: Your GitHub username or organization
   - **Repository**: `PRSM`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi` (optional)

3. Repeat for TestPyPI if desired

#### Workflow Configuration

Create `.github/workflows/release.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read
  id-token: write  # Required for trusted publishing

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to TestPyPI (for pre-releases)
        if: github.event.release.prerelease
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

      - name: Publish to PyPI (for releases)
        if: "!github.event.release.prerelease"
        uses: pypa/gh-action-pypi-publish@release/v1
```

### 5.2 Using GitHub Secrets (Alternative)

If trusted publishing is not available, use GitHub secrets:

#### Setup

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI API token
   - `PYPI_API_TOKEN`: Your PyPI API token

#### Workflow Configuration

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to TestPyPI (for pre-releases)
        if: github.event.release.prerelease
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
        run: twine upload --repository testpypi dist/*

      - name: Publish to PyPI (for releases)
        if: "!github.event.release.prerelease"
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

### 5.3 Triggering a Release

1. Go to your repository → **Releases** → **Draft a new release**
2. Enter a tag version (e.g., `v0.2.0`)
3. Enter a release title
4. Write release notes
5. For testing: check **Set as a pre-release**
6. Click **Publish release**

The workflow will automatically:
- Build the package
- Upload to TestPyPI (for pre-releases)
- Upload to PyPI (for full releases)

---

## 6. Troubleshooting

### 6.1 Common Errors

#### "403 Forbidden" Error

**Cause:** Authentication failed or insufficient permissions.

**Solutions:**
1. Verify your API token is correct and hasn't expired
2. Ensure `username` is set to `__token__` (not your username)
3. Check that the token has the correct scope (project or entire account)
4. Regenerate the token if necessary

```bash
# Test credentials
twine upload --repository testpypi --verbose dist/*
```

#### "400 File already exists" Error

**Cause:** This version has already been uploaded.

**Solutions:**
1. Increment the version number in `pyproject.toml`
2. Rebuild the package: `rm -rf dist/ && python -m build`
3. Upload again

> **Note:** PyPI does not allow overwriting existing versions. Each version is immutable.

#### "400 Invalid classifier" Error

**Cause:** A classifier in `pyproject.toml` is not recognized.

**Solution:** Check the list of valid classifiers at https://pypistats.org/classifiers/

#### "Invalid distribution" Error

**Cause:** Package structure is incorrect.

**Solutions:**
1. Verify `pyproject.toml` is valid
2. Check that required fields are present: `name`, `version`, `description`
3. Run `twine check dist/*` before uploading

#### Package Name Conflict

**Cause:** The package name is already taken.

**Solutions:**
1. Choose a different package name (we use `prsm-network` instead of `prsm`)
2. Update `pyproject.toml`:
   ```toml
   [project]
   name = "prsm-network"
   ```
3. Rebuild and re-upload

### 6.2 Regenerating API Tokens

If you need to regenerate an API token:

1. Log in to PyPI or TestPyPI
2. Go to **Account settings** → **API tokens**
3. Find the token you want to regenerate
4. Click **Remove** to delete it
5. Create a new token following the steps in Section 1.4
6. Update your `~/.pypirc` or environment variables

### 6.3 Verifying Package Contents

To inspect what's in your package before uploading:

```bash
# List wheel contents
unzip -l dist/prsm_network-0.2.0-py3-none-any.whl

# List source distribution contents
tar -tvf dist/prsm_network-0.2.0.tar.gz
```

### 6.4 Testing Locally Before Upload

```bash
# Install the wheel locally
pip install dist/prsm_network-0.2.0-py3-none-any.whl

# Test the installation
prsm --version
python -c "import prsm; print(prsm.__version__)"

# Uninstall when done
pip uninstall prsm-network
```

### 6.5 Getting Help

- **PyPI Documentation**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Python Packaging Guide**: https://packaging.python.org/en/latest/tutorials/packaging-projects/

---

## 7. Quick Reference

### Build Commands
```bash
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
```

### TestPyPI Commands
```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ prsm-network
prsm --version
```

### PyPI Commands
```bash
twine upload dist/*
pip install prsm-network
prsm --version
```

---

## 8. Summary

| Step | Command | Status |
|------|---------|--------|
| Clean artifacts | `rm -rf dist/ build/ *.egg-info` | Required |
| Build package | `python -m build` | Required |
| Validate package | `twine check dist/*` | Recommended |
| Upload to TestPyPI | `twine upload --repository testpypi dist/*` | Testing |
| Verify TestPyPI | `pip install --index-url ... prsm-network` | Testing |
| Upload to PyPI | `twine upload dist/*` | Production |
| Verify PyPI | `pip install prsm-network` | Production |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-06 | Initial comprehensive documentation |
