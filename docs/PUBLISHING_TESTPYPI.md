# TestPyPI Publishing Guide

## Task 4: TestPyPI Dry Run Results

### Upload Status: FAILED (Credentials Required)

The TestPyPI upload attempt failed due to missing API credentials. Twine prompted for an API token but could not read it from the non-interactive terminal.

### Build Artifacts Verified ✓

The following distribution files are ready for upload:
- `dist/prsm-0.2.0-py3-none-any.whl` (97.9 MB)
- `dist/prsm-0.2.0.tar.gz` (96.5 MB)

---

## Package Name Availability

### PyPI (Production)
| Package Name | Status |
|--------------|--------|
| `prsm` | **TAKEN** - Returns 200 from HTML endpoint |
| `prsm-network` | **AVAILABLE** - Returns "Not Found" from JSON API |

### TestPyPI
| Package Name | Status |
|--------------|--------|
| `prsm` | **UNCERTAIN** - Behind client challenge protection |
| `prsm-network` | **LIKELY AVAILABLE** |

**Recommendation:** The `prsm` package name is already taken on PyPI. Consider using `prsm-network` or another unique name for publishing.

---

## How to Set Up TestPyPI Credentials

### Step 1: Create a TestPyPI Account

1. Visit https://test.pypi.org/account/register/
2. Fill in your details and verify your email

### Step 2: Create an API Token

1. Log in to TestPyPI
2. Go to https://test.pypi.org/manage/account/token/
3. Click "Create API token"
4. Set a descriptive name (e.g., "PRSM Publishing Token")
5. Select scope: "Entire account" (for first upload)
6. Copy the token immediately - it starts with `pypi-`

### Step 3: Configure Credentials

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

**Security Note:** Set restrictive permissions:
```bash
chmod 600 ~/.pypirc
```

#### Option B: Using Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-token>
```

#### Option C: Command Line (Not Recommended for Security)

```bash
twine upload --repository testpypi \
  --username __token__ \
  --password <your-testpypi-token> \
  dist/*
```

---

## Publishing Commands

### TestPyPI (Dry Run)

```bash
# With ~/.pypirc configured:
twine upload --repository testpypi dist/*

# With environment variables:
twine upload --repository testpypi dist/*
```

### Verify TestPyPI Upload

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ prsm

# Or with pipx for testing:
pipx install --index-url https://test.pypi.org/simple/ prsm
```

### Production PyPI

```bash
# With ~/.pypirc configured:
twine upload dist/*

# Or explicitly:
twine upload --repository pypi dist/*
```

---

## GitHub Actions CI/CD Integration

For automated publishing via GitHub Actions, use trusted publishing or secrets:

### Option A: Trusted Publishing (Recommended)

1. Go to PyPI/TestPyPI → Publishing settings
2. Add a trusted publisher for your GitHub repository
3. Configure workflow:

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
```

### Option B: GitHub Secrets

Add secrets to your repository:
- `TEST_PYPI_API_TOKEN` - TestPyPI API token
- `PYPI_API_TOKEN` - Production PyPI API token

```yaml
- name: Publish to TestPyPI
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
  run: twine upload --repository testpypi dist/*
```

---

## Troubleshooting

### "403 Forbidden" Error
- Verify your API token is correct
- Ensure token has proper scope
- Check that `username` is set to `__token__`

### "400 File already exists" Error
- Version already uploaded to PyPI
- Increment version in `pyproject.toml` and rebuild

### Package Name Conflict
- The name `prsm` is taken on PyPI
- Use `prsm-network` or another unique name
- Update `pyproject.toml` with new name before building

---

## Next Steps

1. **Create TestPyPI account** at https://test.pypi.org/account/register/
2. **Generate API token** at https://test.pypi.org/manage/account/token/
3. **Configure credentials** in `~/.pypirc` or environment variables
4. **Re-run upload**: `twine upload --repository testpypi dist/*`
5. **Verify installation** from TestPyPI
6. **Decide on package name** - `prsm` is taken, consider `prsm-network`

---

## Summary

| Item | Status |
|------|--------|
| Build artifacts | ✓ Ready |
| Twine installed | ✓ Version 6.2.0 |
| TestPyPI credentials | ✗ Not configured |
| Package name `prsm` | ✗ Taken on PyPI |
| Package name `prsm-network` | ✓ Available |
| Upload to TestPyPI | ⏸ Blocked by credentials |
