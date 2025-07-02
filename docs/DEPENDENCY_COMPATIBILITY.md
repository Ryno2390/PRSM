# Dependency Compatibility Guide

## Known Compatible Versions

This document maintains known compatible versions for critical dependency chains that have version conflicts.

### Web3 + Ethereum Stack

**Issue**: `web3`, `eth-account`, and `ckzg` have conflicting version requirements
- `web3` requires `ckzg<2.0.0`
- `eth-account>=0.13.0` requires `ckzg>=2.0.0`

**✅ Compatible Combination (PRODUCTION READY):**
```
# Generated with pip-compile on Python 3.13
web3==7.12.0
eth-account==0.13.7
eth-abi==5.2.0
eth-hash==0.7.1
eth-keyfile==0.8.1
eth-keys==0.7.0
eth-rlp==2.2.0
eth-typing==5.2.1
eth-utils==5.3.0
parsimonious==0.10.0  # Now Python 3.13 compatible
sphinx==8.2.3
```

**Final Resolution:**
- **parsimonious==0.10.0** now works with Python 3.13 (getargspec issue fixed)
- **pip-compile** successfully resolved all dependencies without conflicts
- **web3>=6.0.0** constraint in requirements.in ensures modern web3 stack
- **sphinx==8.2.3** included for documentation building
- All packages verified compatible with Python 3.13

**Installation Test:**
```bash
python test_ethereum_install.py
```

**Version History:**
- `web3==6.20.4` + `eth-account==0.13.7` + `ckzg==2.1.1` ❌ CONFLICT (ckzg version incompatible)
- `web3==6.15.1` + `eth-account==0.12.3` + `ckzg==1.0.6` ❌ CONFLICT (ckzg==1.0.6 doesn't exist)
- `web3==6.11.3` + `eth-account==0.11.2` + `ckzg==1.0.0` ❌ CONFLICT (ckzg==1.0.0 doesn't exist)
- `web3==6.5.4` + `eth-account==0.8.0` + no ckzg ❌ CONFLICT (web3==6.5.4 doesn't exist)
- `web3==5.31.3` + `eth-account==0.5.9` + no ckzg ✅ COMPATIBLE (verified stable)

### Scientific Computing Stack

**For macOS users with compilation issues:**
```bash
# Use conda for pre-compiled packages
conda install numpy scipy scikit-learn
pip install -r requirements.txt
```

**Compatible versions:**
```
numpy==2.3.1
scipy==1.16.0
scikit-learn==1.6.1
```

### Testing Before Release

Always test dependency installations:
```bash
# Create clean environment
python -m venv test_env
source test_env/bin/activate

# Test installation
pip install -r requirements.txt

# Verify critical imports
python -c "import web3; import eth_account; print('Ethereum stack OK')"
python -c "import numpy; import scipy; import sklearn; print('Scientific stack OK')"
```

## Dependency Conflict Resolution Process

1. **Identify conflict** using `pip-tools` or manual testing
2. **Research compatible versions** on PyPI dependency graphs
3. **Test in clean environment** 
4. **Update this documentation**
5. **Add to CI/CD testing**

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Test dependency installation
  run: |
    pip install -r requirements.txt
    python -c "import web3; import eth_account"
```