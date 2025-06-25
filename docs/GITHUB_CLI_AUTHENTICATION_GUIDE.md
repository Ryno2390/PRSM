# GitHub CLI Authentication Guide

This guide documents the solution for GitHub CLI authentication issues encountered during PRSM development sessions.

## Problem Description

When attempting to push changes to GitHub using `git push` or `gh` commands, you may encounter authentication errors such as:

```
fatal: could not read Username for 'https://github.com': Device not configured
remote: Permission to Ryno2390/PRSM.git denied to Ryno2390.
fatal: unable to access 'https://github.com/Ryno2390/PRSM.git/': The requested URL returned error: 403
```

This typically occurs when:
- Multiple credential helpers are configured in Git
- GitHub CLI authentication has expired
- Git is configured to use HTTPS instead of SSH with proper authentication
- Credential conflicts between different authentication methods
- Token scope limitations or expired tokens

## Troubleshooting Steps Attempted

During our Red Team Safety Monitoring Integration push, we encountered persistent 403 errors despite having valid authentication. Here are the troubleshooting steps we tried:

### ❌ Failed Approaches (For Reference)

1. **Multiple Credential Helper Cleanup**: 
   ```bash
   git config --global --unset-all credential.helper
   git config --global --unset-all credential.username
   gh auth setup-git
   ```
   *Result*: Still had conflicts from system-level credential helpers

2. **Token in URL**: 
   ```bash
   git remote set-url origin https://token@github.com/user/repo.git
   ```
   *Result*: 403 error persisted

3. **Manual Credential Store**: 
   ```bash
   echo "https://user:token@github.com" > ~/.git-credentials
   git config --global credential.helper store
   ```
   *Result*: 403 error persisted

4. **Custom Credential Helper Function**: 
   ```bash
   git config --global credential.helper '!f() { echo "username=user"; echo "password=token"; }; f'
   ```
   *Result*: 403 error persisted

### ✅ Recommended Solution: GitHub CLI Direct Operations

Based on extensive troubleshooting during our Red Team Safety Monitoring Integration, we discovered that traditional Git push methods can encounter persistent authentication issues even with valid tokens. The most reliable approach is to use GitHub CLI directly for repository operations.

## Solution: GitHub CLI Interactive Authentication

### Primary Method: GitHub CLI Direct Operations

When `git push` fails with 403 errors, use GitHub CLI alternatives:

```bash
# Check authentication status
gh auth status

# Verify repository access
gh repo view

# If authentication issues persist, use browser-based auth
gh auth login --hostname github.com --git-protocol https --web
# This will prompt for browser authentication with one-time code

# For pushing changes when git push fails:
# Option 1: Use GitHub CLI repo sync (requires clean working directory)
git stash  # if needed
gh repo sync --force
git stash pop  # if stashed

# Option 2: Create a pull request instead of direct push
git checkout -b feature-branch
gh pr create --title "Your Feature" --body "Description"
```

### Advanced Troubleshooting Method: Commit Verification

If direct push fails, verify your commit is ready and properly formed:

```bash
# Check commit status
git log --oneline -1
git show --name-only

# Verify repository access via API
gh repo view --json viewerPermission

# Check for conflicting credential helpers
git config --get-regexp credential
```

### Step 1: Clear Existing Credential Conflicts

First, check for multiple credential helpers that may be causing conflicts:

```bash
git config --global --get-regexp credential.helper
```

If you see multiple credential helpers, clean them up:

```bash
git config --global --unset-all credential.helper
```

### Step 2: Use GitHub CLI Interactive Authentication

The most reliable method is to use GitHub CLI's interactive authentication:

```bash
gh auth login
```

This command will:
1. Prompt you to choose authentication method (GitHub.com vs GitHub Enterprise)
2. Ask for your preferred protocol (HTTPS vs SSH)
3. Provide authentication options:
   - **Login with a web browser** (Recommended)
   - Paste an authentication token

### Step 3: Choose Web Browser Authentication

When prompted, select **"Login with a web browser"**:

```
? How would you like to authenticate GitHub CLI?
  Login with a web browser
> Paste an authentication token
```

### Step 4: Complete Browser Authentication

1. GitHub CLI will display a one-time code (e.g., `ABCD-1234`)
2. Press Enter to open your default browser
3. If browser doesn't open automatically, navigate to: https://github.com/login/device
4. Enter the one-time code displayed in your terminal
5. Follow the browser prompts to authorize GitHub CLI
6. Return to terminal once authorization is complete

### Step 5: Verify Authentication

Confirm successful authentication:

```bash
gh auth status
```

You should see output similar to:
```
github.com
  ✓ Logged in to github.com as yourusername (oauth_token)
  ✓ Git operations for github.com configured to use https protocol.
```

### Step 6: Test Repository Operations

Test that you can now perform Git operations:

```bash
# Check repository status
gh repo view

# Test pushing changes
git push origin main

# Alternative: Use GitHub CLI for syncing
gh repo sync --force
```

## Alternative Method: Using Personal Access Token

If web browser authentication isn't available:

### Step 1: Create Personal Access Token

1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
2. Generate a new token with appropriate scopes:
   - `repo` (for private repositories)
   - `workflow` (for GitHub Actions)
   - `admin:org` (if working with organization repositories)

### Step 2: Authenticate with Token

```bash
gh auth login --with-token < token.txt
```

Or paste the token when prompted by:
```bash
gh auth login
```

## Troubleshooting Common Issues

### Issue: "gh: command not found"

Install GitHub CLI:

**macOS (Homebrew):**
```bash
brew install gh
```

**Windows:**
```bash
winget install GitHub.cli
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install gh

# CentOS/RHEL
sudo yum install gh
```

### Issue: Git still prompting for credentials

Ensure Git is configured to use GitHub CLI:

```bash
gh auth setup-git
```

### Issue: Multiple Git remotes causing confusion

Check your remote configuration:
```bash
git remote -v
```

Ensure origin points to your GitHub repository:
```bash
git remote set-url origin https://github.com/yourusername/repository.git
```

## Best Practices

1. **Use GitHub CLI for authentication** - More reliable than managing Git credentials manually
2. **Prefer web browser authentication** - Simpler and more secure than token management
3. **Verify authentication status regularly** - Run `gh auth status` before starting work sessions
4. **Keep GitHub CLI updated** - Newer versions have better authentication handling
5. **Document your authentication method** - Consistent approach across development sessions
6. **Use GitHub CLI alternatives when git push fails** - `gh repo sync` or pull requests as backup methods
7. **Clear credential conflicts proactively** - Multiple credential helpers can cause persistent issues

## Production Experience: Red Team Integration Case Study

During our Red Team Safety Monitoring Integration deployment, we encountered persistent 403 authentication errors despite:
- ✅ Valid GitHub CLI authentication (`gh auth status` showed logged in)
- ✅ Valid personal access token with all required scopes
- ✅ Admin permissions on the repository (`gh repo view --json viewerPermission` confirmed)
- ✅ Successful API access to the repository

**Root Cause**: Multiple conflicting credential helpers at system level that couldn't be fully cleared through standard Git configuration commands.

**Solution Applied**: After extensive troubleshooting, we discovered the **fundamental fix**: switching from HTTPS to SSH authentication completely bypasses credential helper conflicts.

## ✅ **FUNDAMENTAL FIX: SSH Authentication**

**The Ultimate Solution**: Switch to SSH authentication to completely bypass credential helper conflicts.

### Quick SSH Setup (Recommended)

```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_github -N ""

# 2. Display public key (copy this)
cat ~/.ssh/id_github.pub

# 3. Add to GitHub: Settings → SSH and GPG keys → New SSH key

# 4. Configure SSH
echo "Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_github
  IdentitiesOnly yes" >> ~/.ssh/config

chmod 600 ~/.ssh/config ~/.ssh/id_github

# 5. Switch remote to SSH
git remote set-url origin git@github.com:username/repository.git

# 6. Test connection
ssh -T git@github.com

# 7. Push works perfectly!
git push origin main
```

**Why SSH is the Fundamental Fix:**
- ✅ Completely bypasses Git credential helpers
- ✅ No conflicts between osxkeychain, GitHub CLI, and other systems
- ✅ More secure and reliable for development
- ✅ Once set up, authentication is seamless
- ✅ **Proven solution** - Successfully deployed Red Team Safety Monitoring Integration

### Success Story
After implementing SSH authentication, our Red Team Safety Monitoring Integration pushed successfully on first try, resolving persistent 403 authentication errors that couldn't be fixed through credential helper management.

## Quick Reference Commands

```bash
# Check authentication status
gh auth status

# Re-authenticate if needed
gh auth login

# Setup Git integration
gh auth setup-git

# View repository information
gh repo view

# Sync repository (alternative to git push)
gh repo sync --force

# Check remote configuration
git remote -v
```

## Session Workflow

For each new coding session:

1. Check authentication: `gh auth status`
2. If authentication expired: `gh auth login`
3. Verify repository access: `gh repo view`
4. Proceed with development work
5. Commit and push changes normally

This workflow ensures consistent, reliable GitHub operations across all development sessions.

---

**Created:** December 18, 2024  
**Last Updated:** December 18, 2024  
**Status:** Verified working solution for PRSM development