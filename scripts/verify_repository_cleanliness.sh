#!/bin/bash
# Repository Cleanliness Verification Script
# Ensures repository maintains audit-ready standards

set -e

echo "🔍 PRSM Repository Cleanliness Verification"
echo "=========================================="

# Check root directory file count
ROOT_FILE_COUNT=$(find . -maxdepth 1 -type f | wc -l)
echo "📁 Root directory files: $ROOT_FILE_COUNT"

if [ "$ROOT_FILE_COUNT" -gt 30 ]; then
    echo "⚠️  WARNING: Root directory has too many files ($ROOT_FILE_COUNT > 30)"
    echo "   Consider moving non-essential files to subdirectories"
fi

# Check for temporary files (excluding archive directory)
echo ""
echo "🧹 Checking for temporary files..."
TEMP_FILES=$(find . -path "./archive" -prune -o \( -name "*.tmp" -o -name "*.cache" -o -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" \) -print 2>/dev/null | wc -l)

if [ "$TEMP_FILES" -eq 0 ]; then
    echo "✅ No temporary files found"
else
    echo "⚠️  Found $TEMP_FILES temporary files:"
    find . -path "./archive" -prune -o \( -name "*.tmp" -o -name "*.cache" -o -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" \) -print 2>/dev/null | head -5
    echo "   Run: find . -name '.DS_Store' -delete to clean"
fi

# Verify essential files exist
echo ""
echo "📋 Verifying essential files..."
ESSENTIAL_FILES=(
    "README.md"
    "LICENSE"
    "requirements.txt" 
    "setup.py"
    "pyproject.toml"
    "Dockerfile"
    "docker-compose.yml"
    "CHANGELOG.md"
    "SECURITY.md"
    "CODE_OF_CONDUCT.md"
    "CONTRIBUTING.md"
)

MISSING_FILES=0
for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (MISSING)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

# Check documentation structure
echo ""
echo "📚 Verifying documentation structure..."
DOC_DIRS=(
    "docs/"
    "docs/api/"
    "docs/architecture/" 
    "docs/audit/"
    "docs/business/"
    "docs/security/"
)

MISSING_DOCS=0
for dir in "${DOC_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir"
    else
        echo "❌ $dir (MISSING)"
        MISSING_DOCS=$((MISSING_DOCS + 1))
    fi
done

# Check code organization
echo ""
echo "🏗️  Verifying code organization..."
CODE_DIRS=(
    "prsm/"
    "lite_browser/"
    "tests/"
    "scripts/"
    "config/"
    "examples/"
)

MISSING_CODE=0
for dir in "${CODE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir"
    else
        echo "❌ $dir (MISSING)"
        MISSING_CODE=$((MISSING_CODE + 1))
    fi
done

# Summary
echo ""
echo "📊 AUDIT READINESS SUMMARY"
echo "=========================="

TOTAL_ISSUES=$((MISSING_FILES + MISSING_DOCS + MISSING_CODE))

if [ "$ROOT_FILE_COUNT" -le 30 ] && [ "$TEMP_FILES" -eq 0 ] && [ "$TOTAL_ISSUES" -eq 0 ]; then
    echo "🎉 Repository is AUDIT-READY!"
    echo "   ✅ Clean root directory ($ROOT_FILE_COUNT files)"
    echo "   ✅ No temporary files"
    echo "   ✅ All essential files present"
    echo "   ✅ Proper documentation structure"
    echo "   ✅ Organized code structure"
    echo ""
    echo "🚀 Ready for investor/developer review!"
    exit 0
else
    echo "⚠️  Repository needs cleanup:"
    [ "$ROOT_FILE_COUNT" -gt 30 ] && echo "   - Too many root files ($ROOT_FILE_COUNT)"
    [ "$TEMP_FILES" -gt 0 ] && echo "   - $TEMP_FILES temporary files found"
    [ "$MISSING_FILES" -gt 0 ] && echo "   - $MISSING_FILES essential files missing"
    [ "$MISSING_DOCS" -gt 0 ] && echo "   - $MISSING_DOCS documentation directories missing"
    [ "$MISSING_CODE" -gt 0 ] && echo "   - $MISSING_CODE code directories missing"
    echo ""
    echo "Please address these issues before external audit."
    exit 1
fi