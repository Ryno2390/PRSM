#!/bin/bash

# PRSM Go SDK Test Runner
# Runs comprehensive tests for the Go SDK

set -e

echo "PRSM Go SDK Test Runner"
echo "======================="
echo

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.19 or later."
    exit 1
fi

# Check Go version
GO_VERSION=$(go version | cut -d' ' -f3 | cut -d'o' -f2)
echo "✓ Go version: $GO_VERSION"

# Check if we're in the right directory
if [ ! -f "go.mod" ]; then
    echo "❌ go.mod not found. Please run this script from the Go SDK root directory."
    exit 1
fi

echo "✓ Found go.mod file"
echo

# Download dependencies
echo "📦 Downloading dependencies..."
go mod tidy
go mod download
echo "✓ Dependencies downloaded"
echo

# Run linting (if golangci-lint is available)
if command -v golangci-lint &> /dev/null; then
    echo "🔍 Running linter..."
    golangci-lint run
    echo "✓ Linting passed"
    echo
else
    echo "⚠️  golangci-lint not found. Skipping linting."
    echo
fi

# Run tests
echo "🧪 Running tests..."
echo

# Run tests with coverage
go test -v -cover ./...

# Check if tests passed
if [ $? -eq 0 ]; then
    echo
    echo "✅ All tests passed!"
else
    echo
    echo "❌ Some tests failed!"
    exit 1
fi

# Run race condition detection
echo
echo "🏁 Running race condition detection..."
go test -race ./...

if [ $? -eq 0 ]; then
    echo "✅ No race conditions detected!"
else
    echo "❌ Race conditions detected!"
    exit 1
fi

# Build examples to ensure they compile
echo
echo "🔨 Building examples..."

cd examples
for example in *.go; do
    if [ -f "$example" ]; then
        echo "  Building $example..."
        go build -o /tmp/$(basename "$example" .go) "$example"
        if [ $? -eq 0 ]; then
            echo "  ✓ $example compiled successfully"
        else
            echo "  ❌ $example failed to compile"
            exit 1
        fi
    fi
done

cd ..
echo "✅ All examples compiled successfully!"

# Generate test coverage report
echo
echo "📊 Generating coverage report..."
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

echo "✓ Coverage report generated: coverage.html"

# Run benchmarks if they exist
echo
echo "⚡ Running benchmarks..."
go test -bench=. -benchmem ./...

echo
echo "🎉 All tests and checks completed successfully!"
echo
echo "Summary:"
echo "- Tests: ✅ Passed"
echo "- Race detection: ✅ Passed"
echo "- Examples: ✅ Compiled"
echo "- Coverage: 📊 Generated"
echo "- Benchmarks: ⚡ Completed"
echo
echo "You can view the coverage report by opening coverage.html in your browser."