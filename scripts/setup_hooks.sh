#!/bin/bash

# Setup script for git hooks
# This script installs the pre-commit hook to enforce Black formatting

set -e  # Exit on any error

echo "🔧 Setting up git hooks for torch-point-ops..."
echo ""
echo "📝 Choose your setup method:"
echo "  1) Simple git hook (basic, local only)"
echo "  2) Pre-commit framework (recommended for teams)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
  2)
    echo "🚀 Setting up pre-commit framework..."
    
    # Check if pre-commit is installed
    if ! command -v pre-commit &> /dev/null; then
        echo "Installing pre-commit..."
        pip install pre-commit
    fi
    
    # Install the pre-commit hooks
    pre-commit install
    
    echo "✅ Pre-commit framework installed!"
    echo ""
    echo "🎯 This will now run on every commit:"
    echo "   • Black formatting"
    echo "   • Ruff linting" 
    echo "   • Various code quality checks"
    echo ""
    echo "💡 To run manually: pre-commit run --all-files"
    exit 0
    ;;
  1)
    echo "🔧 Installing simple git hook..."
    ;;
  *)
    echo "❌ Invalid choice. Defaulting to simple git hook..."
    ;;
esac

# Get the root directory of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"
SCRIPTS_DIR="$REPO_ROOT/scripts"

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "❌ This is not a git repository!"
    exit 1
fi

# Check if the pre-commit hook source exists
if [ ! -f "$SCRIPTS_DIR/pre-commit.hook" ]; then
    echo "❌ Pre-commit hook source not found at $SCRIPTS_DIR/pre-commit.hook"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Copy the pre-commit hook
cp "$SCRIPTS_DIR/pre-commit.hook" "$HOOKS_DIR/pre-commit"

# Make it executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "🎯 The hook will now:"
echo "   • Check Python files with Black before each commit"
echo "   • Auto-format files if needed"
echo "   • Prevent commits with formatting issues"
echo ""
echo "💡 To test the hook:"
echo "   # Make some changes to a Python file"
echo "   git add file.py"
echo "   git commit -m 'test'"
echo ""
echo "🚀 Happy coding!" 