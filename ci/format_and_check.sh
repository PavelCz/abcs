#!/usr/bin/env bash

# Format and check script for ACS library
# This script formats code and runs linting and type checking

# Source files to check
SRC_FILES=(
    "src/acs/"
    "tests/"
    # "examples/"
)
EXCLUDED_FILES=("")

set -x  # echo commands
set -e  # quit immediately on error

echo "=== ACS Code Quality Checks ==="

echo "1. Formatting code with ruff..."
# Run ruff as formatter (black-ish and isort-ish)
ruff format "${SRC_FILES[@]}" --exclude "${EXCLUDED_FILES[@]}"

echo "2. Linting and fixing code with ruff..."
# Run ruff as linter (flake8-ish) with auto-fix
ruff check "${SRC_FILES[@]}" --exclude "${EXCLUDED_FILES[@]}" --fix

echo "3. Type checking with pytype..."
# Run pytype with suppressed debug logging
pytype src/acs/*.py --verbosity=0 2>/dev/null

echo "✅ All checks passed!"