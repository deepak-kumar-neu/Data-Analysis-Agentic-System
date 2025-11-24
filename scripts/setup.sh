#!/bin/bash

# Setup script for Data Analysis Agentic System
# This script sets up the development environment

set -e  # Exit on error

echo "=========================================="
echo "Data Analysis Agentic System - Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing package in development mode..."
pip install -e .

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created - Please edit it with your API keys"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/sample data/cache data/test
mkdir -p results/logs results/reports results/visualizations
mkdir -p docs/images
echo "✓ Directories created"

# Run tests to verify installation
echo ""
echo "Running tests to verify installation..."
if pytest tests/ -v --tb=short 2>/dev/null; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed or no tests found yet"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the system: python src/main.py --help"
echo "4. Run tests: pytest tests/ -v"
echo ""
echo "For more information, see README.md"
echo ""
