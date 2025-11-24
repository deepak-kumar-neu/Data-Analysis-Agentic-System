#!/bin/bash

# Quick Start Script for Data Analysis Agentic System
# This script helps you get started quickly with the system

echo "=================================="
echo "Data Analysis Agentic System"
echo "Quick Start Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"

if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9 or higher."
    exit 1
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅ Dependencies installed"

echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p results
mkdir -p logs
echo "✅ Directories created"

echo ""

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created (please edit with your API keys)"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "You can now run the system in several ways:"
echo ""
echo "1. Simple Example:"
echo "   python examples/simple_example.py"
echo ""
echo "2. Parallel Execution Example:"
echo "   python examples/simple_example.py parallel"
echo ""
echo "3. Custom AI Tool Example:"
echo "   python examples/simple_example.py custom"
echo ""
echo "4. Full CLI:"
echo "   python src/main.py --data-source data/your_file.csv --objective 'Your analysis goal'"
echo ""
echo "For more options:"
echo "   python src/main.py --help"
echo ""
echo "=================================="
